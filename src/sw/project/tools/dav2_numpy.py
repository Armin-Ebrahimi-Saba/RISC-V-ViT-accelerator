# SPDX-License-Identifier: CC0-1.0
# SPDX-FileCopyrightText: 2026 RVLab Student Project
"""
NumPy reimplementation of Depth-Anything V2 Small (ViT-S/14 + DPT head).

This is the blueprint for the C engine that runs on the CV32E40P: every
operation here has a one-to-one counterpart in dav2_ops.c / dav2_engine.c.

Two modes:
  * quant=False -- pure float32, used to prove the reimplementation matches
    the PyTorch reference.
  * quant=True  -- "fake quantisation" is inserted at exactly the points where
    the C engine quantises, so we can measure end-to-end int8 accuracy before
    committing to the C implementation.

Quantisation scheme (mirrors the C engine):
  * weights: per-output-channel symmetric int8
  * activations: per-tensor symmetric int8, scale computed dynamically from the
    actual tensor maximum (no offline calibration needed)
  * accumulation: int32
"""
import numpy as np

# ---------------------------------------------------------------------------
# quantisation primitives
# ---------------------------------------------------------------------------


def qscale(amax, bits=8):
    """Symmetric scale for a given absolute maximum."""
    qmax = (1 << (bits - 1)) - 1
    if amax <= 0:
        return 1.0
    return float(amax) / qmax


def fake_quant_tensor(x, bits=8):
    """Per-tensor symmetric fake quantisation (what the C engine stores)."""
    s = qscale(np.abs(x).max(), bits)
    qmax = (1 << (bits - 1)) - 1
    q = np.clip(np.rint(x / s), -qmax, qmax)
    return q * s


def fake_quant_perchannel(w, axis=0, bits=8):
    """Per-output-channel symmetric fake quantisation of a weight tensor."""
    ax = tuple(i for i in range(w.ndim) if i != axis)
    amax = np.abs(w).max(axis=ax, keepdims=True)
    qmax = (1 << (bits - 1)) - 1
    s = np.where(amax > 0, amax / qmax, 1.0)
    return np.clip(np.rint(w / s), -qmax, qmax) * s


class Q:
    """Quantisation policy shared by the whole model."""

    def __init__(self, enabled=True, act_bits=8, w_bits=8):
        self.enabled = enabled
        self.act_bits = act_bits
        self.w_bits = w_bits

    def act(self, x):
        return fake_quant_tensor(x, self.act_bits) if self.enabled else x

    def w(self, w, axis=0):
        return fake_quant_perchannel(w, axis, self.w_bits) if self.enabled else w


# ---------------------------------------------------------------------------
# elementary ops
# ---------------------------------------------------------------------------


def linear(x, w, b, q):
    """x: (N, Cin), w: (Cout, Cin), b: (Cout,) -> (N, Cout)"""
    y = q.act(x) @ q.w(w, axis=0).T
    if b is not None:
        y = y + b
    return y


def layernorm(x, w, b, eps=1e-6):
    """LayerNorm over the last axis. Kept in higher precision in the C engine
    too (it is O(N*C) and numerically sensitive)."""
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x):
    """Exact erf GELU, matching torch.nn.GELU's default."""
    from scipy.special import erf  # optional; fall back below
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def gelu_np(x):
    """erf-based GELU without scipy (uses the tanh-free erf approximation of
    Abramowitz & Stegun 7.1.26, accurate to ~1.5e-7 -- far below int8 noise).
    The C engine uses a 256-entry LUT, which is exact for int8 inputs."""
    z = x / np.sqrt(2.0)
    sign = np.sign(z)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return 0.5 * x * (1.0 + sign * y)


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def conv2d(x, w, b, stride=1, pad=0, q=None):
    """x: (Cin, H, W), w: (Cout, Cin, kh, kw) -> (Cout, Ho, Wo). im2col + GEMM,
    exactly how the C engine does it."""
    Cin, H, W = x.shape
    Cout, _, kh, kw = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Hp, Wp = x.shape[1], x.shape[2]
    Ho = (Hp - kh) // stride + 1
    Wo = (Wp - kw) // stride + 1
    cols = np.empty((Ho * Wo, Cin * kh * kw), dtype=np.float32)
    idx = 0
    for oy in range(Ho):
        for ox in range(Wo):
            patch = x[:, oy * stride:oy * stride + kh, ox * stride:ox * stride + kw]
            cols[idx] = patch.reshape(-1)
            idx += 1
    wm = w.reshape(Cout, -1)
    if q is not None:
        cols = q.act(cols)
        wm = q.w(wm, axis=0)
    y = cols @ wm.T
    if b is not None:
        y = y + b
    return y.T.reshape(Cout, Ho, Wo)


def conv_transpose2d(x, w, b, stride, q=None):
    """Non-overlapping transposed convolution (kernel == stride), which is the
    only case DAv2's resize_layers need. w: (Cin, Cout, k, k)."""
    Cin, H, W = x.shape
    _, Cout, kh, kw = w.shape
    assert kh == stride and kw == stride, "only non-overlapping case supported"
    wm = w.reshape(Cin, Cout * kh * kw)
    if q is not None:
        x = q.act(x)
        wm = q.w(wm, axis=0)
    # (H*W, Cin) @ (Cin, Cout*kh*kw)
    flat = x.reshape(Cin, H * W).T @ wm
    flat = flat.reshape(H, W, Cout, kh, kw)
    y = flat.transpose(2, 0, 3, 1, 4).reshape(Cout, H * kh, W * kw)
    if b is not None:
        y = y + b[:, None, None]
    return y


def interpolate_bilinear(x, out_h, out_w, align_corners=True):
    """x: (C, H, W) -> (C, out_h, out_w), matching F.interpolate."""
    C, H, W = x.shape
    if align_corners:
        sy = (H - 1) / (out_h - 1) if out_h > 1 else 0.0
        sx = (W - 1) / (out_w - 1) if out_w > 1 else 0.0
        ys = np.arange(out_h) * sy
        xs = np.arange(out_w) * sx
    else:
        sy, sx = H / out_h, W / out_w
        ys = np.clip((np.arange(out_h) + 0.5) * sy - 0.5, 0, None)
        xs = np.clip((np.arange(out_w) + 0.5) * sx - 0.5, 0, None)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, H - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[None, :, None]
    wx = (xs - x0)[None, None, :]
    a = x[:, y0][:, :, x0]
    b = x[:, y0][:, :, x1]
    c = x[:, y1][:, :, x0]
    d = x[:, y1][:, :, x1]
    top = a * (1 - wx) + b * wx
    bot = c * (1 - wx) + d * wx
    return top * (1 - wy) + bot * wy


def relu(x):
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

EMBED_DIM = 384
N_HEADS = 6
HEAD_DIM = EMBED_DIM // N_HEADS
N_BLOCKS = 12
PATCH = 14
INTERMEDIATE = [2, 5, 8, 11]
OUT_CH = [48, 96, 192, 384]
FEATURES = 64


class DAv2Numpy:
    def __init__(self, sd, pos_embed, quant=True, act_bits=8, w_bits=8):
        """sd: dict of numpy arrays (the checkpoint). pos_embed: (1+P, 384)
        already interpolated to the target resolution."""
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in sd.items()}
        self.pos_embed = pos_embed.astype(np.float32)
        self.q = Q(quant, act_bits, w_bits)

    def g(self, name):
        return self.p[name]

    # -- encoder ----------------------------------------------------------
    def patch_embed(self, img):
        """img: (3, H, W) normalised -> (P, 384) patch tokens."""
        w = self.g("pretrained.patch_embed.proj.weight")
        b = self.g("pretrained.patch_embed.proj.bias")
        y = conv2d(img, w, b, stride=PATCH, pad=0, q=self.q)  # (384, ph, pw)
        C, ph, pw = y.shape
        return y.reshape(C, ph * pw).T, ph, pw  # (P, 384)

    def block(self, x, i):
        pfx = f"pretrained.blocks.{i}."
        n = layernorm(x, self.g(pfx + "norm1.weight"), self.g(pfx + "norm1.bias"))
        qkv = linear(n, self.g(pfx + "attn.qkv.weight"), self.g(pfx + "attn.qkv.bias"), self.q)
        N = x.shape[0]
        qkv = qkv.reshape(N, 3, N_HEADS, HEAD_DIM)
        qh = qkv[:, 0].transpose(1, 0, 2)  # (H, N, hd)
        kh = qkv[:, 1].transpose(1, 0, 2)
        vh = qkv[:, 2].transpose(1, 0, 2)
        scale = HEAD_DIM ** -0.5
        attn = np.einsum("hnd,hmd->hnm", self.q.act(qh * scale), self.q.act(kh))
        attn = softmax(attn, axis=-1)
        ctx = np.einsum("hnm,hmd->hnd", self.q.act(attn), self.q.act(vh))
        ctx = ctx.transpose(1, 0, 2).reshape(N, EMBED_DIM)
        a = linear(ctx, self.g(pfx + "attn.proj.weight"), self.g(pfx + "attn.proj.bias"), self.q)
        x = x + a * self.g(pfx + "ls1.gamma")

        n = layernorm(x, self.g(pfx + "norm2.weight"), self.g(pfx + "norm2.bias"))
        h = linear(n, self.g(pfx + "mlp.fc1.weight"), self.g(pfx + "mlp.fc1.bias"), self.q)
        h = gelu_np(h)
        h = linear(h, self.g(pfx + "mlp.fc2.weight"), self.g(pfx + "mlp.fc2.bias"), self.q)
        x = x + h * self.g(pfx + "ls2.gamma")
        return x

    def encoder(self, img):
        tokens, ph, pw = self.patch_embed(img)
        cls = self.g("pretrained.cls_token").reshape(1, EMBED_DIM)
        x = np.concatenate([cls, tokens], axis=0) + self.pos_embed
        outs = []
        for i in range(N_BLOCKS):
            x = self.block(x, i)
            if i in INTERMEDIATE:
                outs.append(x.copy())
        nw = self.g("pretrained.norm.weight")
        nb = self.g("pretrained.norm.bias")
        # final LayerNorm applied to every intermediate output, cls dropped
        return [layernorm(o, nw, nb)[1:] for o in outs], ph, pw

    # -- DPT head ---------------------------------------------------------
    def res_conv_unit(self, x, pfx):
        out = relu(x)
        out = conv2d(out, self.g(pfx + "conv1.weight"), self.g(pfx + "conv1.bias"), 1, 1, self.q)
        out = relu(out)
        out = conv2d(out, self.g(pfx + "conv2.weight"), self.g(pfx + "conv2.bias"), 1, 1, self.q)
        return out + x

    def fusion(self, idx, xs, size):
        pfx = f"depth_head.scratch.refinenet{idx}."
        out = xs[0]
        if len(xs) == 2:
            out = out + self.res_conv_unit(xs[1], pfx + "resConfUnit1.")
        out = self.res_conv_unit(out, pfx + "resConfUnit2.")
        if size is None:
            size = (out.shape[1] * 2, out.shape[2] * 2)
        out = interpolate_bilinear(out, size[0], size[1], align_corners=True)
        return conv2d(out, self.g(pfx + "out_conv.weight"), self.g(pfx + "out_conv.bias"), 1, 0, self.q)

    def head(self, feats, ph, pw):
        proj = []
        for i, f in enumerate(feats):
            x = f.T.reshape(EMBED_DIM, ph, pw)
            x = conv2d(x, self.g(f"depth_head.projects.{i}.weight"),
                       self.g(f"depth_head.projects.{i}.bias"), 1, 0, self.q)
            if i == 0:
                x = conv_transpose2d(x, self.g("depth_head.resize_layers.0.weight"),
                                     self.g("depth_head.resize_layers.0.bias"), 4, self.q)
            elif i == 1:
                x = conv_transpose2d(x, self.g("depth_head.resize_layers.1.weight"),
                                     self.g("depth_head.resize_layers.1.bias"), 2, self.q)
            elif i == 3:
                x = conv2d(x, self.g("depth_head.resize_layers.3.weight"),
                           self.g("depth_head.resize_layers.3.bias"), 2, 1, self.q)
            proj.append(x)

        rn = [conv2d(proj[i], self.g(f"depth_head.scratch.layer{i+1}_rn.weight"), None, 1, 1, self.q)
              for i in range(4)]

        path4 = self.fusion(4, [rn[3]], rn[2].shape[1:])
        path3 = self.fusion(3, [path4, rn[2]], rn[1].shape[1:])
        path2 = self.fusion(2, [path3, rn[1]], rn[0].shape[1:])
        path1 = self.fusion(1, [path2, rn[0]], None)

        out = conv2d(path1, self.g("depth_head.scratch.output_conv1.weight"),
                     self.g("depth_head.scratch.output_conv1.bias"), 1, 1, self.q)
        out = interpolate_bilinear(out, ph * PATCH, pw * PATCH, align_corners=True)
        out = conv2d(out, self.g("depth_head.scratch.output_conv2.0.weight"),
                     self.g("depth_head.scratch.output_conv2.0.bias"), 1, 1, self.q)
        out = relu(out)
        out = conv2d(out, self.g("depth_head.scratch.output_conv2.2.weight"),
                     self.g("depth_head.scratch.output_conv2.2.bias"), 1, 0, self.q)
        out = relu(out)
        return out[0]

    def forward(self, img):
        feats, ph, pw = self.encoder(img)
        return self.head(feats, ph, pw)
