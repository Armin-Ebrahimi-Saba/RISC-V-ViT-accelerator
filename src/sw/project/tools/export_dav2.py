# SPDX-License-Identifier: CC0-1.0
# SPDX-FileCopyrightText: 2026 RVLab Student Project
"""
Export Depth-Anything V2 Small into the flat binary blob that the RISC-V
inference engine reads directly out of DDR3.

Everything that can be done once on the host is done here, so the CV32E40P
never has to:
  * positional-embedding interpolation to the target resolution (bicubic)
  * folding LayerScale gamma into the preceding linear layer's weights+bias
  * folding the attention 1/sqrt(head_dim) factor into the Q rows of qkv
  * reshaping every convolution into a plain (M x K) GEMM weight matrix
  * per-output-channel symmetric int8 quantisation

Blob layout:
    [64 B header][directory of tensor records][64 B-aligned tensor data]

Usage:
    python export_dav2.py --size 126 --image demo01.jpg --out build/dav2

Requirements: torch, numpy, PIL, and a checkout of the Depth-Anything-V2
reference repository with its checkpoint (set DAV2_REF_DIR / DAV2_CKPT, see
dav2_common.py). None of these are in the project's .venv, which only holds
the FPGA flow. Run this in a separate Python environment.

The exported blob is checked into build/dav2/dav2_weights.bin. To run a
different picture WITHOUT this exporter's dependencies, use
dav2_patch_image.py, which rewrites just the two image tensors in a copy of
an existing blob -- the weights do not depend on the picture.

The blob also carries dav2_blob_config.h, generated alongside it, which tells
the C engine the input size, patch grid and token count at compile time.
"""
import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x32564144  # 'DAV2'
VERSION = 2

DT_F32, DT_I8, DT_I16, DT_I32 = 0, 1, 2, 3
DT_SIZE = {DT_F32: 4, DT_I8: 1, DT_I16: 2, DT_I32: 4}

NAME_LEN = 40
REC_SIZE = NAME_LEN + 8 * 4  # 72 bytes
HDR_SIZE = 64
ALIGN = 64

EMBED_DIM = 384
N_HEADS = 6
HEAD_DIM = 64
N_BLOCKS = 12
PATCH = 14
INTERMEDIATE = [2, 5, 8, 11]
OUT_CH = [48, 96, 192, 384]

# Activations are int16 but restricted to a 14-bit magnitude so that a K=1536
# accumulation of act*int8 weights provably cannot overflow int32:
#   1536 * 8191 * 127 = 1.60e9 < 2^31 = 2.15e9
ACT_QMAX = 8191


class Blob:
    def __init__(self):
        self.records = []  # (name, dtype, dims, bytes)

    def add(self, name, arr, dtype):
        assert len(name) < NAME_LEN, name
        a = np.ascontiguousarray(arr)
        if dtype == DT_F32:
            a = a.astype("<f4")
        elif dtype == DT_I8:
            a = a.astype("<i1")
        elif dtype == DT_I16:
            a = a.astype("<i2")
        elif dtype == DT_I32:
            a = a.astype("<i4")
        dims = list(a.shape) + [0] * (4 - a.ndim)
        assert a.ndim <= 4, name
        self.records.append((name, dtype, dims, a.tobytes()))

    def add_quant_weight(self, name, w2d, bias=None):
        """Per-output-channel symmetric int8 quantisation of an (M, K) matrix.
        Emits '<name>.w' (int8 MxK), '<name>.s' (float32 M) and optionally
        '<name>.b' (float32 M)."""
        w2d = np.asarray(w2d, dtype=np.float32)
        amax = np.abs(w2d).max(axis=1)
        scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.rint(w2d / scale[:, None]), -127, 127).astype(np.int8)
        self.add(name + ".w", q, DT_I8)
        self.add(name + ".s", scale, DT_F32)
        if bias is not None:
            self.add(name + ".b", np.asarray(bias, dtype=np.float32), DT_F32)

    def serialize(self):
        n = len(self.records)
        dir_off = HDR_SIZE
        data_off = dir_off + n * REC_SIZE
        data_off = (data_off + ALIGN - 1) // ALIGN * ALIGN

        offsets, blobs, cur = [], [], data_off
        for _, _, _, raw in self.records:
            cur = (cur + ALIGN - 1) // ALIGN * ALIGN
            offsets.append(cur)
            blobs.append(raw)
            cur += len(raw)
        total = cur

        out = bytearray(total)
        struct.pack_into("<12I", out, 0, MAGIC, VERSION, n, dir_off, data_off,
                         total, 0, 0, 0, 0, 0, 0)
        for i, (name, dtype, dims, raw) in enumerate(self.records):
            base = dir_off + i * REC_SIZE
            out[base:base + NAME_LEN] = name.encode().ljust(NAME_LEN, b"\0")
            struct.pack_into("<8I", out, base + NAME_LEN, dtype, len(raw),
                             offsets[i], dims[0], dims[1], dims[2], dims[3], 0)
            out[offsets[i]:offsets[i] + len(raw)] = raw
        return bytes(out)


def conv_to_gemm(w):
    """(Cout, Cin, kh, kw) -> (Cout, kh*kw*Cin).

    The engine keeps feature maps channel-last (NHWC), so an im2col row is laid
    out as (ky, kx, cin) with the channels contiguous. The PyTorch weight is
    (cout, cin, ky, kx), so the spatial and channel axes must be swapped before
    flattening."""
    w = np.asarray(w, dtype=np.float32)
    return w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1).copy()


def convT_to_gemm(w):
    """Non-overlapping ConvTranspose2d weight (Cin, Cout, k, k) into the GEMM
    form the engine uses: one row per output column, ordered (ky, kx, cout) so
    that each input pixel expands into a contiguous k*k*Cout run of NHWC
    output. Result is (k*k*Cout, Cin)."""
    w = np.asarray(w, dtype=np.float32)
    Cin, Cout, kh, kw = w.shape
    return w.transpose(2, 3, 1, 0).reshape(kh * kw * Cout, Cin).copy()


def build(sd, pos_embed, image_chw, size, out_dir):
    b = Blob()
    grid = size // PATCH
    n_tokens = grid * grid + 1

    def g(name):
        return sd[name]

    # -- meta ------------------------------------------------------------
    b.add("pos_embed", pos_embed, DT_F32)                    # (n_tokens, 384)
    b.add("cls_token", g("pretrained.cls_token").reshape(-1), DT_F32)

    # input image, pre-normalised, quantised to the engine's activation format,
    # stored channel-last (H, W, 3) to match the engine's NHWC convention
    image_hwc = np.ascontiguousarray(np.asarray(image_chw).transpose(1, 2, 0))
    img_scale = float(np.abs(image_hwc).max()) / ACT_QMAX
    img_q = np.clip(np.rint(image_hwc / img_scale), -ACT_QMAX, ACT_QMAX)
    b.add("image", img_q, DT_I16)                            # (H, W, 3)
    b.add("image_scale", np.array([img_scale], dtype=np.float32), DT_F32)

    # -- patch embedding -------------------------------------------------
    b.add_quant_weight("patch_embed",
                       conv_to_gemm(g("pretrained.patch_embed.proj.weight")),
                       g("pretrained.patch_embed.proj.bias"))

    # -- transformer blocks ---------------------------------------------
    attn_scale = 1.0 / math.sqrt(HEAD_DIM)
    for i in range(N_BLOCKS):
        p = f"pretrained.blocks.{i}."
        o = f"blk{i}."
        b.add(o + "norm1.w", g(p + "norm1.weight"), DT_F32)
        b.add(o + "norm1.b", g(p + "norm1.bias"), DT_F32)

        # fold 1/sqrt(head_dim) into the Q rows so the engine never scales
        qkv_w = np.array(g(p + "attn.qkv.weight"), dtype=np.float32)
        qkv_b = np.array(g(p + "attn.qkv.bias"), dtype=np.float32)
        qkv_w[:EMBED_DIM] *= attn_scale
        qkv_b[:EMBED_DIM] *= attn_scale
        b.add_quant_weight(o + "qkv", qkv_w, qkv_b)

        # fold LayerScale ls1.gamma into attn.proj (per output channel)
        g1 = np.array(g(p + "ls1.gamma"), dtype=np.float32)
        b.add_quant_weight(o + "proj",
                           np.array(g(p + "attn.proj.weight"), dtype=np.float32) * g1[:, None],
                           np.array(g(p + "attn.proj.bias"), dtype=np.float32) * g1)

        b.add(o + "norm2.w", g(p + "norm2.weight"), DT_F32)
        b.add(o + "norm2.b", g(p + "norm2.bias"), DT_F32)
        b.add_quant_weight(o + "fc1", g(p + "mlp.fc1.weight"), g(p + "mlp.fc1.bias"))

        # fold LayerScale ls2.gamma into mlp.fc2
        g2 = np.array(g(p + "ls2.gamma"), dtype=np.float32)
        b.add_quant_weight(o + "fc2",
                           np.array(g(p + "mlp.fc2.weight"), dtype=np.float32) * g2[:, None],
                           np.array(g(p + "mlp.fc2.bias"), dtype=np.float32) * g2)

    b.add("norm.w", g("pretrained.norm.weight"), DT_F32)
    b.add("norm.b", g("pretrained.norm.bias"), DT_F32)

    # -- DPT head --------------------------------------------------------
    for i in range(4):
        b.add_quant_weight(f"proj{i}",
                           conv_to_gemm(g(f"depth_head.projects.{i}.weight")),
                           g(f"depth_head.projects.{i}.bias"))

    # The transposed-convolution GEMM has one row per (ky, kx, cout) output
    # column, so the per-output-channel bias must be tiled k*k times to line up
    # with those rows.
    b.add_quant_weight("resize0", convT_to_gemm(g("depth_head.resize_layers.0.weight")),
                       np.tile(g("depth_head.resize_layers.0.bias"), 4 * 4))
    b.add_quant_weight("resize1", convT_to_gemm(g("depth_head.resize_layers.1.weight")),
                       np.tile(g("depth_head.resize_layers.1.bias"), 2 * 2))
    b.add_quant_weight("resize3", conv_to_gemm(g("depth_head.resize_layers.3.weight")),
                       g("depth_head.resize_layers.3.bias"))

    for i in range(4):
        b.add_quant_weight(f"rn{i+1}", conv_to_gemm(g(f"depth_head.scratch.layer{i+1}_rn.weight")))

    for i in (1, 2, 3, 4):
        p = f"depth_head.scratch.refinenet{i}."
        o = f"rf{i}."
        for unit in (1, 2):
            for cv in (1, 2):
                src = f"{p}resConfUnit{unit}.conv{cv}."
                b.add_quant_weight(f"{o}u{unit}c{cv}", conv_to_gemm(g(src + "weight")), g(src + "bias"))
        b.add_quant_weight(o + "out", conv_to_gemm(g(p + "out_conv.weight")), g(p + "out_conv.bias"))

    b.add_quant_weight("out1", conv_to_gemm(g("depth_head.scratch.output_conv1.weight")),
                       g("depth_head.scratch.output_conv1.bias"))
    b.add_quant_weight("out2a", conv_to_gemm(g("depth_head.scratch.output_conv2.0.weight")),
                       g("depth_head.scratch.output_conv2.0.bias"))
    b.add_quant_weight("out2b", conv_to_gemm(g("depth_head.scratch.output_conv2.2.weight")),
                       g("depth_head.scratch.output_conv2.2.bias"))

    raw = b.serialize()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dav2_weights.bin").write_bytes(raw)

    cfg_text = f"""/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Generated by tools/export_dav2.py -- do not edit.
 * Regenerate by re-running the exporter with the desired --size.
 */
#ifndef DAV2_BLOB_CONFIG_H
#define DAV2_BLOB_CONFIG_H

#define DAV2_INPUT_SIZE   {size}
#define DAV2_PATCH_GRID   {grid}
#define DAV2_N_TOKENS     {n_tokens}
#define DAV2_N_PATCHES    {grid * grid}
#define DAV2_BLOB_BYTES   {len(raw)}u

#endif
"""
    # The generated config must exist in exactly one place, otherwise a stale
    # copy silently wins the include path. The engine sources live next to it,
    # and both the RISC-V flow build and the host build include it from there.
    src_cfg = Path(__file__).resolve().parent.parent / "dav2_blob_config.h"
    src_cfg.write_text(cfg_text)
    (out_dir / "dav2_blob_config.h").write_text(cfg_text)
    print(f"wrote {out_dir/'dav2_weights.bin'}  ({len(raw)/1e6:.2f} MB, "
          f"{len(b.records)} tensors)")
    print(f"wrote {src_cfg}")
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=126,
                    help="input side length, must be a multiple of 14")
    ap.add_argument("--image", default="demo01.jpg")
    ap.add_argument("--out", default="build/dav2")
    ap.add_argument("--common-dir", default=str(Path(__file__).resolve().parent),
                    help="directory containing dav2_common.py (defaults to this script's directory)")
    args = ap.parse_args()

    assert args.size % PATCH == 0, "input size must be a multiple of 14"

    sys.path.insert(0, args.common_dir)
    import dav2_common as C

    model = C.load_model()
    sd = {k: v.numpy() for k, v in model.state_dict().items()}
    pos_embed = C.interpolated_pos_embed(model, args.size)

    img_path = args.image
    if not Path(img_path).exists():
        img_path = str(C.REF / "assets/examples" / args.image)
    chw, rgb = C.load_image(img_path, args.size)

    out_dir = Path(args.out)
    build(sd, pos_embed, chw, args.size, out_dir)

    # golden float reference for host-side verification
    ref = C.float_reference(model, chw)
    np.save(out_dir / "ref_depth.npy", ref)
    np.save(out_dir / "ref_rgb.npy", rgb)
    print(f"wrote golden reference {out_dir/'ref_depth.npy'} "
          f"(range {ref.min():.3f}..{ref.max():.3f})")


if __name__ == "__main__":
    main()
