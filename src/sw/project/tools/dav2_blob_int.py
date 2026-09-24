"""Convert the float parameters of a weight blob to integers, offline.

The board has no FPU. export_dav2.py writes every weight matrix as int8, but
it keeps the small per-row and per-channel parameters in float32. This
script replaces them by integer forms, so the engine needs no floating
point at all. It reads a version 2 blob and writes a version 3 blob:

  <layer>.s   row scale       float32 (M)    -> int32 (M, 2): (m, sh)
  <layer>.b   row bias        float32 (M)    -> int32 (M, 2): (m, sh)
  *norm*.w    LayerNorm gamma float32 (C)    -> int32 (C): Q15
  *norm*.b    LayerNorm beta  float32 (C)    -> int32 (C): Q16
  cls_token, pos_embed        float32        -> int32: Q24
  image, image_scale          removed (images are sent separately)

With --phi-header it also writes dav2_gelu_phi.h: the normal distribution
function Phi(z) for z = -8 + i/128, i = 0..2048, int32 Q30, which the GELU
table is built from. It is compiled into the program (not put in the blob)
because the boot self-test uses GELU before the weights are loaded.

(m, sh) means the real number m * 2^-sh, with 2^30 <= |m| < 2^31 (or m = 0).
A float32 has a 24-bit mantissa, so this form holds it exactly. It is also
the (multiplier, shift) pair the engine's integer arithmetic uses.

gamma and beta are rounded exactly as the engine used to round them at run
time (float32 product, then +-0.5 in float32, then truncation), so
LayerNorm gets the same integers as before.

Usage:
    python3 dav2_blob_int.py build/dav2/dav2_weights.bin            # in place
    python3 dav2_blob_int.py in.bin out.bin
    python3 dav2_blob_int.py --phi-header src/sw/project/dav2_gelu_phi.h
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x32564144
VERSION_FLOAT = 2
VERSION_INT = 3
DT_F32, DT_I8, DT_I16, DT_I32 = 0, 1, 2, 3
NAME_LEN = 40
REC_SIZE = NAME_LEN + 8 * 4
HDR_SIZE = 64
ALIGN = 64

POS_Q = 24          # cls_token and pos_embed: value = int / 2^24
PHI_Z0 = -8         # gelu_phi covers z in [-8, 8]
PHI_STEP_LOG2 = 7   # step 1/128
PHI_Q = 30


def read_blob(raw):
    magic, ver, n, dir_off, data_off, total = struct.unpack_from("<6I", raw, 0)
    if magic != MAGIC:
        raise SystemExit("not a DAV2 blob")
    recs = []
    for i in range(n):
        o = dir_off + i * REC_SIZE
        name = raw[o:o + NAME_LEN].split(b"\0")[0].decode()
        dt, nb, off, d0, d1, d2, d3, _ = struct.unpack_from("<8I", raw, o + NAME_LEN)
        dims = [d for d in (d0, d1, d2, d3) if d]
        recs.append([name, dt, dims, raw[off:off + nb]])
    return ver, recs


def write_blob(recs, version):
    n = len(recs)
    dir_off = HDR_SIZE
    data_off = (dir_off + n * REC_SIZE + ALIGN - 1) // ALIGN * ALIGN
    offsets, cur = [], data_off
    for r in recs:
        cur = (cur + ALIGN - 1) // ALIGN * ALIGN
        offsets.append(cur)
        cur += len(r[3])
    out = bytearray(cur)
    struct.pack_into("<12I", out, 0, MAGIC, version, n, dir_off, data_off, cur,
                     0, 0, 0, 0, 0, 0)
    for i, (name, dt, dims, data) in enumerate(recs):
        base = dir_off + i * REC_SIZE
        out[base:base + NAME_LEN] = name.encode().ljust(NAME_LEN, b"\0")
        d = list(dims) + [0] * (4 - len(dims))
        struct.pack_into("<8I", out, base + NAME_LEN, dt, len(data), offsets[i],
                         d[0], d[1], d[2], d[3], 0)
        out[offsets[i]:offsets[i] + len(data)] = data
    return bytes(out)


def mant_shift(f32):
    """float32 array -> int32 (N, 2) of (m, sh), value = m * 2^-sh, exact."""
    bits = np.asarray(f32, dtype="<f4").view("<u4").astype(np.int64)
    exp = (bits >> 23) & 0xFF
    if np.any(exp == 0xFF):
        raise SystemExit("inf or nan in a parameter")
    mant = (bits & 0x7FFFFF) | 0x800000
    m = mant << 7                          # 2^30 <= m < 2^31
    sh = 157 - exp                         # 24-bit mantissa * 2^(exp-150) = m * 2^-sh
    zero = exp == 0                        # zero (denormals are treated as zero)
    m = np.where(zero, 0, np.where(bits >> 31, -m, m))
    sh = np.where(zero, 0, sh)
    out = np.stack([m, sh], axis=1).astype("<i4")
    # check the representation reproduces the float exactly
    back = out[:, 0].astype(np.float64) * np.power(2.0, -out[:, 1].astype(np.float64))
    assert np.array_equal(back.astype(np.float32),
                          np.where(zero, 0, np.asarray(f32, np.float32))), "not exact"
    return out


def round_like_engine(f32, factor):
    """int32 of iround(f * factor) as the engine computed it in float32."""
    t = np.asarray(f32, dtype=np.float32) * np.float32(factor)
    r = np.where(np.signbit(t), t - np.float32(0.5), t + np.float32(0.5)).astype(np.float32)
    return np.trunc(r).astype("<i4")


def phi_table():
    n = 16 * (1 << PHI_STEP_LOG2) + 1
    z = PHI_Z0 + np.arange(n, dtype=np.float64) / (1 << PHI_STEP_LOG2)
    phi = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z])
    return np.rint(phi * (1 << PHI_Q)).astype("<i4")


def convert(recs):
    out = []
    for name, dt, dims, data in recs:
        if name in ("image", "image_scale"):
            continue
        if dt != DT_F32:
            out.append([name, dt, dims, data])
            continue
        f = np.frombuffer(data, dtype="<f4")
        if name in ("cls_token", "pos_embed"):
            if np.abs(f).max() >= (1 << (31 - POS_Q)):
                raise SystemExit(f"{name} too large for Q{POS_Q}")
            q = np.rint(f.astype(np.float64) * (1 << POS_Q)).astype("<i4")
            out.append([name, DT_I32, dims, q.tobytes()])
        elif "norm" in name and name.endswith(".w"):
            out.append([name, DT_I32, dims, round_like_engine(f, 32768.0).tobytes()])
        elif "norm" in name and name.endswith(".b"):
            out.append([name, DT_I32, dims, round_like_engine(f, 65536.0).tobytes()])
        elif name.endswith(".s") or name.endswith(".b"):
            out.append([name, DT_I32, [len(f), 2], mant_shift(f).tobytes()])
        else:
            raise SystemExit(f"float tensor '{name}' has no integer form defined")
    return out


def write_phi_header(path):
    t = phi_table()
    lines = []
    for i in range(0, len(t), 8):
        lines.append("    " + ", ".join("%d" % v for v in t[i:i + 8]) + ",")
    Path(path).write_text(
        "/* SPDX-License-Identifier: CC0-1.0\n"
        " * SPDX-FileCopyrightText: 2026 RVLab Student Project\n"
        " *\n"
        " * Generated by tools/dav2_blob_int.py --phi-header -- do not edit.\n"
        " *\n"
        " * The normal distribution function Phi(z) = (1 + erf(z / sqrt 2)) / 2\n"
        " * for z = %d + i / %d, i = 0..%d, in Q%d. GELU(x) = x * Phi(x).\n"
        " */\n"
        "#ifndef DAV2_GELU_PHI_H\n#define DAV2_GELU_PHI_H\n\n"
        "#include <stdint.h>\n\n"
        "static const int32_t dav2_gelu_phi[%d] = {\n%s\n};\n\n#endif\n"
        % (PHI_Z0, 1 << PHI_STEP_LOG2, len(t) - 1, PHI_Q, len(t), "\n".join(lines)))
    print(f"wrote {path} ({len(t)} entries)")


def main():
    if sys.argv[1] == "--phi-header":
        write_phi_header(sys.argv[2])
        return
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    ver, recs = read_blob(src.read_bytes())
    if ver == VERSION_INT:
        print(f"{src} is already version {VERSION_INT}; nothing to do")
        if dst != src:
            dst.write_bytes(src.read_bytes())
        return
    if ver != VERSION_FLOAT:
        raise SystemExit(f"{src}: version {ver}, expected {VERSION_FLOAT}")
    new = convert(recs)
    raw = write_blob(new, VERSION_INT)
    dst.write_bytes(raw)
    n_f32 = sum(1 for r in new if r[1] == DT_F32)
    print(f"wrote {dst}: version {VERSION_INT}, {len(new)} tensors, "
          f"{len(raw) / 1e6:.2f} MB, {n_f32} float tensors left")


if __name__ == "__main__":
    main()
