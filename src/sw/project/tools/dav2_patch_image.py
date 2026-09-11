"""Put a different input image into an existing weight blob.

The exporter bakes the input image into the blob as two tensors, `image`
(int16, H x W x 3, channel-last) and `image_scale` (one float32). Everything
else in the 24.87 MB is weights, and none of it depends on the picture. So a
new image only needs those two tensors rewritten in place -- the directory,
offsets and sizes are unchanged -- and it needs no torch, no numpy and no
model checkout, which this machine does not have.

Preprocessing reproduces export_dav2.py exactly: ImageNet normalisation,
then quantise so that the largest magnitude maps to ACT_QMAX.

Sources:
  --synth NAME    a generated test pattern (see SYNTH)
  --ppm FILE      a binary P6 PPM, any size, resized bilinearly
"""
import argparse
import struct
import sys
from pathlib import Path

ACT_QMAX = 8191
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REC = 72
NAME_LEN = 40


def read_dir(blob):
    magic, ver, n, dir_off = struct.unpack_from("<4I", blob, 0)
    assert magic == 0x32564144, "not a DAV2 blob"
    recs = {}
    for i in range(n):
        off = dir_off + i * REC
        name = blob[off:off + NAME_LEN].split(b"\0")[0].decode()
        dtype, nbytes, offset = struct.unpack_from("<3I", blob, off + NAME_LEN)
        dims = struct.unpack_from("<4I", blob, off + NAME_LEN + 12)
        recs[name] = (dtype, nbytes, offset, dims)
    return recs


# ---------------------------------------------------------------- sources

def synth(name, size):
    """Return rgb as a list of (r, g, b) floats in 0..1, row-major."""
    px = []
    for y in range(size):
        for x in range(size):
            u, v = x / (size - 1), y / (size - 1)
            if name == "gradient":          # bright top-left to dark bottom-right
                g = 1.0 - (u + v) / 2
                px.append((g, g, g))
            elif name == "checker":         # 7x7 checkerboard
                c = 1.0 if ((x * 7 // size) + (y * 7 // size)) % 2 == 0 else 0.15
                px.append((c, c, c))
            elif name == "sphere":          # a shaded ball on a floor
                dx, dy = u - 0.5, v - 0.45
                r2 = dx * dx + dy * dy
                if r2 < 0.09:
                    z = (0.09 - r2) ** 0.5 / 0.3
                    s = 0.3 + 0.7 * max(0.0, (-0.6 * dx - 0.5 * dy + 0.6 * z))
                    px.append((s * 0.9, s * 0.4, s * 0.3))
                else:
                    f = 0.25 + 0.5 * v                # floor lighter toward viewer
                    px.append((f * 0.5, f * 0.6, f * 0.7))
            elif name == "corridor":        # one-point perspective
                cx, cy = abs(u - 0.5), abs(v - 0.5)
                d = max(cx, cy) * 2
                s = 0.15 + 0.8 * d
                wall = 0.3 * ((x // 9 + y // 9) % 2)
                px.append((s + wall * 0.2, s, s - wall * 0.1))
            else:
                raise SystemExit("unknown synth %r" % name)
    return px


def read_ppm(path, size):
    data = Path(path).read_bytes()
    # P6 header: magic, width, height, maxval, then binary RGB
    tokens = []
    i = 0
    while len(tokens) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            while data[i:i + 1] not in (b"\n", b""):
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    i += 1
    assert tokens[0] == b"P6", "only binary P6 PPM is supported"
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    raw = data[i:i + w * h * 3]
    src = [(raw[k] / maxval, raw[k + 1] / maxval, raw[k + 2] / maxval)
           for k in range(0, w * h * 3, 3)]

    def at(xx, yy):
        return src[min(h - 1, yy) * w + min(w - 1, xx)]

    px = []
    for y in range(size):
        for x in range(size):
            fx, fy = x * (w - 1) / (size - 1), y * (h - 1) / (size - 1)
            x0, y0 = int(fx), int(fy)
            ax, ay = fx - x0, fy - y0
            p00, p10, p01, p11 = at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1)
            px.append(tuple(
                (p00[c] * (1 - ax) + p10[c] * ax) * (1 - ay)
                + (p01[c] * (1 - ax) + p11[c] * ax) * ay for c in range(3)))
    return px


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", default="build/dav2/dav2_weights.bin")
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synth", choices=["gradient", "checker", "sphere", "corridor"])
    src.add_argument("--ppm")
    args = ap.parse_args()

    blob = bytearray(Path(args.blob).read_bytes())
    recs = read_dir(blob)
    _, nb_img, off_img, dims = recs["image"]
    _, _, off_scale, _ = recs["image_scale"]
    H, W, C = dims[0], dims[1], dims[2]
    assert H == W and C == 3, dims
    assert nb_img == H * W * 3 * 2

    rgb = synth(args.synth, H) if args.synth else read_ppm(args.ppm, H)

    # ImageNet normalise, exactly as export_dav2.py / dav2_common.load_image
    norm = [((p[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]) for p in rgb for c in range(3)]
    amax = max(abs(v) for v in norm)
    scale = amax / ACT_QMAX
    q = [max(-ACT_QMAX, min(ACT_QMAX, int(round(v / scale)))) for v in norm]

    struct.pack_into("<%dh" % len(q), blob, off_img, *q)
    struct.pack_into("<f", blob, off_scale, scale)
    Path(args.out).write_bytes(blob)
    print("wrote %s: image %dx%dx3 from %s, scale %.6g"
          % (args.out, H, W, args.synth or args.ppm, scale))


if __name__ == "__main__":
    main()
