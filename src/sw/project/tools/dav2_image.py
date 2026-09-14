"""Turn a picture into the bytes the engine wants, and back.

The engine takes its input image separately from the weights: a buffer of
size x size x 3 int16 pixels in HWC order (height, width, channel), followed by
one float32 scale, such that real_value = int16 * scale. This module produces
that buffer -- the ".dav2img" format -- from a photograph, a synthetic scene,
or a raw array, and reads it back for display.

Preprocessing reproduces export_dav2.py / dav2_common.load_image exactly:
bilinear resize to size x size, divide by 255, ImageNet mean/std normalise,
then quantise so the largest magnitude maps to ACT_QMAX. Any drift here would
make the board disagree with the PyTorch reference for reasons that have
nothing to do with the hardware, so the constants are copied, not retyped.

Usage:
    python3 dav2_image.py photo.jpg out.dav2img            # from a picture
    python3 dav2_image.py --synth road out.dav2img         # built-in scene
    python3 dav2_image.py --show out.dav2img preview.png   # render to PNG
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ACT_QMAX = 8191
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def quantise(rgb01):
    """rgb01: float32 (H, W, 3) in 0..1. Returns (int16 HWC array, scale)."""
    x = (rgb01.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    scale = float(np.abs(x).max()) / ACT_QMAX
    q = np.clip(np.rint(x / scale), -ACT_QMAX, ACT_QMAX).astype(np.int16)
    return np.ascontiguousarray(q), scale


def from_file(path, size):
    """Photograph -> (int16 HWC, scale), resized like the exporter does."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    return quantise(rgb)


def from_synth(name, size):
    """One of dav2_patch_image.py's generated scenes -> (int16 HWC, scale)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dav2_patch_image import synth
    px = synth(name, size)
    rgb = np.clip(np.array(px, dtype=np.float32).reshape(size, size, 3), 0.0, 1.0)
    return quantise(rgb)


def pack(q, scale):
    """int16 HWC + scale -> the bytes the board receives."""
    return q.astype("<i2").tobytes() + struct.pack("<f", scale)


def unpack(data, size):
    """The inverse: bytes -> (int16 HWC array, scale)."""
    n = size * size * 3
    q = np.frombuffer(data[: n * 2], dtype="<i2").reshape(size, size, 3)
    (scale,) = struct.unpack_from("<f", data, n * 2)
    return q, scale


def to_rgb8(q, scale):
    """Undo normalisation for display. Returns uint8 (H, W, 3)."""
    x = q.astype(np.float32) * scale * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="picture file (jpg/png/...)")
    ap.add_argument("out")
    ap.add_argument("--synth", help="generated scene instead of a picture")
    ap.add_argument("--size", type=int, default=126)
    ap.add_argument("--show", action="store_true",
                    help="treat src as a .dav2img and render it to out (PNG)")
    args = ap.parse_args()

    if args.show:
        from PIL import Image
        q, scale = unpack(Path(args.src).read_bytes(), args.size)
        Image.fromarray(to_rgb8(q, scale)).save(args.out)
        print(f"wrote {args.out}")
        return

    if args.synth:
        q, scale = from_synth(args.synth, args.size)
        src = args.synth
    elif args.src:
        q, scale = from_file(args.src, args.size)
        src = args.src
    else:
        ap.error("give a picture file or --synth NAME")

    Path(args.out).write_bytes(pack(q, scale))
    print(f"wrote {args.out}: {args.size}x{args.size}x3 int16 + scale {scale:.6g}, from {src}")


if __name__ == "__main__":
    main()
