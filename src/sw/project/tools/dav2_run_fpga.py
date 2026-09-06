# SPDX-License-Identifier: CC0-1.0
# SPDX-FileCopyrightText: 2026 RVLab Student Project
"""
Run Depth-Anything V2 on the Nexys Video board.

Sequence:
  1. start OpenOCD against the FPGA, reset-halt the CPU and load the program
  2. wait for the program to initialise DDR3 and print DAV2_WAITING_FOR_WEIGHTS
  3. push the ~25 MB weight blob into DDR3 at 0x80000000 over JTAG
  4. write the handshake word so the program starts inference
  5. stream stdout, capture the hex-encoded depth map and save it as .npy

Usage (from the repository root, with the rvlab environment set up):

    python src/sw/project/tools/dav2_run_fpga.py \
        --elf build/sw_project/build/sw.elf \
        --blob build/dav2/dav2_weights.bin \
        --out build/dav2/fpga_depth.npy

Note on speed: the blob crosses JTAG one OpenOCD `load_image` at a time. This
is the slow part of the run (minutes), which is why the weights are loaded once
and the program then loops on inference rather than reloading per frame.
"""
import argparse
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from flow.tools import openocd  # noqa: E402

BLOB_ADDR = 0x80000000
GO_ADDR = 0x8F000000
GO_MAGIC = 0xD00DFEED


def wait_for_marker(ocd, marker, timeout=120.0):
    """Poll hostio until `marker` appears in the program's stdout."""
    buf = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        chunk = ocd.hostio_read_str() if hasattr(ocd, "hostio_read_str") else None
        if chunk:
            buf += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
            if marker in buf:
                return buf
        time.sleep(0.05)
    raise TimeoutError(f"marker {marker!r} not seen within {timeout}s")


def load_blob(ocd, blob_path):
    """Write the weight blob into DDR3 via OpenOCD's system bus access."""
    size = Path(blob_path).stat().st_size
    print(f"loading {size/1e6:.2f} MB of weights to 0x{BLOB_ADDR:08x} ...")
    t0 = time.time()
    # OpenOCD's load_image is far faster than word-at-a-time writes because it
    # batches the transfer inside the OpenOCD process.
    ocd.cmd(f"load_image {{{blob_path}}} 0x{BLOB_ADDR:08x} bin")
    dt = time.time() - t0
    print(f"weights loaded in {dt:.1f}s ({size/1e6/max(dt,1e-9):.2f} MB/s)")


def verify_blob(ocd, blob_path):
    """Spot-check the header so a truncated transfer is caught early."""
    raw = Path(blob_path).read_bytes()[:24]
    magic, version, n_tensors = struct.unpack_from("<3I", raw)
    got_magic = ocd.readword(BLOB_ADDR)
    got_version = ocd.readword(BLOB_ADDR + 4)
    got_n = ocd.readword(BLOB_ADDR + 8)
    if (got_magic, got_version, got_n) != (magic, version, n_tensors):
        raise RuntimeError(
            f"blob header mismatch in DDR3: got {got_magic:08x}/{got_version}/{got_n}, "
            f"expected {magic:08x}/{version}/{n_tensors}")
    print(f"blob header verified in DDR3 ({n_tensors} tensors)")


def parse_result(text, out_path):
    """Extract the hex-encoded float depth map printed by the program."""
    try:
        body = text.split("DAV2_RESULT_BEGIN", 1)[1]
        header, body = body.split("\n", 1)
        body = body.split("DAV2_RESULT_END", 1)[0]
    except IndexError:
        print("no depth map found in program output")
        return
    count = int(header.strip())
    hexdata = "".join(body.split())
    raw = bytes.fromhex(hexdata)
    if len(raw) != count * 4:
        print(f"warning: expected {count*4} bytes, got {len(raw)}")
    import numpy as np
    side = int(round(count ** 0.5))
    depth = np.frombuffer(raw, dtype="<f4").reshape(side, side)
    np.save(out_path, depth)
    print(f"saved {out_path} ({side}x{side}, range {depth.min():.4f}..{depth.max():.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", default="build/sw_project/build/sw.elf")
    ap.add_argument("--blob", default="build/dav2/dav2_weights.bin")
    ap.add_argument("--out", default="build/dav2/fpga_depth.npy")
    ap.add_argument("--cfg", default="src/design/openocd/fpga.cfg")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="seconds to wait for inference to finish")
    args = ap.parse_args()

    with openocd.start(REPO / args.cfg) as ocd:
        ocd.run_prog(Path(args.elf))
        wait_for_marker(ocd, "DAV2_WAITING_FOR_WEIGHTS")
        load_blob(ocd, str(Path(args.blob).resolve()))
        verify_blob(ocd, args.blob)
        ocd.writeword(GO_ADDR, GO_MAGIC)
        print("handshake written; inference running (this takes minutes)")
        text = wait_for_marker(ocd, "DAV2_RESULT_END", timeout=args.timeout)
        parse_result(text, args.out)


if __name__ == "__main__":
    main()
