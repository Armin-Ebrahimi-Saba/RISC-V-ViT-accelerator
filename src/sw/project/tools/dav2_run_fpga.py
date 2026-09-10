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
import subprocess
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from flow.tools import openocd  # noqa: E402
from flow.tools.openocd import Hostio  # noqa: E402
from flow.tools.riscv_debug_helper import (  # noqa: E402
    reset_halt_rvlab_cpu, DMControl, DMStatus)

BLOB_ADDR = 0x80000000
GO_ADDR = 0x8F000000
GO_MAGIC = 0xD00DFEED


def read_hostio(ocd):
    """Drain the program's stdout ring buffer and return it as text.

    OpenOcd.hostio_read() prints straight to our stdout, which is no use when
    we need to match markers, so this returns the characters instead.

    Both indices are read from the device on every poll rather than kept in a
    host-side mirror. A mirror desynchronises whenever the device restarts
    without the host knowing (a debugger reset, say), and the ring cannot then
    recover: the device computes occupancy as (widx - ridx) & (SIZE-1), so a
    read index even one byte ahead of the write index reads as SIZE-1 --
    permanently "full". The device then spins in obuf_putc (hostio.c:23)
    forever and the program looks hung at whatever line it last printed.
    Measured on hardware as widx=18, ridx=19, enqueued=1023, ring empty.
    """
    widx = ocd.readword(Hostio.OBUF_WIDX)
    ridx = ocd.readword(Hostio.OBUF_RIDX)
    navail = (widx - ridx) & (Hostio.OBUF_SIZE - 1)
    if navail == 0:
        ocd.obuf_ridx = ridx          # keep the lab helper's mirror honest
        return ""

    out = []
    pos = ridx
    wordaddr_last = -1
    word = 0
    for _ in range(navail):
        wordaddr = Hostio.OBUF + (pos & ~3)
        if wordaddr != wordaddr_last:
            word = ocd.readword(wordaddr)
            wordaddr_last = wordaddr
        out.append(chr((word >> ((pos & 3) * 8)) & 0xff))
        pos = (pos + 1) & (Hostio.OBUF_SIZE - 1)

    # pos == widx here, so the ring is always left exactly empty and a skew
    # can never accumulate.
    ocd.writeword(Hostio.OBUF_RIDX, pos)
    ocd.obuf_ridx = pos
    return "".join(out)


def start_program(ocd, elf_filename):
    """Load and start the program without an interactive console.

    OpenOcd.run_prog() puts stdin into raw mode, so it needs a terminal and
    fails under redirection. This is the same sequence minus the console.
    """
    reset_halt_rvlab_cpu(ocd)
    ocd.cmd("lpriscv1.tap.0 arp_examine")
    reset_halt_rvlab_cpu(ocd)
    ocd.cmd("tcl_trace off")
    # Must precede any memory access: without sysbus mode the read/write
    # commands fail silently, so hostio_clear() would leave the ring-buffer
    # indices untouched and we would read stale output from a previous run.
    ocd.cmd("riscv set_mem_access sysbus")

    # Absolute path, braced: OpenOCD resolves relative paths against its own
    # working directory, not ours. A path it cannot open makes load_image and
    # verify_image fail without raising, and `resume` then runs whatever was
    # already in BRAM -- which is the reference program baked into the
    # bitstream, not ours. Check the output rather than trust it.
    elf = str(Path(elf_filename).resolve())
    print(f"loading {elf} ...", flush=True)
    out = ocd.cmd(f"load_image {{{elf}}} 0 elf")
    if "error" in out.lower() or "can't" in out.lower():
        raise RuntimeError(f"load_image failed: {out.strip()}")
    out = ocd.cmd(f"verify_image {{{elf}}} 0 elf")
    if "error" in out.lower() or "differ" in out.lower():
        raise RuntimeError(f"verify_image failed: {out.strip()}")
    ocd.cmd("reg pc 0x80")
    ocd.hostio_clear()
    ocd.cmd("resume")
    print("program started", flush=True)


def dm_halt(ocd, timeout=5.0):
    """Halt the core by driving the debug module directly.

    OpenOCD's own `halt` is a no-op on this target: it returns an empty
    string and `targets` keeps reporting running, so every subsequent `reg`
    read fails. The reset-halt path in flow/tools does work, and it works by
    writing DMControl over dmi, so do the same thing with haltreq instead of
    ndmreset -- that stops the core where it is rather than restarting it.
    Afterwards force OpenOCD to re-poll so its cached target state agrees and
    `reg` is accepted.
    """
    ctrl = DMControl(ocd)
    status = DMStatus(ocd)
    ctrl.dmactive = 1
    ctrl.haltreq = 1
    ctrl.write()

    deadline = time.time() + timeout
    st = status.read()
    while not st.allhalted and time.time() < deadline:
        st = status.read()
    halted = bool(st.allhalted)

    ctrl.haltreq = 0
    ctrl.dmactive = 1
    ctrl.write()

    ocd.cmd("lpriscv1.tap.0 arp_poll")
    ocd.cmd("poll")
    return halted, st


def wait_for_marker(ocd, marker, timeout=120.0, buf=""):
    """Poll hostio until `marker` appears in the program's stdout.

    Returns the text accumulated so far. Also gives up if the program exits
    before the marker shows, which otherwise looks identical to a hang.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        chunk = read_hostio(ocd)
        if chunk:
            buf += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
            if marker in buf:
                return buf
            continue
        if ocd.readword(Hostio.FLAGS) & 1:
            retval = ocd.readword(Hostio.RETVAL)
            raise RuntimeError(
                f"program exited (return value {retval}) before {marker!r} appeared")
        time.sleep(0.05)
    raise TimeoutError(f"marker {marker!r} not seen within {timeout}s")


def go_address(elf_path):
    """Address of the handshake flag, looked up by symbol.

    The flag is a BRAM variable rather than a fixed DDR3 address: once the
    blob has been pushed through system bus access, the CPU no longer observes
    debugger writes to DDR3, so a DDR3 handshake word is never seen.
    """
    out = subprocess.run(
        ["riscv-none-elf-nm", str(elf_path)],
        capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "dav2_go":
            return int(parts[0], 16)
    raise RuntimeError("symbol dav2_go not found in " + str(elf_path))


def ensure_running(ocd):
    """Resume the CPU if an image load left it halted.

    OpenOCD halts the target to perform load_image and does not resume it
    afterwards. A handshake write then lands correctly but the CPU is frozen,
    so it never leaves its wait loop -- silence that looks exactly like a hang
    inside the model.
    """
    state = ocd.cmd("targets")
    if "halted" in state.lower():
        print("target halted by the image load; resuming", flush=True)
        ocd.cmd("resume")
        return True
    return False


def load_blob(ocd, blob_path, chunk_bytes=2 * 1024 * 1024):
    """Write the weight blob into DDR3 via OpenOCD's system bus access.

    This is by far the slowest step of a board run. The blob lands in DDR3
    through system bus access, which costs a USB round-trip per 32-bit word --
    measured at roughly an hour for 25 MB, not the minutes an ELF load into
    BRAM takes. So transfer in chunks and report progress: a silent hour is
    indistinguishable from a hang, which is exactly the mistake to avoid.
    """
    size = Path(blob_path).stat().st_size
    print(f"loading {size/1e6:.2f} MB of weights to 0x{BLOB_ADDR:08x} "
          f"in {chunk_bytes//1024//1024} MB chunks ...", flush=True)
    t0 = time.time()
    done = 0
    while done < size:
        n = min(chunk_bytes, size - done)
        # min_addr/max_length select the slice of the image to write, so the
        # file is opened once per chunk but only that window is transferred.
        ocd.cmd(f"load_image {{{blob_path}}} 0x{BLOB_ADDR:08x} bin "
                f"0x{BLOB_ADDR + done:08x} 0x{n:x}")
        done += n
        el = time.time() - t0
        rate = done / 1e6 / max(el, 1e-9)
        eta = (size - done) / 1e6 / max(rate, 1e-9)
        print(f"  {done/1e6:6.2f}/{size/1e6:.2f} MB  "
              f"{rate:.3f} MB/s  elapsed {el/60:.1f} min  eta {eta/60:.1f} min",
              flush=True)
    dt = time.time() - t0
    print(f"weights loaded in {dt/60:.1f} min ({size/1e6/max(dt,1e-9):.3f} MB/s)",
          flush=True)


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
    print(f"blob header verified in DDR3 ({n_tensors} tensors)", flush=True)


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
        start_program(ocd, Path(args.elf))
        text = wait_for_marker(ocd, "DAV2_WAITING_FOR_WEIGHTS")
        load_blob(ocd, str(Path(args.blob).resolve()))
        verify_blob(ocd, args.blob)
        ensure_running(ocd)
        go = go_address(Path(args.elf).resolve())
        print(f"handshake flag at 0x{go:08x} (BRAM)", flush=True)
        ocd.writeword(go, GO_MAGIC)
        print("handshake written; inference running", flush=True)
        try:
            text = wait_for_marker(ocd, "DAV2_RESULT_END", timeout=args.timeout,
                                   buf=text)
        except TimeoutError:
            # The core is stalled. Halt it and read where -- far better than
            # inferring the location from which printf was last seen.
            print("\n--- inference did not finish; probing the core ---",
                  flush=True)
            # Sysbus reads work while the core runs, so sample the live state
            # first: this needs no halt and therefore cannot perturb anything.
            GEMM = 0x20010000
            names = ("status", "dbg", "dbg2", "dbg3", "dbg4", "cycles")
            offs  = (0x00, 0x2c, 0x30, 0x34, 0x38, 0x28)
            for k in range(4):
                vals = [ocd.readword(GEMM + o) for o in offs]
                w = ocd.readword(Hostio.OBUF_WIDX)
                r = ocd.readword(Hostio.OBUF_RIDX)
                print("  sample%d widx=%d ridx=%d enq=%d | %s" % (
                    k, w, r, (w - r) & (Hostio.OBUF_SIZE - 1),
                    " ".join("%s=%08x" % (n, v)
                             for n, v in zip(names, vals))), flush=True)
                time.sleep(1.0)

            # Only then try to stop it. A core wedged on a bus access that
            # never completes cannot retire and so cannot accept a halt;
            # "targets" staying at running is itself the diagnosis.
            halted, st = dm_halt(ocd)
            print("  dm halt: allhalted=%d anyrunning=%d anyunavail=%d "
                  "anyhavereset=%d" % (st.allhalted, st.anyrunning,
                                       st.anyunavail, st.anyhavereset),
                  flush=True)
            print("  targets:", ocd.cmd("targets").strip()[-30:], flush=True)
            if not halted:
                print("  core did not halt via the debug module either",
                      flush=True)
            for r in ("pc", "ra", "sp", "a0", "a1", "a2", "a3", "a4", "a5"):
                print("  %-4s %s" % (r, ocd.cmd("reg " + r).strip()), flush=True)
            for csr in ("dpc", "dcsr", "mcause", "mepc", "mtval", "mstatus", "mcycle"):
                try:
                    print("  %-8s %s" % (csr, ocd.cmd("reg " + csr).strip()),
                          flush=True)
                except Exception as exc:      # not all CSRs are exposed
                    print("  %-8s <%s>" % (csr, exc), flush=True)
            for i in range(6):
                try:
                    ocd.cmd("step")
                    print("  step%d %s" % (i, ocd.cmd("reg pc").strip()),
                          flush=True)
                except Exception as exc:
                    print("  step%d <%s>" % (i, exc), flush=True)
            raise
        parse_result(text, args.out)


if __name__ == "__main__":
    main()
