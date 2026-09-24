"""Turn the board's soft-float call-site report into functions and lines.

The program, built with build_flags.txt, prints after each frame one line
per call site:  FPSITE <return address> <calls> <cycles>  and then
FPSITE_END. This script reads a runner log, takes the last such block,
maps each return address to the calling function and source line with
addr2line on the same sw.elf, and prints totals per function and the
largest call sites.

Usage:
    python3 dav2_floatprof_map.py run.log [build/sw_project/build/sw.elf]
"""
import collections
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def addr2line_tool():
    for name in ("riscv-none-elf-addr2line", "riscv32-unknown-elf-addr2line",
                 "riscv64-unknown-elf-addr2line"):
        if shutil.which(name):
            return name
    xpack = sorted(Path.home().glob("Public/xpack-riscv-none-elf-gcc-*/bin/riscv-none-elf-addr2line"))
    if xpack:
        return str(xpack[-1])
    raise SystemExit("no RISC-V addr2line found")


def main():
    log = Path(sys.argv[1])
    elf = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "build/sw_project/build/sw.elf"

    # the last complete block of FPSITE lines
    block, cur = [], []
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"\s*FPSITE ([0-9a-f]{8}) (\d+) (\d+)", line)
        if m:
            cur.append((int(m.group(1), 16), int(m.group(2)), int(m.group(3))))
        elif line.strip().startswith("FPSITE_END"):
            block, cur = cur, []
    if not block:
        raise SystemExit("no FPSITE block in %s" % log)

    # the return address is after the call; pc - 2 lies inside the call
    # instruction for both compressed and full-size calls
    pcs = ["%x" % (pc - 2) for pc, _, _ in block]
    out = subprocess.run([addr2line_tool(), "-f", "-C", "-e", str(elf)] + pcs,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    where = [(out[2 * i], Path(out[2 * i + 1].split(" ")[0]).name) for i in range(len(block))]

    tot_calls = sum(c for _, c, _ in block)
    tot_cyc = sum(y for _, _, y in block)
    print("%d call sites, %d calls, %.1f Mcycles in soft-float routines"
          % (len(block), tot_calls, tot_cyc / 1e6))

    per_fn = collections.defaultdict(lambda: [0, 0])
    for (pc, c, y), (fn, _) in zip(block, where):
        per_fn[fn][0] += c
        per_fn[fn][1] += y
    print("\nper calling function:")
    print("  %-28s %10s %9s %6s" % ("function", "calls", "Mcycles", "share"))
    for fn, (c, y) in sorted(per_fn.items(), key=lambda kv: -kv[1][1]):
        print("  %-28s %10d %9.2f %5.1f%%" % (fn, c, y / 1e6, 100.0 * y / tot_cyc))

    print("\nlargest call sites:")
    print("  %-28s %-26s %10s %9s" % ("function", "line", "calls", "Mcycles"))
    rows = sorted(zip(block, where), key=lambda r: -r[0][2])
    for (pc, c, y), (fn, loc) in rows[:20]:
        print("  %-28s %-26s %10d %9.2f" % (fn, loc, c, y / 1e6))


if __name__ == "__main__":
    main()
