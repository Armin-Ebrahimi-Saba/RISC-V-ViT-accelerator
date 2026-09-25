"""Estimate CV32E40P cycles of the engine without the board.

Runs a bare-metal RV32IMC program (built by build.sh) in the Unicorn CPU
emulator and counts cycles with a simple timing model:

  1 per instruction, +3 for a taken branch, +1 for a jump, +5 for
  mulh/mulhsu/mulhu, +34 for div/rem; per data access +4 (load) and +2
  (store) in the on-chip RAM, +0.2 in the DDR3 region (cache hits).

The constants reproduce the board's boot-time microbenchmark (main.c,
dav2_bench) within 0.3 cycles per iteration: run calib.elf. The
accelerator is replaced by stubs. In bench.elf its jobs finish at once. In
frame.elf each job runs for a time from the block's design (see frame.c)
while the CPU goes on, and the CPU waits for it where the driver would.
For the round-seven engine the whole frame comes out at 373 Mcycles
against 360 measured on the board (4 % high).

Usage (Unicorn and pyelftools: pip install unicorn pyelftools):
  python3 emu.py calib.elf                  microbenchmark, per phase
  python3 emu.py bench.elf [phase]          kernels on synthetic data;
                                            with a phase number, a
                                            source-line profile of it
  python3 emu.py frame.elf -1 blob img      one whole frame (about 1 min)
  BYFN=1 python3 emu.py frame.elf 1 blob img  ... with a per-function profile

Harness protocol: a write to 0x10000000 marks a phase; reads of
0x10000004/8 return the cycle counter (dav2_cycles); writes to
0x1000000c/10 report a counter id and value; 0x10000014 an output hash.
"""
import sys
from unicorn import Uc, UC_ARCH_RISCV, UC_MODE_RISCV32, UC_HOOK_CODE, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE
from unicorn.riscv_const import UC_RISCV_REG_PC
from elftools.elf.elffile import ELFFile

P = dict(br=3, jmp=1, mulh=5, div=34, ld_bram=4, ld_ddr=0.2, st_bram=2, st_ddr=0.2)

def classify(ins, size):
    """-> (extra cycles, is_control)"""
    if size == 2:
        q, f3 = ins & 3, (ins >> 13) & 7
        if q == 1 and f3 in (1, 5):
            return P["jmp"], False            # c.jal, c.j: always taken
        if q == 1 and f3 in (6, 7):
            return 0, True                    # c.beqz, c.bnez
        if q == 2 and f3 == 4 and ((ins >> 2) & 31) == 0 and ((ins >> 7) & 31) != 0:
            return P["jmp"], False            # c.jr, c.jalr
        return 0, False
    op = ins & 0x7f
    if op == 0x63:
        return 0, True
    if op in (0x6f, 0x67):
        return P["jmp"], False
    if op == 0x33 and (ins >> 25) == 1:
        f3 = (ins >> 12) & 7
        if f3 in (1, 2, 3):
            return P["mulh"], False
        if f3 >= 4:
            return P["div"], False
    return 0, False

def run(elf_path, limit=4_000_000_000, profile_phase=-1, files=()):
    uc = Uc(UC_ARCH_RISCV, UC_MODE_RISCV32)
    uc.mem_map(0, 1 << 20)
    uc.mem_map(0x80000000, 64 << 20)
    for addr, path in files:
        uc.mem_write(addr, open(path, "rb").read())
    st = {"cyc": 0, "exp": None, "marks": [], "ins": 0, "profile_phase": profile_phase}

    def mmio_read(uc, off, size, ud):
        c = int(st["cyc"]); return (c & 0xffffffff) if off == 4 else (c >> 32)
    def mmio_write(uc, off, size, value, ud):
        if off == 24:
            st["cyc"] += value                  # a stubbed job's estimated time
        if off == 20:
            print("output hash %08x" % value)
        if off == 12:
            st["subid"] = value
        if off == 16:
            st.setdefault("sub", []).append((st["subid"], value))
        if off == 0:
            st["marks"].append((value, st["cyc"], st["ins"]))
            st["on"] = (value == st.get("profile_phase", -1))
            if value == 0xdead:
                uc.emu_stop()
    uc.mmio_map(0x10000000, 0x1000, mmio_read, None, mmio_write, None)

    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
        for seg in elf.iter_segments():
            if seg["p_type"] == "PT_LOAD" and seg["p_filesz"]:
                uc.mem_write(seg["p_paddr"], seg.data())
        entry = elf.header["e_entry"]

    cache = {}
    prof = st["prof"] = {}
    st["pc"] = 0
    st["on"] = False
    def on_code(uc, addr, size, ud):
        e = st["exp"]
        c = 0
        if e is not None and addr != e:
            c = P["br"]
            if st["on"]:
                prof[st["pc"]] = prof.get(st["pc"], 0) + c
        k = cache.get(addr)
        if k is None:
            ins = int.from_bytes(uc.mem_read(addr, size), "little")
            k = cache[addr] = classify(ins, size)
        st["cyc"] += c + 1 + k[0]
        if st["on"]:
            prof[addr] = prof.get(addr, 0) + 1 + k[0]
        st["pc"] = addr
        st["ins"] += 1
        st["exp"] = addr + size if k[1] else None
    def on_read(uc, acc, addr, size, value, ud):
        c = P["ld_ddr"] if addr >= 0x80000000 else P["ld_bram"]
        st["cyc"] += c
        if st["on"]:
            prof[st["pc"]] = prof.get(st["pc"], 0) + c
    def on_write(uc, acc, addr, size, value, ud):
        c = P["st_ddr"] if addr >= 0x80000000 else P["st_bram"]
        st["cyc"] += c
        if st["on"]:
            prof[st["pc"]] = prof.get(st["pc"], 0) + c
    uc.hook_add(UC_HOOK_CODE, on_code)
    uc.hook_add(UC_HOOK_MEM_READ, on_read, begin=0, end=0x0fffffff)
    uc.hook_add(UC_HOOK_MEM_READ, on_read, begin=0x80000000, end=0xffffffff)
    uc.hook_add(UC_HOOK_MEM_WRITE, on_write, begin=0, end=0x0fffffff)
    uc.hook_add(UC_HOOK_MEM_WRITE, on_write, begin=0x80000000, end=0xffffffff)
    uc.emu_start(entry, 0xffffffff, count=limit)
    run.prof = st["prof"]
    return st["marks"], st.get("sub", [])

import os
BYFN = os.environ.get("BYFN") == "1"
ONLY = os.environ.get("ONLY")          # only code whose outer function is this

def line_profile(elf, prof, top=25):
    import pickle
    pickle.dump(prof, open(elf + ".prof", "wb"))
    import subprocess, glob, collections
    import shutil, os
    a2l = shutil.which("riscv-none-elf-addr2line") or sorted(glob.glob(
        os.path.expanduser("~/Public/xpack-riscv-none-elf-gcc-*/bin/riscv-none-elf-addr2line")))[-1]
    addrs = sorted(prof)
    # -i lists the inline chain, innermost first; an address without code
    # after each address marks where its chain ends
    args = []
    for a in addrs:
        args += ["%x" % a, "fffffff0"]
    out = subprocess.run([a2l, "-f", "-i", "-e", elf] + args, capture_output=True,
                         text=True).stdout.splitlines()
    by = collections.Counter()
    k = 0
    for a in addrs:
        chain = []
        while not (out[k] == "??" and out[k + 1].startswith("??:")):
            chain.append((out[k], out[k + 1]))
            k += 2
        k += 2
        fn, loc = chain[0] if chain else ("??", "??")
        outer = chain[-1][0] if chain else "??"
        if ONLY and outer != ONLY:
            continue
        key = outer if BYFN else "%s %s" % (fn, loc.split("/")[-1].split(" ")[0])
        by[key] += prof[a]
    tot = sum(by.values())
    for loc, c in by.most_common(top):
        print("  %-44s %9.0f %5.1f%%" % (loc, c, 100.0 * c / tot))

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "reuse":
    import pickle
    line_profile(sys.argv[1], pickle.load(open(sys.argv[1] + ".prof", "rb")), 30)
    sys.exit(0)

if __name__ == "__main__":
    ph = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    files = []
    if len(sys.argv) > 4:
        files = [(0x80000000, sys.argv[3]), (0x81E00000, sys.argv[4])]
    marks, sub = run(sys.argv[1], profile_phase=ph, files=files)
    if files:
        pn = ["gemm (accel)", "gemm (cpu)", "requantise", "attention", "layernorm", "gelu",
              "add/relu", "im2col", "interpolate", "other"]
        tot = marks[-2][1] - marks[-3][1] if len(marks) >= 3 else 0
        print("frame: %.1f Mcycles, %.2f s at 50 MHz (CPU, and accelerator jobs from frame.c's model)" % (tot / 1e6, tot / 50e6))
        subn = ["ln stats", "ln gamma/beta", "ln requant", "rq range", "rq params", "att prep q,k",
                "att prep v", "att softmax", "att normalise", "att wait", "gelu table"]
        kn = ["GEMM (encoder)", "GEMM (convolutions)", "GEMM (attention)", "requant",
              "requant + add", "requant int16 in", "requant context"]
        for d, v in sub:
            if d >= 200:
                print("    %-18s %8.1f Mcycles" % (subn[d - 200], v / 1e6))
            elif d >= 100:
                print("  accelerator %-20s %8.1f Mcycles busy" % (kn[d - 100], v / 1e6))
            else:
                print("  %-14s %8.1f Mcycles" % (pn[d], v / 1e6))
        if ph >= 0:
            line_profile(sys.argv[1], run.prof, 30)
        sys.exit(0)
    if "calib" in sys.argv[1]:
        cn = {1: "alu (3 ops + loop)", 2: "32 alu ops unrolled, per 4 ops", 3: "mulh + add + loop",
              4: "lw from BRAM", 5: "lw from DDR3, seq", 6: "lw from DDR3, stride",
              7: "sw to DDR3, seq", 8: "lh + sh on BRAM"}
        for a, b in zip(marks, marks[1:]):
            if a[0] in cn:
                print("  %-34s %5.1f cycles per iteration" % (cn[a[0]], (b[1] - a[1]) / 4096))
        sys.exit(0)
    names = {1: "layernorm affine", 2: "layernorm plain", 3: "requant qkv M=1152",
             4: "requant proj+res M=384", 5: "gelu 82x1536", 6: "attention 6 heads",
             7: "add 82x384", 8: "interpolate 72->126 x32"}
    for a, b in zip(marks, marks[1:]):
        if a[0] not in (0, 0xdead):
            print("%-24s %10.0f cycles %9d instructions" % (names.get(a[0], hex(a[0])), b[1] - a[1], b[2] - a[2]))
    subn = ["ln stats", "ln gamma/beta", "ln requant", "rq range", "rq params", "att prep q,k",
            "att prep v", "att softmax", "att normalise", "att wait", "gelu table"]
    for d, v in sub:
        print("  detail %-16s %10d" % (subn[d] if d < len(subn) else d, v))
    if ph >= 0:
        print("line profile of phase %d:" % ph)
        line_profile(sys.argv[1], run.prof)
