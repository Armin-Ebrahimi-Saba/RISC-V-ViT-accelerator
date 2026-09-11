# Handoff — current state

For whoever picks this up next. Everything here is as of the last commit on
`student-gemm-accelerator`; the other documents in this directory carry the
detail.

| Want to know… | Read |
|---|---|
| how the system fits together | `ARCHITECTURE.md` |
| when things happen, cycle by cycle | `DATAFLOW.md` |
| what went wrong and how each thing was found | `DEBUGGING.md` |
| rules to carry to the next project | `LESSONS.md` |
| how to build, simulate, and run | `../CLAUDE.md` § Run |

---

## 1. Where it stands

**The model runs end to end on the FPGA and its output is bit-exact with the
host build.** Verified on five images: the demo photograph and four synthetic
scenes. All 15876 pixels identical in every case; correlation 0.999836 with
the PyTorch reference, which is the same figure the host build achieves.

| Measurement | Value |
|---|---|
| Frame time | 93.6 s (4.681 G cycles at 50 MHz) |
| Frames per second | 0.0107 |
| Weight load (once per session, JTAG) | ~66 s at 0.37 MB/s |
| Accelerator vs CPU on the matmul | ~36× |
| Timing, last build | pnr WNS +0.287 ns, WHS +0.018 ns, 0 failing |
| Resources | LUT ~12.4 %, BRAM 23.0 %, DSP 3.1 % |

Three defects were found and fixed in the platform's DDR3 path. None was in
the accelerator or the model code. They are described fully in
`DEBUGGING.md`; in one line each:

1. **Request mux** (`rvlab_tlul_ddr.sv`) — `a_ready` came from a block that
   never accepted anything, so requests vanished and the CPU wedged.
2. **Cache write-back** (`rvlab_ddr_block_cache.sv`) — a line evicted the
   cycle after it was written was written back with stale contents.
3. **Prefetcher** (`rvlab_ddr_prefetch.sv`) — returned another address's
   data when regions collided in the cache. **Bypassed, not repaired.**

---

## 2. What is switched off or worked around

These are deliberate and documented in the RTL, but they are not finished
work.

- **Prefetcher bypassed** — `USE_PREFETCH = 0` in `rvlab_tlul_ddr.sv`. Costs
  read bandwidth. Repairing it needs the aliasing test
  (`rvlab_ddr_alias_tb.sv`) to pass with it in the path.
- **Response skid buffer bypassed** — `USE_RSP_HOLD = 0` in the same file.
  `student_tl_rsp_hold.sv` is correct and was verified, but once the request
  mux was fixed it was no longer needed for the CPU path, and it changes
  `a_ready` timing in ways that complicated bisecting. Left in the tree,
  switchable.
- **Accelerator retry timer** (`RETRY_CYCLES = 2048` in `student_gemm.sv`) —
  covers the cache's `d_ready` defect for reads. The defect itself is
  unfixed; the CPU never trips it because it is single-outstanding.
- **Start-up self-checks** — `DAV2_STARTUP_CHECKS 0` in `main.c`. Set to 1
  to run the DDR3 random/sequential tests and the per-shape accelerator
  comparison at boot. They cost a few seconds.

---

## 3. What is not done

- **The `mcycle` wrap in the reported frame time.** `main.c` prints
  "inference finished in N kcycles" by differencing two 32-bit reads across
  an interval that wraps every 85.9 s, so the number is wrong by multiples
  of that. The correct figure comes from summing the per-stage timestamps
  with wrap correction, which `dav2_run_fpga.py` does not yet do for you.
- **`sim_ddrmodel_xsim` has no start condition.** The fast whole-SoC
  simulation now elaborates and loads the blob, but the program waits for a
  host to write `dav2_go` and no host exists in simulation. Do not use the
  "blob header looks valid" shortcut — it broke hardware (see
  `DEBUGGING.md`). A `+dav2_autostart` plusarg read by the testbench and
  poked into the flag is the right shape.
- **`student_gemm_ddrpath_tb` fails wholesale.** With the prefetcher
  bypassed to match the board it drives the DUT but reports whole rows
  unwritten and jobs not finishing — a different failure from anything seen
  on hardware, so its environment is wrong. Suspect `ddr3_blk_model`'s
  `DEPTH(1)` against `MAX_INFLIGHT(1)`.
- **Photographs as input.** `dav2_patch_image.py` takes any binary P6 PPM
  and rewrites the image tensors in a blob without torch. This machine has
  no PIL or ImageMagick, so no photo other than the baked-in demo has been
  run. `convert photo.jpg photo.ppm` elsewhere, then
  `--ppm photo.ppm`.
- **Performance.** The accelerator has removed the matmuls from the
  critical path; the CPU's element-wise work (LayerNorm, softmax, GELU,
  requantisation) in software floating point is now the larger share.
  Nothing has been done about it.

---

## 4. Running it

```
source .venv/bin/activate
flow rvlab_fpga_top.program                              # board must be plugged in
python -u src/sw/project/tools/dav2_run_fpga.py --timeout 900
```

writes `build/dav2/fpga_depth.npy`. For another image:

```
python3 src/sw/project/tools/dav2_patch_image.py --synth sphere --out build/x.bin
python -u src/sw/project/tools/dav2_run_fpga.py --blob build/x.bin --out build/x.npy --timeout 900
```

Check any result against the oracle:

```
make -C src/sw/project/host
./src/sw/project/host/dav2_host build/x.bin build/x_host.bin
```

and compare the floats; they must be identical.

---

## 5. Things that will bite you

The full list is in `LESSONS.md`. The five most likely to cost an afternoon:

- `pkill -f "[x]simk"` leaves the `xsim --gui` window alive with a dead
  kernel behind it; it looks hung. Use `pkill -f "[x]sim"`.
- Any change to `src/design/reggen/*.hjson` shifts later register offsets.
  Rebuild **both** `libsys` and `sw_project`, or software writes the DDR3
  reset bit into a read-only register and DDR3 never comes up.
- `with OpenOcd() as ocd:` sends `shutdown` on exit and kills the OpenOCD a
  live board run depends on. Connect and close the socket by hand.
- A CPU wedged on a bus access cannot be halted. Do not spend time on the
  debugger; read the watchdog register at `DDR_CTRL0 + 0x4/0x8` over JTAG.
- A test that passes is only evidence if you have seen it fail. Run the
  negative control.
