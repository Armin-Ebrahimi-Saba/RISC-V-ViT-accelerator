# Handoff: DDR3 stall on the Depth-Anything V2 / GEMM accelerator build

State as of the end of the session that fixed the long-standing hang. Read
`LESSONS.md` for the portable rules and `report_errors.md` for the full
narrative; this file is only what the next person needs to continue.

---

## 1. Where things stand

The model had never reached `block 1/12` on hardware. It now does. The run in
`build/dav2sim/fixrun.log` passed patch embedding and blocks 1-3 and was still
advancing (~200 M cycles per block) when the session ended.

The root cause was **not** in the accelerator or in the model code.

### The defect

`src/rtl/ddr3/rvlab_tlul_ddr.sv` built its TL-UL response from the error
responder by default:

```systemverilog
tl_o = err_resp_rsp;               // a_ready came from here
if (cache_rsp.d_valid) tl_o = cache_rsp;
```

`tlul_err_resp.sv:38` holds `a_ready` high whenever it is idle. Once
calibration completes, `err_resp_req.a_valid` is forced to zero, so the error
responder never accepts anything -- but its `a_ready` was still what the bus
saw. Any request issued while the cache was not ready was handshaked away by
the bus and **accepted by nobody**, so no response for it could ever exist.

The CPU then wedged on a bus access it could never retire. A core that cannot
retire cannot enter debug mode, which is why `halt` never worked and no `pc`
was ever obtainable all session.

### The fix

`a_ready` now comes from whichever module will actually take the request:

```systemverilog
tl_o.a_ready = ctrl_calib_complete ? cache_rsp.a_ready : err_resp_rsp.a_ready;
```

---

## 2. What found it, and what did not

**What found it:** a hardware debug register. `src/rtl/student/student_tl_watch.sv`
snoops the DDR3 TL-UL port and latches the oldest unanswered request; it is
exposed through `ddr_ctrl` as `wdog_addr` (`DDR_CTRL0 + 0x4`) and `wdog_stat`
(`+0x8`, packed `{stall_cycles[15:0], src[7:0], outstanding[4:0], opcode[2:0]}`).
This works on a wedged system because **JTAG system-bus reads keep working when
the core cannot be halted**. One line gave the answer:

```
addr=82045670 PutFullData source=71 outstanding=30 stalled=65535 (SATURATED)
```

Decoding `source`: low 2 bits index `sm1_11`'s hosts (0 corei, 1 cored,
2 dbgsba, 3 student_host); the next bit indexes `student.sv`'s socket (0 DMA,
1 GEMM); the top bit is student_gemm's write flag. So: an accelerator write,
never answered, with 30 transactions outstanding on the port.

**What did not find it, and why** -- worth knowing before repeating it:

- *The debugger.* Structurally unable to answer: there was no architectural
  state to read. Confirmed via `DMControl.haltreq` directly, not just
  OpenOCD's `halt`: `allhalted=0 anyrunning=1 anyunavail=0`.
- *The host-native build* (`src/sw/project/host/dav2_host`). Runs the whole
  model in ~4 s and proves the C terminates. It has no bus, so it can only
  ever find algorithmic bugs.
- *`src/tb/rvlab_ddr_dready_tb.sv`.* Proved a mechanism I had hypothesised was
  real, using stimulus I wrote myself. That is confirmation, not localisation;
  it could not tell me whether that mechanism was the one occurring.

---

## 3. Two earlier fixes, both real, neither sufficient

1. **`rvlab_ddr_block_cache` ignores front-end `d_ready`** and pulses `d_valid`
   for one cycle, so a response is lost if the host is unready that cycle.
   Proven in `src/tb/rvlab_ddr_dready_tb.sv` (set `USE_SKID = 0` to see it
   fail, `1` to see it pass). Worked around by
   `src/rtl/student/student_tl_rsp_hold.sv`, a response skid buffer that ties
   the cache's `d_ready` high and throttles the A channel with a credit
   counter.

   **Caveat:** that skid buffer *amplified* the `a_ready` defect above. Its
   credit throttle deasserts the cache-side `a_ready` regularly, and while the
   mux ignored `a_ready`, every request in those windows was swallowed. It is
   correct now that the mux is fixed, but if you are bisecting bus behaviour,
   know that it changes `a_ready` duty cycle.

2. **`student_gemm` lost-response retry** (`RETRY_CYCLES = 2048`). Still
   present and still useful, but note it recovers *reads*; the transaction the
   watchdog caught was a write.

---

## 4. Traps that cost real time here

- **`openocd.start` resets the core.** Killing the runner to attach means every
  reading you take is a freshly restarted program, not the stalled one. Attach
  to the *running* OpenOCD instead.
- **`OpenOcd.__exit__` sends `shutdown`.** Using `with OpenOcd() as ocd:` to
  "just peek" kills the OpenOCD a live run depends on. Connect and close the
  socket by hand (see the fixed `scratchpad/peek.py` pattern).
- **Never start inference on "the blob header looks valid".** The magic word is
  at the start of the blob, so it lands with the first chunk and the model runs
  against a blob still being written. Symptom: DDR3 read-back fails outright a
  few seconds later. Only `dav2_go` means the transfer finished.
- **Adding a register to a reggen block shifts later offsets.** `wdog_addr`
  and `wdog_stat` pushed `ctrl` from `+0x4` to `+0xc`; a stale `sw.elf` then
  writes the DDR3 reset bit into a read-only register and DDR3 never comes out
  of reset. Rebuild `libsys` and `sw_project` after any hjson change.
- **`mcycle` is 32-bit** and wraps every 85.9 s at 50 MHz. Cycle counts across
  the weight load are wrapped; do not read a drop as a speedup.
- **Console output is a hang generator.** The hostio ring is 1 kB drained over
  JTAG; a device that outruns it spins in `obuf_putc` (`hostio.c:23`) and looks
  exactly like a hung program. Keep target printing minimal.

---

## 5. Uncommitted work

```
 M CLAUDE.md
 M flow/system_tb.py            new sim_ddrmodel_xsim task
 M flow/tools/xsim.py           one --testplusarg per entry (see below)
 M src/design/reggen/ddr_ctrl.hjson   wdog_addr / wdog_stat
 M src/rtl/ddr3/rvlab_tlul_ddr.sv     THE FIX + watchdog + behavioural guard
 M src/sw/project/main.c        handshake restored to strict dav2_go
 M src/sw/project/tools/dav2_run_fpga.py   dm_halt + watchdog readout
 M src/tb/ddr3_blk_model.sv     backdoor blob load via +ddr_blob
?? src/rtl/student/student_tl_watch.sv
```

`flow/tools/xsim.py` had a real bug: all plusargs were joined into a single
`--testplusarg`, so the first absorbed the name of the second and both were
lost. Invisible until a testbench opens the resulting path and gets fd 0.

---

## 5b. Open bug: one lost accumulator word at N=81

The bus hang is fixed and the model now runs to completion, but the depth map
does **not** match the host build: Pearson r = 0.19 against both the host and
the PyTorch reference, while host-vs-reference is r = 0.9998. So the C engine
is right and the target diverges.

What has been ruled out, each by measurement:

- **Weights.** The device-side FNV checksum over all 24,871,428 bytes read
  through the CPU's own path is `fdd83e93`, identical to the file.
- **Kernels.** `dav2_selftest` prints `3ac57cd2` on both host and target.
- **The accelerator's logic.** `student_gemm_tb` runs the exact failing shape,
  81x588x384, against ideal memory and passes 32776 words.
- **The skid buffer.** Bypassing `student_tl_rsp_hold` changes nothing --
  byte-identical failure with and without it.

What remains: on hardware, `dav2_accel_bigcheck` finds accumulator words the
accelerator never wrote (`hw == 0` against a pre-zeroed buffer), so a
PutFullData is lost on the DDR3 path. It is perfectly deterministic -- same
index, same values, six consecutive runs.

Characterisation so far, all with M=384, K=588 unless noted:

| N | result | | N | result |
|---|---|---|---|---|
| 17 | clean | | 79 | 1 word lost |
| 33 | clean | | 80 | clean |
| 49 | 2 words lost | | 81 | 1 word lost |
| 65 | 2 words lost | | 82 | clean |
| 97 | 2 words lost | | 83 | clean |

K does not matter (81x384, 81x588 and 81x592 all fail). Size alone does not
matter: 82x384x1536 writes 125,952 words and loses none. Shrinking the failing
case did not work -- 81x588x64 still fails, but 81x588x16 and everything
smaller is clean -- so there is not yet a case small enough to simulate
against the real cache with waveforms. Finding one is the way in.

## 6. Next steps

1. **Let the current run finish** and confirm `DAV2_RESULT_END` plus a written
   depth map. Nothing has ever produced `build/dav2/fpga_depth.npy`.

   **New, unexplained, and needing attention:** now that the run gets this far
   it prints blob-lookup complaints it never previously reached, e.g.
   `tensor 'blk1.norm1.b' not found in blob` and `'blk6.fc2' has k=0,
   expected 1536`. Either some tensors are legitimately absent and the engine
   tolerates it, or blob reads are returning wrong data. Check against the
   host build, which reads the same file correctly:
   `./src/sw/project/host/dav2_host build/dav2/dav2_weights.bin out.bin`.
   If the host is clean and the target is not, suspect the DDR3 read path --
   the cache's `d_ready` defect is worked around, not fixed.
2. **FPS measured: 0.0102, i.e. 98.4 s per frame** (4.920 G cycles at 50 MHz,
   summed over the stage timestamps with wrap correction). Note the program's
   own "inference finished in N kcycles" line is wrong -- it differences two
   32-bit `mcycle` reads across an interval that wraps several times, and
   under-reports by whole multiples of 85.9 s. Worth fixing in `main.c`.
3. **Commit.** Note `CLAUDE.md` says: do not add yourself as a contributor.
4. **Unfinished: the fast system simulation.** `flow systb_project.sim_ddrmodel_xsim`
   swaps the DDR3 controller, PHY and chip model for `ddr3_blk_model`
   (`RVLAB_DDR_BEHAVIOURAL`), which should run at the measured ~4 kcycles/s
   instead of DDR3's ~600x slower rate. It is **not yet verified**: the
   `ddr3_blk_model: loaded ...` line never appeared, so the behavioural back
   end may not have been in the path. `initial $display` markers now name the
   active back end -- check which one prints. It also needs a start condition
   that does not depend on a host writing `dav2_go` (see the trap above; the
   blob-header shortcut is not acceptable).
5. **Investigate whether `student_tl_rsp_hold` is still needed** now that the
   mux is fixed. It is correct either way, but the cache's `d_ready` defect it
   works around is real and unfixed, so probably keep it.
