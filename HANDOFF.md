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

## 5b. RESOLVED: the lost write was a cache write-back bug

The model now runs end to end on the FPGA and is **bit-exact with the host
build**: all 15876 pixels identical, r = 1.000000 against the host and
r = 0.999836 against the PyTorch reference. Frame time 93.6 s (0.0107 FPS).

### The defect

`rvlab_ddr_block_cache.sv` issued dirty-line write-backs with
`a_data: data_rdata_raw` -- the raw RAM output, which is one cycle stale when
the line being evicted was written on the immediately preceding access. The
forwarded `data_rdata` (write-first, via `data_wen_q`) exists for exactly that
window and was already used for the front-end response, but never for the
back-end write-back. Two back-to-back misses to the same set -- write a line,
then evict it -- sent the pre-write contents to DDR3.

That is why every lost word was a tile's final row: it is the write issued
immediately before the next tile's first access evicts it. And it is why the
accelerator's counters showed `issued == acked` with the data gone -- the
cache acks the write, then writes back the wrong bytes.

Fix: `a_data: data_rdata`. One line, plus gating the data/dirty lookups on
`stall` to match the tag lookup (correct, but not the mechanism).

### How it was found, and a mistake on the way

Found by making `rvlab_ddr_alias_tb` drive the bus **pipelined**, presenting
the next request the moment the previous is accepted, as a CPU does.
`tlul_test_host` serialises transactions and can never hit the window, which
is why the aliasing test passed on unfixed RTL until then. The negative
control was essential: the first "fix" (stall-gating the lookups) passed on a
test that also passed unfixed, i.e. proved nothing.

The mistake: the "DDR3 random r/w loses 111 words" result recorded in commit
8dcdee1 was **the checker, not the memory**. The random sequence hits some
addresses twice and the naive check compared against the earlier write.
111 was exactly the number of repeated addresses. With repeats skipped, DDR3
passes 0/65536 on random, in-cache and sequential patterns. Do not trust a
memory test that does not account for duplicate addresses.

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
2. **FPS measured: 0.0107, i.e. 93.6 s per frame** (4.681 G cycles at 50 MHz,
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
