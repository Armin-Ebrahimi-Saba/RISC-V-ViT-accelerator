# Lessons from the rvlab int8 GEMM accelerator

Portable notes from bringing a custom TL-UL accelerator up on a CV32E40P SoC
with DDR3 on an Artix-7. Each entry is *symptom -> cause -> fix*, with the
generalisable rule last. Written to be useful on the next project, not to
document this one; the full narrative is in `report_errors.md`.

---

## 1. Platform / bus integration

**A device that pulses `d_valid` for one cycle and ignores `d_ready` loses
responses.**
`rvlab_ddr_block_cache` asserts `d_valid: hit` combinationally and never reads
its front-end `d_ready` (all `d_ready` uses in the file are back-end). TL-UL
requires a device to hold `d_valid` until `d_ready`. A response therefore
vanishes whenever the fabric is busy that cycle. Measured on hardware as
`accepted=444, responses=443` — exactly one read accepted and never answered.
Triggered by dirty evictions: tiles whose writes hit lost nothing; the tile
whose every write missed lost 383 responses for 384 writes.

*Fix, when the RTL is not yours to change:* a retry timer in your own master —
remember the address and source of the request in flight, re-issue it if no
response arrives within N cycles.

> **Rule:** before designing a pipelined master, grep the target device for
> `d_ready` and confirm it actually uses it. A CPU that is single-outstanding
> and always-ready will never expose this, so "it works for the CPU" proves
> nothing about your block.

**A device may track only one transaction.**
The same cache holds a single `ancillary_q` (the source ID) and back-pressures
with `a_ready = ~stall`. An 8-deep outstanding scheme buys nothing against it.

> **Rule:** read the endpoint's outstanding-transaction capability *before*
> sizing your reorder buffer. Latency-hiding is worthless if the device
> serialises anyway.

**A device may require the request to stay asserted for the whole
transaction.**
That cache's tag lookup is stall-gated (`tag_mem[stall ? access_idx_q :
access_idx]`) but its dirty bit is not (`dirty_mem[access_idx]`, the live bus
index). Release `a_valid` after `a_ready` — legal TL-UL — and it reloads the
dirty flag from an unrelated set and evicts the same line forever. Measured:
164 backend accepts per 4000 cycles, `dirty` stuck at 1.

**Nested sockets steal source-ID bits.**
`tlul_socket_m1` shifts the host's `a_source` up and puts its own index in the
low bits. Budget: one local socket takes `$clog2(M)` bits, the crossbar's takes
its own. Check `top_pkg::TL_AIW` against your slot count plus every socket
between you and the device.

**Back-pressure from a register block can reach the CPU's instruction fetch
combinationally.**
Adding a socket put the GEMM register adapter's `outstanding` flag 21 logic
levels from `if_stage_i/instr_rdata_id_o_reg` — 78% routing, WNS −0.077 ns.
Fixed with `HReqPass(1'b0)`, `HRspPass(1'b0)` on our *own* socket
instantiation, no third-party edit: +0.268 ns.

> **Rule:** a socket inserted for address decode is also inserted into the
> critical path. Register its host side by default.

## 2. Debugger / DDR3 quirks

**A handshake flag must not live in memory the debugger writes.**
Once the debugger pushed a blob into DDR3 via system bus access, the CPU
stopped observing debugger writes to DDR3 — it spun on a stale value forever
while a debugger read of the same address returned the new one. Moving the flag
to a BRAM variable, located by `nm`, fixed it immediately.

**Set the memory access mode before any memory access.**
OpenOCD's `riscv set_mem_access sysbus` is issued inside `load_image`. Anything
that reads or writes memory *before* that — e.g. clearing a hostio ring —
fails **silently**. Symptom: stale output from a previous run, mistaken for the
wrong program running.

**Relative paths are resolved by the other process.**
`load_image <relative>` resolves against OpenOCD's cwd, not yours. It and
`verify_image` fail without raising, and `resume` then runs whatever was
already in memory — a plausible-looking wrong program. Always pass absolute,
braced paths and check the returned text.

**Vivado cable drivers:** `No matching targets found` with the USB node
`root:root 0664` means the udev rules were never installed. Vivado's
`install_drivers` calls `./install_digilent.sh` relative to cwd — run it *from
its own directory* or it exits 127. Replug the board afterwards; udev only
applies rules when a device appears.

## 3. RTL bugs worth knowing

**Width truncation in comparisons.** `rb_cnt_q != SW'(OUTSTANDING)` with
`SW=$clog2(8)=3` truncates 8 to 0 and the guard never fires. Introduce
`CW = SW+1` for any counter that must represent the full count.

**Don't-care fields still get checked.** `a_data` is a don't-care on a `Get`,
but TL-UL FIFOs push the whole struct through a `prim_fifo_sync` whose
`DataKnown_A` assertion fails on any X. Driving `a_data: sel_wr ? wr_data :
32'd0` cost nothing and removed 1708 assertion failures.

**A stale index into an array of accumulators.** An index cleared only on one
FSM path retained `n_rows_q` across a state transition; on a *partial* tile
that selected an accumulator whose backing RAM was never written — X. Full
tiles wrapped to index 0 and looked fine.

> **Rule:** partial/last-iteration cases are where indices go stale. Test a
> partial *first* tile, not just a partial last one.

**reggen only flattens single-field registers** to `hw2reg.<reg>.d`. A
multi-field register generates `hw2reg.<reg>.<field>.d`. Collapse debug
registers to one 32-bit field with the layout in the description.

**SystemVerilog declaration order.** `xvlog` rejects use-before-declaration in
continuous assignments even where synthesis accepts it. Keep `assign`s below
the signals they read.

## 4. Host tooling

**Interactive helpers need a TTY.** A lab helper that calls
`termios.tcgetattr(sys.stdin)` dies under redirection. Write a non-interactive
variant rather than running everything in a terminal.

**`hasattr` turns a typo into a silent hang.** Code called
`ocd.hostio_read_str()` — a method that never existed — guarded by `hasattr`,
so it degraded to `None` and polled forever. Never guard a call you believe
exists.

**`print()` block-buffers under redirection.** 50 minutes were spent watching a
transfer that had finished in 67 seconds. Use `flush=True` or `python -u`.

**Chunk long transfers and report progress.** One opaque 25 MB `load_image`
makes "slow" indistinguishable from "hung". Chunked with rate and ETA, the same
transfer is self-explanatory.

**Never leave a completion poll unbounded.** `while (REG32(STATUS) & BUSY);`
turns any hardware stall into a silent freeze. Bound it, print the state on
timeout, and fall back.

**`pkill -f <pattern>` matches its own command line.** Repeated exit-144
self-kills. Use `pgrep -f "[d]av2_run"`, and never put the kill in a command
whose text contains the pattern.

**Kill the parent, not the child.** Killing `xsimk` leaves its `flow` parent
waiting forever; seven accumulated and exhausted 30 GB of RAM. Kill the
top-level process.

## 5. Process — the expensive lessons

**Make the test fail before you make it pass.** A "regression-clean" fix was
shipped to hardware where it deadlocked the accelerator at boot: the testbench
instantiated the module *without* the new parameter, so it defaulted off and
the test exercised the old path. Verifying a parameter you did not enable is
worse than not testing, because it manufactures confidence.

**Instrument; do not infer from silence.** Hours were spent reporting "the CPU
takes 12 minutes on element-wise work" when the CPU was *hung*. The arithmetic
never supported it — 110,000 element operations cannot take 36 billion cycles —
and one `printf` inside the loop settled in five minutes what days of reasoning
had not.

**Check the arithmetic on your own claims.** Every wrong diagnosis here
survived because nobody, including me, divided the cycle count by the work.

**Vary one thing.** The useful results all came from controlled A/B: hold
`a_valid` vs release it; prefetcher in vs out; accelerator on vs forced off;
2048 vs 32768 elements. The useless hours came from reasoning about symptoms.

**The board can be the better instrument.** A deterministic 3-minute hardware
reproduction beat a hand-written memory model that took a day and produced
misleading results — the model was never faithful enough to trust, and every
mismatch looked like a design bug.

**Add debug registers early.** `dbg`/`dbg2`/`dbg3`/`dbg4` exposing FSM state,
the pending address, and accept/response counters found the root cause in one
run after everything else had failed. `accepted=444, responses=443` is a fact;
everything before it was a theory.

**Watchdogs must trigger on the right thing.** A stall watchdog keyed on bus
idleness never fired, because the cache was *livelocked* — busy, making no
progress. Key it on "no response reached the consumer" and count activity in
the window to separate livelock from deadlock.

## 6. Things that were true but did not matter

Recorded so they are not re-investigated: address drift during a stall was a
real bug in our master and fixing it changed nothing on the board; the
prefetcher's combinational `d_anc` looked wrong but is correct by construction
(it asserts `a_ready` and `d_valid` together); the accelerator was never
deadlocked at any outstanding depth; and the requantisation access pattern is
not the trigger — all ten synthetic variants pass at full scale while the real
loop hangs.
