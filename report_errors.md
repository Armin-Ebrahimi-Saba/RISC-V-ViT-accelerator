# Error log — accelerator bring-up

Every failure hit while bringing the int8 GEMM accelerator up on the Nexys
Video board, with its cause and fix. Ordered by kind, not by time.

---

## 1. Environment and tooling

| Error | Cause | Fix |
|---|---|---|
| `HjsonDecodeError` line 131 | Compact one-line `fields:` form — hjson unquoted values run to end of line | Verbose multi-line field blocks |
| `undefined macro 'INIT_MEM_FILE'` | Not defined for standalone analysis | `-d INIT_MEM_FILE=empty.mem` |
| `undefined macro 'ASSERT'` | `prim_assert.sv` not compiled first | Compile it before the generated `reg_top` |
| `Module <BUFG> not found`, `'glbl' is not declared` | Missing Xilinx sim libraries | `-L unisims_ver -L secureip`, compile `glbl.v`, add `work.glbl` as a top |
| `Could not find RISC-V GCC toolchain` | Not on PATH | `export PATH="$HOME/.local/opt/riscv/bin:$PATH"` |
| `openocd` / `xterm` not found | Not installed | `apt install openocd xterm` |
| `No matching targets found on connected servers` | No Xilinx udev rules; USB node was `root:root 0664` | Vivado `install_drivers`, then replug the board |
| `install_drivers` rc=127 | Calls `./install_digilent.sh` relative to the current directory | Run it from its own directory |
| Repeated exit 144 | `pkill -f <pattern>` matched its own command line | `pgrep -f "[d]av2_run"`, and never put the kill in a command containing the name |

## 2. RTL defects

**Partial-tile X — 1708 assertion failures in system simulation.**
`t_q` was cleared only at `klast`, so the `ST_DRAIN → ST_MAC` transition left it
at `n_rows_q`. On a full tile that wraps to `acc_q[0]` harmlessly; on a
**partial** tile it indexed an accumulator whose A-tile row was never loaded, so
`wr_data` was X — and `a_data` was loaded into every request, reads included.

    a_data: sel_wr ? wr_data : 32'd0     // never put accumulator data on a Get
    t_q <= '0;                           // clear on leaving ST_DRAIN

The testbench missed it because it needs a partial *first* tile **and** a weight
stream long enough that reads are still being issued after the first drain. Added
`N=8, K=32, M=16` to both testbenches and confirmed it fails (98 errors) against
the unfixed RTL before restoring the fix.

Others: `rb_cnt_q != SW'(OUTSTANDING)` truncated 8 to 0 (added `CW = SW+1`);
`signed'()` replaced with `$signed()`; `unique case` on a pre-reset X replaced
with a plain `case`; `dbg` collapsed to one field because reggen only flattens
single-field registers to `.d`.

## 3. Host tooling defects

| Error | Cause | Fix |
|---|---|---|
| `termios.error: Inappropriate ioctl` | Called the lab's interactive `run_prog()`, which needs a TTY | Wrote `start_program()` — same sequence without the console |
| Marker never matched | Called `ocd.hostio_read_str()`, **a method that never existed**; `hasattr` degraded it silently to `None` | Wrote `read_hostio()`, mirroring the ring-buffer protocol but returning text |
| Board ran `test_rvlab`, not our program | ELF path was relative and OpenOCD resolves against *its own* cwd; `load_image` and `verify_image` failed **silently** | Absolute braced path, and check the command output |
| Stale output; program looked wedged | `hostio_clear()` runs *before* `load_image()`, which is what issues `riscv set_mem_access sysbus`; without it the writes were discarded silently | Set sysbus mode first, clear immediately before `resume` |
| No progress for ~50 min | `print()` block-buffers under redirection | `flush=True`, `python -u` |
| Program hung forever | `while (REG32(GEMM_STATUS) & STATUS_BUSY);` unbounded | Bounded wait, diagnostics, permanent fallback to the CPU kernel |
| **CPU never saw the handshake** | The `go` word lived at `0x8F000000` in DDR3; once the debugger has pushed the blob via system bus access, the CPU stops observing debugger writes to DDR3 | Moved the flag to a BRAM variable `dav2_go`, located by `nm` |

## 4. Integration and platform

**Timing failure, WNS −0.077 ns, 1 of 49,977 endpoints.**
The GEMM register block's back-pressure reached the CPU's instruction fetch
combinationally through the socket added in `student.sv` — 21 logic levels, 78%
routing. Fixed with `HReqPass(1'b0)`, `HRspPass(1'b0)` on our own instantiation
(no third-party source touched): **+0.268 ns**, later +0.370.

**Accelerator hangs on its first DDR3 job.**
A debug register added to the block showed `state=ST_LOAD_A, rb_cnt=8,
rd_left` frozen across two samples: 9 reads issued, 1 answered.
`rvlab_ddr_prefetch` asserts `a_ready` and `d_valid` together and derives the
response's ancillary combinationally from the request *currently presented*, so
it holds no per-transaction state. A pipelined host loses every response after
the first. Added a `MAX_INFLIGHT` parameter; `student.sv` uses `MAX_INFLIGHT(1)`.
This never showed up before because the CPU stalls on every load and is
single-outstanding by construction — the accelerator is the first
multi-outstanding master in this SoC — and because BRAM's `tlul_adapter_sram`
tracks transactions properly, which is why the boot self-test passed.

## 5. Diagnoses that were wrong

Recorded because each cost time, and each was settled by building a test rather
than reasoning further about the symptom.

* **"The JTAG blob transfer takes 1–2 hours."** It takes **67 s**. Buffered
  output hid the completion, so a finished transfer was mistaken for a slow one.
* **"The CPU can't see the handshake — cache coherency."** It sees it fine; the
  real cause was the sysbus-mode ordering above.
* **"`load_image` halts the CPU and never resumes it."** `targets` reported
  `running` both before and after.
* **"A genuine deadlock in the accelerator."** The stall watchdog (3000 idle
  cycles) never fired, and the reproduced job completed bit-exact. The block is
  latency-bound, not stalled: 0.116 MAC/cycle against a slow memory versus 12.4
  against a fast one, same RTL.
* **"An `nt=1` partial-tile edge case."** `N=1`, `N=17` (the 16→1 transition)
  and `N=16` all pass at `MAX_INFLIGHT=1` against single-outstanding memory.
* **"The lost write is in `student_gemm.sv`'s drain logic."** Per-job write
  counters showed `issued == acked == owed` on the failing job. The accelerator
  issued every write; the cache lost one on write-back (section 15).
* **"DDR3 loses 111 words under random access."** The test's checker compared
  against a superseded write; 111 was the number of repeated addresses. Memory
  passes 0/65536 with the checker fixed (section 16).
* **"Stall-gating the cache's data/dirty lookups fixes the lost write."** It
  passed a test that also passed on unfixed RTL. The real mechanism was the
  write-back data source (section 15).

## 6. The board hang — root cause and fix

**Symptom.** The accelerator froze on the sixth tile of patch embedding,
deterministically: `state=ST_MAC, rb_cnt=1, rd_left=56299`, identical across
runs. Ruled out along the way, each by evidence rather than argument:
outstanding-request concurrency, shape/partial-tile dependence, and address
drift during a stall.

**The measurement that cracked it.** Debug registers exposing the bus counters:

    bus: areq_valid=0 a_ready=1 d_valid=0 err=0
         addr=8003bd50 accepted=444 responses=443

Exactly one response missing. The read was accepted and never answered — not a
stall, not back-pressure, not an error.

**Root cause.** `rvlab_ddr_block_cache` asserts its front-end response for one
cycle without ever consulting `d_ready`:

    fe_rsp_o = '{ a_ready: ~stall, d_valid: hit, d_anc: ancillary_q, ... };

Every `d_ready` in that file is back-end. `rvlab_ddr_cache.sv:33` dutifully
forwards `tl_i.d_ready` down and it is discarded. TL-UL requires a device to
hold `d_valid` until `d_ready`; this one does not, so a response is lost
whenever the fabric is not ready that cycle. The CPU never exposes it: it is
single-outstanding and always ready.

Corroborated independently by UberDDR3 issue #16, "Wishbone ACK sometimes
missing", reporting the same class of failure after "several tens to hundreds
of transactions" (our DDR3 controller is UberDDR3, GPL-3.0, Angelo Jacobo).

**When it happens.** Per-tile hardware measurements:

    tile:   549611 cycles, 61152 beats, retries=0,   rsps=1760    (nt=16)
    tile:   549327 cycles, 61152 beats, retries=0,   rsps=1760
    tile:   549328 cycles, 61152 beats, retries=0,   rsps=1760
    tile:   549406 cycles, 61152 beats, retries=0,   rsps=1760
    tile:   549432 cycles, 61152 beats, retries=0,   rsps=1760
    tile: 25567606 cycles, 56742 beats, retries=383, rsps=57126   (nt=1)

Tiles with `nt=16` lose nothing. The `nt=1` tile issues 384 writes — one per
weight row, each advancing the C pointer by `c_stride` = 324 bytes, so **every
write lands in a fresh line and forces a dirty eviction** — and loses 383
responses. Essentially every eviction costs the following read its response.
With `nt=16` the 16 writes are consecutive words, so evictions are rare.

**Fix (in student files only).** `student_gemm` remembers the address and slot
of the read in flight; if no response arrives within `RETRY_CYCLES`, it
re-issues that exact read under its original source ID. Recovery, not a cure —
the defect is in platform RTL that must not be modified.

Validated in the order that matters: a memory model that drops the Nth read
response (`tlul_test_mem.DROP_NTH_RD`) makes the block hang exactly as hardware
did; with retry it completes bit-exact; and the healthy-memory regression is
unchanged at 1672 words / 31,615 cycles.

**Throughput.** 9.0 cycles/beat on clean tiles. `RETRY_CYCLES` was initially
65,536 — a guess made before latency could be measured — costing
383 x 65,536 = 25.1M cycles, 98% of that tile. Retuned to 2048 from the
measured 8.3 cycles/beat.

## 7. A fix that was right in simulation and wrong on the board

`rvlab_ddr_block_cache` also reloads its dirty flag from the live bus index
while stalled, though its tag lookup is stall-gated. A host whose address
drifts mid-transaction makes it evict the same line forever — confirmed by A/B
in simulation (83 us holding the request; 164 accepts per 4000 cycles
releasing it). `student_gemm` did drift and no longer does.

**It made no difference on the board.** A genuine bug, not this one.

## 8. STRICT_SERIAL — a self-inflicted deadlock

An attempt to avoid response collisions by never having a read and a write
outstanding together **deadlocked**: `ST_DRAIN` must issue writes, but the read
engine prefetches beats consumed only in `ST_MAC`, so `rb_cnt` never falls to
zero and no write can ever issue.

Worse than the bug was the process: the regression test was reported clean, but
the testbench instantiates `student_gemm` without that parameter, so it
defaulted off and the test exercised the old path entirely. A parameter that
was never enabled was declared safe and put on hardware, where it disabled the
accelerator at boot. Reverted, with the dead end recorded at the declaration.

## 9. Still open (superseded — see section 17 for the current state)

The accelerator works on hardware. **The bottleneck is now the scalar CPU
code**: the accelerator finishes patch embedding's GEMM in ~0.5 s, while the

## 10. The CPU hangs on the same defect

Measured with `dav2_progress` probes at every phase boundary of patch
embedding, and then inside the loops:

    [patch embedding]        t=3354700 kcycles
    [  pe: dav2_qw done]     +13k        0.3 ms
    [    qgemm: setup done]  +2,369k     47 ms   im2col + allocation
    [    qgemm: matmul done] +4,054k     81 ms   accelerator, 6 tiles
    [    qgemm: amax done]   +962k       19 ms   min/max scan
    (requantisation)         never completes

Everything up to the requantisation takes **148 ms**. The requantisation loop
never finishes -- and probes inside it show it dies **within its first outer
iteration**, roughly 384 strided reads and 384 sequential 2-byte writes in.

The CPU is not slow here, it is **stalled**. It has no retry: one lost load
response stalls the core forever.

> **Superseded — see section 12.** The observation (stalled, not slow) was
> right. The attribution was wrong: the trigger was not the cache's lost
> responses but a request-acceptance defect one level up, in the DDR3 request
> mux. Everything below about *which loop* stalls is therefore describing
> where the CPU happened to be standing when the port jammed, not a property
> of requantisation.

**This corrects several earlier statements in this log and in conversation.**
"Patch embedding takes ~12 minutes of CPU element-wise work" and the earlier
">13 minutes" were both hangs recorded as slowness, inferred from silence
because nothing was instrumented inside the loops. The arithmetic never
supported them: 110,000 element operations cannot take 36 billion cycles.

**Attempted fix that did not work.** The requantisation wrote with a stride of
`M*2` = 768 bytes, so nearly every store missed and forced a dirty eviction.
Reordering the loop (n outer, m inner) makes the stores sequential and strides
the reads instead, which should turn dirty evictions into clean fills. Verified
bit-identical on a full host inference. **It still hangs**, at the same place.
So stride is not the trigger. The reorder is kept: it is a genuine locality
improvement and costs nothing.

**What is not established.** Which access patterns are fatal. Pure-looking
loops survive: im2col completes 47,628 elements, the min/max scan completes.
Both requantisation orders die in a few hundred accesses. But im2col also
interleaves reads (image) and writes (arena) across two DDR3 regions, so
"interleaved read/write" is *not* a correct discriminator, and I do not have
one. The loss rate the accelerator measured -- 383 in 57,126, about 1 in 150 --
is consistent with the CPU dying within its first few hundred interleaved
accesses, but consistency is not proof.

## 11. The real root cause — a request nobody accepted

`rvlab_tlul_ddr.sv` built its TL-UL response by starting from the error
responder and overriding it when the cache answered:

    tl_o = err_resp_rsp;                 // a_ready comes from HERE
    if (cache_rsp.d_valid) tl_o = cache_rsp;

`tlul_err_resp.sv:38` holds `a_ready` high whenever it is idle. Once
calibration completes, `err_resp_req.a_valid` is forced to `0` so the error
responder never *takes* a request — but its `a_ready` was still what the bus
saw. Any request issued while the cache was not ready was therefore
handshaked away by the fabric (`a_valid && a_ready`) and **accepted by
nobody**. No response for it can ever exist.

The fix is to source `a_ready` from whichever module will actually accept:

    tl_o.a_ready = ctrl_calib_complete ? cache_rsp.a_ready : err_resp_rsp.a_ready;

**Effect.** Before: the run never reached `block 1/12` in any attempt, across
the whole bring-up. After: patch embedding, then blocks 1 through 7 and
counting, at ~250 M cycles per block.

**The skid buffer made this worse, not better.** `student_tl_rsp_hold`
(section 6) throttles the A channel with a credit counter, so it deasserts the
cache-side `a_ready` routinely — and with the mux ignoring that signal, every
request issued in those windows was swallowed. An intermittent defect became a
systematic one. A correct fix to one layer can amplify a defect in another;
re-measure after every such change rather than assuming monotone improvement.

> **Rule:** when a response never arrives, do not assume the response was
> lost. Check first that the request was ever *accepted*. A mux that selects a
> response source must select the matching `a_ready`, or the two halves of the
> handshake describe different modules.

## 12. How it was finally found — a register, not a debugger

Every software instrument failed, for one structural reason: **cv32e40p
cannot enter debug mode while an outstanding bus access cannot retire.**
Driving `DMControl.haltreq` over DMI directly (bypassing OpenOCD's `halt`)
returned `allhalted=0, anyrunning=1, anyunavail=0` — the debug module was
talking to the hart fine; the hart simply could not stop. There was no `pc`
to read, and there never would be. Days of PC-hunting were chasing a reading
that does not exist in this failure mode.

What still works on a wedged system is **JTAG system-bus access**. So the
answer was to record in hardware what the core can no longer be asked.
`src/rtl/student/student_tl_watch.sv` snoops the DDR3 TL-UL port and latches
the oldest unanswered request; `ddr_ctrl` exposes it at `+0x4` (address) and
`+0x8` (`{stall_cycles, source, outstanding, opcode}`), with a saturating
stall counter so a pinned maximum is unambiguous. One line named the culprit:

    ddr watchdog: addr=82045670 PutFullData source=71 outstanding=30
                  stalled=65535 cycles (SATURATED -- never answered)

Decoding `source=71`: low 2 bits = 3 → `student_host` on `sm1_11`; next bit =
1 → `student_gemm` inside `student.sv`; top bit set → its write engine. A
*write*, from the accelerator, to the activation arena — while the accelerator
itself reported `state=IDLE` with zero outstanding writes. `outstanding=30`
saturating was the tell: requests were leaving and nothing was coming back.

> **Rule:** when the debugger structurally cannot answer, add a register that
> can. A dozen flops on a bus port, readable over sysbus, beat every software
> probe on this project — and unlike printf instrumentation, an observer that
> drives nothing cannot perturb the bug it is watching.

## 13. Two self-inflicted failures worth remembering

**A probe that killed the thing it measured.** A "read-only" script used
`with OpenOcd() as ocd:` — whose `__exit__` sends `shutdown`. It terminated
the OpenOCD the live run depended on, leaving the runner blocked on a dead
socket. Connect and close the socket by hand when attaching to a session you
must not disturb.

**A start condition that fired too early.** To let a simulation start without
a host, `main.c` accepted "blob header valid" as an alternative to the
`dav2_go` handshake. The magic word is at the *start* of the blob, so it
lands with the first chunk: the CPU began inference against a blob still being
written underneath it, and DDR3 read-back failed outright seconds later.
Reverted. Only the flag means the transfer finished.

## 14. The prefetcher returns the wrong line under aliasing

Once section 11's hang was gone, the run reached the transformer blocks and
began reporting tensors as absent that are demonstrably present:

    dav2: tensor 'blk1.qkv.w' not found in blob
    dav2: 'blk6.fc1' has k=2143289344, expected 384      (0x7FC00000, a NaN pattern)

Each miss failed **twice in a row** when `dav2_find` was made to retry, so it
was not a transient bad read. A device-side FNV checksum over all 24,871,428
blob bytes, read through the CPU's own path, matched the file exactly. So the
weights were correct in DDR3 and the CPU's *cached* reads of them were wrong.

**Cause.** `rvlab_ddr_prefetch` returns the alias partner's line, another
set's line, or zeros when two regions collide in the direct-mapped cache. The
blob at `0x80000000` and the arena at `0x82000000` differ only in tag, so
every set aliases, and the 21 kB blob directory is scanned linearly on every
lookup while the accelerator writes the arena.

**Proof.** `src/tb/rvlab_ddr_alias_tb.sv` walks that pattern against the real
`rvlab_ddr_cache` + `rvlab_ddr_prefetch` with `ddr3_blk_model` behind them.
With the prefetcher in the path, 65 of 256 reads return wrong data
(`got 820100a5 want 800100a5 <-- ARENA'S DATA`). With it bypassed, 256/256
pass — so the cache is not at fault.

**Fix.** `USE_PREFETCH = 1'b0` in `rvlab_tlul_ddr.sv`. This is a bypass, not
a repair; it costs read bandwidth. On hardware the miss count went from 61
per run to 0. The prefetcher's invalidation logic *looks* correct on
inspection (PUT matches invalidate, pending entries go `Stale`), which is why
it is bypassed rather than patched — the fault was not localised inside it.

## 15. One lost accumulator word — a cache write-back bug

With lookups fixed, inference completed but the depth map was wrong:
r = 0.19 against the host build, which itself matches PyTorch at r = 0.9998.
The engine is correct; the target diverges.

**Narrowing it.** `dav2_accel_bigcheck` compares the accelerator against the
CPU kernel at model shapes, all buffers in DDR3. Four of five shapes were
bit-exact; patch embedding, 81×588×384, lost **exactly one word** of 31,104,
deterministically (`hw 0`, `sw -4690841`, same index six runs running).
A shape sweep showed N is the trigger, not K: 49, 65, 79, 81, 97 fail;
17, 33, 80, 82, 83 pass. The same shape passes in `student_gemm_tb` against
ideal memory, so the accelerator's logic was clean and the fault needed the
real cache's back-pressure.

Three more measurements, each ruling something out:

- Evicting the line and re-reading still gave 0 → **lost, not read stale**.
- Every lost word was a tile's final row (`n ≡ 15 mod 16`, or the single row
  of a partial last tile) → the write issued immediately before the next
  tile's first access evicts it.
- Per-job counters packed into `dbg2` (`{writes issued, writes acked}`)
  showed `issued == acked == nt × M` on the failing job → the accelerator put
  the write on the bus and the bus acknowledged it. **This exonerated
  `student_gemm.sv`** and moved the fault downstream.

**Cause.** `rvlab_ddr_block_cache.sv` issued dirty-line write-backs with
`a_data: data_rdata_raw` — the raw RAM output, which is one cycle stale when
the line being evicted was written on the immediately preceding access. The
write-first-forwarded `data_rdata` (via `data_wen_q`) exists for exactly that
window and was already used for the front-end response, but never for the
back-end write-back. Two back-to-back misses to the same set — write a line,
then evict it — sent the pre-write contents to DDR3. The cache acks the
write, then writes back the wrong bytes; that is why `issued == acked` with
the data gone.

**Fix.** One line: `a_data: data_rdata`. (Also gated the data/dirty lookups
on `stall` to match the tag lookup — correct, but *not* the mechanism; see
below.)

**Proof, and why the first attempt proved nothing.** The aliasing testbench
passed on unfixed RTL because `tlul_test_host` waits for each response before
presenting the next request, so a stalled miss never has a second request
behind it — the exact condition the defect needs. Driving the bus
**pipelined** (next request presented the moment the previous is accepted, as
a CPU does) made it fail on unfixed RTL at a single word, `80004340`, whose
write was immediately followed by a write to the same set. The stall-gating
"fix" passed that test — and so did unfixed RTL, i.e. the test had no power
until the driver was pipelined. The negative control is what separated a
real fix from a plausible one.

**Result.** Every GEMM shape bit-exact. Full inference output identical to
the host in all 15,876 pixels — r = 1.000000 against host, r = 0.999836
against PyTorch. 93.6 s per frame, 0.0107 FPS.

This fix is in platform RTL (`rvlab_ddr_block_cache.sv`, RVLab code, not a
vendored third-party library) and is worth upstreaming.

## 16. A finding that was wrong: "DDR3 loses 111 words"

Recorded in commit `8dcdee1` and retracted in `1c8aa6f`. A random
write-then-read test over the 64 MB arena reported 111 of 65,536 words wrong,
stable across runs, with the wrong value always being *another valid word*.
Two controls — random access confined to 8 kB (no evictions) and sequential
access across the arena — both passed, which was read as "only out-of-order
eviction loses data".

**It was the checker.** The random sequence picks some addresses more than
once, and the naive check compared each op against its own value rather than
the *last* value written there. The number of repeated addresses in the
sequence is exactly 111. The "wrong" values were correct: they were the later
write. The controls "isolated" nothing — sequential never repeats an address
and the in-cache test compared immediately after each write.

With a repeat-address bitmap, DDR3 passes **0/65536** on random, in-cache and
sequential patterns. The memory was never at fault.

> Rule: a memory test that does not account for duplicate addresses is
> measuring its own generator. Build the last-writer check in before
> trusting a single number from it — the simulation testbench had this check
> from the start and the hardware test did not.

## 17. Where this leaves the project

**Done and verified on hardware.** The full Depth-Anything V2 inference runs
on the FPGA and is bit-exact with the host reference. Three real defects were
found and fixed along the way, none of them in the accelerator or the model:

| Defect | Where | Fix | Section |
|---|---|---|---|
| `a_ready` taken from the idle error responder; requests accepted by nobody | `rvlab_tlul_ddr.sv` | source `a_ready` from the module that will accept | 11 |
| prefetcher returns aliased lines | `rvlab_ddr_prefetch.sv` | bypassed (`USE_PREFETCH=0`) | 14 |
| write-back uses stale RAM output for a line written the previous cycle | `rvlab_ddr_block_cache.sv` | `a_data: data_rdata` | 15 |

Plus the cache's `d_ready` defect (section 6), still worked around by
`student_tl_rsp_hold` rather than fixed at source.

**Instruments that made the difference**, in order of leverage:

1. `student_tl_watch` — hardware watchdog register, readable over JTAG
   sysbus on a wedged core. Found section 11 in one line.
2. `dav2_accel_bigcheck` — accelerator vs CPU kernel at model shapes in DDR3.
   Localised section 15 to one word, then exonerated the accelerator.
3. `rvlab_ddr_alias_tb` driven **pipelined** — reproduced section 15 in
   simulation. Serialised driving could not.
4. The negative control, every time. Two "fixes" in this project passed
   tests that also passed unfixed.

**Not done.**

- `rvlab_ddr_prefetch` is bypassed, not repaired. Read bandwidth is lower
  than it could be.
- `student_gemm`'s retry re-issues reads only; a dropped write ack would
  still wedge the block (ultrareview finding, pre-existing). Now that
  requests are no longer swallowed it may never trigger, but it is a real
  hang path.
- `main.c` prints `inference finished in N kcycles` from two 32-bit `mcycle`
  reads across an interval that wraps; it under-reports by whole multiples
  of 85.9 s. Sum the stage timestamps with wrap correction instead.
- `sim_ddrmodel_xsim` works (blob loads, behavioural back end reports) but
  still needs a start condition that does not depend on a host writing
  `dav2_go`. The blob-header shortcut is **not** acceptable (section 13).
- `student_gemm_ddrpath_tb` fails wholesale in its current environment and
  does not model the board; do not trust it until that is fixed.
- Frame time is 93.6 s. The accelerator is ~36× faster than the CPU kernel
  for GEMM, so the remaining time is CPU-side float bookkeeping and the
  JTAG-bound console; profiling has not been done.

**All work is committed** through `1c8aa6f` on `student-gemm-accelerator`,
pushed to the `armin` remote.
