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

## 9. Still open

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
response stalls the core forever. This is the same platform defect the
accelerator works around.

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

## 11. Where this leaves the project

**Working, on hardware:** bitstream and timing; the quantised kernels
(`3ac57cd2`, matching host and simulation); the accelerator, reachable and
correct, cross-checked against software at boot; DDR3 calibration and memtest;
the 24.87 MB weight load; and the accelerator running real DDR3 work at
**9.0 cycles/beat**, surviving lost responses via retry.

**Blocking a full frame:** the CPU stalls in requantisation, the first
element-wise stage after the first GEMM. No FPGA depth map exists. The only
complete inference remains the host x86 build, correlation 0.99984 against
PyTorch.

**The honest summary:** the accelerator problem is solved. What replaced it is
a platform-level memory defect that the accelerator can now survive and the CPU
cannot, and working around it from the student files alone has not yet been
achieved.
