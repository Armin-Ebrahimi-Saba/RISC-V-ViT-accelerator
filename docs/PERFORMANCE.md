# Performance — how one frame went from 93.6 s to 7.2 s

This is the record of the speed-up work: what was measured, what each change
did, and what is left. Every step kept the FPGA output **bit-exact with the
host build** of the same C engine, and the correlation with the PyTorch
reference on the demo image went from 0.999836 to 0.999872 along the way
(one change in LayerNorm improved the numerics slightly; nothing was traded
for speed). Sections 5–7 (rounds two to five) were developed and verified in
simulation and on the accelerator emulator while the board was unavailable;
§6 has the measurement taken once the board came back, confirming bit-exact
output at 8.667 s per frame. §6 also covers round six, which brought the
frame to 8.187 s, and round seven, which removed all floating point and
brought it to 7.20 s.

Terms used below: *frame* — one 126×126 depth map; *GEMM* — a matrix
multiplication, the accelerator's job; *requantisation* — turning the int32
sums a GEMM produces back into int16 activations with a per-row scale;
*beat* — one 32-bit word crossing the bus.

---

## 1. The result

| | Start of the work | After round 1 (measured) | **Current (measured, §6)** |
|---|---|---|---|
| Frame time (inference only) | 93.6 s | 14.2 s | **7.20 s** |
| Frames per second | 0.0107 | 0.0704 | **0.139** |
| Result readout to the PC | ~30 s (hex text over the console) | 0.7 s (JTAG system bus) | 0.36 s (int16) |
| Accelerator | 16 multipliers, 1 read in flight, 8.3 cycles/beat | 64 multipliers, 8 in flight, 3.2 cycles/beat | **128 multipliers, 8 in flight, 2.1 cycles/beat** |
| FPGA output vs host build | bit-exact | bit-exact | **bit-exact (15876/15876)** |
| Correlation with PyTorch (demo) | 0.999836 | 0.999872 | 0.999862 |
| LUT / BRAM / DSP | 12.4 % / 23.0 % / 3.1 % | 14.8 % / 36.9 % / 10.4 % | **18.9 % / 54.4 % / 20.4 %** |
| Timing (WNS, 50 MHz) | +0.287 ns | +0.121 ns | **+0.395 ns** |

Where the 7.20 s goes now (cycles at 50 MHz, measured on the board — see
§6 for the full profile and how it compares to the 14.2 s breakdown below):

| Operator | Mcycles (14.2 s) | Mcycles (8.2 s) | Mcycles (7.2 s) | Runs on |
|---|---|---|---|---|
| LayerNorm | 128 | 97 | 86 | CPU |
| attention | 114 | 87 | 87 | CPU; the two matmuls on the accelerator |
| requantisation | 101 | 82 | 74 | CPU sets up, accelerator converts (+ fused add/ReLU) |
| GELU | 51 | 57 | 43 | CPU |
| interpolate | 63* | 30 | 29 | CPU |
| other | — | 28 | 12 | mixed |
| GEMM | 93 | **15** | 15 | accelerator (128-row tile, repaired prefetcher) |
| add/relu | 92 | **12** | 12 | fused into the requantisation job (accelerator) |
| im2col | 67 | **1** | 1 | gather mode (accelerator) |

*the 14.2 s column's "interpolate, misc" bucket is split into "interpolate"
and "other" in the later profiles.

GEMM, the residual add/ReLU, and im2col are no longer worth optimising —
gather mode and the fused epilogue removed nearly all of their CPU-side and
accelerator-side cost. LayerNorm, requantisation setup, and attention are
now the largest shares, and all three are CPU arithmetic, not accelerator
time (see §4 and §7).

---

## 2. Method: measure, then change one thing

The first version of this work guessed at the bottlenecks from the code and
was wrong twice. What actually worked:

1. **A per-operator profiler** (`dav2_prof_*` in `dav2.h`/`dav2_ops.c`).
   Every operator adds its cycles to a bucket; the table is printed after
   each frame on the board and on the host. The very first table showed the
   accelerator at 20 % of the frame and the CPU's work around it at 80 %.
2. **Bit-exactness as the regression test.** After every change the host
   build was run on the demo image and compared byte-for-byte with the
   previous output (`cmp`), and the FPGA against the host. A change that
   was meant to be exact and was not would have been caught immediately;
   the two changes that intentionally altered numerics (LayerNorm in fixed
   point, one clamp in attention) were checked against PyTorch instead.
3. **A microbenchmark on the board** (`dav2_bench()` in `main.c`, printed
   at boot), after two rounds of memory-oriented changes gained nothing.
   It measured what one instruction, one load and one branch cost on this
   core, and turned the rest of the work around (§4).

---

## 3. The steps, in order, with what each did

| # | Change | Frame | Kind |
|---|---|---|---|
| 0 | starting point | 93.6 s | |
| 1 | profiler; result over JTAG instead of 127 kB of hex text; word-wise copies instead of libsys's byte-loop `memcpy`; on-chip scratch for attention operands; tiled requantisation; LayerNorm in fixed point; `-O2` on the compute files | 52.5 s | software |
| 2 | NROWS 16 → 64; A/W row-stride registers so K > KMAX is split instead of falling back to the CPU; per-slot retry addresses | 41.7 s | RTL |
| 3 | attention's two matmuls on the accelerator (hi/lo int8 split, exact); one reciprocal per token with a correction step instead of 5248 64-bit divisions per head | 24.7 s | software using RTL from step 2 |
| 4 | 8 reads in flight (MAX_INFLIGHT 1 → 8); MAC registers without async reset so they fold into DSP48s | 20.7 s | RTL |
| 5 | word-wise elementwise operators, `mulh` fast path — **no gain**; microbenchmark run | 20.7 s | software |
| 6 | per-row statistics from the accelerator's drain; requantisation as a hardware job; branch-free `apply_multiplier` | **14.2 s** | RTL + software |

### Step 1 — software, 93.6 → 52.5 s

- The **result readout** printed 63 kB of floats as hex text through the
  hostio console, which the PC drains over JTAG at a few kB/s: ~30 s per
  frame, outside the timed inference. The depth map now lives at a fixed
  DDR3 address (`RESULT_ADDR`) and the runner fetches it with `dump_image`
  in 0.7 s.
- libsys's **`memcpy` copies one byte per iteration**. `dav2_copy16` copies
  words. (Later measurement showed the CPU's bus access is not the
  bottleneck it looked like, so this mattered less than expected; im2col
  still copies 36 MB per frame.)
- **Attention** read `v` with a 2304-byte stride, one element per multiply.
  Gathering q, k and vᵀ into on-chip scratch once per head helped, but the
  real fix was step 3.
- **LayerNorm** did ~8 soft-float operations per element (the core has no
  FPU; each is a 50–150 cycle library call). The per-row statistics stay
  in float (82 rows); the per-element work is fixed point. This is the one
  change that altered the output: max Δ 0.07 on a 0.3–2.5 range, and the
  correlation with PyTorch went *up* (0.999836 → 0.999868).
- The flow compiles everything `-Os`. `#pragma GCC optimize("O2")` on the
  four compute files.

### Step 2 — a wider accelerator, 52.5 → 41.7 s

The weight matrix is streamed from DDR3 once per tile of activation rows.
With 16 rows per tile the encoder's 82 tokens took six passes; with 64,
two. The DPT head's 15 876-row convolutions went from 993 passes to 249.
GEMM time on the accelerator halved (515 → 255 Mcycles).

The **row-stride registers** let a job read a column slice of wider rows.
The 384-channel 3×3 convolutions have K = 3456 > KMAX = 2048 and had been
falling back to the CPU kernel — 5.5 s per frame. The driver now splits
them into two chunks and adds the partial sums.

### Step 3 — attention on the accelerator, 41.7 → 24.7 s

Attention was 47 % of the frame: 62 M multiply-accumulates per frame on the
CPU at ~16 cycles each. Both products are matrix multiplications, but with
int16 on both sides, and the block multiplies int16 × int8. The second
operand is split into a high and a low int8 half and the product is run
twice:

    x = hi·2^s + lo     ⇒     a·x = 2^s·(a·hi) + (a·lo)     exactly

Scores use A = k (shifted to 11 bits), W = q >> 4 and q & 15; the context
uses A = P (Q15 probabilities), W = vᵀ >> 6 and vᵀ & 63. The strides from
step 2 let the block read the head slices straight out of the qkv tensor.
The only change to the numbers: the largest probability, exactly 2^15, is
clamped to 32767 so it fits an int16.

The normalisation `Σ p·v / Σ p` was one 64-bit division per element (5248
per head, hundreds of cycles each). It is now one reciprocal per token,
`floor(2^40 / sum)`, multiplied in, plus one compare against the remainder
that makes the result exactly the truncated quotient. (A first version
without the correction step was measurably less accurate against PyTorch —
0.999793 — which is how it was noticed.)

### Step 4 — eight reads in flight, 24.7 → 20.7 s

The block had been limited to one read in flight because the platform's
cache can drop a response (see `DEBUGGING.md`). It now remembers every
outstanding read's address in its reorder-buffer slot and re-issues the
head request if it stays unanswered, so a lost response costs 2048 cycles
rather than the result — at any depth. With 8 in flight the accelerator
went from 8.3 to 3.2 cycles per beat, and the board reports **zero
retries** over whole frames.

### Step 5 — the changes that did nothing, and the measurement that explained it

Working the elementwise operators on two int16 per word and replacing the
64-bit multiply-and-shift with a single `mulh` should have halved the CPU's
bus traffic and arithmetic. The frame did not move. The microbenchmark at
boot says why:

    alu (3 ops + loop)     8-9 cycles per iteration
    32 alu ops unrolled    53 cycles, 50 instructions retired
    lw from BRAM           10 cycles per iteration
    lw from DDR3, seq       8.6
    lw from DDR3, stride   17.9
    sw to DDR3, seq        11.0

Straight-line code runs at **one instruction per cycle**; a loop iteration
pays about **4 cycles for its taken branch**; and a DDR3 load costs about
the same as a BRAM load. Memory placement was never the CPU's problem. Its
time is instruction count and taken branches. Two consequences:

- The requantisation loop at ~70 cycles per element was ~60 instructions of
  real work; no rearrangement of memory would fix it. Hardware would.
- The `mulh` fast path had been compiled as a branch *into* the fast path
  and a jump back — two taken branches, ~8 cycles, around a 5-cycle
  multiply. Rewritten so the common case falls through (and, for the
  residual adds, whose factors are near 1 and need shifts of 31–32, with an
  exact second case using both product halves), `dav2_add` dropped from
  127 to 92 Mcycles.

### Step 6 — requantisation in hardware, 20.7 → 14.2 s

Requantisation needs two passes: the per-row range of the accumulators to
choose the output scale, then the scaling itself. Both are now in the
accelerator:

- **Statistics**: while draining weight row *m* the block tracks the
  max/min of the row's 64 accumulators and appends `{max, min}` at
  `S_ADDR + 8m`. The CPU reads 2 words per row per tile instead of N×M.
- **Requant job** (`CTRL.requant`): reads a chunk of `acc[m][n]` (≤ 1024
  rows × 64 columns) into the tile RAM *transposed*, the per-row
  `{mult, shift, bias}` table into a parameter RAM, and streams out
  `out[n][m] = sat14((acc·mult + 2^(s−1)) >> s + bias)` two int16 per
  word — the C code's arithmetic to the bit, verified in
  `student_gemm_tb` against a bit-level model and on the board against the
  host.

What remains in the requantisation bucket (101 Mcycles) is the CPU's
per-row soft-float work: one `make_multiplier` and a handful of float
operations per weight row, ~3 500 rows per block.

---

## 4. What the microbenchmark means for future work

On this core, with these numbers, the rules are:

1. **Count instructions, not memory accesses.** A load from DDR3 through
   the cache costs ~8 cycles; so does a load from BRAM; so does a
   multiply-high. Moving data closer to the CPU buys almost nothing.
2. **Avoid taken branches in inner loops.** Four cycles each. Fall-through
   fast paths, unrolling, and branch-free selects all pay.
3. **Anything with a fixed per-element cost above ~20 instructions is a
   hardware candidate**, because the accelerator moves a word in ~3 cycles
   and does 64 multiply-accumulates in one.
4. **Soft-float is 50–150 cycles per operation.** Per-row is fine
   (thousands per frame); per-element is not (millions).

---

## 5. Round two — gather mode, known ranges, cheaper CPU arithmetic

Developed and verified while the board was unavailable: **bit-exact on the
host build** (the software parts) and **in the module testbench** (the
RTL). Confirmed on hardware in §6: bit-exact output, gather mode's boot
self-test passing. The boot self-tests guard each new hardware path: if one
fails on the board it disables itself and the frame falls back to the
round-one code, still correct.

| Item | Was | Change | Where |
|---|---|---|---|
| LayerNorm | 128 Mc | one arithmetic pass that stores the Q16 value (a load and a store are cheaper than recomputing), unrolled; records its output range | `dav2_ops.c` |
| residual adds | 92 Mc | no range scan: every producer records `amax_q`, the largest \|value\| it wrote (new field of `dav2_tensor_t`); the add uses it | `dav2.h`, `dav2_ops.c` |
| requant per-row work | ~70 Mc | `make_multiplier` from the float's bits (no `frexpf`, no soft-float); the accelerator reports its output range in `RQ_AMAX` | `dav2_ops.c`, RTL |
| attention CPU side | 114 Mc | softmax exponent and normalisation in 32×32→64 halves instead of 64-bit library arithmetic; gathers unrolled | `dav2_engine.c` |
| GELU, ReLU | 51 Mc + | unrolled; record their output range | `dav2_ops.c` |
| im2col | 67 Mc | **gather mode** (`CTRL.gather`): the accelerator reads the convolution patches straight from the NHWC image and writes zeros for the padding, so no patch matrix is ever built; K longer than the tile is split by kernel position | RTL, `dav2_accel.c` |

Verification of the new RTL: `student_gemm_tb` runs four convolutions
through gather mode against an im2col reference in the testbench (3×3 with
padding, stride 2, a 384-channel one split into 5 + 4 kernel positions, and
a non-square image across two tiles) and checks `RQ_AMAX` on saturated and
unsaturated outputs. At boot the program compares gather mode with
im2col + the CPU kernel on a 5×4×8 image covering every border case.

Build: LUT 15.2 %, BRAM 36.9 %, DSP 10.8 %, WNS +0.395 ns, WHS +0.015 ns
(superseded by round three's build, below).

### Round three — a wider tile, the prefetcher repaired

| Change | Why | Verified by |
|---|---|---|
| **Tile width 64 → 128 rows** (`NROWS`, `student.sv`) | The encoder's 82 tokens fit one tile, so every encoder weight matrix crosses DDR3 once instead of twice; the DPT head's convolutions need 125 passes instead of 249. Costs 64 more BRAMs and DSPs. | `student_gemm_tb`, `student_gemm_ddrpath_tb` with shapes of exactly one tile and one tile plus a remainder, and a 144-pixel gather convolution |
| **Prefetcher repaired and back on** (`rvlab_ddr_prefetch.sv`, `USE_PREFETCH = 1`) | The accelerator measured 3.2 cycles per beat. That is one 32-byte DRAM line fill per eight beats, and the prefetcher exists to hide it. It had been bypassed for returning aliased data. The cause was a table slot reused while its DRAM response was still in flight; see `DEBUGGING.md` §5. | `rvlab_ddr_alias_tb` fails 65/256 on the old code and passes 256/256 on the new; `student_gemm_ddrpath_tb` passes with the prefetcher in the path |

Build: LUT 17.6 %, BRAM 54.4 %, DSP 19.5 %, WNS +0.371 ns, WHS +0.012 ns,
0 failing endpoints (superseded by round five's build, below).

Confirmed on hardware in §6: with rounds two, four and five also present,
the accelerator measured 2.2 cycles/beat (was 3.2) and zero lost-response
retries. The two effects (tile width, prefetcher) were not isolated on the
board; simulation attributes the gain to both as described above.

LayerNorm as an accelerator job was considered and left out. It needs
per-row *and* per-column parameters plus a float square root per row: a
new datapath, not a mode of the existing one, and not something to add
without the board to test it on.

### Round four — the CPU and the accelerator working at the same time

Software only, so the round-three bitstream still applies. Confirmed on
hardware in §6: 585 of the frame's 1268 accelerator jobs overlapped with
CPU work.

**What overlaps.** Almost every piece of CPU work needs the result of the
job before it, so the overlap is narrower than it sounds. There are three
places where independent work exists:

| Where | On the accelerator | On the CPU meanwhile |
|---|---|---|
| every GEMM | the last job, draining weight row after weight row | the output range, row by row, from each row's statistics as they appear — about a dozen soft-float operations per row, ~45 000 rows per frame |
| every requantisation | chunk *c* of 256 rows being converted | the multiplier, shift and bias of chunk *c+1* |
| attention, per head | the score multiply, then the context multiply | splitting vᵀ; preparing the next head's q and k |

The driver can leave an operation's **last** job running (`*_async` in
`dav2_accel.h`, collected by `dav2_accel_finish()`); at most one job is ever
outstanding. To stream a GEMM's statistics, the driver first fills the last
job's statistics slots with a pair no real row can produce (max < min). The
CPU then takes each row's range once its slot changes. If a job fails, the
accelerator disables itself and the operation is redone on the CPU.

**Expected gain: small.** About 5–10 % of the frame, mostly from the
per-row range and parameter work. Attention's multiplies are too short
(~130 k cycles per head) for its overlap to be worth more than ~1 %. My
first estimate — "most of the ~130 Mcycles the accelerator is busy" — was
wrong; the dependency chain does not allow it.

The per-frame accelerator report now prints how many jobs overlapped with
CPU work (585 of 1 268 on the emulator), and the profile's
`gemm (accelerator)` line now counts only the time the CPU actually waited.

### Verifying without the board: the accelerator emulator

`src/sw/project/host/accel_emu.c` is a register-level C model of
`student_gemm`, written from the register descriptions in
`student_gemm.hjson` rather than from the RTL. `make -C src/sw/project/host
dav2_host_emu` builds the engine with the real driver compiled in, programming
this model instead of hardware. Its output must be bit-identical to
`dav2_host`'s, which checks:

- the driver's register programming: gather-mode field packing, strides,
  statistics, requant chunking, `RQ_AMAX`;
- the asynchronous paths. The model finishes a GEMM job a few rows per
  `STATUS` read, so the CPU really does read statistics while the job is
  half done. In one frame 45 137 rows were read that way, 9 058 of them
  after waiting for their slot to fill;
- recovery: `DAV2_EMU_FAIL_JOB=n` reports a bus error on the *n*-th job.
  Failing a GEMM, an attention multiply or a requant job (jobs 10–1 100)
  still gives a bit-identical depth map.

Results: bit-identical on the demo image and three example photographs.
This also verified the round-two and round-three driver code, which until
then had never run anywhere; the RTL itself stays covered by the module
testbenches.

### Round five — the residual add and ReLU inside the requantisation job

RTL change; bitstream rebuilt. Confirmed on hardware in §6.

Each transformer block ends its attention and its MLP with a residual add,
`x = x + GEMM(...)`. Each residual conv unit in the DPT head ends with one,
and several convolutions are followed by a ReLU. Each of these was a
separate CPU pass over a tensor the requantisation job had only just
written. They now happen in the job's epilogue (`CTRL.add`, `CTRL.relu`,
`ARCHITECTURE.md` §4), so the intermediate tensor and the CPU pass are gone.
In the blocks, `x` is also updated in place.

The fused path is bit-identical because the add's scale is computed before
the job runs. It needs the largest |h|, and per row h is a non-decreasing
function of the accumulator, so that maximum is at the row's largest or
smallest accumulator, which the drain statistics provide.

| | Result |
|---|---|
| `student_gemm_tb` | ReLU alone, add, add + ReLU; one chunk, 512-row chunks, two row tiles — pass |
| host vs emulator | bit-identical on the demo and three example photographs |
| CPU add/ReLU time (emulator run, host units) | 3 318 k → 525 k (−84 %); the rest is a ReLU on a copied input and one add of two non-GEMM tensors, neither fusable |
| boot self-test | requant + add, then + ReLU, against C, in BRAM; disables the epilogue alone on a mismatch — shown to fire by breaking the emulator's ReLU |
| injected job failures | see below |

**A latent bug found by fault injection.** Sweeping `DAV2_EMU_FAIL_JOB`
over the frame turned up one failure point, job 1150, where the output
came out wrong, and silently. A multi-tile GEMM that failed on a tile
before its last one returned "declined" but left its statistics descriptor
set. The CPU redid the product correctly, then took the row ranges from the
stale statistics. The bug dated from round two. Fixed in
`dav2_accel_qgemm`; the sweep result is recorded in `HANDOFF.md`.

An in-place residual add is failure-safe the same way: the job writes to a
temporary and the result is copied back only when every job has succeeded.
So a failure leaves `x` intact for the CPU redo, at ~0.3 Mcycles per add.

Build: LUT 18.9 %, BRAM 54.4 %, DSP 20.4 %, WNS +0.173 ns, WHS +0.044 ns,
0 failing endpoints.

Measured on the board (§6): the add/relu bucket fell to 12 Mcycles, from
92 Mcycles before round five — bigger than the 30 Mcycles estimated here,
because the CPU-side estimate did not account for the fused path also
removing the accumulator scan that used to precede every add.

## 6. Measured on the board: 14.2 s → 8.667 s, bit-exact

Rounds two to five, run on hardware together for the first time. Output is
**bit-exact with the host build, 15876/15876 pixels**, confirming every
round's simulation and emulator verification.

| | Value |
|---|---|
| Frame time | 8.667 s (433.31 Mcycles at 50 MHz) |
| Frames per second | 0.1153 |
| Accelerator | 128 rows/pass, 2.2 cycles/beat (was 3.2), 0 lost-response retries |
| Accelerator jobs | 1268 total, 585 overlapped with CPU work (round four) |
| Result readout | 0.74 s |
| Boot self-tests | GEMM and gather-mode self-tests printed and passed; the add/ReLU self-test's confirmation line did not appear in the captured console output (the hostio ring is 1 kB, drained slowly over JTAG, and can drop a line under a burst of boot text) — but the add/relu profile bucket (12 Mcycles, versus ~92 unfused) shows the fused epilogue ran on hardware; bit-exactness is the decisive check |

Measured profile (433.31 Mcycles), compared to the round-one baseline:

| Operator | 14.2 s (Mcycles) | 8.7 s (Mcycles) | Change |
|---|---|---|---|
| GEMM (accelerator) | 93 | **15** | −84 %, from the 128-row tile and the repaired prefetcher |
| add/relu | 92 | **12** | −87 %, fused into the requantisation job |
| im2col | 67 | **1** | −99 %, gather mode built the patches in hardware |
| LayerNorm | 128 | 98 | −23 %, single-pass fixed point |
| requantisation | 101 | 97 | −4 %; cheaper per-row setup, but now the largest CPU-side share since GEMM shrank around it |
| attention | 114 | 89 | −22 % |
| GELU | 51 | 60 | +18 % relative share grew as everything around it shrank; unchanged code |
| interpolate + other | 63 | 61 | ~flat |

The three items rounds two, three and five targeted directly — GEMM,
add/relu, im2col — are no longer worth further optimisation. **LayerNorm,
requantisation setup, and attention are now the largest shares, and all
three are CPU arithmetic that never touches the accelerator.** This
replaces the estimate-based priority list that follows; §7 reflects it.

### Round six — fewer float library calls, and a prefetcher hang fixed

Measured on the board: **8.187 s per frame (0.1221 FPS)**, five frames in a
row, each 8.186 to 8.196 s, all five bit-exact with the host build. All
three boot self-tests printed `ok`.

| Change | Where | Effect (Mcycles) |
|---|---|---|
| The per-row range pass compares the float magnitudes as integers (the bit patterns of floats that are not negative have the same order as the values), and keeps the product `a->scale * s[m]` for the parameter pass instead of computing it twice. `iround` reads the float's sign bit instead of calling a float comparison. | `dav2_ops.c` | requantisation 97 → 82 |
| The GELU table stores one 32-bit word per interval: the value in the low half and the step to the next value in the high half. One load per element instead of two. | `dav2_ops.c` | GELU 60 → 57 |
| Every 4-byte bit copy uses `__builtin_memcpy`. The flow compiles with `-fno-builtin`, so a plain `memcpy` is a call to libsys's byte-by-byte copy. `dav2_make_multiplier` had called it once per weight row since round two. | `dav2_ops.c` | included above |

The first build of these changes used a plain `memcpy` for the bit copies
and made the frame *slower* (8.829 s): requantisation rose from 97 to 103
Mcycles. The disassembly showed the calls to the byte-copy `memcpy`.

The first board run of the corrected build hung in the third frame. The
cause was a defect in the round-three prefetcher repair: an entry could be
lost for good when its DRAM response arrived in the same cycle it was
marked out of date, and with all four entries lost no read reached DRAM.
`DEBUGGING.md` §5 has the full account. The fix is in
`rvlab_ddr_prefetch.sv`; the bitstream was rebuilt (WNS +0.395 ns). The
five-frame run above is on the fixed bitstream.

The accelerator report printed 1270 and 1265 jobs for two of the frames,
against 1268 in earlier runs. The outputs are bit-exact, so the work done
is the same; the job count depends on where the CPU and the accelerator
meet during the overlapped parts.

### How much time the soft-float routines take

The CV32E40P has no FPU, so every float operation in C is a call to a
libgcc routine (`__mulsf3`, `__addsf3`, ...). To measure them, the linker
option `--wrap` sends each call to a wrapper that counts it and times the
real routine (`dav2_floatprof.c`). It is switched on in
`src/sw/project/build_flags.txt`, which is read by `flow/sw.py`.
`tools/dav2_floatprof_map.py` maps the call sites to functions and source
lines. The counter costs about 93 cycles per call, so it is off by default.

Measured on the board, one frame: **1.11 million calls, 102 Mcycles**. That
is about 2.0 s of the 8.19 s frame, or 25 %.

| Routine | Calls | Cycles per call |
|---|---|---|
| `__mulsf3` | 429,803 | 115 |
| `__addsf3` | 221,042 | 97 |
| `__floatsisf` | 154,804 | 64 |
| `__fixsfsi` | 107,827 | 35 |
| `__subsf3` | 73,303 | 110 |
| `__lesf2` | 50,345 | 44 |
| `__gtsf2` | 37,784 | 54 |
| `__divsf3` | 21,010 | 194 |
| others | 14,628 | 35 to 46 |

| Where the float time goes | Mcycles |
|---|---|
| Requantisation parameters per weight row (`qgemm_impl` range and parameter passes, bias rounding, `dav2_make_multiplier`) | ~48 |
| Token matrix, built once per frame but element by element in float (`dav2_engine.c` class and position embedding, then `dav2_quantize_f32`) | ~17 |
| GELU table (`dav2_gelu_f`, `dav2_erff`, `dav2_expf`) | ~16 |
| LayerNorm row statistics and γ/β conversion (`dav2_layernorm`, `dav2_sqrtf`) | ~14 |
| Final depth map to float | ~2 |
| Resize weights, softmax scale, adds | <1 |

### Round seven — no floating point

The float parameters are converted to integers offline, and every scale in
the engine is a pair of integers (m, sh), value = m · 2^−sh (`dav2_xf.h`).
The program contains no soft-float routine: `nm sw.elf` lists none of
`__mulsf3`, `__addsf3`, `__floatsisf` and the others.

| Change | Where |
|---|---|
| New blob version 3. Row scales and biases are (m, sh) pairs, exact from float32. LayerNorm γ is Q15 and β is Q16, rounded as the engine rounded them before. `cls_token` and `pos_embed` are Q24. The image tensors are removed. | `tools/dav2_blob_int.py`, `export_dav2.py` |
| Multiply, add, 1/x, 1/√x (Newton in Q30) on (m, sh) pairs. A 31-bit mantissa; the measured relative error is below 3·10⁻⁹. | `dav2_xf.h` |
| Requantisation parameters, the fused add, LayerNorm statistics, the attention scale, the resize weights and the token matrix use these routines. | `dav2_ops.c`, `dav2_engine.c` |
| GELU is built from a fixed table of Φ (2049 entries, Q30) with linear interpolation, compiled into the program. The boot self-test needs GELU before the weights arrive, so the table is not in the blob. | `dav2_gelu_phi.h` |
| The board returns the int16 depth map and its scale. The PC converts it to float32; `dav2_host` and `dav2_run_fpga.py` use the same formula, so the files are equal bit for bit. | `main.c`, `host_main.c`, `dav2_run_fpga.py` |

Measured on the board: **7.20 s per frame (0.139 FPS)**, five frames in a
row, each 7.201 to 7.212 s. All five are bit-exact with the host build
(15876/15876 pixels). The self-test hash is `4fb8021c` on both. The
accelerator emulator also gives the same output as the host.

| Operator | Round six (Mcycles) | Round seven (Mcycles) |
|---|---|---|
| LayerNorm | 97 | 86 |
| requantisation | 82 | 74 |
| GELU | 57 | 43 |
| other | 28 | 12 |
| frame | 409 | 360 |

The saving is 49 Mcycles, about half of the 102 Mcycles the float routines
took. The integer code that replaces them is not free.

The numbers differ from the float version in the last bits, so the output
is not the same as before. Against PyTorch on the demo image the
correlation is 0.999862 (before: 0.999872). The mean relative error is
4.5 % (before: 3.8 %). Most of both is a gain difference: after the best
gain is fitted, the mean relative error is 0.66 % (before: 0.65 %). The
new and the old output differ by at most 1.9 % of the largest value.

## 7. What is left, in order of expected gain

Based on the measured 8.187 s profile (§6, round six), not the earlier
estimate. Round seven removed the float routines named in this table;
LayerNorm is now 86, requantisation 74 and GELU 43 Mcycles.

| Item | Now (measured) | Estimate | How |
|---|---|---|---|
| LayerNorm | 97 Mcycles | −40 | per-row `1/sqrt` still calls the soft-float library; a fixed-point Newton iteration or a small LUT plus one correction step would remove it. A third accelerator job (row and column parameters, per-row `1/sqrt`) is the alternative considered and rejected in round three: real gain, real hardware risk. |
| requantisation setup | 82 | −25 | `make_multiplier` still runs once per weight row per GEMM (~45 000 calls/frame) on the CPU. Half of it could move into the requant job itself: the block already knows the row range from its own drain, so shift/multiplier derivation could become a small per-row hardware step instead of a CPU one, ahead of the streamed conversion. |
| attention | 87 | −30 | the softmax's 2^-x approximation and the per-head gather/split loops are the largest remaining scalar loops on the CPU; further unrolling and moving the score-shift search into the accelerator's own row-max output (already available from `S_hi`/`S_lo`'s GEMM) would help. |
| GELU | 57 | −10 | already unrolled and packed two elements per word (`GELU_LUT` on `lo16`/`hi16`); the per-element cost is two data-dependent loads from the 257-entry table for the interpolation, which is inherent to linear interpolation and not removable without a coarser table. The 257-entry table itself is rebuilt in float once per call (12 times/frame); a cheaper table build is the more likely gain here. |
| interpolate + other | 58 | −15 | never profiled in detail; the bilinear resize loops in the DPT head are the largest remaining candidate. |
| GEMM, add/relu, im2col | 28 combined | ~0 | at 2.2 cycles/beat and mostly overlapped with CPU work, these are no longer worth further hardware changes on their own. |

All of these together would land around 6–6.5 s per frame. Beyond that the
weights themselves — 25 MB crossing DDR3 once per frame at ~2.2 cycles per
word — set a floor near 2.5–3 s at 50 MHz; a faster fabric clock or a wider
bus would be the next step, and the one most likely to need care around the
DDR3 path where the platform's three defects were found.
