# Performance — how one frame went from 93.6 s to 7.2 s (measured) and about 1.44 s (estimated)

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

### Round eight — CPU work, estimated without the board (not yet measured)

The board was not connected for this round. The changes were checked on
the PC, and their cycles were estimated with a new tool,
`tools/cyclemodel/`. It runs the engine's RISC-V build in the Unicorn CPU
emulator with a timing model of the CV32E40P:

| Cost | Cycles |
|---|---|
| any instruction | 1 |
| taken branch | +3 |
| `mulh`, `mulhu`, `mulhsu` | +5 |
| load / store in the on-chip RAM | +4 / +2 |
| load / store in DDR3 (cache hit) | +0.2 / +0.2 |

These constants reproduce all eight results of the board's boot
microbenchmark within 0.3 cycles per iteration. The accelerator is
replaced by stubs whose jobs finish at once, so the tool counts CPU time
only. For the committed round-seven engine it gives these totals per
frame, next to the board's profile:

| Operator | Tool (Mcycles) | Board (Mcycles) |
|---|---|---|
| attention | 88.3 | 86.6 |
| layernorm | 85.3 | 86.4 |
| gelu | 43.8 | 43.3 |
| interpolate | 28.3 | 29.1 |
| add/relu | 13.4 | 12.3 |
| requantise | 59.4 | 74.2 |

The requantise bucket differs because on the board it also contains the
time the CPU waits for the drain.

One result of the model changed the plan. A load from the on-chip RAM
costs about 4 cycles more than a DDR3 cache hit (board: 11.0 against 7.1
cycles per loop iteration). Staging data in `dav2_scratch` or on the stack
is therefore slower than reading it again from DDR3.

| Change | Where | Tool, per call (Mcycles) |
|---|---|---|
| Requantisation parameters: every row scale and bias is shifted to one exponent per matrix. The range pass and the parameter pass then need a few `mulh` per row instead of the general (m, sh) routines. All rows of a GEMM share one shift; the multipliers are not normalised, which the block and `apply_multiplier` allow. | `dav2_ops.c` | qkv GEMM 1.36 → 0.33 |
| Softmax divides each row of probabilities by its sum (one reciprocal per row). The context is then `(64 c_hi + c_lo + 2^14) >> 15` in 32 bits, without a 64-bit division trick per element. The scores are read twice from DDR3 instead of stored in on-chip RAM. The shift is fixed per head, and the loops are unrolled. | `dav2_engine.c` | attention, one block 6.9 → 4.6 |
| q and k are read directly from DDR3, two values per load. Their range is the OR of the absolute values (the shift depends only on the highest set bit), without compares. | `dav2_engine.c` | in the line above |
| LayerNorm with γ and β: squares summed in 32 bits, 64 at a time; every product is one `mulh` with the shift fixed per row or per tensor. The output range comes from the extremes of y. | `dav2_ops.c` | 3.34 → 2.04 |
| The final LayerNorm (4 calls) has no γ and β: they are folded offline into proj0..3 (`dav2_blob_int.py`, blob version 4). | `dav2_ops.c`, `dav2_blob_int.py` | 3.34 → 1.10 |
| GELU: the table is in the DDR3 arena instead of on the stack, and the output range is not tracked (only fc2's GEMM reads the output, and it needs the scale alone). | `dav2_ops.c` | 3.63 → 2.14 |
| Bilinear resize: separable, each source row resampled horizontally once and kept in one of two row buffers; two channels per word. Same arithmetic, the same output bit for bit. | `dav2_ops.c` | 72→126 × 32: 15.1 → 9.2 |

Folding γ and β of the block LayerNorms into qkv and fc1 was also tried.
It was rejected: γ spreads the columns of W, and the per-row int8 scale
gets coarser. Measured with the NumPy model (`dav2_numpy.py`, 14-bit
activations) against its float version over five images, the mean of
1 − r rose from 1.6e-4 to 1.0e-3. With 16-bit weights the folded model is
as good as the unfolded one, so the loss is the weights' quantisation.
Only the final norm's fold is neutral (1.5e-4).

Estimated whole frame, CPU time only (tool):

| Operator | Round seven (Mcycles) | Round eight (Mcycles) |
|---|---|---|
| requantise | 59.4 | 16.5 |
| attention | 88.3 | 57.1 |
| layernorm | 85.3 | 62.7 |
| gelu | 43.8 | 25.9 |
| interpolate | 28.3 | 16.8 |
| add/relu | 13.4 | 13.1 |
| other | 10.8 | 10.8 |
| total | 329.2 | 202.9 |

On the board round seven took 360 Mcycles, of which about 15 were waits
for the accelerator. If the waits stay the same, round eight would take
about 220 to 235 Mcycles, 4.4 to 4.7 s per frame. The waits may grow,
because the CPU now finishes its work sooner. This has to be measured.

Checked on the PC: `dav2_host` and `dav2_host_emu` give the same output
on all five test images. The results changed in the last bits, as
expected. Against the float model over five images, the mean of 1 − r is
1.9e-4 (round seven: 3.1e-4). On the demo image the correlation with
PyTorch is 0.999897 (round seven: 0.999862). `sw.elf` still contains no
soft-float routine. The profile now also prints a detail table (row
statistics, parameter passes, the phases of attention).

### Round nine — small exact changes (estimated, not yet measured)

The board was still not connected. Every change below gives the same
output as round eight, bit for bit (host build, accelerator emulator,
five test images).

| Change | Where | Tool (Mcycles) |
|---|---|---|
| GELU: a direct table with one int16 per input value (32 kB in the DDR3 arena), each entry the interpolated value of round eight. The element loop is one load per element. | `dav2_ops.c` | one call 2.14 → 1.25 |
| `dav2_add`: the int16 inputs are moved up by 16 bits, so the shift becomes at least 33 and `apply_multiplier`'s fast branch (one `mulh`) applies. The same values. | `dav2_ops.c` | add/relu per frame 13.1 → 10.1, with the next row |
| `dav2_copy_relu`: the copy and the ReLU at the start of each residual conv unit are one pass. | `dav2_ops.c`, `dav2_engine.c` | in the line above |
| Tensor lookup by name starts after the previous hit. The engine asks for the tensors almost in file order, so most lookups compare one or two names instead of up to 300. | `dav2_blob.c` | about 3 per frame |
| Softmax: the reciprocal of the row sum is a 32-bit division instead of a 64-bit library call. | `dav2_engine.c` | about 1.5 per frame |

The GELU estimate does not include cache misses: the tool counts every
DDR3 access as a hit. The table is 32 kB and the cache 16 kB, so some
lookups will miss. The activations cluster near zero, so most lookups use
a small part of the table.

Two changes to LayerNorm were tried and not kept:

- The output scale from a bound known after the statistics
  (max|z| * max|g| + max|b|), so that g and b combine with the output
  scale into one factor per channel: 2.04 → 1.81 Mcycles per call, but the
  mean of 1 − r over five images rose from 1.9e-4 to 2.0e-4. The gain,
  about 5 Mcycles per frame, did not justify a coarser output.
- The same with a 32-bit `mul` instead of the second `mulh`: 1.70 Mcycles
  per call, but 1 − r rose to 2.4e-4.

Estimated whole frame, CPU time only: 202.9 → 183.8 Mcycles.

| Operator | Round eight | Round nine |
|---|---|---|
| requantise | 16.5 | 16.5 |
| attention | 57.1 | 53.7 |
| layernorm | 62.7 | 62.7 |
| gelu | 25.9 | 15.2 |
| add/relu | 13.1 | 10.1 |
| interpolate | 16.8 | 16.8 |
| other | 10.8 | 8.8 |
| total | 202.9 | 183.8 |

With about 15 Mcycles of accelerator waits this is about 200 Mcycles,
4.0 s per frame, if the board confirms the model. LayerNorm and attention
are now 63 % of the CPU time. The software options for them that remain
either lose accuracy (above) or save 1 to 2 Mcycles. The next large step
is hardware: the LayerNorm row statistics or the softmax on the
accelerator.

### Round ten — LayerNorm on the requantisation job (estimated, not yet measured)

The board was still not connected.

| Change | Where | Tool (Mcycles) |
|---|---|---|
| LayerNorm with γ and β: the second half, out = (z γ[c] + β[c]) / scale, is a per-channel multiply, add and saturation, which is exactly the accelerator's requantisation job with the channels as its rows. The CPU computes only z = (x − mean) r in Q16 (one `mulh` per element), channel by channel, and keeps each channel's smallest and largest z. These give the exact output range before any output exists. The job applies γ, β and the scale, saturates, writes the output token by token and reports its range. Without an accelerator the CPU runs the same integer arithmetic. The tool charges the job 3 cycles per element (1.5 bus beats at 2 cycles). | `dav2_ops.c` | one call 2.04 → 1.26 |
| Attention: k is no longer shifted to 11 bits and copied per head. The score GEMMs read it in place from qkv with its row stride, at 14 bits: 64 · 2047 · 8191 = 1.07·10^9 < 2^31. Only q is shifted and split. The score difference in the softmax is unsigned, since it can now exceed 2^31. | `dav2_engine.c` | one block 4.61 → 4.27 |
| Softmax: 2^−f from a 1024-entry table (Q15, error below 11 units) instead of a parabola (error up to about 50 units), in the DDR3 arena. More accurate; about the same speed. | `dav2_engine.c` | in the line above |

The accuracy test was extended to 11 images: the demo photo, all eight
synthetic scenes and two photographs. Some synthetic scenes have an almost
flat depth, so their 1 − r is larger and varies more. Mean of 1 − r
against the float model:

| Engine | Mean of 1 − r (11 images) |
|---|---|
| round seven | 2.16e-3 |
| round nine | 1.69e-3 |
| round ten | 1.69e-3 |

On the demo image the correlation with PyTorch is 0.999898.

Estimated whole frame, CPU time only: 183.8 → 161.2 Mcycles.

| Operator | Round nine | Round ten |
|---|---|---|
| requantise | 16.5 | 16.5 |
| attention | 53.7 | 47.7 |
| layernorm | 62.7 | 46.1 |
| gelu | 15.2 | 15.2 |
| add/relu | 10.1 | 10.1 |
| interpolate | 16.8 | 16.8 |
| other | 8.8 | 8.8 |
| total | 183.8 | 161.2 |

With about 15 Mcycles of accelerator waits this is about 176 Mcycles,
3.5 s per frame, if the board confirms the model. The boot self-test hash
is now `7ca6973c`.

### Round eleven — exact fixes found in the frame profile (estimated, not yet measured)

The board was still not connected.

| Change | Where | Tool (Mcycles per frame) |
|---|---|---|
| LayerNorm: round ten's fast loop needed every row's shift to be at least 33; in real data some rows have a smaller shift, and then the whole call took the slow path. A row with a small shift has a small spread, so (x − mean) 2^16 can be moved up by 33 − shift bits (it stays below 2^23) and one `mulh` gives exactly the same value. Every row now takes the fast loop. Output unchanged. | `dav2_ops.c` | layernorm 46.1 → 35.1 |
| Softmax exponent t = d kf with one 32-bit multiply: only d below the underflow cutoff matters, and there d >> j has 16 bits. The error is below 2^-11 in t, under the exp2 table's resolution. | `dav2_engine.c` | attention 47.7 → 42.5, with the next row |
| The residual updates write into a second buffer for x and swap the two, instead of writing a temporary and copying it back (24 copies of 31.5 k values per frame). A failed job still leaves x intact. Output unchanged. | `dav2_engine.c` | in the line above, and requantise 16.5 → 15.0 |
| The DPT fusion block adds its input directly instead of a copy of it; the copy had lost the known range and made `dav2_add` scan both operands. Output unchanged. | `dav2_engine.c` | add/relu 10.1 → 6.8 |

Estimated whole frame, CPU time only: 161.2 → 140.0 Mcycles; with about
15 Mcycles of accelerator waits about 155 Mcycles, 3.1 s per frame.

**Accuracy, measured more carefully.** Small rounding changes that do not
change the accuracy of any kernel still move the per-image 1 − r by up to
a factor 2, because the quantisation errors of the whole network add up
differently. A handful of images therefore cannot show a change of 10 %.
Two methods were used for this round:

- The softmax exponent was checked at the kernel level: on 3000 random
  score rows, the largest error of a probability against float softmax is
  24 Q15 units for both the exact and the 32-bit form, and the mean error
  3.3e-5 against 3.7e-5.
- A second test set of 24 natural images (random crops, scales and flips
  of the two photographs) was compared image by image (mean of
  log(1 − r) ratios, ± one standard error):

| Comparison | Change of 1 − r |
|---|---|
| round eleven against round ten | −2 % ± 4 % |
| round ten against round nine | −3 % ± 5 % |
| round nine against round seven | +13 % ± 8 % |

Rounds ten and eleven do not change the accuracy. Rounds eight and nine
may cost a little on photographs (less than two standard errors; on the
synthetic scenes they were better). In absolute terms all of them stay
near r = 0.999: the median 1 − r over the 24 crops is 4.1e-4 for both
round seven and round eleven.

### Round twelve — GELU and LayerNorm on the accelerator (hardware; estimated, not yet measured)

The FPGA had most of its resources free (LUT 18.9 %, BRAM 54.4 %, DSP
20.4 %), so this round moved work from the CPU into `student_gemm`. Two
features were added to the requantisation job; both are announced in
`CAPS` and have a boot self-test, and without them the same arithmetic
runs on the CPU.

| Change | Where | Tool (Mcycles per frame) |
|---|---|---|
| **Lookup table** (`CTRL.lut`, `CTRL.lut_load`, `LUT_ADDR`): after saturation, add and ReLU, out = LUT[v + 8192], a 16384 × int16 table in 8 BRAM36, one more pipeline stage (q12). The first job of a GEMM loads the table (8192 words), later jobs reuse it. fc1's requantisation applies GELU this way (`dav2_qgemm_gelu`); the CPU only builds the table (the same table as before, so the same output). | `student_gemm.sv`, `dav2_ops.c`, `dav2_engine.c` | gelu 15.2 → 2.5 |
| **int16 input** (`CTRL.a16`): the job reads int16 values, two per word; the loader writes each word into two tile rows, the output stage takes the half by the column's parity. LayerNorm with γ and β is now two such jobs: z = (x − mean) r at a common 14-bit scale (tokens as rows), then z γ + β (channels as rows). The CPU keeps the row statistics and the channels' ranges, so the output range stays exact. | `student_gemm.sv`, `dav2_ops.c` | layernorm 35.1 → 31.6 |

Estimated whole frame, CPU time only: 140.0 → 128.8 Mcycles; with about
15 Mcycles of accelerator waits about 145 Mcycles, 2.9 s per frame. The
new jobs are charged in the tool at 2 to 3 cycles per element, as bus
time the CPU waits for.

Checks:

- `student_gemm_tb` has new tests for both features: the table alone,
  with ReLU, with the add, over several chunks (loaded once, reused) and
  two row tiles; int16 input in LayerNorm's two shapes (three column
  tiles; two row chunks) with negative multipliers, and with the table. A
  plain job after them does not use the table. PASSED, 579,005 words
  checked.
- `dav2_host` (CPU arithmetic) and `dav2_host_emu` (the accelerator
  emulator, which models both features) give the same output on all 11
  test images.
- Accuracy on the 24 photo crops against round eleven: +6 % ± 6 % in
  1 − r, within the noise; the median 1 − r is unchanged (4.1e-4). The
  two-job LayerNorm keeps z at 14 bits: its largest error in a unit test
  is 2 output steps instead of 1.
- Bitstream: timing met, WNS +0.395 ns, WHS +0.029 ns, 0 failing endpoints. Resources: LUT 19.4 % (25,935, +615), BRAM 56.6 % (206.5 tiles, +8 for the table), DSP 20.4 % (151, unchanged). The DRC report still lists 20 REQP-1840 warnings (RAMB18 async control check) on the parameter RAM's enable pins, driven by the state machine and the read buffer, which have an asynchronous reset; this kind existed before. The ones this round's counters caused on RAM address pins were removed (the counters rq_m_q, rq_p_cnt_q and lut_cnt_q no longer have a reset: every job sets them).

LayerNorm is now mostly the CPU's range tracking: a compare per value in
the statistics pass and in the channel pass (a taken branch costs 3
cycles). Row statistics written by the job itself, as the GEMM drain
already does, would remove the channel pass.

### Round thirteen — int16 weights for attention (hardware; estimated, not yet measured)

The block's multipliers are DSP48E1 slices, 25 × 18 bits, but the weights
were int8. Attention has no int8 operand, so q and v were split into two
int8 halves and every product ran twice.

| Change | Where | Tool (Mcycles per frame) |
|---|---|---|
| **int16 weights** (`CTRL.w16`): two weights per word; the weight selector takes 16-bit halves, the weight pipeline registers are 16 bits, the multiply is 16 × 16 in the same DSPs (DSP count unchanged, 151). `W_STRIDE` 0 means K·2. CAPS bit 26, boot self-test. | `student_gemm.sv`, `dav2_accel.c` | — |
| Attention with int16 weights: scores S = k·q in one GEMM (k in place as A, q shifted to 11 bits as the weights: 64 · 8191 · 2047 < 2^31); its row statistics give each query's maximum, so the softmax has no max pass and reads one matrix. Context C = P·v in one GEMM (fits int32 because P sums to 2^15). The requantisation job writes round(C / 2^15) straight into the head's slice of ctx (a new output stride). Per head 3 jobs instead of 4; no int8 split of q and v. | `dav2_engine.c`, `dav2_accel.c` | attention 47.4 → 25.2 |

The numbers are the same as round twelve's bit for bit: k·q = 16 (k·q_hi)
+ k·q_lo, and round(C · 2^30 / 2^45) = (C + 2^14) >> 15. `dav2_host` and
`dav2_host_emu` give round twelve's output on all 11 test images. With an
older bitstream (no CAPS bit 26) the engine keeps the split path.

Estimated whole frame, CPU time only: 128.8 → 106.6 Mcycles; with about
15 Mcycles of accelerator waits about 122 Mcycles, 2.4 s per frame.

Checks: `student_gemm_tb` has int16-weight GEMMs in attention's two shapes
(K = 64 scores, K = 84 context with strided rows) and over two tiles:
PASSED, 592,677 words checked. Bitstream: timing met, WNS +0.395 ns,
WHS +0.041 ns; LUT 19.5 %, BRAM 56.6 %, DSP 20.4 %.

### Round fourteen — row statistics from the requantisation job (hardware; estimated, not yet measured)

| Change | Where | Tool (Mcycles per frame, CPU) |
|---|---|---|
| **Output row statistics** (`CTRL.ostats`): the job keeps each output row's largest and smallest final value in a 128-word LUT RAM and, after the output, writes one word per row to `S_ADDR` (max in bits 31:16). A new state `RQ_STATS`; CAPS bit 27, boot self-test. | `student_gemm.sv`, `dav2_accel.c` | — |
| The residual updates (proj, fc2) ask for them, so x arrives at LayerNorm with each token's range (`dav2_tensor_t.rst`); LayerNorm's first job gives each channel's range the same way. LayerNorm's token pass then has no compare per value, and its channel pass is gone. The ranges are exact, so the output is the same as round thirteen's, bit for bit. | `dav2_ops.c`, `dav2_engine.c` | layernorm 31.6 → 16.6 |

`student_gemm_tb`: statistics on LayerNorm's first-job shape (three
tiles), with ReLU and with the table. PASSED, 593,163 words. Bitstream:
timing met, WNS +0.264 ns (the critical path is in the DDR3 controller,
not in `student_gemm`); LUT 19.5 %, BRAM 56.6 %, DSP 20.4 %.

**The frame-time estimates of rounds ten to thirteen were too low.** They
added about 15 Mcycles of accelerator waits to the CPU time, the value
measured in round seven. But in round seven most of the accelerator's 68
Mcycles ran while the CPU worked on the requantisation parameters (74
Mcycles then); that CPU work is now 15 Mcycles, so the CPU waits more.
`tools/cyclemodel/` now models the accelerator too: each stubbed job runs
for a time from the block's design (per weight row K MAC cycles and the
drain writes; requantisation one element per cycle or its bus words; 2.1
cycles per bus word as measured), and the CPU waits for it where the
driver would. Checked against the board: for the round-seven engine it
gives 373 Mcycles (7.46 s) against 360 Mcycles (7.20 s) measured, 4 %
high.

| Engine | Model (Mcycles) | Model (s) | CPU waiting (Mcycles) |
|---|---|---|---|
| round seven | 373.2 | 7.46 | 38.8 |
| round fourteen | 155.7 | 3.11 | 62.7 |

So the current estimate is **about 3.0 s per frame** (3.11 s, less the
model's 4 %). The accelerator is now the limit. Its modelled busy time is
83 Mcycles: encoder GEMMs 34, convolution GEMMs 23, attention GEMMs 3.5,
requantisation 23.

### Round fifteen — two weights per cycle in each MAC lane (hardware; estimated, not yet measured)

A GEMM spent K MAC cycles per weight row, one weight per cycle, while the
bus could bring about two: a word holds four int8 weights and takes 2.1
cycles. One A-tile word already holds a[k] and a[k+1], so each lane now
multiplies a[k]·w[k] and a[k+1]·w[k+1], each in its own DSP, and adds
both products to its accumulator.
int8 weights come in pairs (bytes 0,1 then 2,3 of a word), int16 weights
as a word's two halves. No register or software change.

The first version wrote acc + a·w + a1·w1 as one expression. Synthesis
then stayed in "Cross Boundary and Area Optimization" for over an hour
(the step normally takes a minute), most likely trying to fold the
three-input add into the DSPs. With the products registered and the
attributes `use_dsp = "yes"` on the products and `use_dsp = "no"` on the
accumulators, the step takes 69 seconds.

The next build failed in placement: 91,498 LUT RAM cells for 46,200
sites. All 128 A-tile row RAMs had become LUT RAM. The DSP48 had taken
both `a_s2` and the RAM read register `a_q` as its two input registers.
A block RAM needs its read register, so synthesis fell back to LUT RAM,
which can read without one. In round fourteen a multiplexer between
`a_q` and `a_s2` prevented this. `(* keep = "true" *)` on `a_s2` and
`a1_s2` now keeps them out of the DSP, and the row RAMs are block RAMs
again.

That build met timing, but each lane DSP then had no input register and
no multiplier register (about 550 DPIP-1 and DPOP-2 warnings). The lane
pipeline now has three more stages: `a_s3`/`w_s3` (the DSP's AREG and
BREG), `m0_s4` (MREG) and `p0_s5` (PREG). The DSPs are fully pipelined,
and the lane warnings are gone. The cost is 3 cycles per weight row
(the pipeline tail before the drain), about 0.3 Mcycles per frame.

`student_gemm_tb` PASSED (every GEMM, convolution and int16-weight test);
593,163 words. The tile N = 20, K = 64, M = 30 takes 3175 cycles instead
of 4059.
Bitstream: timing met, WNS +0.287 ns, WHS +0.030 ns; LUT 22.4 %, BRAM 56.6 %, DSP 37.7 % (279, of which 256 in the MAC lanes).

`tools/cyclemodel/frame.c` now has the weights per MAC cycle (`MACW`,
at least the row's weight words at the bus rate) and the pipeline tail
(`PIPE`). `FRAME_FLAGS="-DMACW=1 -DPIPE=2" ./build.sh` models round
fourteen's block.

Model: 155.9 → 142.6 Mcycles per frame, **about 2.75 s** (2.85 s, less
the model's 4 %). The accelerator's GEMM time falls from 60.6 to 47.3
Mcycles: encoder 34.4 → 24.2, convolutions 22.6 → 19.5, attention 3.6
(int16 weights, limited by the bus before and after).

**More weights per cycle would not help.** With two per cycle a weight
row already needs 0.525 K cycles on the bus against 0.5 K MAC cycles; the
model gives the same frame for four per cycle. What limits a GEMM now is
bus traffic: for an encoder row (K = 384, 82 tokens) the 96 weight words
and the 84 words of int32 results written by the drain, and then the
requantisation job reads the results back. The next step is to keep the
results on chip.

### Round sixteen — GEMM results kept on chip (hardware; estimated, not yet measured)

A GEMM's int32 results went to DDR3 in the drain and came back in the
requantisation job: 2 of the 2.5 bus words per output. For the encoder
(82 tokens, one tile) the whole result fits on chip: fc1's is the largest,
1536 × 82 words.

| Change | Where | Model (Mcycles per frame) |
|---|---|---|
| **Result RAM** (`CTRL.onchip`, 131,072 words in 128 BRAM36): a GEMM job drains its accumulators into it, one word per cycle and without the bus (C_ADDR, C_STRIDE count its words); only the row statistics go to DDR3. A requantisation job loads its input from it, one word per cycle (A_ADDR, A_STRIDE count its words). CAPS bit 28, boot self-test (GEMM, then requantisation from the RAM, against the CPU). | `student_gemm.sv`, `dav2_accel.c` | — |
| `qgemm_impl` uses it for every plain GEMM with N ≤ 128 and N·M ≤ 131,072 whose requantisation runs on the block: the encoder's qkv, proj, fc1, fc2 and the DPT projections. If a job fails or a chunk is declined, the whole GEMM is redone without the RAM (the results are not in DDR3). | `dav2_ops.c` | accelerator GEMM (encoder) 24.2 → 17.5, requantisation 13.8 → 7.8, requantisation with add 4.2 → 2.6 |

Same output as before, bit for bit (11 images, host and emulator; the
emulator models the RAM). Bus errors injected into jobs 20 to 23, 40, 41
and 300 of a frame all end with the correct output. `student_gemm_tb`:
GEMM into the RAM and requantisation from it over several chunks, the
statistics, and memory left untouched. PASSED, 644,163 words. Bitstream:
timing met, WNS +0.395 ns (DDR3 controller), WHS +0.024 ns; LUT 23.7 %, BRAM 91.6 % (334.5 of 365 tiles), DSP 37.7 %.

The first build of this RAM failed in placement for the same reason as
round fifteen's first build (the A-tile rows in LUT RAM); the `keep`
there fixed both. With 128 more BRAM36 the lane array spreads out, and
the valid bit `v_s5`, which enables all 4096 accumulator bits, had 19 ns
of routing (slack +0.08 ns). `(* max_fanout = 64 *)` on it makes
synthesis copy the register; the worst `sys_clk` path then has +1.67 ns.

Model: 142.6 → 128.4 Mcycles per frame, **about 2.47 s** (2.57 s, less
the model's 4 %).

### Round seventeen — convolutions: tap reuse and the result RAM (hardware; estimated, not yet measured)

The convolution GEMMs were the largest accelerator item (19.5 Mcycles in
the model), and 13.4 Mcycles of that was loading the gathered A tile: each
output pixel reads its k·k·C/2 patch words over the bus at 2.1 cycles per
word. The largest ones are the head's 3 × 3 convolutions with few output
channels (126 × 126 × 32 → 32: 4.8 of 6.5 Mcycles in the A load).

| Change | Where | Model (Mcycles per frame) |
|---|---|---|
| **Tap reuse** (`CTRL.greuse`, CAPS bit 29). With stride 1 and all k·k positions in one job, a pixel's taps kx = 0..k−2 are its left neighbour's taps kx+1, zeros included. When the neighbour is the previous tile row, the gather writer copies these words from that row through the tile RAM's read port (source word = destination word + C/2), one per cycle, and the issuer does not read them. The gather writes pass one register stage (the copy's read takes a cycle); a 128-word FIFO between the reorder buffer and the writer lets the reads of the new column run ahead while the writer copies. For k = 3 a pixel costs K/2 cycles instead of 1.05 K. | `student_gemm.sv` | accelerator GEMM (convolutions) 18.4 → 11.8 |
| **Convolutions into the result RAM**: a single-chunk convolution with N·M ≤ 131,072 drains every tile into the RAM (C_ADDR = n0, C_STRIDE = N, in words); the requantisation job already read chunks of N > 128 from it. Most of the DPT head's convolutions fit; the two largest (72 × 72 and 126 × 126) do not. | `dav2_accel.c`, `dav2_ops.c` | convolutions 19.5 → 18.4, requantisation 7.9 → 7.1 |

`int run;` in `qgemm_impl` (round sixteen) was read uninitialised for a
plain GEMM that does not use the result RAM. It is now 0.

The boot self-test runs the gather test twice: without tap reuse, then
with it ("tap-reuse self-test ok"); a mismatch turns reuse off. The
emulator accepts `CTRL.greuse` and checks that no job reaches past the
result RAM. Same output as before, bit for bit (11 images); bus errors
injected into jobs 20 to 23, 40, 41, 300, 700, 1000 and 1250 all end with
the correct output. `student_gemm_tb`: every convolution again with reuse,
plus k = 5, k = 2, k = 1, four tiles of a 20 × 20 image and tiles that
start inside an output row; with stride 2 or a split kernel the block
ignores the bit. PASSED, 653,861 words. The 20 × 20 × 32 → 8 convolution
takes 67,181 cycles instead of 106,455 in the testbench, whose memory is
faster than DDR3. Bitstream: timing met, WNS +0.303 ns (DDR3 controller),
worst `sys_clk` path +1.49 ns, WHS +0.024 ns; LUT 25.9 % (+3,000 for the
copy multiplexer and the FIFO), BRAM 91.6 %, DSP 37.7 %.

The model's stub for convolutions had let the small ones use the result
RAM before the driver did; without that, round sixteen is 128.7 Mcycles,
not 128.4. Model: 128.7 → 122.2 Mcycles per frame, **about 2.35 s**
(2.44 s, less the model's 4 %). `FRAME_FLAGS="-DREUSE_C=0
-DONCHIP_CONV=0"` models round sixteen.

### Round eighteen — LayerNorm's statistics from the block, and CPU fixes (hardware; estimated, not yet measured)

`tools/cyclemodel/frame.c` now also reports the sub-profile counters
(`DAV2_SUB_*`). They showed which CPU work runs while the accelerator is
idle: the softmax (16.7 Mcycles, the block waits 0.7) and LayerNorm's
statistics pass (7.5), but not the requantisation range pass (10.2),
which overlaps the GEMM job it reads. A 32-bit version of that range
pass made the CPU part faster and the frame no shorter, so it was not
kept.

| Change | Where | Model (Mcycles per frame) |
|---|---|---|
| `dav2_qw` asks for a tensor's data and then for its dims under the same name. `find_index` started after the previous hit, so the second lookup scanned the whole directory (about 298 string compares). It now starts at the previous hit. | `dav2_blob.c` | 122.2 → 120.8 |
| Softmax: one test for the table index and the shift (u = t >> 6), two scores per step with one word store, and the normalisation two values per word (min(q, 32767) = q − (q >> 15), since q ≤ 32768). Same values. | `dav2_engine.c` | 120.8 → 120.3 |
| **Output row sums** (`CTRL.osums`, CAPS bit 30). With `CTRL.ostats` the requantisation job also sums each output row's final values and their squares (a DSP48 with all its registers for the square, then 32-bit and 40-bit sums in logic) and writes four words per row: {max, min}, sum, and the sum of squares in two words. The residual updates ask for them (`dav2_tensor_t.rsum`), so LayerNorm has its mean and variance without a pass over x. They are exact integers, the same as `row_stats` computes. | `student_gemm.sv`, `dav2_ops.c`, `dav2_engine.c` | ln stats 7.8 → 4.0; 120.4 → 117.3 |
| Token embedding: the rounding of patch × scale in 32-bit pieces (mul, mulhu and a carry) when the shift is 16 to 32, instead of a 64-bit shift by a variable; the second pass without the unused branch. Same values. | `dav2_ops.c` | 4.8 → 3.0; 117.3 → 115.5 |

The statistics phase waits until the last row's sums have left their
three pipeline stages. The first build put the 40-bit sum into a DSP48 as
an adder with an unregistered feedback path (DPREG-4); `use_dsp = "no"`
on both sums keeps them in logic.

Checks: same output, bit for bit, on 11 images (host and emulator); bus
errors injected into jobs 20 to 23, 40, 41, 300, 700, 1000 and 1250 all
end with the correct output. The boot self-test gains "row-sums self-test
ok" (an int32 requantisation with its four words per row against the
CPU). `student_gemm_tb`: the sums with and without ReLU and the table,
over three tiles, and with 2 and 4 rows, where the statistics follow the
output at once. PASSED, 656,177 words. Bitstream: timing met, WNS +0.395 ns (DDR3 controller), worst `sys_clk` path +0.78 ns, WHS +0.029 ns; LUT 27.9 %, BRAM 91.6 %, DSP 37.8 % (one more, for the square).

Model: 122.2 → 115.5 Mcycles per frame, **about 2.22 s** (2.31 s, less
the model's 4 %).

### Round nineteen — the DPT head's elementwise work, and CPU arithmetic (hardware; estimated, not yet measured)

After round eighteen the block was busy for 47 of 115.5 Mcycles; the frame
was limited by CPU work done while the block waited.

| Change | Where | Model (Mcycles) |
|---|---|---|
| **ReLU while gathering** (`CTRL.grelu`, CAPS bit 31): a gather job clears the negative int16 halves of every image word it reads from the bus. A residual conv unit's first convolution now reads relu(x) this way, and the CPU copy `dav2_copy_relu` is gone. Without the bit (an older bitstream) `qgemm_impl` makes the copy as before. | `student_gemm.sv`, `dav2_accel.c`, `dav2_ops.c` | add/relu 6.8 → 3.8 |
| `dav2_conv2d_into`: a convolution writes into a tensor the caller allocated. The residual conv units and the fusion blocks' output convolutions no longer copy their result. | `dav2_ops.c`, `dav2_engine.c` | copies 2.1 → 0.75 |
| The fusion blocks' `dav2_add` on the CPU: round(v m / 2^s) as (v mh + ((v ml) >> 16) + 2^(s−17)) >> (s−16) with m = mh 2^16 + ml, two 1-cycle multiplies instead of a 6-cycle mulh. On the block it is not possible: the requantisation job reads its input transposed (in[m][n]) but the residual in output order. | `dav2_ops.c` | add 3.9 → 3.6 |
| `xf_norm` counts leading zeros inline instead of calling `__clzdi2`; the rows' means in LayerNorm take two 32-bit divisions instead of a 64-bit one (`mean_q16`; both checked against the old code on all sums a row can have); the GELU table is built with one halfword store per entry. | `dav2_xf.h`, `dav2_ops.c` | `__clzdi2` 1.05 → 0, `__divdi3` 0.97 → 0.49, GELU table 2.97 → 2.69 |
| Interpolation: two output rows (and two output pixels) between the same two source rows (pixels) share one load and one unpacking, as top + ((d w) >> 8) with d = bottom − top. This covers 106 of the 126 rows of 72 → 126 and all but the copied rows of the 2× steps. | `dav2_ops.c` | interpolate 16.8 → 15.4 |

All of these give the same output as before, bit for bit (11 images, host
and emulator; also an emulator without CAPS bit 31, which takes the copy
path). Bus errors injected into 13 jobs from 20 to 1290 all end with the
correct output. The boot self-test runs the gather test a third time with
ReLU ("gather-ReLU self-test ok"). `student_gemm_tb`: convolutions with
ReLU, with and without reuse, split and stride 2. PASSED, 661,075 words.
Bitstream: timing met, WNS +0.282 ns (DDR3 controller), worst `sys_clk`
path +1.43 ns, WHS +0.033 ns; LUT 26.3 %, BRAM 91.6 %, DSP 37.8 %; the
warnings are those of round eighteen.

The model had counted its own loop in `gemm_core` (2.6 Mcycles) as CPU
time since round seventeen; it now counts the tile rows with a formula.

Model: 115.5 → 107.6 Mcycles per frame, and 105.1 without the model's own
loop: **about 2.0 s** (2.10 s, less the model's 4 %).

### Round twenty — the softmax's exponential on the block (software; estimated, not yet measured)

The softmax was the largest CPU item left (16.2 Mcycles, with the block
idle). Its exponential now runs in a requantisation job through the lookup
table; no RTL change was needed.

The job reads the score matrix S[query][key] with one parameter row per
query: multiplier and shift of kf · 512 (kf = the scores' scale times
log2 e), and the bias −r(smax), where r is the job's own rounding and smax
the query's largest score from the score GEMM's statistics. So
v = r(s) − r(smax) is 0 at the largest score and negative elsewhere, and
the table maps v to 2^(v/512) in Q15. The table is the CPU's formula,
exp2_tab[k & 1023] >> (k >> 10) at k = −2v; it does not depend on any
scale, is built once per frame, and is loaded with the first head of each
block (fc1's GELU table replaces it in between).

The requantisation job transposes: it writes P^T[key][query]. The CPU then
forms each query's sum from the columns (for a word w = p(q) + 2^16 p(q+1)
it adds w and w >> 16, eight queries at a time in registers) and writes
P[query][key] divided by the sum, two queries and two keys per step. With
inv = (2^31 − 2^16) / sum no clamp is needed. The CPU work per score is
about 7 cycles instead of 30. A design without this CPU pass would need
four jobs per head: the per-query division cannot be expressed in a
requantisation job, whose parameters are per input row.

| | Before | After |
|---|---|---|
| softmax (CPU, including the wait for the job) | 16.2 | 8.3 |
| accelerator, exponential jobs | — | 1.8 |
| jobs per frame | 1304 | 1376 |
| frame (model) | 105.1 | 97.5 |

**The output is no longer bit-identical to `dav2_host`.** The exponent is
now on a grid of 2^−9 (rounded) instead of 2^−10 (truncated), and the
normalisation rounds inv down instead of to nearest. `dav2_host` has no
lookup table and keeps the CPU softmax; `dav2_host_emu` and the board use
the job. Against the float references nothing measurable changes (new /
old, geometric mean of the per-image ratio, 95 % interval):

| Set | 1 − r | gain-fitted error |
|---|---|---|
| 24 photo crops | 0.961 [0.856, 1.079] | 0.998 [0.920, 1.083] |
| 11 test images | 0.879 [0.768, 1.007] | 0.904 [0.789, 1.035] |

New and old outputs differ by 1 − r ≈ 3e-5, against 1.3e-3 for either
against float. A job that fails turns the block off for the rest of the
frame, as before; the remaining heads then use the CPU softmax, so the
result is valid either way (bus errors injected into 14 jobs from 20 to
1370: correlation 0.99987 or better with the undisturbed result). The
boot self-test hash does not involve attention and stays `c1bf94c1`.

Model: 105.1 → 97.5 Mcycles per frame, **about 1.9 s** (1.95 s, less the
model's 4 %).

### Round twenty-one — the DPT head's interpolations while the block works (software; estimated, not yet measured)

The head's two large upsamplings (36 → 72 in the last fusion block, 72 →
126 before the head's 3 × 3 convolution) ran on the CPU while the block
waited, and then the block ran the next convolution while the CPU waited.
The convolution's tiles (128 pixels each) only need the input rows up to
their last output row, plus the kernel's reach.

A **producer** (`dav2_producer_t`, `dav2.h`) now makes such an input row by
row. The interpolation is one (`dav2_interp_begin`, one step per output
row, or two that share their source rows). The driver asks for the rows a
tile reads before it starts the tile (`dav2_producer_need`: pixels
`n0 + nt` for a GEMM, rows up to `oy·stride − pad + k − 1` for a gather),
and lets the producer make the next rows while it waits for a job
(`dav2_producer_idle`). `qgemm_impl` completes the producer after the
accelerator call and before every path that reads the whole input (the
ReLU copy, im2col, the CPU GEMM). The result is the same, bit for bit:
the same jobs, the same order, the same data.

The emulator now reads a GEMM's A tile when the job starts, as the block
does, instead of row by row during the job; a producer that is late would
give a wrong result there. Checks: the 11 images and the demo are
bit-identical to round twenty (emulator and host), also with a build in
which the waits make no rows (`-DDAV2_PRODUCER_NO_IDLE`, only
`dav2_producer_need`). With one input row too few for a gather, or only
the earlier tiles' rows for a GEMM, the output differs, so the check
works. Bus errors injected into 17 jobs, 12 of them in the head, all end
with a valid result.

Model: 97.5 → 92.0 Mcycles per frame, **about 1.77 s** (1.84 s, less the
model's 4 %). The model starts each job at once and lets the CPU produce
while it runs, which is the ideal overlap; on the board a tile may wait
for its rows, and the CPU makes a row (about 40,000 cycles) at a time.

### Round twenty-two — no copies of q, no GELU table on the CPU (hardware; estimated, not yet measured)

| Change | Where | Model (Mcycles) |
|---|---|---|
| **q's range without a pass over q.** `dav2_qgemm_colext` returns each output column's largest and smallest value. The requantisation is non-decreasing in the accumulator (multipliers ≥ 0), so these are the requantised extremes of the column's accumulators, which the range pass reads anyway. The shift of a head's q depends only on the bit length of its largest \|q\|, which these give exactly. | `dav2_ops.c`, `dav2_engine.c` | q preparation 5.0 → 2.6; frame 92.0 → 90.0 |
| **Weight shift** (`CTRL.wsh`, 4 bits): int16 weights enter the multipliers as w >>> wsh. The score GEMM reads q straight from qkv (rows 1152 apart) and the block applies the head's shift, so the shifted copy of q is gone. | `student_gemm.sv`, `dav2_accel.c`, `dav2_engine.c` | q preparation 2.6 → 0.1 |
| **Interpolating lookup table** (`CTRL.lutint`): the table holds the 257 GELU points, word i = {L[i+1], L[i]} (256 words), and the block computes L[i] + (((L[i+1] − L[i]) · f) >> 6), the formula by which the CPU built the 16,384 entries. The CPU no longer builds them, and a table load is 256 words instead of 8192. | `student_gemm.sv`, `dav2_ops.c` | gelu 2.7 → 1.7 |

All three give the same output, bit for bit (11 images, host and emulator).
The CAPS register has no free bit left, so the two block features are
found by boot self-tests: "weight-shift self-test ok" (an int16-weight
GEMM with the weights shifted by 3) and "interpolating-table self-test ok"
(the table in the same 16,384-entry buffer as the direct table's test, so
an older block, which loads 8192 words, reads nothing outside it). An
emulator of the older block reports both as not present and gives the same
output. The 17 × 6 multiply of the table is kept in logic
(`use_dsp = "no"`); in a DSP48 it had no registers.

`student_gemm_tb`: int16-weight GEMMs with the weights shifted by 2 and 5,
the interpolating table loaded and then reused, with ReLU and over two
tiles, and the direct table again afterwards. PASSED, 680,835 words.
Bitstream: timing met, WNS +0.395 ns (DDR3 controller), worst `sys_clk` path +0.667 ns (lookup table through the interpolation), WHS +0.026 ns; LUT 26.9 %, BRAM 91.6 %, DSP 37.8 %; the warnings are those of round twenty-one. A first build had only +0.103 ns on the path from the result RAM across the chip to the tile RAM (a path from round sixteen that earlier placements gave 1.0 to 1.5 ns); the on-chip requantisation load now has a register stage there.

The model now also counts the table loads (it had left out the 8192 words
per GELU before, about 0.2 Mcycles). Model: 92.0 → 88.0 Mcycles per frame,
**about 1.7 s** (1.76 s, less the model's 4 %).

### Round twenty-three — attention heads in a pipeline, the DPT taps in the background (software; estimated, not yet measured)

| Change | Where | Model (Mcycles) |
|---|---|---|
| **Attention in a pipeline.** The CPU's softmax pass of head h (about 90,000 cycles) now runs while the block computes head h+1's scores, and head h+1's v^T is made while the block computes head h's context. The score matrix, its statistics and v^T have two buffers each. Per head: collect the exponential job, start the next head's scores, the softmax pass, the context GEMM and the next v^T, the context requantisation, the next head's exponential job. | `dav2_engine.c` | attention 13.8 → 12.5; frame 88.0 → 86.7 |
| **The DPT taps in the background.** After blocks 3, 6, 9 and 12 the final LayerNorm of x (without the class token) was a CPU pass with the block idle. `dav2_lnplain_t` splits it: the row statistics and the output scale at once, the rows as background work (`dav2_background_set`), written straight into `feats[j]` (no copy). The steps run in the accelerator waits, also in `qgemm_impl`'s wait for row statistics; `residual_update` completes them before x changes, and the encoder before the head. | `dav2_ops.c`, `dav2_engine.c` | layernorm 11.3 → 10.9; frame 86.7 → 85.3 |

The model's wait (`wait_done`) now makes producer and background steps
while a job runs, as the driver's wait loops do.

Both keep the output the same, bit for bit (11 images, host and emulator;
also the emulator of the older block, and a build in which the waits make
no steps, `-DDAV2_PRODUCER_NO_IDLE`). Bus errors injected into 26 jobs,
most of them in the attention of the first blocks, all end with a valid
result.

Model: 88.0 → 85.3 Mcycles per frame, **about 1.64 s** (1.71 s, less the
model's 4 %).

### Round twenty-four — a finer attention schedule (software; estimated, not yet measured)

Round twenty-three overlapped the softmax pass of head h with the scores
of head h+1, but per head the CPU still waited for the exponential job
and for the context requantisation (synchronous). Now the work of two
heads is interleaved:

| CPU | block |
|---|---|
| v^T of head h+1 | context requantisation of head h−1 (now left running) |
| first half of head h's P | scores of head h+1 |
| second half of head h's P | exponential job of head h+1 |
| column sums of head h+1 | context GEMM of head h |

The softmax pass is split into its column sums and two halves of the
normalisation (by queries); P^T has two buffers. The context
requantisation is left running (`dav2_accel_requant_stride_async`) and
collected before the next context GEMM overwrites its input; if it failed,
the CPU makes that head's context from the intact input. The exponential
job of head h+1 is collected before the context GEMM, so that a failure is
known and that head's softmax runs on the CPU.

Same output, bit for bit (11 images, host, emulator and the emulator of
the older block). Bus errors injected into 25 jobs (20 of them in the first
block's attention) all end with a valid result.

Model: attention 13.8 (round twenty-two) → 10.5 Mcycles; frame 85.3 →
83.7 Mcycles, **about 1.61 s** (1.67 s, less the model's 4 %).

### Round twenty-five — where the block waits, and LayerNorm's arithmetic (software; estimated, not yet measured)

`tools/cyclemodel/frame.c` now also reports how long the block is idle
before each kind of job (the CPU works, the block waits). For round
twenty-four, in Mcycles:

| Block idle before | Mcycles | The CPU meanwhile |
|---|---|---|
| requantisation | 12.1 | mostly the part of the head's interpolations that the convolutions do not cover |
| int16 requantisation (LayerNorm's jobs) | 9.3 | LayerNorm's row statistics and channel pass; the token embedding before the first |
| convolution GEMMs | 4.2 | mostly the fusion blocks' adds |
| attention GEMMs | 3.7 | the softmax pass |

LayerNorm's per-row arithmetic, the same values as before:

- `xf_rsqrt` with 32-bit operands and 64-bit products. X < 2^32; y
  starts in [2^29, 2^30] and stays below 2^30; x·y² ≤ 1.41 at the start and
  ≤ 1 after the first Newton step. The start value's division (X − 2^30) / 6
  was a 64-bit division, a library call (`__divdi3`); it is now 32-bit.
  Checked against the old code on 31.7 million inputs (121 exponents):
  identical.
- The channel pass: lo·g·ZS as lo·G with G = g·ZS per channel, in two 32-bit
  partial products (exact, the value is below 2^57); the bias parameter
  with the mulh form of the rounding shift when the shift is ≥ 33 (exact,
  the rounding constant is then a multiple of 2^32). Checked on 20 million
  random values: identical.

Same output, bit for bit (11 images, host and emulator); the boot
self-test hash stays `c1bf94c1`.

Model: ln stats 3.5 → 2.9 Mcycles; frame 83.7 → 82.8 Mcycles, **about
1.59 s** (1.66 s, less the model's 4 %).

### Round twenty-six — the fusion blocks' adds as producers (software; estimated, not yet measured)

A fusion block adds its two inputs and passes the sum to a residual conv
unit, whose first convolution gathers it with ReLU. The add
(`dav2_add_t`: the scale first, then one image row per step) is now a
producer for that convolution, as the interpolations are for theirs
(round twenty-one): the tiles start as their rows exist and the CPU adds
the next rows while the block works. The unit gets the sum's tensor itself,
whose largest |value| the last step sets (its second convolution's
residual add needs it). Same output, bit for bit (11 images, host and
emulator, also with `-DDAV2_PRODUCER_NO_IDLE`); bus errors injected into
five jobs of the head end with a valid result.

Model: block idle before convolution GEMMs 4.2 → 0.5 Mcycles; frame 82.8
→ 82.0 Mcycles, **about 1.57 s** (1.64 s, less the model's 4 %).

### Round twenty-seven — the interpolation's vertical step on the block (hardware; estimated, not yet measured)

The DPT head's bilinear upsamplings were the largest CPU item left
(15.1 Mcycles, of which the convolutions hid about 5.5). Their vertical
step is now a small job of the block.

**LERP job** (`CTRL.lerp`): the A-load brings two tile rows (N_ROWS = 2,
A_STRIDE apart, K_LEN int16 each); then word j of the output is, per int16
lane, t + (((b − t) · w) >>> 8) with w = `ADD_MULT_X`[8:0] — exactly the
CPU's (t (256 − w) + b w) >> 8, since 256 t is a multiple of 256. A new
state `ST_LERP` reads word j of both rows and writes its interpolation a
cycle later, two cycles per word at most, the bus's rate anyway. The two
17 × 9 multiplies are in logic. No CAPS bit: the boot self-test finds it
("LERP self-test ok"); an older block reads the registers as a GEMM of two
rows and one weight row that writes two words into the test buffer, and
the self-test then reports it as not present.

**The interpolation producer** (round twenty-one) has a second mode: the
CPU makes the horizontal rows, all of them into one buffer (plus a spare
row) in the waits; before a convolution tile the producer's new `need`
callback runs the LERP jobs for the output rows the tile reads (in chunks
of at most 2048 int16), while the waits for them make more horizontal rows.
The tile loops write their registers again after each producer call. A job
that fails leaves its chunk to the CPU.

Checks: same output, bit for bit (11 images, host and emulator; with
`-DDAV2_PRODUCER_NO_IDLE`; with an emulator of a block without the job,
which falls back to the CPU). With one output row too few made, the output
differs. Bus errors injected into 8 of the head's jobs end with the same
output. `student_gemm_tb`: rows of 2016, 64, 8 and 2048 int16, weights 0,
1, 128, 147, 255. PASSED, 685,695 words. Bitstream: timing met, WNS +0.395 ns (DDR3 controller), worst `sys_clk` path +0.837 ns, WHS +0.012 ns; LUT 27.1 %, BRAM 91.6 %, DSP 37.8 %; the warnings are those of round twenty-two.

Model: interpolation (CPU) 15.1 → 5.7 Mcycles, block +3.0 (LERP);
frame 82.0 → 76.7 Mcycles, **about 1.47 s** (1.53 s, less the model's
4 %). The accelerator now runs about 1946 jobs per frame.

### Round twenty-eight — the attention's matrices in the result RAM (software; estimated, not yet measured)

The result RAM (131,072 words of BRAM, round sixteen) held only the
encoder's GEMM results; it was unused during the attention. Per head, the
score matrix S (82 × 82 int32) went to DDR3 and came back for the
exponential job, and the context C (64 × 82 int32) went to DDR3 and came
back for its requantisation: about 24K bus words per head, 1.7M per frame.

Now the score GEMM drains S into the result RAM at word 0, and the context
GEMM drains C at word 16384 (`dav2_accel_gemm16_cr_async`: `CTRL.w16 |
CTRL.onchip`, the statistics still go to DDR3). The exponential job and
the context requantisation read them from there
(`dav2_accel_requant_lut_cr_async`, `dav2_accel_requant_stride_cr_async`).
A new driver offset `accel_cr_base` places a job's data in the result RAM.
No hardware change: both modes already exist. Only the jobs read S and C.
When a job is declined or fails, the CPU makes the lost matrix again in
DDR3 (S from q and k, C from P and v^T; a v^T slot that already holds
the next head's is made again) and continues as before. A boot self-test
("attention result-RAM self-test ok") runs a 20 × 64 × 8 int16-weight
GEMM at word 16384 and its strided requantisation against the CPU; if it
fails, the attention keeps S and C in DDR3.

A bug from round twenty is fixed: `dav2_accel_requant_lut_async` gave the
deferred job the address of a local `amax`, and `dav2_accel_finish` wrote
through it after the function had returned (found by AddressSanitizer).
It now uses a static sink.

Checks: same output, bit for bit (11 images, emulator; also with an
emulator of a block without the result RAM). Bus errors injected into
each of the first 230 jobs (four encoder blocks), with AddressSanitizer
and UBSan: no errors, and every output is within 1.3·10⁻⁴ of the
reference (1 − r).

Model: accelerator busy in the attention 6.6 → 3.2 Mcycles; the attention
is now limited by the CPU (softmax sums and normalisation 6.1 Mcycles,
v^T 2.2), so the frame gains less: 76.7 → 76.1 Mcycles, **about 1.46 s**.

### Round twenty-nine — v^T by a transposition job (software; estimated, not yet measured)

The context GEMM needs v^T of each head (64 rows of the tokens). The CPU
made it from qkv, 2.2 Mcycles per frame, while the attention was waiting
for the CPU. The int16-input requantisation job already transposes: it
reads in[m][n] and writes out[n][m]. With identity parameters (mult
2^30, shift 30, bias 0: round(v · 2^30 / 2^30) = v, and |v| ≤ 8191 is not
saturated) it is a transposition. A new driver pitch, `accel_in_pitch`,
lets the job read rows of qkv (A_STRIDE = qkv row in bytes), so the job
reads head h's 64 v columns of each token and writes v^T rows Kp apart
(`dav2_accel_transpose16_async`). No hardware change. The padding columns
n..Kp−1 are zeroed once per attention call. The job is collected before
the next one starts; a failure leaves v^T to the CPU. A boot self-test
("transposition self-test ok") checks 12 rows of 16 int16 with a pitch of
40 into rows 14 apart.

Checks: same output, bit for bit (11 images, emulator; also with emulators
of a block without int16 input and without the result RAM). Bus errors
injected into each of the first 240 jobs, with AddressSanitizer and UBSan:
no errors, every output within 1.3·10⁻⁴ of the reference (1 − r).

Model: v^T on the CPU 2.2 → 0 Mcycles (1.0 now waits for the job), block
+0.8 (transpose), block idle before attention GEMMs 5.1 → 2.6; frame
76.1 → 75.0 Mcycles, **about 1.44 s**. About 2021 jobs per frame.

## 7. What is left, in order of expected gain

Based on the measured 8.187 s profile (§6, round six), not the earlier
estimate. Round seven removed the float routines named in this table;
LayerNorm is now 86, requantisation 74 and GELU 43 Mcycles. Round eight
addresses most rows of this table; its estimate is in §6.

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
