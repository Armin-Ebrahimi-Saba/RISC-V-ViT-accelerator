# Performance — how one frame went from 93.6 s to 14.2 s

This is the record of the speed-up work: what was measured, what each change
did, and what is left. Every step kept the FPGA output **bit-exact with the
host build** of the same C engine, and the correlation with the PyTorch
reference on the demo image went from 0.999836 to 0.999872 along the way
(one change in LayerNorm improved the numerics slightly; nothing was traded
for speed).

Terms used below: *frame* — one 126×126 depth map; *GEMM* — a matrix
multiplication, the accelerator's job; *requantisation* — turning the int32
sums a GEMM produces back into int16 activations with a per-row scale;
*beat* — one 32-bit word crossing the bus.

---

## 1. The result

| | Start of the work | Now |
|---|---|---|
| Frame time (inference only) | 93.6 s | **14.2 s** |
| Frames per second | 0.0107 | **0.0704** |
| Result readout to the PC | ~30 s (hex text over the console) | 0.7 s (JTAG system bus) |
| Accelerator | 16 multipliers, 1 read in flight, 8.3 cycles/beat | 64 multipliers, 8 in flight, 3.2 cycles/beat |
| FPGA output vs host build | bit-exact | bit-exact |
| Correlation with PyTorch (demo) | 0.999836 | 0.999872 |
| LUT / BRAM / DSP | 12.4 % / 23.0 % / 3.1 % | 14.8 % / 36.9 % / 10.4 % |
| Timing (WNS, 50 MHz) | +0.287 ns | +0.121 ns |

Where the 14.2 s goes now (cycles at 50 MHz, from the profile the program
prints after every frame):

| Operator | Mcycles | Share | Runs on |
|---|---|---|---|
| LayerNorm | 128 | 18 % | CPU |
| attention (gathers, softmax, normalisation) | 114 | 16 % | CPU; the two matmuls on the accelerator |
| requantisation (per-row parameters, job wait) | 101 | 14 % | mostly accelerator |
| GEMM | 93 | 13 % | accelerator |
| residual adds, ReLU | 92 | 13 % | CPU |
| im2col (convolution patch gather) | 67 | 9 % | CPU |
| GELU | 51 | 7 % | CPU |
| interpolate, misc | 63 | 9 % | CPU |

The profile is flat. No single operator is worth more than a fifth of the
frame any more, and everything left on the CPU is instruction count, not
memory (see §4).

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

## 5. What is left, in order of expected gain

| Item | Now | Estimate | How |
|---|---|---|---|
| LayerNorm per element | 128 Mcycles | −80 | one pass storing the Q16 value instead of two computing it (memory is cheap, instructions are not); or a third accelerator job |
| attention CPU side | 114 | −60 | gather/split loops unrolled; softmax exponent in a small LUT |
| per-row float work in requant | ~70 | −50 | integer `make_multiplier` (frexp is a bit scan) |
| residual adds | 92 | −40 | fuse the range scan into the previous operator's output pass |
| im2col | 67 | −60 | a gather mode in the accelerator's A-load (read rows at a stride with a window) |
| GELU | 51 | −30 | unroll; LUT index arithmetic on packed pairs |
| GEMM | 93 | −40 | repair or replace the bypassed prefetcher; larger cache lines |

All of these together would land around 6–7 s per frame. Beyond that the
weights themselves — 25 MB crossing DDR3 twice per frame at ~3 cycles per
word — set a floor near 3 s at 50 MHz; a faster fabric clock or a wider
bus would be the next step.
