<!--
SPDX-License-Identifier: CC0-1.0
SPDX-FileCopyrightText: 2026 RVLab Student Project
-->

# Depth-Anything V2 (Small) on the RVLab RISC-V SoC

Board: Digilent Nexys Video, Xilinx Artix-7 **XC7A200T**
CPU: **CV32E40P**, RV32IMC, no FPU, no SIMD, no vector unit
Model: **Depth-Anything V2 Small** — ViT-S/14 encoder + DPT head, **24,785,089 parameters**
Weights: **int8 in DDR3** at `0x8000_0000`
Compute: **entirely on the RISC-V core** (no hardware accelerator)

**Status: the model runs and is numerically verified.** The C engine reproduces
the PyTorch reference with **correlation 0.9998** and 2.3 % relative RMSE at
126×126 input.

---

## 1. Result summary

| Metric | Value |
| --- | --- |
| Model | Depth-Anything V2 Small, all 24.8 M pretrained weights |
| Weight format | per-output-channel symmetric int8 → **24.87 MB blob in DDR3** |
| Activation format | int16 (14-bit range), per-tensor dynamic scale |
| Accumulator | int32, provably non-overflowing |
| Input resolution | 126×126 (81 patches), compile-time configurable |
| Accuracy vs PyTorch float | **corr 0.9998, rel. RMSE 2.26 %** (demo01) |
| Accuracy across 4 test images | corr **0.9959 – 0.9999** |
| Activation memory (arena peak) | **15.34 MB** (DDR3) |
| Program footprint | 18.9 KB .text, 1.8 KB .data, 54.9 KB .bss (fits 256 KB BRAM) |
| Compute per frame | **2.412 G MAC** at 126×126 |
| Estimated frame time | ~1.6–3.2 min at 100 MHz (4–8 cycles/MAC) |
| Bitstream | timing met, **WNS +0.100 ns**, 0 failing endpoints |
| Floating point in inner loops | **none** — all per-element work is integer |
| GEMM accelerator (§6c) | 16 MAC/cycle, **12.4 measured in unit sim**, not yet synthesised |

Sections 1–6b describe the pure-software port, which is complete and verified.
**§6c** covers the int8 GEMM accelerator added afterwards: implemented and
unit-verified, but not yet taken through synthesis or onto hardware.

Accuracy across the test images:

| Image | corr vs PyTorch | rel. RMSE | max abs err |
| --- | --- | --- | --- |
| demo01 | 0.99984 | 2.26 % | 0.197 |
| demo02 | 0.99966 | 3.83 % | 0.057 |
| demo05 | 0.99588 | 13.82 % | 0.045 |
| demo09 | 0.99990 | 6.77 % | 0.052 |

(demo05's high relative RMSE is an artefact of its small dynamic range — its
absolute error is the second smallest of the four.)

Rendered side-by-side comparisons (input, PyTorch reference, this engine, error)
are in `build/dav2/comparison.png`. The reference and engine columns are
visually indistinguishable; the error column is normalised to its own maximum,
which exaggerates it.

### Resolution scaling

Input size is a compile-time parameter; all three were built and verified:

| Input | Patches | MACs | Arena peak | Est. frame time @100 MHz | corr vs its own float ref |
| --- | --- | --- | --- | --- | --- |
| 70×70 | 25 | 0.761 G | 4.74 MB | ~46 s | 0.99934 |
| **126×126** | **81** | **2.412 G** | **15.34 MB** | **~145 s** | **0.99984** |
| 154×154 | 121 | 3.591 G | 22.91 MB | ~216 s | 0.99939 |

(Frame-time estimates assume 6 cycles/MAC — one int16 load, one int8 load, a
multiply and an add on a single-issue in-order core, with the loop unrolled by
4. They exclude DDR3 stalls, so treat them as a lower bound.)

---

## 2. Constraints established before writing code

### 2.1 What the platform provides

From `docs/design_ref/`:

| Resource | Value |
| --- | --- |
| Main BRAM | 256 KB @ `0x0000_0000` |
| DDR3 | 512 MB @ `0x8000_0000` |
| CPU | CV32E40P, RV32IMC |
| Stack top | `0x0003_8000` |
| hostio (stdout/stdin) | `0x0003_F000` |
| DDR3 controller regs | `0x1f00_1000` |

The DDR3 path is UberDDR3 behind a 16 KB, 256-bit-wide, **direct-mapped**
last-level cache with a TL-UL adapter. Direct-mapped is the important word: it
drove the decision to keep every weight access strictly sequential (§4.3).

### 2.2 What the model costs, honestly

ViT-S/14: 24.8 M parameters, patch 14, embed dim 384, 12 blocks, 6 heads; DPT
decoder fusing blocks 3/6/9/12.

* **Storage** — int8 weights are ~25 MB. They fit DDR3 with room to spare, and
  they *must* live there: 25 MB does not fit in 256 KB of BRAM. The task's
  requirement that weights live in DDR3 is therefore also a necessity.
* **Compute** — at the native 518×518 the model is ~40 GFLOPs/frame. A scalar
  RV32IMC at ~100 MHz sustains on the order of 10–50 MOPS. Native resolution
  would take **hours to days per frame**.

Since the user's instruction is that the RISC-V core does the computation, the
only available lever is input resolution. ViT-S/14 needs a multiple of 14:

| Input | Tokens | Rough cost vs native |
| --- | --- | --- |
| 518×518 | 1369 | 1.00× |
| 154×154 | 121 | ~0.09× |
| **126×126** | **81** | **~0.06×** |
| 70×70 | 25 | ~0.02× |

126×126 was chosen as the default. At that size the network is ~2 G MAC/frame.

**A floor that resolution cannot remove:** every one of the ~25 MB of weights
must cross the DDR3 bus at least once per inference, regardless of input size.
That alone sets a lower bound of roughly 0.3–0.5 s/frame at realistic TL-UL
bandwidth. This is a **functional** port, not a real-time one — frame time is
expected in the minutes. That is the physically achievable result for this
model on this core without an accelerator.

---

## 3. Environment setup

### 3.1 What was missing

* Vivado 2022.2 present (XSim available); **no QuestaSim** → simulation via XSim.
* **No RISC-V toolchain** → nothing could be built at all.
* **No PyTorch/numpy** → no way to read or quantise the checkpoint.
* **No board attached** (`lsusb` shows no Digilent device). The user confirmed
  the board exists but is not currently plugged in, so everything is built and
  verified up to the bitstream, and the board run is scripted and documented.

### 3.2 RISC-V toolchain

`flow/tools/build_sw.py` searches PATH for `riscv-none-elf-`,
`riscv32-unknown-elf-`, `riscv-none-embed-`, and builds with
`-march=rv32imc_zicsr -mabi=ilp32 -Os -nostdlib`.

The docs suggest unpacking the xpack toolchain into `~/bin/riscv`, but
**`/home/armin/bin` is root-owned and not user-writable**, so it went to
`~/.local/opt/riscv` instead. Installed: xpack riscv-none-elf-gcc **15.2.0-1**.

```bash
export PATH=~/.local/opt/riscv/bin:$PATH
export PATH=~/bin/Vivado/2022.2/bin:$PATH
export XILINX_VIVADO=~/bin/Vivado/2022.2
```

Baseline check: `flow sw_minimal build` succeeded, confirming the toolchain
works with the existing flow before any model code was written.

### 3.3 Host Python environment

Model preparation uses a separate venv (numpy, pillow, safetensors,
huggingface_hub, CPU torch), kept out of the repo's own `.venv`, which exists
only for PyDesignFlow/Sphinx. System Python is 3.14.4; torch **2.13.0+cpu**
installs against it. First attempt failed because the PyTorch CPU wheel index
does not host `safetensors`/`huggingface_hub` — fixed by installing the general
packages from PyPI and only torch from the PyTorch index.

Checkpoint: `depth-anything/Depth-Anything-V2-Small`,
`depth_anything_v2_vits.pth`, verified at 24,785,089 parameters.

---

## 4. Design

### 4.1 De-risking before writing C

Writing a few thousand lines of quantised C and *then* discovering that int8
destroys a ViT would have been the expensive way to fail. So the first
deliverable was **`tools/dav2_numpy.py`**, a NumPy reimplementation of the whole
model that mirrors, op for op, what the C would do.

Two checks, in order:

1. **Is the reimplementation right?** In float mode it matched PyTorch to
   **2.9e-05** max abs error — pure float32 rounding noise. The blueprint is
   correct.
2. **Does quantisation survive?** With "fake quantisation" inserted at exactly
   the points where the C engine quantises:

| Config | max err | rel. RMSE | correlation |
| --- | --- | --- | --- |
| int8 act / int8 weight | 1.018 | 6.31 % | 0.98107 |
| **int16 act / int8 weight** | 0.211 | 2.74 % | **0.99977** |
| int16 act / int16 weight | 0.004 | 0.01 % | 1.00000 |

This is the single most valuable measurement in the project. It shows int8
*activations* are what cost accuracy in a ViT (the well-known LayerNorm outlier
problem), while int8 *weights* are nearly free. Keeping weights at int8 —
which is what the DDR3 storage requirement is really about — while running
activations at int16 buys back almost all the accuracy at no arithmetic cost:
on RV32IMC a 16×8 multiply and an 8×8 multiply are both a single `mul`.

### 4.2 The overflow analysis that fixed the activation width

int16 activations raise a real hazard. The worst-case accumulation is the MLP's
second layer, K = 1536:

```
K * act_max * weight_max  <  2^31 ?
1536 * 32767 * 127 = 6.39e9   >  2.15e9   OVERFLOW
1536 *  8191 * 127 = 1.60e9   <  2.15e9   safe
```

So activations are stored in int16 but **restricted to a 14-bit magnitude**
(±8191). Measured accuracy is identical to full int16 (corr 0.99975 vs 0.99977),
and int32 accumulation is now provably safe rather than empirically safe.

The same analysis applies inside attention, where q·k has K = 64. Full-range
14-bit q and k would give 64·8191² = 4.3e9 — overflow again. Q and K are
therefore rescaled to 11 bits (±2047) before the score computation:
64·2047² = 2.7e8, very safe. The attention·V product needs no such treatment
because the probabilities sum to 1 by construction, bounding the result at
32768·max|v|.

### 4.3 Numeric pipeline

Per GEMM (which, after im2col, is *every* layer in the network):

1. **Accumulate** int32: `acc[n][m] = Σ a[n][k]·w[m][k]`, tracking per-channel
   min/max as a side effect.
2. **Derive the output scale exactly**, including bias, from those per-channel
   extremes — O(channels) float work. Because the true range is known, nothing
   clips and no calibration dataset is needed.
3. **Requantise** with a per-channel integer multiplier and shift
   (`(acc·mult) >> shift`, gemmlowp style).

Float is used *only* in step 2 — a few hundred operations per layer. Every
per-element operation is integer. This is why the missing FPU costs almost
nothing.

Consequences of dynamic (rather than calibrated) scaling: no calibration
dataset, no offline activation statistics, and robustness to unusual inputs.

### 4.4 What was folded away offline

The exporter (`tools/export_dav2.py`) does everything that can be done once on
the host, so the CV32E40P never repeats it:

* **Positional embedding interpolation** — DINOv2 bicubic-interpolates a 37×37
  position grid to the target grid with a `+0.1` offset trick. Since resolution
  is fixed at build time, the interpolated table is precomputed and stored.
* **LayerScale** (`ls1.gamma`, `ls2.gamma`) folded into the preceding weight
  matrix and bias. Note this must fold into the *weights*, not the scales:
  gamma can be negative, and a per-channel quantisation scale cannot be.
* **The attention `1/sqrt(head_dim)` factor** folded into the Q rows of `qkv`.
* **Every convolution reshaped into a plain (M×K) GEMM matrix**, so the engine
  needs exactly one matrix kernel.

### 4.5 Memory layout: NHWC

Feature maps are **channel-last**. A tensor is an (n × c) matrix of rows
(pixels or tokens) by channels. This makes an im2col row contiguous and keeps
weight streaming sequential — which matters specifically because the DDR3
last-level cache is direct-mapped and would thrash on strided access.

GEMM tiles activation rows into a 16 KB BRAM buffer and streams the weight rows
from DDR3 sequentially, so the fast memory holds the data that is reused and the
slow memory is only ever read forwards.

### 4.6 Elementwise kernels

* **GELU** — the input is a 14-bit quantised value, so GELU is just a map from
  a bounded integer range. A 257-entry table with linear interpolation is built
  per layer (257 float evaluations) and then applied with two adds and a shift
  per element, instead of 126k erf evaluations.
* **Softmax** — fixed point throughout: `2^-x` via integer split into integer
  and fractional parts, with a parabolic correction on the fractional part.
* **LayerNorm** — mean and variance accumulated in int32/int64, then one
  reciprocal square root per row; the per-element normalisation runs in float
  because there are only ~31k elements per LayerNorm and 28 of them in total.
* **Bilinear resize** — `align_corners=true` to match `F.interpolate`, with Q8
  integer weights.

### 4.7 No libm

The rvlab build links `-nostdlib` with only `-lgcc`. libgcc provides soft-float
arithmetic but **no transcendentals**, so `sqrtf`, `expf`, `erff`, `frexpf` and
`ldexpf` are implemented in `dav2_mathf.c`. They are used on both host and
target so the two builds stay bit-identical. Verified against libm: sqrt
1.2e-07, exp 9.8e-07, erf 3.0e-07 max relative error — all far below int8 noise.

---

## 5. Bringing it up: three real bugs

The engine did not work the first time. All three failures were found by
comparing against the NumPy blueprint, which is exactly why the blueprint was
written first.

### Bug 1 — transposed-convolution bias overrun

*Symptom:* output values of 1e38; correlation 0.17.

A `-DDAV2_TRACE` mode prints the dynamic range of every intermediate tensor.
The first tensor with an absurd scale was `resize0`, the 4× transposed
convolution in the DPT head. Its GEMM has 768 rows (`k·k·Cout` = 4·4·48), but
the exported bias had only 48 entries — one per output channel. The requantiser
read `bias[m]` for m up to 767, i.e. off the end of the array.

*Fix:* tile the bias `k·k` times in the exporter so it lines up with the GEMM
rows, which are ordered `(ky, kx, cout)`.

*Lesson:* reshaping a convolution into a GEMM changes what "output channel"
means, and the bias has to follow that change.

### Bug 2 — softmax `2^-x` correction had the wrong sign

Found by inspection while chasing bug 3. Linear interpolation between 1.0 and
0.5 *overestimates* `2^-x` because the function is convex, so the parabolic
correction must be subtracted, not added; the coefficient was wrong too. At
x = 0.5 the old code returned 25130 where 23170 was correct (+8.5 %).

*Fix:* `p = lin - ((x(1-x) · 5623) >> 16)`, where 5623 = 4 × the peak gap
(0.75 − 2^-0.5 = 0.0429 → 1406 in Q15). Now accurate to ~0.2 %.

### Bug 3 — the attention softmax was uniform (the real one)

*Symptom:* after fixing bug 1, end-to-end correlation was still only 0.67.

Bisection down the network: patch embedding correct (corr 0.9999) → block 0
already wrong (0.69). Bisection *inside* block 0: `norm1` correct, `qkv`
correct (0.999986 — so the weight folding was right), but **`ctx` correlation
0.14 with magnitude 4.5× too small**. A too-small context vector is the
signature of a softmax that is too flat: it averages all the value vectors
instead of selecting.

Instrumenting the score computation showed the scores themselves were exactly
right — integer logit spread 14.590 against NumPy's 14.594. The bug was in the
conversion to the fixed-point exponent:

```c
int32_t k_q16 = (int32_t)(kf * 65536.0f + 0.5f);   /* kf ~ 1.2e-5 */
int64_t t_q16 = (d * (int64_t)k_q16) >> 16;        /* extra shift */
```

Two compounding errors. `kf·65536` was 0.806, which rounds to **1** — a
24 % error with no fractional resolution left. And the `>> 16` was an extra
shift that did not belong, since `k_q16` already carried the Q16 factor. The
result: every exponent collapsed to ~0, every probability to ~1.0, and the
softmax became uniform.

*Fix:* carry `kf` as a (multiplier, shift) pair using the same
`dav2_make_multiplier` helper the requantiser already used and had tested:

```c
dav2_make_multiplier(kf, &kmult, &kshift);
t_q16 = (d * (int64_t)kmult) >> (kshift - 16);
```

`ctx` correlation went from 0.14 to **0.9998**, all 12 blocks to ≥ 0.998, and
end-to-end to **0.99984**.

*Lesson:* a small float carried in fixed point needs a dynamic exponent. Fixing
the constant to Q16 silently destroyed it. The general failure mode — rounding
a sub-unity scale factor into a small integer — is worth watching for anywhere
in quantised code.

### Verification method

Before the engine was trusted, each kernel was unit-tested against NumPy on
random data (`optest.c` / `optest.py`):

```
OK  conv 1x1 s1 p0        corr 0.999991     OK  interp 5x5->9x9      corr 1.000000
OK  conv 3x3 s1 p1        corr 0.999980     OK  interp 9x9->18x18    corr 0.999997
OK  conv 3x3 s2 p1        corr 0.999978     OK  interp 72x72->126x126 corr 0.999996
OK  conv 14x14 s14 p0     corr 0.999976     OK  gelu                 corr 1.000000
OK  convT stride 2        corr 0.999992     OK  layernorm            corr 1.000000
OK  convT stride 4        corr 0.999990
```

This is what localised bug 3 so quickly: since every kernel was individually
correct, the fault had to be in the engine's wiring, not the maths.

---

### Verifying the RISC-V build without a board

The host build proves the *algorithm*. It does not prove that the **RISC-V**
build computes the same thing: different compiler, 32-bit target, and libgcc
soft-float instead of hardware float. No RISC-V emulator was available on this
machine (no qemu-user, no spike, and no 32-bit glibc headers for an `-m32`
cross-check), so verification uses `dav2_selftest.c`:

* it runs every kernel — GEMM with and without bias, LayerNorm (int64 + rsqrt),
  GELU (LUT build via erf/exp), residual add with mismatched scales, conv
  stride 1 and 2, transposed conv, bilinear resize — on a deterministic
  xorshift32 sequence;
* it hashes every output tensor **including the float scale bit patterns**,
  so the soft-float path is covered too;
* both builds print the same FNV-1a checksum. Since every kernel is integer and
  the float bookkeeping is IEEE-754 on both sides, the checksums must be
  identical. A mismatch would point at the toolchain or a 32-bit assumption,
  not at the model.

**Result — the two builds agree exactly:**

```
host (x86-64, hardware float):        DAV2_SELFTEST 3ac57cd2
RISC-V (CV32E40P in XSim RTL sim):    DAV2_SELFTEST 3ac57cd2
```

This confirms that the RISC-V build of every kernel — integer arithmetic,
int64 paths, and the whole soft-float bookkeeping chain — is bit-identical to
the host build that was validated against PyTorch. The accuracy numbers in §1
therefore carry over to the target.

The RISC-V side runs the same routine inside an XSim RTL simulation of the full
SoC (added `sim_rtl_xsim_batch` for this, since the stock XSim tasks are
GUI-only). It needs no DDR3 and no weights, so it works in plain RTL
simulation. It is slow there — 17 minutes wall clock for ~87 ms of simulated
time, most of it the GELU table's 257 soft-float erf/exp evaluations; on real
hardware the same self-test takes milliseconds.

The run ends with `Error: DDR not present` and `[FAIL] test_sw`. **This is the
expected outcome, not a failure of the port:** the `sim_rtl_xsim` targets build
from `srcs_noddr`, which deliberately excludes the DDR3 controller from RTL
simulation for speed (see `docs/design_ref/ddr3_memory.rst`). The program
correctly detects the missing controller and exits non-zero, and the testbench
reports that as a failing program. This ordering is deliberate: the self-test
runs *before* `ddr_init()` precisely so that it is usable in a DDR3-less
simulation. ### The DDR3-inclusive simulation: attempted, not completed

To exercise `ddr_init()` and the DDR3 read/write probe in RTL, a
`sim_rtl_xsim_ddr_batch` run was started (a second headless task added for the
DDR3-inclusive source set). **It did not finish**, and the reason is worth
recording:

* it confirmed the DDR3 model responds and the UberDDR3 controller runs its
  calibration reads, and the program boots and prints its banner;
* but it reached only ~10.5 ms of simulated time in ~80 minutes, versus the
  ~87 ms the no-DDR3 run needed to complete. Adding the DDR3 model slows
  simulation by roughly 40x, projecting to **~12 hours** for the same program.

The run was terminated before completing. This is precisely the trade-off
`docs/design_ref/ddr3_memory.rst` describes when it says the DDR3 interface is
excluded from RTL simulation by default. **The DDR3 path is therefore verified
by construction and by code review, not by simulation:** `ddr_init()` is the
stock rvlab routine used unmodified by `test_rvlab` and `monitor`, and the
program's own startup probe (write/read-back at `0x8200_0000`) checks the path
at run time on real hardware before any weights are trusted. Confirming it end
to end is a task for the board, not the simulator.

## 6. Hardware results

The design needed **no RTL changes** — the user's requirement is that the
RISC-V core does the computation, so this is a pure software port onto the
stock rvlab SoC. The bitstream was nevertheless built end to end to confirm the
flow works and the board is ready.

*(An int8 GEMM accelerator was added later as a separate step; see §6c. The
numbers in this section and in §6b are from the pre-accelerator bitstream and
have not yet been re-measured with it.)*

| Metric | Value |
| --- | --- |
| Worst negative slack | **+0.100 ns** (timing met) |
| Failing endpoints | 0 of 41635 |
| Worst hold slack | +0.055 ns |
| Slice LUTs | 15526 / 133800 (11.6 %) |
| Slice registers | 11781 / 269200 (4.4 %) |
| Block RAM | 68 / 365 (18.6 %) |
| DSPs | **5 / 740 (0.7 %)** |

That last row is the interesting one: 735 unused DSP slices, i.e. the XC7A200T
has ample room for the int8 MAC array that would make this fast. The engine is
already shaped for it — a single kernel (`dav2_qgemm`) accounts for essentially
all 2.4 G MACs, so an accelerator has exactly one attachment point.

## 6b. Post-implementation report review

TASK.md asks for the synthesis / PNR / bitstream reports to be checked for
warnings and fixed *without changing third-party libraries if possible*. All
listed files were reviewed:

| Report | Result |
| --- | --- |
| `bitstream/*.io_report.txt` | **Clean.** "No pins missing in reference / observed"; all 155 pins match the pincheck CSV. |
| `syn` + `pnr` `*.drc.txt` | 16 violations, **all severity Warning**, 0 Errors, 0 Critical Warnings. |
| `syn` + `pnr` `*.methodology.txt` | 43 violations, all Warning/Advisory. |
| `*.qor_assessment.txt` | **Score 5/5 — "Design runs will meet timing."** Methodology Check Details table is empty. |
| `*.timing_summary.txt` | Setup, hold and pulse-width all met, 0 failing endpoints of 41635. `check_timing` reports 19 `no_input_delay` and 52 `no_output_delay` (HIGH). |

### Where the 59 warnings actually live

Every one of them is in vendored or shared IP — **none in `src/rtl/student/`, and
this project added no RTL at all**:

| Rule | Count | Location |
| --- | --- | --- |
| DPIR-1 (async driver blocks DSP merge) | 32 | CV32E40P multiplier |
| DPIP-1 / DPOP-1 / DPOP-2 (DSP pipelining) | 15 | CV32E40P multiplier |
| SYNTH-10 (wide multiplier) | 3 | CV32E40P multiplier |
| SYNTH-5 (distributed RAM mapping) | 4 | DDR3 last-level cache |
| RTGT-1 (RAM retargeting, Advisory) | 4 | DDR3 block manager |
| CHECK-1 (list of disabled rules) | 1 | informational, not a violation |

### Why these were deliberately not "fixed"

* **The 50 CV32E40P warnings cannot be fixed without editing the CPU.** DPIR-1
  states the cause explicitly: the core's registers use *asynchronous* reset,
  and DSP48 registers support only synchronous reset, so they cannot be merged.
  Fixing it means changing a verified third-party CPU's reset architecture.
  DPIP/DPOP would require adding pipeline stages to the multiplier, changing
  its latency — which the core's control logic depends on.
* **SYNTH-5 is Vivado making a deliberate good choice.** Its own text: mapped to
  distributed RAM "because ... the chosen mapping will yield a better timing."
  Reverting it would make timing worse.
* **RTGT-1 is advisory** and would save ~12 LUTs out of 133,800 (0.009 %).

### The one safe experiment that was tried

Synthesis already runs `directive="PerformanceOptimized"`. The one
behaviour-preserving lever not in use was register retiming, which moves
*existing* registers across combinational logic without changing cycle counts.
It was tested (`synth_design -retiming`) and made **no difference whatsoever**:
DRC 16 → 16, methodology 43 → 43, WNS 0.100 → 0.100 ns, LUTs unchanged. This is
exactly what DPIR-1 predicts — the async resets block the DSP merge, so there is
nothing for retiming to move. The change was reverted rather than left in shared
flow code as a no-op.

### The unconstrained I/O, and why fabricating constraints would be wrong

`check_timing` flags 19 inputs and 52 outputs with no delay constraint. This is
**not an oversight**: `src/design/xdc/rvlab_fpga_top.xdc` constrains exactly the
interface where timing matters — JTAG (`create_clock` on TCK, input delay on
TDI/TMS, output delay on TDO, false path on TRST). The remaining ports (LEDs,
switches, PMOD, OLED, audio codec, SD, HDMI, Ethernet) are routed through the
`userio` struct to the **student module**, so their correct timing depends on
what a given project drives them with.

Inventing `set_input_delay`/`set_output_delay` numbers for them would
manufacture constraints from no data — either creating false violations that
mask real ones, or granting false confidence. Blanket `set_false_path` would
silence the HIGH warning but would also hide genuine timing problems for any
future project that does drive Ethernet or HDMI. Both were rejected as
dishonest fixes to a shared course constraint file.

### One option left on the table (deliberately not taken)

The QoR report flags 158 cells carrying DONT_TOUCH/MARK_DEBUG, which blocks
optimisation. These come from `(* mark_debug = "true" *)` attributes left in
`src/rtl/ddr3/ddr3_controller.v` (UberDDR3, third-party). They *could* be
cleared from project-level XDC via `set_property MARK_DEBUG false`, which would
respect the "don't edit third-party sources" rule and might recover some LUTs
and timing margin. It was **not** done, because it alters synthesis of the DDR3
controller — the one block this project could neither exercise in simulation
(too slow, §5) nor test on hardware (no board). Trading unverifiable risk in the
DDR3 path for a handful of LUTs is a bad bargain when the entire model depends
on DDR3 working. It is recorded here as a decision for whoever has the board.

**Conclusion: no warning found in these reports is a defect, and none can be
fixed without either editing third-party IP or fabricating constraints. The
design meets timing with zero failing endpoints and the best possible QoR score,
so the correct action was to leave it alone.**

---

## 6c. The int8 GEMM accelerator

Everything up to §6b is the pure-software port that was originally asked for.
The accelerator was added afterwards, as a separate step, once that port was
verified. The relationship between the two is deliberate: **the software kernel
was not replaced.** It is still compiled in, still the reference, and still the
path taken whenever the accelerator declines a shape. The accelerator is an
offload for one function, not a rewrite.

### Why exactly one attachment point

The trace build reports the whole frame at 126×126 as **2.412 G MAC** and
**5.273 M requantised output elements**. Every one of those MACs goes through
`dav2_qgemm()` — attention projections, MLPs, and all convolutions, because
`dav2_conv2d`/`dav2_conv_transpose` are im2col wrappers around the same GEMM.
So there is a single function to accelerate, and it was already shaped for it
(NHWC layout, contiguous weight rows, int8×int16→int32).

> A note on clock rate: `sys_clk` is **50 MHz** (20 ns, `rvlab_fpga_top.xdc`),
> not the 100 MHz the §1 estimate table assumes. Double those frame times.
> Figures in this section use 50 MHz.

### Architecture

`src/rtl/student/student_gemm.sv`, parameterised `NROWS=16`, `KMAX=2048`,
`OUTSTANDING=8`. It is a TL-UL device (registers) *and* a TL-UL host (memory).

The dataflow is weight-stationary-inverted: the **activations** stay put and
the weights stream past them.

1. **LOAD_A** — a tile of `NROWS` activation rows is read into `NROWS` private
   block RAMs, one per row (1024 × 32 bit each, so one RAMB36 per row).
2. **MAC** — the weight matrix is read as *one contiguous byte stream*. Each
   32-bit beat carries four int8 weights; each weight is broadcast to all 16
   multipliers, which pair it with their own activation. One weight row takes
   K cycles and produces 16 int32 accumulators.
3. **DRAIN** — those 16 accumulators are written back as one contiguous run,
   and the next weight row starts.

All three streams are sequential in memory. That was the design driver, not an
accident: the DDR3 last-level cache is direct-mapped, so a strided access
pattern would thrash it.

The MAC is a 3-stage pipeline — present the tile-RAM address, select the half
of the 32-bit word addressed by `k[0]`, then multiply-accumulate — so the
multiply and the accumulate map onto a DSP48E1 with its A, B and P registers
and nothing in the fabric between them. 16 DSPs, 16 BRAMs.

### The three decisions worth explaining

**Why the output is transposed.** The accumulator block is stored `[m][n]`, not
`[n][m]`. In `[n][m]` the accelerator would emit 16 words at stride `M*4` per
weight row — 16 separate cache lines, revisited on every subsequent `m`, on a
direct-mapped cache. In `[m][n]` it emits 16 *adjacent* words. The software
kernel was changed to produce the same layout so the two paths stay
interchangeable, and it turned out to suit the CPU too: the requantisation pass
can now hoist `mult`/`shift`/`bias` out of its inner loop, because they are
constant per `m`.

**Why there is a reorder buffer.** One 32-bit beat feeds 4 × 16 = 64 MACs, so
at 16 MACs/cycle the block needs ≈ 0.3 beats/cycle to stay compute-bound. With
a single outstanding request it would instead pay the full memory latency per
beat and run at roughly a third of that. So the read side keeps up to 8
requests in flight, tagging each with its slot index in `a_source` and
reassembling them in order. This also means the block does **not** assume the
memory answers in order — the unit testbench answers out of order on purpose.

**Why `OUTSTANDING` is 8 and not more.** The TL-UL sockets between the student
port and DDR3 steal source-ID bits: the local 2-way `tlul_socket_m1` takes one
and the crossbar's 4-way socket takes two, leaving 5 usable bits. 8 read slots
plus a write flag uses 4, keeping a bit of margin. More slots would only help
the LOAD_A phase, which for real model shapes (M = 384…1536) is ~2 % of the
job. Not worth spending the last margin bit on.

### Integration

`student.sv` now splits the fast device window by address and merges the two
host ports, using the stock TL-UL sockets rather than hand-written arbitration:

| Address | Device |
| --- | --- |
| `0x2000_0000` | `student_dma` (unchanged) |
| `0x2001_0000` | `student_gemm` |

`tlul_socket_1n` steers requests down, `tlul_socket_m1` merges host traffic up.
The DMA exercise is untouched and still works. `student_tlul_mux.sv` — the
lab's own stub — was deliberately left alone rather than co-opted.

### Software contract and fallback

`dav2_accel.c` drives it. `dav2_accel_qgemm()` returns 0 — meaning "I did
nothing, run the software kernel" — whenever K is not a multiple of 4, K
exceeds `KMAX`, any pointer is unaligned, or a bus error was latched. The
caller in `dav2_qgemm()` is one line:

```c
if (!dav2_accel_qgemm(a, wt, acc))
    dav2_qgemm_cpu(a->v, wt->w, acc, N, K, M);
```

Capabilities are read from a `caps` register rather than hard-coded, so the
same binary runs on a bitstream with a different `NROWS`, or with no
accelerator at all.

There is also a **start-up cross-check**: `dav2_accel_check()` runs a fixed
20 × 64 × 6 GEMM through both paths and compares every word. It needs only
BRAM, so it runs before `ddr_init()` and works in a DDR3-less simulation. If it
fails, the accelerator is disabled and inference still produces the correct
answer, slowly. A silently-wrong accelerator would show up as a subtly wrong
depth map, which is the worst possible failure mode for this project.

### Verification

**Unit testbench** (`src/tb/student_gemm_tb.sv`, run under XSim) drives the
registers exactly as the C driver does, against a TL-UL memory
(`src/tb/tlul_test_mem.sv`) that accepts 8 requests in flight, uses random
2–12 cycle latency, and **answers out of order**:

| Case | Result |
| --- | --- |
| `caps` matches parameters | ok |
| N=4 K=8 M=3 (smaller than a tile) | ok |
| N=16 K=64 M=8 (exactly one tile) | ok |
| N=20 K=64 M=6 (two tiles, partial second) | ok |
| N=17 K=128 M=4 (partial tile of one row) | ok |
| N=16 K=384 M=12 | ok |
| N=16 K=384 M=64 | ok |

**1544 output words checked, all bit-exact, 0 errors.**

**Regression of the software path.** The `[m][n]` layout change touched the
verified engine, so the host build was re-run against the same weights: the
output is **bit-identical** to the pre-change result (max abs diff 0.0), still
corr 0.99984 vs PyTorch. The deterministic kernel checksum is still `3ac57cd2`.

**Elaboration.** `rvlab_fpga_top` elaborates clean with the new blocks in
place; the RISC-V program builds (.bss 57.4 KB, still well inside 256 KB BRAM).

### Measured throughput, and what it should mean

From the largest testbench case (N=16, K=384, M=64): **393 216 MACs in 31 610
cycles = 12.4 MAC/cycle**, i.e. 78 % of the 16 MAC/cycle peak, against a
deliberately hostile memory model. At 50 MHz that is ≈ 620 MMAC/s versus a peak
of 800 MMAC/s.

| | CPU only | With accelerator |
| --- | --- | --- |
| MAC rate | ~8 MMAC/s (6 cycles/MAC @ 50 MHz) | ~620 MMAC/s measured in sim |
| GEMM time for 2.412 G MAC | ~290 s | **~4 s** |

That is roughly a **75× speed-up on the GEMM**. It is emphatically *not* a 75×
speed-up on the frame, and this is where honesty matters more than the headline:
once the GEMM drops to ~4 s, the frame is dominated by what is still on the
CPU — the 5.273 M-element requantisation and min/max passes, LayerNorm, GELU,
softmax, im2col copies and interpolation — plus the hard floor from §2.2, that
all ~25 MB of weights must cross the DDR3 bus at least once per frame. Expect
seconds per frame rather than minutes, but the exact number is **not known
until it runs on hardware**, and nothing here should be read as a measurement
of frame time.

### What has *not* been done

Per instruction, the accelerator has been implemented and unit-verified but
**not run in the full flow**: no synthesis, place-and-route or bitstream has
been rebuilt with it, no system-level RTL simulation has been run, and it has
not been on the board. The §6/§6b hardware numbers are all still from the
pre-accelerator bitstream. Two things will need checking on the next
synthesis run:

* **Timing.** `sys_clk` closed at +0.496 ns slack before. The MAC array is
  DSP-resident and should be easy at 50 MHz, but the added `tlul_socket_1n` /
  `tlul_socket_m1` layer sits in the bus path and has not been timed.
* **The §6b report review must be redone**, since it is a per-build task.

## 7. Files

Student-owned code, all under `src/sw/project/`:

| File | Role |
| --- | --- |
| `dav2.h` | numeric contract, tensor/weight types, op API |
| `dav2_ops.c` | GEMM, LayerNorm, GELU, add, im2col, conv, resize, arena |
| `dav2_engine.c` | ViT-S/14 encoder + DPT head |
| `dav2_blob.c` | weight-blob directory lookup |
| `dav2_mathf.c` | sqrtf/expf/erff/frexpf/ldexpf (no libm on target) |
| `dav2_selftest.c` | deterministic kernel checksum, host vs RISC-V |
| `dav2_accel.h/.c` | driver + start-up cross-check for the GEMM accelerator |
| `main.c` | SoC entry: DDR3 init, handshake, inference, ASCII depth map |
| `host/` | native build of the same engine for verification |
| `tools/dav2_numpy.py` | NumPy blueprint (validated against PyTorch) |
| `tools/export_dav2.py` | quantiser and blob writer |
| `tools/dav2_run_fpga.py` | JTAG loader + board run script |

Student-owned hardware (§6c):

| File | Role |
| --- | --- |
| `src/rtl/student/student_gemm.sv` | the int8 GEMM accelerator |
| `src/rtl/student/student.sv` | now splits the fast window and merges host ports |
| `src/design/reggen/student_gemm.hjson` | its register map + software contract |
| `src/tb/student_gemm_tb.sv` | unit testbench, checks every output word |
| `src/tb/tlul_test_mem.sv` | out-of-order TL-UL memory model for that testbench |

Additive changes outside the student area, three lines in total:
`flow/system_tb.py` gained a `sim_rtl_xsim_batch` task (the existing XSim tasks
open a GUI and there was no headless variant — QuestaSim has one, XSim did
not), and `flow/__init__.py` lists `student_gemm_tb` alongside the other
module-level testbenches. No third-party source was modified.

---

## 8. Running it

### Build and verify on the host

```bash
# 1. export weights + golden reference (needs the host python env)
python src/sw/project/tools/export_dav2.py --size 126 --image demo01.jpg \
       --out build/dav2 --common-dir <dir containing dav2_common.py>

# 2. build and run the engine natively, then compare against PyTorch
cd src/sw/project/host && make && make run
```

### Build for the SoC

```bash
export PATH=~/.local/opt/riscv/bin:$PATH
cp build/dav2/dav2_blob_config.h src/sw/project/
flow sw_project build
```

### Exercise the GEMM accelerator on its own

```bash
flow student_gemm_tb sim_rtl_xsim    # or sim_rtl_questa
```

Expect `student_gemm_tb PASSED (1544 words checked)`.

### Rebuild the bitstream with the accelerator

Not yet done — see §6c, "What has *not* been done".

```bash
flow rvlab_fpga_top syn pnr bitstream
# then redo the §6b report review: it is a per-build task
```

### Run on the Nexys Video board

```bash
flow rvlab_fpga_top program          # load the bitstream
python src/sw/project/tools/dav2_run_fpga.py \
    --elf build/sw_project/build/sw.elf \
    --blob build/dav2/dav2_weights.bin \
    --out build/dav2/fpga_depth.npy
```

The program initialises DDR3, prints `DAV2_WAITING_FOR_WEIGHTS`, and blocks.
The script then pushes the 24.87 MB blob to `0x8000_0000` over JTAG, verifies
the header in place, writes the handshake word at `0x8F00_0000`, and streams
the result back. Weight loading over JTAG is the slow step (minutes) — which
is why the weights are loaded once and inference runs afterwards.

---

## 9. Honest limitations

* **Not real time.** See §2.2. The GEMM accelerator of §6c removes the
  arithmetic bottleneck (~75× on that kernel in simulation), but the floor
  from §2.2 remains: all ~25 MB of weights must cross the DDR3 bus at least
  once per frame. Expect seconds, not milliseconds.
* **The accelerator has not been through the FPGA flow.** It is unit-verified
  in simulation only: no synthesis, place-and-route, bitstream, system-level
  simulation or board run. Its timing and resource cost are therefore
  estimates (16 DSPs, 16 BRAMs), not measurements.
* **Post-accelerator frame time is unmeasured.** The §6c estimate of "seconds"
  is arithmetic, not a reading. The non-GEMM tail — 5.273 M requantised
  elements, LayerNorm, GELU, softmax, im2col — was never profiled separately,
  and it is now the likely limiter.
* **Single fixed image.** The input is baked into the weight blob by the
  exporter. Feeding a live camera would mean writing the preprocessed frame to
  a DDR3 address and re-running, which the engine already supports structurally
  (the image is just another tensor in the blob).
* **The board run is scripted but not executed** — no board was attached during
  this work. Everything up to and including the bitstream is verified here; the
  hardware step is documented above.
* **demo05-type images** (small depth dynamic range) show higher *relative*
  error, though absolute error stays small and correlation stays above 0.995.

## 6d. First system simulation with the accelerator

`flow systb_project sim_rtl_xsim_batch` (no DDR3 model, ~17 min) run with the
current binary. This is the first time the accelerator RTL and its C driver
executed together on the CV32E40P.

The first attempt produced **1708 assertion failures** — 616 `DataKnown_A` and
392 `DataFlow_A` — in `student_i.host_merge_i` and propagating into
`xbar_main_i`. The software results were nevertheless correct.

**Root cause.** `t_q`, the drain-phase row index, is cleared only at `klast`
inside `ST_MAC`. On the `ST_DRAIN -> ST_MAC` transition it retains `n_rows_q`,
so for the whole next pass `wr_data = acc_q[n_rows_q]`. With a full tile
(`n_rows_q == NROWS == 16`) that wraps to `acc_q[0]` and is harmless; with a
**partial** tile it indexes an accumulator whose A-tile RAM was never written,
which is X. `a_data` was then loaded into every request unconditionally,
including reads, and the TL-UL FIFOs assert on X anywhere in the payload.

**Fix** (both in `student_gemm.sv`):

    a_data: sel_wr ? wr_data : 32'd0,   // never put accumulator data on a Get
    t_q <= '0;                          // clear the index when leaving ST_DRAIN

**Why the unit testbench missed it.** The bug needs a partial *first* tile and
a weight stream long enough that reads are still being issued after the first
drain. `N=4,K=8,M=3` has only 6 weight beats, all issued during the first pass;
every other partial tile in the suite was a *second* tile whose upper rows still
held defined data from the tile before. The software self-test's first job is
`N=8,K=32,M=16` — 128 beats, partial first tile. That case is now in both
testbenches, and was confirmed to fail (98 errors) against the unfixed RTL
before the fix was restored.

This was never a wrong-answer bug: `a_data` is a don't-care on a read, so every
computed value was correct throughout. It was a protocol-cleanliness defect that
would have pushed X into the crossbar FIFOs.

**Result after the fix:**

    hostio: DAV2_SELFTEST 3ac57cd2                    <- matches the host build
    hostio: GEMM accelerator: 16 rows/pass, K<=2048   <- caps via both sockets
    hostio: GEMM accelerator: self-test ok (20x64x6)  <- HW vs SW on the core
    hostio: Error: DDR not present.                   <- expected, no DDR3 model

0 assertion failures, 0 X events. The trailing `[FAIL] test_sw` is the
testbench reporting the program's nonzero exit, which is the absent DDR3 model,
not a software fault.

Newly established: the accelerator is reachable at `0x20010000` through the
added socket layer; the driver operates it correctly from C; and hardware and
software agree numerically on the real CPU. Still outstanding: synthesis,
timing closure, and any run with DDR3 or on the board.

**New files:** `src/tb/student_gemm_soc_tb.sv` (DUT behind a real
`tlul_socket_m1`), plus a simulation-only X monitor inside `student_gemm.sv`
guarded by `` `ifndef SYNTHESIS ``.

## 6e. FPGA build with the accelerator

`flow rvlab_fpga_top syn pnr bitstream`, all stages rebuilt with `--clean`.

**First attempt: timing failed.** WNS `-0.077 ns`, one failing endpoint of 49,977:

    Source:      core_i/student_i/gemm_i/reg_top_i/u_reg_if/outstanding_reg
    Destination: core_i/cpu_i/.../if_stage_i/instr_rdata_id_o_reg[17]
    Data Path Delay: 20.169 ns (logic 4.421 ns 21.9%, route 15.748 ns 78.1%)
    Logic Levels: 21

The GEMM register adapter's `outstanding` flag gates `a_ready`, and that
back-pressure reached the crossbar -- and from there the CPU's instruction
fetch -- combinationally through the `tlul_socket_1n` added in `student.sv`.
Only 4.4 ns is logic; the other 15.7 ns is routing across the die.

**Fix:** `HReqPass(1'b0)`, `HRspPass(1'b0)` on that socket instantiation, making
its host-side FIFOs registering rather than pass-through. This is a parameter on
our own instantiation -- no third-party source was modified. It costs one cycle
each way on fast-window accesses, which are register pokes.

**After the fix: timing met.**

    | Metric        | Before  | After   |
    | WNS (ns)      | -0.077  | +0.268  |
    | Failing paths | 1       | 0       |
    | WHS (ns)      | +0.021  | +0.037  |
    | Slice LUTs    | 17,152  | 17,049  |

**Resources** (XC7A200T): LUTs 17,049 / 133,800 (12.7%); Block RAM 84 / 365
(23.0%); DSP48E1 23 / 740 (3.1%). Of these the accelerator contributes 16 DSPs
(the MAC array) and 16 RAMB36 (the A-tile RAMs), which confirms both inferred
as hard macros rather than degrading into LUT logic.

### Report review (TASK.md repeating item)

* `bitstream/rvlab_fpga_top.io_report.txt` -- clean: no pins missing in either
  direction, all 152 pins matched against the reference.
* `syn/vivado.log`, `pnr/vivado.log`, `bitstream/vivado.log` -- **0 critical
  warnings** in all three.
* `pnr/rvlab_fpga_top.drc.txt` -- 88 violations, all severity **Warning**, none
  Error or Critical: 87 DSP pipelining advisories (DPIP-1/DPOP-1/DPOP-2) plus
  one disabled-check note. 72 belong to the accelerator, 15 to the CV32E40P's
  own multiplier (pre-existing, third-party).

  **Not acted on, deliberately.** These advise using the DSP48E1's internal
  MREG/PREG registers. Doing so means adding pipeline stages inside the MAC
  array and re-verifying the accumulator timing, and it buys Fmax we do not
  need: timing is met with 268 ps to spare at 50 MHz and DSP usage is 3.1%.
  The warnings are worth revisiting only if the clock is raised.

* Synthesis warnings from `student_gemm.sv` are benign and expected:
  `areq_q_reg[a_valid]` and `areq_q_reg[d_ready]` are reported as removed
  because both are driven separately in the output `always_comb`; the
  `tl_host_i` `d_param`/`d_size`/`d_source[7:3]` "no load" notices are the
  response fields the block legitimately ignores.

The bitstream is at `build/rvlab_fpga_top/bitstream/rvlab_fpga_top.bit`. It has
not been loaded onto a board.
