# How the model runs on the FPGA

A plain-language walkthrough. No prior knowledge assumed.
For the engineering detail and measurements, see `report.md`.

---

## 0. The one-paragraph version

We are running **Depth-Anything V2 Small** — a neural network that turns a
photo into a depth map (how far away each pixel is) — on an FPGA board. There
is no operating system, no Python, no PyTorch on the board. A small CPU that we
built into the FPGA runs a plain C program that does the maths itself. Because
that CPU is slow at multiplication-heavy work, we also built a small piece of
custom hardware next to it that does the multiplications in bulk.

---

## 1. The role of the CPU

The chip on the board is an **Artix-7 FPGA**. An FPGA is a blank slate of
programmable logic — it is not a processor until you describe one and load that
description onto it.

What we load includes a CPU called **CV32E40P**. It is a small 32-bit RISC-V
core running at **50 MHz**. For context, a laptop CPU runs at ~3000 MHz, has
many cores, and can do many operations at once. This one:

- runs at 50 MHz,
- has **one** core, executing **one** instruction at a time, in order,
- has **no floating-point unit** — decimal arithmetic must be emulated in
  software, which is slow,
- has **no SIMD** — no instruction that multiplies 8 numbers at once.

The CPU is the **conductor**, not the workhorse. It:

1. walks through the network layer by layer, in the right order,
2. fetches the right weights for each layer,
3. does the "shaping" work — rearranging data, normalisation, activation
   functions,
4. hands the heavy multiplication to the accelerator and waits for it.

## 2. The GEMM accelerator

**GEMM** = **GE**neral **M**atrix **M**ultiply. It is the thing neural networks
spend nearly all their time doing: multiply a big grid of numbers by another
big grid, and add up the results.

One frame of this model needs **2.4 billion multiply-and-add operations**. The
CPU can do roughly one per cycle at 50 MHz, so on its own that is about
**5 minutes per image**.

So we built custom hardware — `student_gemm`, described in
`src/rtl/student/student_gemm.sv` — and loaded it onto the FPGA alongside the
CPU. It contains **16 multipliers side by side**, all working in the same
cycle. It also fetches its own data from memory instead of being spoon-fed by
the CPU.

Result: roughly **75x faster** on the multiplication part — about 4 seconds
instead of 290.

Two things worth understanding:

- **It is not a "neural network chip."** It knows nothing about layers or
  attention. It multiplies matrices. That is deliberately all it does.
- **The CPU is still needed.** Once the accelerator lands the multiplication
  work, everything *else* — normalisation, activation functions, data
  rearrangement — becomes the slow part. Speeding up one stage moves the
  bottleneck; it does not remove it.

**How the CPU talks to it:** the accelerator appears to the CPU as a handful of
memory addresses starting at `0x20010000`. The CPU writes "here is matrix A,
here is matrix B, put the answer here, go", then polls an address until a
"busy" bit clears. This is the normal way software drives hardware.

## 3. DRAM (the DDR3 memory)

The model's weights are **25 MB**. That has to live somewhere.

The board has **512 MB of DDR3 DRAM** — a separate memory chip next to the
FPGA. It is large but comparatively slow and far away.

In our memory map:

| Address | Contents |
|---|---|
| `0x80000000` | the 25 MB weight blob |
| `0x82000000` | 64 MB scratch space for intermediate results |
| `0x8F000000` | a single "the weights are loaded, you may start" flag |

**This is the real speed limit.** Every frame, all 25 MB of weights must travel
from DDR3 into the chip. Even at full memory bandwidth that takes time, and no
amount of extra multipliers changes it. This is why the accelerator was
designed to read memory in long straight runs rather than jumping around — the
memory system is much faster at sequential access.

## 4. Where the model is stored

Two different memories, for two different things:

- **The weights (25 MB)** live in **DDR3**, because nothing else is big enough.
- **The program (~80 KB)** lives in **BRAM**, on-chip.

Getting 25 MB onto the board is itself a problem: the board has no disk, no
network, no filesystem. So the weights are pushed in from a PC over the
**JTAG** debug cable — the same cable used to program the FPGA. This takes
several minutes, so it is done **once**, and the program then processes images
without reloading.

The handshake works like this: the program boots, prints
`DAV2_WAITING_FOR_WEIGHTS`, and spins on that flag word at `0x8F000000`. The PC
pushes the weights, then writes a magic value to the flag. The program sees it
and begins.

## 5. BRAM (the on-chip memory)

**BRAM** = **B**lock **RAM**: small, fast memory blocks built into the FPGA
itself. Fast and close, but there is little of it.

We use it for two things:

**The program memory — 256 KB.** The entire C program must fit here:

| Section | Size | What it is |
|---|---|---|
| `.text` | 19,770 B | the instructions |
| `.data` | 2,136 B | initialised variables |
| `.bss` | 58,828 B | working buffers |

About 80 KB of 256 KB. This budget is why there is no inference framework on
the device — see §8.

**Inside the accelerator.** It holds a working tile of 16 rows of data in 16
private BRAMs, so its 16 multipliers can each read their own number in the same
cycle. This is the core trick: DDR3 could never feed 16 multipliers directly,
so data is pulled in once, then reused many times from BRAM.

The pattern — *slow big memory, fast small memory, move data between them
carefully* — is the central problem in accelerator design.

## 6. The software side

Roughly 2,000 lines of C, all hand-written, in `src/sw/project/`:

| File | Role |
|---|---|
| `main.c` | boot, self-tests, DDR3 init, the weight handshake, print the result |
| `dav2_engine.c` | the network itself — which layer follows which |
| `dav2_ops.c` | the mathematical kernels |
| `dav2_accel.c` | the driver that operates the accelerator |
| `dav2_blob.c` | reads weights out of the 25 MB blob |
| `dav2_mathf.c` | `sqrt`, `exp`, `erf` — hand-written, because the build does not link a maths library |

Three ideas hold it together:

**Quantisation.** The original model uses 32-bit floating-point numbers. Our
CPU has no floating-point hardware. So weights are converted to **8-bit
integers** and intermediate values to **16-bit integers**. This makes the model
4x smaller and lets the accelerator use cheap integer multipliers. Accuracy
cost is small — the output correlates **0.99984** with the PyTorch original.

**One choke point.** Every matrix multiply in the network — and there are many
different layers — is routed through a single function, `dav2_qgemm()`. That is
why adding the accelerator required changing essentially one line:

```c
if (!dav2_accel_qgemm(a, wt, acc))          /* try the hardware */
    dav2_qgemm_cpu(a->v, wt->w, acc, N, K, M);   /* else do it in software */
```

**The software path never leaves.** If the accelerator cannot handle a shape,
software does it. At boot, `dav2_accel_check()` runs one small matrix multiply
*both* ways and compares. If they disagree, the accelerator is switched off and
the model still produces a correct — merely slower — answer.

There is no `malloc`. Memory is handed out by bumping a pointer through the
64 MB scratch region and released in bulk.

## 7. Tools used

Nothing exotic; all of it standard.

| Stage | Tool |
|---|---|
| Original model, weight export | **PyTorch**, **NumPy** (on the PC, offline) |
| Compile C for the board | **riscv-none-elf-gcc** |
| Compile the same C for the PC | **gcc** — for testing |
| Turn our hardware description into a chip layout | **Vivado** (Xilinx) |
| Simulate the hardware before building it | **XSim** |
| Load program and weights onto the board | **OpenOCD**, over JTAG |
| Run all of the above in order | **PyDesignFlow** (`flow`) |

The trick that makes development bearable: **the same C sources compile for
both the board and the PC**. A bug in the maths shows up on a laptop in
seconds, instead of after a minutes-long load onto hardware.

## 8. Libraries and frameworks used

**On the board: none.**

No TensorFlow Lite, no ONNX Runtime, no CMSIS-NN, no IREE, no PyTorch. Not even
the C standard maths library. The program links with `-nostdlib` and includes
only `stdint.h`, `stdio.h`, `string.h` and the board's own header.

To be precise about *why*: this is the project's choice, not a missing
toolchain. `riscv-none-elf-g++` 15.2.0 is installed, and `libstdc++.a`,
`libc.a` and `libm.a` all exist for the exact `rv32imc/ilp32` multilib the
board uses; libsys even ships a `malloc` (`src/sw/sys/malloc.c`). The build
declines all of it.

The reasons:

1. **No room.** 256 KB total for everything.
2. **Nothing to lower onto.** Frameworks earn their speed by using SIMD
   instructions. This CPU has none.
3. **They cannot reach our accelerator.** `student_gemm` is hardware we
   invented; no existing framework knows it exists. Teaching one would be more
   work than the 2,000 lines of C.
4. **The remaining reasons are about value, not feasibility.** A framework
   would have to be ported onto a `-nostdlib` build and would then run the
   portable reference kernels, which are slower than the hand-written ones.

**Off the board**, standard tools do the heavy lifting: PyTorch and NumPy for
the reference model and the weight export.

---

## Current status — what has actually been proven

Being precise about this, because "it works" would be overstating it:

| Claim | Status |
|---|---|
| The maths is correct | **Verified** on a PC vs PyTorch (correlation 0.99984) |
| The maths is correct *on the CPU* | **Verified** in simulation (checksum matched the PC exactly) |
| The accelerator computes correctly | **Verified** in simulation — 1,544 values, all exact |
| Accelerator + CPU together | **In progress** — being simulated now |
| Runs on the physical board | **Not yet** — the FPGA layout has not been rebuilt |
| A real depth map from real hardware | **Not yet** |

The pieces are each verified. They have not yet run together on real hardware.
