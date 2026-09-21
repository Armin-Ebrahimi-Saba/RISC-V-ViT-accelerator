# System architecture

How the pieces of this design fit together. Written for an engineer who has
not seen the project before; every specialised term is explained the first
time it is used.

The goal of the design is to run **Depth-Anything V2 Small** — a neural
network that takes a photograph and estimates how far away every pixel is —
on a small RISC-V processor inside an FPGA, with the heavy matrix arithmetic
offloaded to a custom hardware block.

---

## 1. The board and the chip

The hardware is a Digilent Nexys Video board carrying a Xilinx **Artix-7
XC7A200T** FPGA and a 512 MB DDR3 memory chip.

An *FPGA* (field-programmable gate array) is a chip whose logic is not fixed
at manufacture: a "bitstream" loaded at power-on wires its internal resources
into whatever circuit you describe. Everything in this document below the
DDR3 chip — the processor, the bus, the accelerator, the memory controller —
is built from that fabric. Nothing here is a fixed silicon CPU.

The design runs at **50 MHz**. That sounds slow next to a desktop CPU, and it
is: it is the clock at which a soft processor synthesised into this fabric
closes timing with margin. The whole SoC uses 14.8 % of the chip's logic,
36.9 % of its block RAM (mostly the accelerator's 256 kB tile) and 10.4 % of
its DSP multipliers (77 of 740); timing is met with 0.12 ns to spare.

---

## 2. The SoC at a glance

An *SoC* (system on chip) is the collection of processor, memories,
peripherals and interconnect that together make a computer. Here it is:

<figure>
<svg viewBox="0 0 860 470" role="img" aria-label="SoC block diagram: four bus masters on the left feed a crossbar, which routes to four device ports on the right; the accelerator is both a device and a master" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;font-size:12px">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>
<g fill="none" stroke="currentColor" stroke-width="1.4">
<!-- masters -->
<rect x="20" y="40" width="150" height="110" rx="6"/>
<rect x="20" y="180" width="150" height="60" rx="6"/>
<rect x="20" y="270" width="150" height="110" rx="6"/>
<!-- crossbar -->
<rect x="260" y="40" width="120" height="340" rx="6"/>
<!-- devices -->
<rect x="470" y="40" width="150" height="60" rx="6"/>
<rect x="470" y="120" width="150" height="60" rx="6"/>
<rect x="470" y="200" width="150" height="60" rx="6"/>
<rect x="470" y="280" width="150" height="100" rx="6"/>
<rect x="690" y="200" width="150" height="60" rx="6"/>
<!-- master -> xbar -->
<line x1="170" y1="70" x2="260" y2="70" marker-end="url(#ah)"/>
<line x1="170" y1="120" x2="260" y2="120" marker-end="url(#ah)"/>
<line x1="170" y1="210" x2="260" y2="210" marker-end="url(#ah)"/>
<!-- xbar -> devices -->
<line x1="380" y1="70" x2="470" y2="70" marker-end="url(#ah)"/>
<line x1="380" y1="150" x2="470" y2="150" marker-end="url(#ah)"/>
<line x1="380" y1="230" x2="470" y2="230" marker-end="url(#ah)"/>
<line x1="380" y1="310" x2="470" y2="310" marker-end="url(#ah)"/>
<!-- ddr -> chip -->
<line x1="620" y1="230" x2="690" y2="230" marker-end="url(#ah)"/>
<!-- accelerator as master drives the crossbar -->
<line x1="170" y1="325" x2="260" y2="325" marker-end="url(#ah)" stroke="#d9480f" stroke-width="2"/>
<!-- the device-side registers and the master-side engine are one block -->
<path d="M545 380 L545 420 L95 420 L95 380" stroke="#d9480f" stroke-width="2" stroke-dasharray="6 3"/>
</g>
<g fill="currentColor">
<text x="95" y="62" text-anchor="middle" font-weight="600">CV32E40P</text>
<text x="95" y="80" text-anchor="middle">RISC-V CPU</text>
<text x="95" y="103" text-anchor="middle" font-size="11">instruction port</text>
<text x="95" y="128" text-anchor="middle" font-size="11">data port</text>
<text x="95" y="203" text-anchor="middle" font-weight="600">Debug module</text>
<text x="95" y="222" text-anchor="middle" font-size="11">JTAG → system bus</text>
<text x="95" y="295" text-anchor="middle" font-weight="600">GEMM accelerator</text>
<text x="95" y="313" text-anchor="middle" font-size="11">(as a bus master)</text>
<text x="95" y="345" text-anchor="middle" font-size="11">reads A and W,</text>
<text x="95" y="360" text-anchor="middle" font-size="11">writes C</text>
<text x="320" y="200" text-anchor="middle" font-weight="600">TL-UL</text>
<text x="320" y="218" text-anchor="middle" font-weight="600">crossbar</text>
<text x="320" y="245" text-anchor="middle" font-size="11">4 masters</text>
<text x="320" y="260" text-anchor="middle" font-size="11">4 devices</text>
<text x="545" y="65" text-anchor="middle" font-weight="600">BRAM</text>
<text x="545" y="83" text-anchor="middle" font-size="11">256 kB on-chip, program</text>
<text x="545" y="145" text-anchor="middle" font-weight="600">Peripherals</text>
<text x="545" y="163" text-anchor="middle" font-size="11">timer, DDR3 control regs</text>
<text x="545" y="225" text-anchor="middle" font-weight="600">DDR3 path</text>
<text x="545" y="243" text-anchor="middle" font-size="11">cache → controller</text>
<text x="545" y="305" text-anchor="middle" font-weight="600">Student device</text>
<text x="545" y="323" text-anchor="middle" font-size="11">GEMM accelerator regs</text>
<text x="545" y="341" text-anchor="middle" font-size="11">DMA regs</text>
<text x="545" y="365" text-anchor="middle" font-size="11">(as a bus device)</text>
<text x="765" y="225" text-anchor="middle" font-weight="600">DDR3 chip</text>
<text x="765" y="243" text-anchor="middle" font-size="11">512 MB, off-chip</text>
<text x="320" y="440" text-anchor="middle" font-size="11" fill="#d9480f">dashed: the same block — programmed as a device, fetching as a master</text>
</g>
</svg>
<figcaption>Everything talks through one crossbar. The accelerator appears twice: as a <em>device</em> the CPU programs through registers, and as a <em>master</em> that fetches its own operands from DDR3 — the orange arrow and dashed link. That double role is why bus bugs affected it and the CPU differently.</figcaption>
</figure>

**Bus masters** — the things that start transactions:

- **CV32E40P** — the RISC-V processor. RV32IMC: 32-bit, integer, multiply,
  compressed instructions. No floating-point unit, so every `float` in the C
  code is emulated in software. It has two bus ports, one for fetching
  instructions and one for data.
- **Debug module** — reachable over JTAG (a serial debug link from the host
  PC). It can halt the CPU and read its registers, and it can also read and
  write memory directly on the bus without the CPU's involvement. That second
  ability turned out to be the most valuable debugging tool in the project.
- **GEMM accelerator** — the custom hardware block, which fetches its own
  operands with up to eight reads in flight. It has two job types: a matrix
  product, and the requantisation of a product's int32 result to int16 (§4).

**Bus devices** — the things that respond:

- **BRAM** — 256 kB of on-chip memory. Holds the program, its stack, and a
  40 kB scratch buffer the C kernels use for anything reused many times
  inside one operator (an attention head's q and k, LayerNorm's per-channel
  tables). Single-cycle from the CPU's point of view, unlike DDR3.
- **Peripherals** — a timer and the DDR3 controller's status registers.
- **DDR3 path** — the cache and controller in front of the external memory.
  Holds the 25 MB of neural-network weights and all the working data.
- **Student device** — the register interface of the accelerator and a DMA
  block.

### The bus: TL-UL

The interconnect is **TileLink Uncached Lightweight (TL-UL)**, a simple
request/response bus from the OpenTitan project. Each transaction is a
*request* on the "A channel" (address, opcode, data) answered by exactly one
*response* on the "D channel". Each channel has a `valid` from the sender and
a `ready` from the receiver; a transfer happens only in a cycle where both are
high. That two-signal handshake is the single most important rule in this
design, and two of the three bugs found were violations of it.

---

## 3. Memory map

<figure>
<svg viewBox="0 0 700 300" role="img" aria-label="Memory map: BRAM at 0x0000_0000, peripherals around 0x1F00_0000, accelerator registers at 0x2001_0000, and DDR3 from 0x8000_0000 holding the weight blob, then the image-in and depth-out slots, then the activation arena" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;font-size:12px">
<g fill="none" stroke="currentColor" stroke-width="1.4">
<rect x="20" y="30" width="660" height="40" rx="4"/>
<rect x="20" y="90" width="660" height="40" rx="4"/>
<rect x="20" y="150" width="660" height="40" rx="4"/>
<rect x="20" y="210" width="240" height="70" rx="4"/>
<rect x="265" y="210" width="170" height="70" rx="4"/>
<rect x="440" y="210" width="240" height="70" rx="4"/>
</g>
<g fill="currentColor">
<text x="30" y="55" font-family="ui-monospace,monospace">0x0000_0000</text>
<text x="180" y="55">BRAM — program, stack, hostio console ring (256 kB)</text>
<text x="30" y="115" font-family="ui-monospace,monospace">0x1F00_0000</text>
<text x="180" y="115">peripherals — timer, DDR3 control + watchdog registers</text>
<text x="30" y="175" font-family="ui-monospace,monospace">0x2001_0000</text>
<text x="180" y="175">GEMM accelerator registers (student device)</text>
<text x="30" y="235" font-family="ui-monospace,monospace">0x8000_0000</text>
<text x="30" y="255" font-weight="600">weight blob, 24.87 MB</text>
<text x="30" y="272" font-size="11">int8 weights + directory, over JTAG once</text>
<text x="275" y="228" font-family="ui-monospace,monospace" font-size="11">0x81E0_0000</text>
<text x="275" y="243" font-size="11">image in, 95 kB</text>
<text x="275" y="260" font-family="ui-monospace,monospace" font-size="11">0x81F1_0000</text>
<text x="275" y="275" font-size="11">depth map out, 63 kB</text>
<text x="450" y="235" font-family="ui-monospace,monospace">0x8200_0000</text>
<text x="450" y="255" font-weight="600">activation arena, 64 MB</text>
<text x="450" y="272" font-size="11">bump allocator for intermediate tensors</text>
</g>
</svg>
<figcaption>The blob and the arena are 32 MB apart in DDR3. That distance matters: the cache in front of DDR3 is indexed by address bits [13:5], so both regions map onto the same 512 cache lines and constantly evict each other.</figcaption>
</figure>

Four regions in DDR3 do all the work:

- **The weight blob** at `0x8000_0000`. A single file produced by
  `export_dav2.py` containing every tensor the network needs — weights,
  biases, scales — plus a directory at the front so the C code can find them
  by name. It is loaded once over JTAG (about a minute).
- **The input image** at `0x81E0_0000`, in the gap between blob and arena:
  126×126×3 int16 pixels then one float scale, 95 kB. It is *not* part of the
  blob, so a new picture is a quarter-second transfer rather than a new
  minute-long weight load. The program serves frames: it waits for the host
  to write an image here and raise a flag, runs it, publishes the result,
  and waits for the next.
- **The depth map** at `0x81F1_0000`, also in the gap: 126×126 float32,
  63 kB. The host reads it straight off the bus over JTAG in 0.7 s (it used
  to be printed as hex text through the console, ~30 s).
- **The activation arena** at `0x8200_0000`. Working memory for the tensors
  the network produces as it runs. A *bump allocator* hands out space by
  advancing a pointer and can only free it all at once, which suits a
  network whose memory needs are known and layered.

---

## 4. The GEMM accelerator

*GEMM* means general matrix multiply. Every matrix multiplication in the
network — and a transformer is almost nothing but matrix multiplications —
funnels through one C function, `dav2_qgemm()`. That function hands the work
to the accelerator.

The block computes `C = A × Wᵀ` where `A` is an int16 activation matrix
(N rows × K), `W` is an int8 weight matrix (M rows × K), and `C` is the int32
result (M × N). *int8/int16/int32* are 8-, 16- and 32-bit integers; the
network was quantised so all the arithmetic is integer, because the CPU has
no floating-point hardware.

<figure>
<svg viewBox="0 0 860 400" role="img" aria-label="Accelerator dataflow: A tile of 64 rows is loaded once into on-chip memory, then weight rows stream through 64 multiply-accumulate units in parallel, and each completed weight row's 64 results are drained to DDR3 together with their max and min" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;font-size:12px">
<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>
<g fill="none" stroke="currentColor" stroke-width="1.4">
<rect x="20" y="40" width="130" height="320" rx="6"/>
<rect x="240" y="40" width="150" height="120" rx="6"/>
<rect x="240" y="220" width="150" height="70" rx="6"/>
<rect x="480" y="40" width="200" height="320" rx="6"/>
<rect x="500" y="90" width="160" height="26" rx="3"/>
<rect x="500" y="124" width="160" height="26" rx="3"/>
<rect x="500" y="158" width="160" height="26" rx="3"/>
<rect x="500" y="290" width="160" height="26" rx="3"/>
<rect x="730" y="150" width="110" height="100" rx="6"/>
<line x1="150" y1="100" x2="240" y2="100" marker-end="url(#ah2)"/>
<line x1="150" y1="255" x2="240" y2="255" marker-end="url(#ah2)"/>
<line x1="390" y1="100" x2="480" y2="100" marker-end="url(#ah2)"/>
<line x1="390" y1="255" x2="480" y2="255" marker-end="url(#ah2)" stroke-dasharray="5 3"/>
<line x1="680" y1="200" x2="730" y2="200" marker-end="url(#ah2)"/>
<path d="M785 250 L785 385 L180 385 L180 340 L152 340" marker-end="url(#ah2)"/>
</g>
<g fill="currentColor">
<text x="85" y="65" text-anchor="middle" font-weight="600">DDR3</text>
<text x="85" y="95" text-anchor="middle" font-size="11">A: activations</text>
<text x="85" y="110" text-anchor="middle" font-size="11">int16, N×K</text>
<text x="85" y="250" text-anchor="middle" font-size="11">W: weights</text>
<text x="85" y="265" text-anchor="middle" font-size="11">int8, M×K</text>
<text x="85" y="340" text-anchor="middle" font-size="11">C: results, int32</text>
<text x="195" y="90" text-anchor="middle" font-size="11">load once</text>
<text x="195" y="245" text-anchor="middle" font-size="11">stream</text>
<text x="315" y="65" text-anchor="middle" font-weight="600">A tile</text>
<text x="315" y="85" text-anchor="middle" font-size="11">64 rows × K</text>
<text x="315" y="103" text-anchor="middle" font-size="11">on-chip BRAM</text>
<text x="315" y="130" text-anchor="middle" font-size="11">read every cycle</text>
<text x="315" y="145" text-anchor="middle" font-size="11">for each W row</text>
<text x="315" y="245" text-anchor="middle" font-weight="600">W beat</text>
<text x="315" y="265" text-anchor="middle" font-size="11">4 × int8 per word</text>
<text x="435" y="245" text-anchor="middle" font-size="11">broadcast</text>
<text x="580" y="65" text-anchor="middle" font-weight="600">64 MAC units</text>
<text x="580" y="107" text-anchor="middle" font-size="11">row 0: acc += A[0][k]·W[m][k]</text>
<text x="580" y="141" text-anchor="middle" font-size="11">row 1: acc += A[1][k]·W[m][k]</text>
<text x="580" y="175" text-anchor="middle" font-size="11">row 2: acc += A[2][k]·W[m][k]</text>
<text x="580" y="230" text-anchor="middle">⋮</text>
<text x="580" y="307" text-anchor="middle" font-size="11">row 63</text>
<text x="580" y="345" text-anchor="middle" font-size="11">all 64 in parallel, one k per cycle</text>
<text x="785" y="175" text-anchor="middle" font-weight="600">drain</text>
<text x="785" y="200" text-anchor="middle" font-size="11">64 results</text>
<text x="785" y="215" text-anchor="middle" font-size="11">per W row</text>
<text x="785" y="230" text-anchor="middle" font-size="11">→ C column</text>
<text x="785" y="244" text-anchor="middle" font-size="11">+ {max, min}</text>
<text x="480" y="378" text-anchor="middle" font-size="11">write C[m][0..63] (and {max,min}) back to DDR3</text>
</g>
</svg>
<figcaption>Reuse is the whole idea: 64 rows of A are loaded once and held on-chip, then every weight row streams past all 64 at once. Each int8 weight is multiplied against 64 activations the cycle it arrives, so the weight — the dominant memory traffic — crosses the bus once per tile: twice for the encoder's 82 tokens.</figcaption>
</figure>

The execution of one *GEMM job* (one 64-row tile of A against all of W):

1. **Load** the A tile — up to 64 rows × K int16 — from DDR3 into on-chip
   BRAM (256 kB, 64 block RAMs). Rows may lie `A_STRIDE` bytes apart, so
   the tile can be a column slice of a wider matrix.
2. **Stream** W: each 32-bit word carries four int8 weights. As each arrives
   it is broadcast to all 64 MAC units, each of which multiplies it against
   its own row of A and accumulates. W rows may also be strided.
3. **Drain**: when a weight row `m` is finished, the accumulated results
   are written to DDR3 as column `m` of C. With `S_ADDR` set, the row's max
   and min go out after them — the range the requantisation needs, so the
   CPU never reads C back for it.
4. Repeat 2–3 for every row of W. Then the CPU starts the next tile.

A matrix with N=82 rows needs two jobs: one full tile and one of 18 rows.
A reduction longer than the tile RAM (K = 3456 in the 384-channel 3×3
convolutions, against KMAX = 2048) is run as two column-slice jobs whose
partial sums the CPU adds.

The block reads with **eight requests in flight** through a reorder buffer
indexed by the bus source ID, which is what brought it from 8.3 to 3.2
cycles per beat. It measures 3.2 because a 32-byte cache line is filled from
DRAM once per eight beats; the bypassed prefetcher (§5) would hide that.

### The requantisation job

Every GEMM's int32 result has to become int16 activations again:
`out[n][m] = sat14((C[m][n]·mult[m] + 2^(s−1)) >> s + bias[m])`, with a
multiplier, shift and bias per weight row chosen by the CPU from the row
ranges above. That used to be ~70 CPU cycles per element — a third of the
frame — and is now the block's second job type (`CTRL.requant`):

1. Read the per-row `{mult, shift, bias}` table (`P_ADDR`) into a small
   parameter RAM.
2. Read a chunk of C — up to 1024 rows × 64 columns — into the A-tile RAM
   **transposed**: RAM row = n, word = m. This is what turns the m-major
   result into n-major output for free.
3. Stream out: one element per cycle through a seven-stage pipeline
   (multiply in DSPs, round, shift, add bias, saturate), two int16 packed
   per word, into a 16-entry FIFO that the write engine drains.

The arithmetic matches the C code to the bit; `student_gemm_tb` checks it
against a bit-level model and the board against the host build.

The block has a register interface the CPU programs (addresses and strides
of A, W and C; K, M and the tile row count; `S_ADDR`, `P_ADDR`; a control
word with start and requant bits; status) and four debug registers that
expose internal counters. Peak is 64 multiply-accumulates per cycle; the
block is memory-bound, not compute-bound, so the tile width buys fewer
passes over the weights rather than more arithmetic per pass.

### Retry timer

The block also carries a **retry timer**: every outstanding read's address
is kept in its reorder-buffer slot, and if the oldest one gets no response
within 2048 cycles of bus silence it is re-issued. This exists because the
platform's cache can drop a response (§5); the accelerator could survive
that, the CPU could not. With eight reads in flight the board reports zero
retries per frame.

---

## 5. The DDR3 path — where the bugs lived

Between the crossbar and the memory chip sit four blocks. All three defects
fixed in this project were in this path, so it is worth seeing in full.

<figure>
<svg viewBox="0 0 900 420" role="img" aria-label="The DDR3 path: crossbar port feeds a request mux, then a write-back cache, then (bypassed) prefetcher, then clock-domain crossing FIFO, block manager, and the UberDDR3 controller and PHY driving the DDR3 chip. Three defects are marked." style="max-width:100%;height:auto;font-family:system-ui,sans-serif;font-size:12px">
<defs><marker id="ah3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>
<g fill="none" stroke="currentColor" stroke-width="1.4">
<rect x="20" y="120" width="90" height="60" rx="6"/>
<rect x="150" y="120" width="100" height="60" rx="6"/>
<rect x="290" y="100" width="130" height="100" rx="6"/>
<rect x="460" y="120" width="100" height="60" rx="6" stroke-dasharray="6 3"/>
<rect x="600" y="120" width="80" height="60" rx="6"/>
<rect x="720" y="120" width="80" height="60" rx="6"/>
<rect x="830" y="100" width="60" height="100" rx="6"/>
<line x1="110" y1="150" x2="150" y2="150" marker-end="url(#ah3)"/>
<line x1="250" y1="150" x2="290" y2="150" marker-end="url(#ah3)"/>
<line x1="420" y1="150" x2="460" y2="150" marker-end="url(#ah3)"/>
<line x1="560" y1="150" x2="600" y2="150" marker-end="url(#ah3)"/>
<line x1="680" y1="150" x2="720" y2="150" marker-end="url(#ah3)"/>
<line x1="800" y1="150" x2="830" y2="150" marker-end="url(#ah3)"/>
<!-- bypass around prefetch -->
<path d="M420 130 Q 510 70 600 130" stroke-dasharray="6 3"/>
<!-- clock boundary -->
<line x1="590" y1="60" x2="590" y2="225" stroke-dasharray="3 4"/>
<!-- defect markers -->
<circle cx="200" cy="105" r="11" fill="#d9480f" stroke="none"/>
<circle cx="355" cy="85" r="11" fill="#d9480f" stroke="none"/>
<circle cx="510" cy="105" r="11" fill="#d9480f" stroke="none"/>
</g>
<g fill="currentColor">
<text x="65" y="145" text-anchor="middle" font-size="11">crossbar</text>
<text x="65" y="160" text-anchor="middle" font-size="11">DDR port</text>
<text x="200" y="145" text-anchor="middle" font-size="11">request</text>
<text x="200" y="160" text-anchor="middle" font-size="11">mux</text>
<text x="355" y="130" text-anchor="middle" font-weight="600">cache</text>
<text x="355" y="150" text-anchor="middle" font-size="11">16 kB, write-back</text>
<text x="355" y="165" text-anchor="middle" font-size="11">direct-mapped</text>
<text x="355" y="185" text-anchor="middle" font-size="11">512 × 32-byte lines</text>
<text x="510" y="145" text-anchor="middle" font-size="11">prefetcher</text>
<text x="510" y="78" text-anchor="middle" font-size="11">bypass (in use)</text>
<text x="510" y="160" text-anchor="middle" font-size="11">(bypassed)</text>
<text x="640" y="145" text-anchor="middle" font-size="11">CDC</text>
<text x="640" y="160" text-anchor="middle" font-size="11">FIFO</text>
<text x="760" y="145" text-anchor="middle" font-size="11">block</text>
<text x="760" y="160" text-anchor="middle" font-size="11">manager</text>
<text x="860" y="140" text-anchor="middle" font-size="11">UberDDR3</text>
<text x="860" y="155" text-anchor="middle" font-size="11">ctrl</text>
<text x="860" y="170" text-anchor="middle" font-size="11">+ PHY</text>
<text x="590" y="50" text-anchor="middle" font-size="11">50 MHz | 100 MHz</text>
<text x="200" y="109" text-anchor="middle" font-size="11" fill="#fff" font-weight="700">1</text>
<text x="355" y="89" text-anchor="middle" font-size="11" fill="#fff" font-weight="700">2</text>
<text x="510" y="109" text-anchor="middle" font-size="11" fill="#fff" font-weight="700">3</text>
<text x="20" y="260" font-weight="600" fill="#d9480f">1</text>
<text x="40" y="260" font-size="12">Request mux took its "ready" from the wrong block, so requests were accepted by nobody. CPU hung, unhaltable.</text>
<text x="20" y="290" font-weight="600" fill="#d9480f">2</text>
<text x="40" y="290" font-size="12">Cache wrote back a line's stale contents when that line had been written the cycle before. One word lost per tile.</text>
<text x="20" y="320" font-weight="600" fill="#d9480f">3</text>
<text x="40" y="320" font-size="12">Prefetcher returned another address's data when two regions collided in the cache. Bypassed rather than repaired.</text>
<text x="20" y="360" font-size="12">Also in the cache: it ignores the bus "ready" on responses, so a busy receiver can lose one. Covered by the accelerator's retry timer.</text>
</g>
</svg>
<figcaption>The dashed vertical line is a clock-domain crossing: the controller runs at 100 MHz, the rest at 50 MHz. All three fixed defects (orange) sit on the 50 MHz side, in platform RTL rather than the accelerator.</figcaption>
</figure>

Terms used above:

- **Cache** — a small fast memory that keeps copies of recently used data so
  the next access is served locally instead of from slow DDR3.
- **Direct-mapped** — each address can live in exactly one cache line,
  chosen by address bits [13:5]. Two addresses that agree in those bits
  *collide*: bringing one in throws the other out.
- **Write-back** — writes go into the cache and are marked *dirty*; the
  updated line is only written to DDR3 later, when it is thrown out
  (*evicted*) to make room. This is efficient but means an
  "acknowledged" write is not yet in memory.
- **Prefetcher** — guesses which lines will be wanted next and fetches them
  early. Ours returns wrong data under collisions and is switched off.
- **CDC FIFO** — a queue that safely passes data between two clock domains.
- **Block manager** — converts the cache's 32-byte line requests into the
  controller's Wishbone bus transactions and tracks up to 16 in flight.
- **UberDDR3** — a third-party open-source DDR3 controller (formally
  verified upstream; no defect was found in it).

The cache's failure mode deserves one more sentence, because it was the last
and hardest to find. When the line being evicted was written on the
*immediately preceding* cycle, the cache read the line's contents from its
RAM before that write had landed there, and sent the old contents to DDR3.
It only ever happened on two back-to-back misses to the same cache line —
write one, evict it for the next — which is exactly what the accelerator's
last write of a tile followed by the next tile's first access produces.

---

## 6. How the pieces are exercised

| Level | What runs | Where the DDR3 is | Time |
|---|---|---|---|
| Host reference | the C engine natively on the PC | `malloc` | ~4 s |
| Module testbench | one RTL block against a behavioural memory (`student_gemm_tb`: GEMM shapes, strided operands, row statistics, the requantisation job against a bit-level model; `student_gemm_ddrpath_tb`: the same block through the real cache with eight reads in flight) | `ddr3_blk_model.sv` | seconds–minutes |
| System testbench | the whole SoC, no DDR3 | none | ~17 min for 4 M cycles |
| System + behavioural DDR3 | the whole SoC, cache and block manager real | `ddr3_blk_model.sv` | same rate |
| Full DDR3 simulation | everything including the JEDEC chip model | real | ~600× slower; avoid |
| Board | the bitstream on the FPGA | real | 14.2 s per frame |

The host reference is the *oracle*: the same C source compiled for the PC.
Because every kernel is integer and the float bookkeeping is IEEE single
precision on both sides, the board's output is required to be bit-identical
to it — and is, on every image tried. That equality was also the regression
test for the speed-up work: every change was checked against the previous
host output byte for byte (`PERFORMANCE.md`).

The program also measures the board itself: a per-operator cycle profile
printed after every frame, and a microbenchmark at boot giving cycles per
instruction, per load and per taken branch on this core. Both exist because
guessing where the time went was wrong twice.

---

## 7. Where things are

| Path | Contents |
|---|---|
| `src/rtl/student/student_gemm.sv` | the accelerator |
| `src/rtl/student/student_tl_watch.sv` | stalled-transaction watchdog register |
| `src/rtl/student/student_tl_rsp_hold.sv` | response skid buffer (currently bypassed) |
| `src/rtl/ddr3/rvlab_tlul_ddr.sv` | the DDR3 path top; request mux fix; prefetch bypass |
| `src/rtl/ddr3/rvlab_ddr_block_cache.sv` | the cache; write-back fix |
| `src/sw/project/` | the C inference engine; `dav2_accel.c` is the accelerator driver, `dav2_ops.c` the kernels, `main.c` the frame server, profiler hooks and microbenchmark |
| `src/sw/project/tools/` | exporter, board runner, image preprocessing (`dav2_image.py`) |
| `src/tb/` | testbenches |
| `docs/DATAFLOW.md` | timing diagrams of the bus and one inference; the complete numbered flow of a frame |
| `docs/DEBUGGING.md` | how every defect was found |
| `docs/LESSONS.md` | portable rules for the next project |
| `docs/HANDOFF.md` | current state and open items |
| `docs/PERFORMANCE.md` | the speed-up from 93.6 s to 14.2 s: profile, each step, what is left |
