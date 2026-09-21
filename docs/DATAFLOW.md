# How data flows — timing diagrams

Companion to `ARCHITECTURE.md`. Where that document shows *what* the blocks
are, this one shows *when* things happen: the bus handshake cycle by cycle,
one inference from weights to depth map, and the exact cycle sequence of the
bug that was hardest to find.

A *timing diagram* shows signals as horizontal traces against time running
left to right, with vertical divisions marking clock cycles. A signal drawn
high is 1, low is 0. Reading one is mostly a matter of finding the cycle
where two things are true at once.

---

## 1. The TL-UL handshake

Every transfer on the bus, in either direction, obeys one rule: **data moves
in a cycle where `valid` and `ready` are both high.** The sender raises
`valid` and holds its data; the receiver raises `ready` when it can take it.
Neither may assume the other. In particular, once `valid` is up the sender
must keep the data stable until the transfer happens — it may not withdraw or
change it.

![Timing diagram of a TL-UL request and response. The request handshakes in cycle 3 when a_valid and a_ready are both high; the response handshakes in cycle 6 when d_valid and d_ready are both high. In cycle 5, d_valid is high but d_ready is low, and a compliant device holds the response.](../img/tlul_handshake.svg)

*Cycle 3: the request transfers. Cycle 5: the device offers a response but the receiver is not ready, so the device must keep offering it. Cycle 6: it transfers. The platform cache did not hold — it dropped the response after cycle 5 — and that single omission is the "d_ready defect" the accelerator's retry timer exists to survive.*


What each of the three fixed bugs did to this picture:

- **Request mux (bug 1):** the `a_ready` the CPU saw came from a block that
  was never going to take the request. Cycle 3 looked like a transfer to the
  CPU; nothing on the other side recorded it. No response could ever come.
- **d_ready defect (worked around):** the device pulsed `d_valid` for cycle 5
  only. If the receiver was busy that one cycle, the response was gone.
- **Write-back (bug 2):** the handshake was perfect; the *data* on the
  write-back was one cycle stale. Timing diagrams of handshakes cannot show
  this one — §3 below can.

---

## 2. One inference, end to end

Wall-clock view of a single frame on the board. Total 94 s; 50 MHz clock.

![Gantt-style timeline of one inference: weight load over JTAG about 66 seconds, patch embedding under a second, twelve transformer blocks about 0.7 seconds each, DPT head about five seconds. Within each block, GEMMs and requantisation run on the accelerator and the element-wise work on the CPU.](../img/inference_timeline.svg)

*The JTAG weight load dominates the wall clock but is a one-time setup cost. Inside the model, every block alternates accelerator jobs (GEMM, requantisation) with CPU element-wise work; after the speed-up the two are of similar size.*


Terms:

- **Patch embedding** — the first layer, which cuts the image into 14×14
  patches and turns each into a 384-number vector.
- **Transformer block** — the repeated unit of a vision transformer:
  self-attention (each patch looks at every other patch) followed by an MLP
  (a two-layer per-patch network). Twelve of them, in sequence.
- **qkv / proj / fc1 / fc2** — the four matrix multiplies in one block.
- **DPT head** — the decoder that turns the transformer's features back into
  a full-resolution depth map.
- **LayerNorm, GELU, softmax** — element-wise normalisation and activation
  functions. Cheap per element, but there are millions of elements and they
  run on a scalar CPU that retires about one instruction per cycle; they are
  in fixed point now, the per-row statistics excepted.

---

## 3. The whole computation, step by step

Every operation of one frame, in the order it runs, numbered 1–35. Each box
says what is computed, which tensors it reads (**R**) and writes (**W**),
with their shapes, and — by the coloured tag — which memory they live in.
Blue boxes run on the CPU, orange on the accelerator, grey on the PC.

Parallelism is drawn with fork/join bars:

- **Solid orange bars** mark work that is truly concurrent. Inside every
  accelerator job (expanded once, in step 3) three engines run at the same
  time: the read engine with up to eight requests in flight, the 64
  multiply-accumulate units, and the write engine with up to eight writes
  awaiting acknowledgement. That overlap is why the block moves a word every
  3.2 cycles instead of every 8.3.
- **Dashed grey bars** mark work that is *independent* — it could run in
  parallel — but is executed one piece after another today: the six
  attention heads, the two half-products in each attention GEMM (10a/10b,
  12a/12b), and the four DPT feature levels (22–25). Everything else is a
  strict chain: each step needs the previous step's output.
- The CPU and the accelerator are not overlapped either: the CPU polls the
  status register while a job runs. `PERFORMANCE.md` §5 lists what could be
  overlapped and what it would gain.

![Every computation of one frame in order: 35 numbered operations with the tensors they read and write and the memory they live in; fork/join bars for parallel and independent work](../img/computation.svg)

*Steps 7–21 repeat for each of the twelve transformer blocks; steps 9–13 for each of the six heads within a block. "GEMM + requant" everywhere means steps 3–5: a matrix product on the accelerator, the per-row range it reports, the CPU's per-row parameters, and the accelerator's requantisation job. (`img/computation.png` is the same figure as a bitmap.)*

### The same frame as a process

The figure below is the complementary view: the same frame in four lanes (PC,
CPU, accelerator, memory), with the block and the attention head expanded and
the two accelerator jobs detailed underneath — where §3's figure is about
*what is computed*, this one is about *who does it and where the data goes*.

![One frame, start to finish: PC, CPU, accelerator and memory lanes, steps numbered 1–8, the transformer block and attention expanded, GEMM and requantisation expanded underneath](../img/flow.svg)

*Steps 1–2 happen once per session; 3–8 once per image. The A and B expansions are what every "GEMM" and "requant" in the upper part stands for.*

---

## 4. The write-back bug, cycle by cycle

This is the sequence that cost the most to find, drawn at the level where it
becomes obvious. Two back-to-back writes to addresses that collide in the
cache (same line index, different tags). The first write misses, is brought
in, and lands in the cache RAM. The second write misses the same line the
very next cycle and must evict the first — writing its contents back to DDR3.

![Cycle diagram of the cache write-back bug. In cycle N the first write lands in the data RAM. In cycle N+1 the second write to the same line misses and the cache issues the write-back using data_rdata_raw, which is the RAM's output from before the cycle-N write landed. The forwarded signal data_rdata already had the correct value and was the fix.](../img/writeback_bug.svg)

*Write A, then evict A one cycle later. The eviction reads the RAM before A's write is visible and sends the previous contents to DDR3. A's data is lost. Every lost word on the board was a tile's final write, because that is the write followed immediately by the next tile's first miss to the same line.*


Why it hid so long:

- The accelerator's own counters said every write was issued and
  acknowledged — which was true. The cache acknowledges a write when it
  lands in the *cache*, before the write-back. The ack was honest; the
  write-back was wrong.
- It needs two misses to the same line on *consecutive* cycles. A testbench
  driver that waits for each response before issuing the next request can
  never produce that. The bug only appeared in simulation once the driver
  was rewritten to present the next request the moment the previous one was
  accepted, as a real CPU does.
- Reading the word back after evicting the cache showed the loss was
  permanent, which ruled out "written but read stale" and pointed at the
  write-back rather than the read path.

---

## 5. How a stalled bus was diagnosed without halting the CPU

Not a timing diagram but a data-flow one, because the mechanism is the
point: when the CPU is wedged on a bus access it can never retire, the
debugger cannot halt it, so the usual "read the program counter" is
impossible. A hardware register that watches the bus was the way in.

![Data flow of the watchdog diagnosis: the CPU is wedged and cannot be halted, but the debug module reads memory over the bus directly; the watchdog register on the DDR3 port has latched the oldest unanswered request, and reading it over JTAG names the stalled address, opcode and master.](../img/watchdog_probe.svg)

*The CPU's stalled access blocks the CPU, not the bus. The debug module's own bus port still works, so a register that has been watching the DDR3 port can be read out and says which request never came back — from which master, to which address, for how long.*


This is the diagnostic that broke the longest impasse in the project. The
register (`student_tl_watch.sv`, exposed at `DDR_CTRL0 + 0x4/0x8`) reported
in a single line an accelerator write, thirty transactions outstanding on the
port, and a saturated stall counter — enough to go straight to the request
mux and find bug 1.
