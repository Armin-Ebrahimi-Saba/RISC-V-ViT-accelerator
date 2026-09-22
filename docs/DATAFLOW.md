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

One rule governs every transfer on the bus, in either direction: **data
moves only in a cycle where `valid` and `ready` are both high.** The sender
raises `valid` and holds its data steady; the receiver raises `ready` once it
can accept it. Neither side may assume the other is watching — and once
`valid` is up, the sender is not allowed to change or withdraw the data until
the transfer actually happens.

![Timing diagram of a TL-UL request and response. The request handshakes in cycle 3 when a_valid and a_ready are both high; the response handshakes in cycle 6 when d_valid and d_ready are both high. In cycle 5, d_valid is high but d_ready is low, and a compliant device holds the response.](../img/tlul_handshake.svg)

*Cycle 3: the request goes through, since a_valid and a_ready are both high. Cycle 5: the device offers its response, but the receiver isn't ready yet — a compliant device would keep offering it. Cycle 6: the response finally goes through. The platform's cache does not behave like this: it drops the response after cycle 5 instead of holding it. That single misbehaviour is the "d_ready defect" the accelerator's retry timer exists to survive.*

Each of the three fixed bugs broke this picture in a different way:

- **Request mux (bug 1)** — the CPU saw `a_ready` go high from a block that
  was never actually going to accept the request. Cycle 3 looked like a
  transfer to the CPU, but nothing on the other side recorded it, so no
  response could ever arrive.
- **d_ready defect (worked around, not fixed)** — the device raises
  `d_valid` for exactly one cycle (cycle 5 here) and never checks whether the
  receiver is ready. If the receiver is busy that one cycle, the response is
  simply gone.
- **Write-back (bug 2)** — the handshake itself was fine; the *data* riding
  on the write-back was one cycle stale. A handshake diagram can't show a
  bug like this — §4 below does.

---

## 2. One inference, end to end

The wall-clock view of running a single frame on the board: 94 seconds
total, at a 50 MHz clock.

![Gantt-style timeline of one inference: weight load over JTAG about 66 seconds, patch embedding under a second, twelve transformer blocks about 0.7 seconds each, DPT head about five seconds. Within each block, GEMMs and requantisation run on the accelerator and the element-wise work on the CPU.](../img/inference_timeline.svg)

*The JTAG weight load takes up most of the wall clock, but it's a one-time setup cost, not something every frame pays. Inside the model itself, every transformer block alternates accelerator jobs (GEMM, requantisation) with CPU element-wise work — after the speed-up work described in PERFORMANCE.md, the two now take about the same amount of time.*

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

One full picture, numbered 1–35 in the order things happen for a single
frame. Each box names an operation, which tensors it reads (**R**) and
writes (**W**) and their shapes, and — by its colour — where it runs: blue
for the CPU, orange for the accelerator, grey for the PC. A coloured tag
under each box (blob, image, arena, scratch, tile, result, pc) says which
memory it touched — see the memory map in `ARCHITECTURE.md` §3.

Parallelism is drawn with fork/join bars, and most of the computation has
none — it's a straight chain where each step needs the output of the one
before it:

- **Solid orange bars** mark work that genuinely happens at the same time.
  This occurs in exactly one place: inside every accelerator job (detailed
  once, in step 3), where three engines run concurrently — a read engine
  with up to eight requests in flight, the 128 multiply-accumulate units, and
  a write engine with up to eight writes awaiting acknowledgement. That
  overlap is the whole reason the block moves one word every 3.2 cycles
  instead of every 8.3.
- **Dashed grey bars** mark work that *could* run in parallel — because the
  pieces don't depend on each other — but today runs one after another: the
  six attention heads, the two half-products of each attention GEMM
  (10a/10b and 12a/12b), and the four DPT feature levels (22–25).
- The CPU and the accelerator are never overlapped either — while a job
  runs, the CPU just polls the status register instead of doing other useful
  work. `PERFORMANCE.md` §5 lists what overlapping them would gain.

![Every computation of one frame in order: 35 numbered operations with the tensors they read and write and the memory they live in; fork/join bars for parallel and independent work](../img/computation.svg)

*Steps 7–21 repeat for each of the twelve transformer blocks; steps 9–13 for each of the six heads within a block. "GEMM + requant" everywhere means steps 3–5: a matrix product on the accelerator, the per-row range it reports back, the CPU turning that range into per-row scaling parameters, and the accelerator's requantisation job that applies them. (`img/computation.png` is the same figure as a bitmap.)*

### Step by step, in plain terms

The table groups the 35 steps by what part of the network they belong to.
"GEMM+requant" stands for the same three-step pattern every time: the
accelerator multiplies two matrices, the CPU computes per-row scaling from
the ranges the accelerator reported, and the accelerator scales the int32
result back down to int16 (see §4 of `ARCHITECTURE.md`).

**Input (once per frame)**

| # | What happens |
|---|---|
| 1 | The host resizes the photo to 126×126, normalises it the way the model was trained to expect, and quantises it to int16, then sends it to the board over JTAG. |
| 2 | The CPU cuts the image into 81 non-overlapping 14×14 patches (`im2col`) — the standard way to turn an image into a sequence a transformer can consume. |

**Patch embedding (once per frame)**

| # | What happens |
|---|---|
| 3 | GEMM+requant: each patch is turned into a 384-number vector by multiplying it against the patch-embedding weights. Two accelerator tiles (64 + 17 patches). |
| 4 | The CPU computes the requantisation table from the accelerator's per-row ranges. |
| 5 | The accelerator's requantisation job produces `emb`, the embedded patches, as int16. |
| 6 | The CPU builds the 82-token sequence: a learned class token plus the 81 embedded patches, each with its learned position added. This is the `x` matrix every transformer block updates. |

**Transformer encoder — steps 7–21 repeat for each of the 12 blocks**

| # | What happens |
|---|---|
| 7 | LayerNorm 1: the CPU rescales each token's row to zero mean and unit variance (in float, for accuracy), producing `n`. |
| 8 | GEMM+requant: `n` is multiplied by the block's qkv weights, producing query, key and value vectors for all 6 attention heads at once. |
| 9–13 | Attention, repeated for each of the 6 heads (see below) — independent work, run one head after another today. |
| 14 | GEMM+requant: the combined attention output (`ctx`) is projected back to 384 dimensions. |
| 15 | Residual add: `x = x + attn`, so the block's output builds on its input rather than replacing it. |
| 16 | LayerNorm 2, same idea as step 7, producing `n2`. |
| 17 | GEMM+requant: the MLP's first layer expands `n2` from 384 to 1536 dimensions. |
| 18 | GELU: a smooth nonlinearity applied element-wise, via a 257-entry lookup table with linear interpolation (fast on a CPU with no floating-point unit). |
| 19 | GEMM+requant: the MLP's second layer compresses back down to 384 dimensions. |
| 20 | Residual add, same idea as step 15; `x` now holds this block's output and feeds the next block's step 7. |
| 21 | After blocks 3, 6, 9 and 12 only: a normalised copy of `x` is saved as one of the four feature maps the decoder needs. |

*Attention (steps 9–13, once per head):* the accelerator only multiplies
int8 by int16, but `q`, `k` and `v` are int16, so each is split into an
8-bit high and low half and the product run twice — `a·x = 2ˢ·(a·hi) +
(a·lo)` — which is exact, not an approximation.

| # | What happens |
|---|---|
| 9 | The CPU gathers this head's slice of q/k/v and splits q and v into their high/low int8 halves. |
| 10a/10b | Two independent GEMMs compute the attention scores from the high and low halves; run one after another today. |
| 11 | Softmax: the CPU turns each row of scores into probabilities (fixed-point, per query token). |
| 12a/12b | Two more independent GEMMs combine the probabilities with `v`'s high/low halves into the raw context. |
| 13 | The CPU normalises by each row's probability sum and writes this head's slice of `ctx`. |

**DPT decoder — steps 22–25 build four feature maps, 26–29 fuse them**

The four feature maps saved in step 21 are at four different resolutions
(9×9 up to 36×36 after this stage); the decoder folds them together from
coarsest to finest.

| # | What happens |
|---|---|
| 22–25 | One per feature level (independent, run one after another today): a 1×1 convolution (GEMM+requant) projects the features to fewer channels, an upsample or downsample brings it to that level's target resolution, then a 3×3 convolution (im2col on the CPU, GEMM+requant on the accelerator) refines it. |
| 26–29 | Strictly sequential, coarsest level first: each step's `ResidualConvUnit` (two 3×3 convolutions with ReLUs, plus the original input added back) refines the previous level's output, adds it to this level's own feature map, upsamples on the CPU, and projects with a 1×1 convolution. |

**Output (once per frame)**

| # | What happens |
|---|---|
| 30 | A 3×3 convolution narrows 64 channels down to 32. |
| 31 | Bilinear upsampling (CPU) from 72×72 to the full 126×126. |
| 32 | Another 3×3 convolution at 32 channels, then a ReLU. |
| 33 | A final 1×1 convolution collapses 32 channels to the single depth channel, then a ReLU. |
| 34 | The CPU converts the fixed-point result to a float depth map and raises the "done" flag. |
| 35 | The host reads the depth map back over JTAG (0.7 s), saves it, and renders it beside the PyTorch reference. |

Across all 35 steps, one frame runs 2,275 accelerator jobs, reads the 25 MB
weight blob from DDR3 twice (once per activation tile), uses a 15 MB peak of
the 64 MB scratch arena, and takes 14.2 s at 50 MHz.

### The same frame as a process

The figure below is a complementary view of the same frame: instead of *what
is computed* (§3's figure above), it shows *who does it and where the data
goes*, laid out in four lanes — PC, CPU, accelerator, and memory.

![One frame, start to finish: PC, CPU, accelerator and memory lanes, steps numbered 1–8, the transformer block and attention expanded, GEMM and requantisation expanded underneath](../img/flow.svg)

Steps 1–2 happen once per session (boot and weight load); 3–8 repeat once
per image. Step 5 (one transformer block) and its sub-step 5c (one attention
head) are expanded into their own numbered rows; "GEMM" and "requant" boxes
everywhere point to the A and B expansions at the bottom, which are what
every such box in the rest of the diagram stands for.

| # | Who | What happens |
|---|---|---|
| 1 | Host | Loads the 25 MB weight blob over JTAG — once per session, about 66 s — and spot-checks a few words to confirm it landed correctly. |
| 2 | CPU | Boots, initialises DDR3, runs a small self-test GEMM against a known answer, then waits for the host's "go" flag. |
| 3 | Host | Sends one image (0.3 s) and raises the "run a frame" flag. |
| 4 | CPU + accelerator | Patch embedding: `im2col`, then GEMM+requant (→ A, B), then the CPU adds position embeddings and the class token. |
| 5 | CPU + accelerator | One transformer block, sub-steps 5a–5k: LayerNorm, qkv GEMM, six attention heads (5c), projection GEMM, residual add, LayerNorm, MLP up GEMM, GELU, MLP down GEMM, residual add, and — on blocks 3/6/9/12 — the feature tap. Runs strictly in that order; the CPU polls STATUS during every accelerator job rather than doing something else. |
| 6 | CPU + accelerator | The DPT head: four feature levels each get a 1×1 convolution, a resize, and a 3×3 convolution (6a); the levels are then fused coarsest-to-finest with residual conv units and 1×1 convolutions (6b); a final 3×3 → upsample → 3×3 → 1×1 chain produces the depth map (6c). Every convolution is `im2col` (CPU) + GEMM + requant (→ A, B) — 2,275 accelerator jobs in total per frame. |
| 7 | CPU | Publishes the depth map to its DDR3 result slot, prints timing, and raises the "done" flag. |
| 8 | Host | Reads the result over JTAG (0.7 s), saves it, and compares it against the host build — then loops back to step 3 for the next image. |

**A — one GEMM on the accelerator**, what every "GEMM" box above means:

| # | What happens |
|---|---|
| A1 | The CPU programs the job's registers — A/W/C addresses and strides, K, M, row count, `S_ADDR` for stats — and sets `CTRL.start`, then polls `STATUS`. |
| A2 | The accelerator loads a 128-row tile of A into on-chip RAM, up to 8 reads in flight through a reorder buffer; a lost response is automatically re-issued. |
| A3 | It streams W a word at a time (4 packed int8 weights each), broadcasting each to all 128 multiply-accumulate units — 128 MACs per cycle, K cycles per weight row. |
| A4 | It drains that row's 64 int32 sums to column m of C, then (if stats are on) that row's max and min. |
| A5 | It moves to the next row, or — once all M rows are done — reports `STATUS.done`; the CPU then starts the next 128-row tile back at A2. |

Everything above A3 runs concurrently inside the block, as noted in §3.

**B — requantisation**, what every "requant" box above means, run right
after every GEMM:

| # | What happens |
|---|---|
| B1 | From each row's {max, min}, the CPU computes one output scale for the whole tensor, then a per-row {multiplier, shift, bias} table. |
| B2 | The accelerator reads the int32 result chunk transposed into its tile RAM, and the parameter table into a small RAM alongside it. |
| B3 | It streams the scaled, rounded, saturated int16 result out, two values packed per 32-bit word. |

---

## 4. The write-back bug, cycle by cycle

This was the most expensive bug to track down, so it's drawn at the level
where the mistake actually becomes visible: two back-to-back writes whose
addresses collide in the cache (same line index, different tags — meaning
they map to the same cache slot but hold different data). The first write
misses, is fetched from DDR3, and lands in the cache's RAM. The very next
cycle, the second write misses that same line and must evict the first one
first — which means writing the first write's data back to DDR3.

![Cycle diagram of the cache write-back bug. In cycle N the first write lands in the data RAM. In cycle N+1 the second write to the same line misses and the cache issues the write-back using data_rdata_raw, which is the RAM's output from before the cycle-N write landed. The forwarded signal data_rdata already had the correct value and was the fix.](../img/writeback_bug.svg)

What happens, cycle by cycle:

1. **Cycle N — write A lands.** A write to a line already in the cache (or
   just missed in) updates the cache's data RAM. From this cycle on, the RAM
   holds A's new value.
2. **Cycle N+1 — write B misses the same line.** B's address maps to the
   same cache slot as A but is a different address, so it's a miss: the
   line must be evicted to make room, and since it was written it's dirty —
   its contents have to go back to DDR3 first.
3. **The eviction reads the wrong copy.** The cache's write-back logic reads
   the line's contents from `data_rdata_raw`, the RAM's registered output
   from *before* cycle N's write is visible on that signal — one cycle too
   early. It sends that stale, pre-A value to DDR3 instead of A's actual
   data.
4. **A is gone.** DDR3 now holds A's old value where it should hold A's new
   value, and nothing else will ever rewrite it — the loss is silent and
   permanent.
5. **The fix.** A signal called `data_rdata` already existed alongside
   `data_rdata_raw` — the same read, but forwarded so it reflects a
   same-cycle write. Using it for the write-back instead closes the gap.

*Every lost word on the board was a tile's final write, because that write
is always immediately followed by the next tile's first access missing the
same cache line — exactly the two-cycle pattern above.*

Why it hid so long:

- The accelerator's own counters said every write was issued and
  acknowledged — which was true. The cache acknowledges a write the moment
  it lands in the *cache*, before the write-back happens. The
  acknowledgement was honest; the write-back was wrong.
- It only shows up on two misses to the same line on *consecutive* cycles. A
  testbench driver that waits for each response before sending the next
  request can never produce that pattern. The bug only appeared once the
  driver was rewritten to present its next request the moment the previous
  one was accepted — the way a real CPU, or the accelerator, actually
  behaves.
- Reading the word back after the cache had evicted it (so the read had to
  go to DDR3) showed the loss was permanent. That ruled out "written but
  briefly read stale" and pointed straight at the write-back path rather
  than the read path.

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
