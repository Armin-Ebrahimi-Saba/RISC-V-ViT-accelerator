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

<figure>
<svg viewBox="0 0 760 300" role="img" aria-label="Timing diagram of a TL-UL request and response. The request handshakes in cycle 3 when a_valid and a_ready are both high; the response handshakes in cycle 6 when d_valid and d_ready are both high. In cycle 5, d_valid is high but d_ready is low, and a compliant device holds the response." style="max-width:100%;height:auto;font-family:ui-monospace,monospace;font-size:12px">
<g stroke="currentColor" stroke-width="1" opacity="0.25">
<line x1="140" y1="30" x2="140" y2="280"/><line x1="220" y1="30" x2="220" y2="280"/>
<line x1="300" y1="30" x2="300" y2="280"/><line x1="380" y1="30" x2="380" y2="280"/>
<line x1="460" y1="30" x2="460" y2="280"/><line x1="540" y1="30" x2="540" y2="280"/>
<line x1="620" y1="30" x2="620" y2="280"/><line x1="700" y1="30" x2="700" y2="280"/>
</g>
<g fill="currentColor" font-size="11">
<text x="180" y="24" text-anchor="middle">1</text><text x="260" y="24" text-anchor="middle">2</text>
<text x="340" y="24" text-anchor="middle">3</text><text x="420" y="24" text-anchor="middle">4</text>
<text x="500" y="24" text-anchor="middle">5</text><text x="580" y="24" text-anchor="middle">6</text>
<text x="660" y="24" text-anchor="middle">7</text>
</g>
<g fill="none" stroke="currentColor" stroke-width="2">
<!-- a_valid: high 2..3 -->
<polyline points="140,80 220,80 220,50 380,50 380,80 700,80"/>
<!-- a_ready: high 3 -->
<polyline points="140,125 300,125 300,95 380,95 380,125 700,125"/>
<!-- d_valid: high 5..6 -->
<polyline points="140,185 460,185 460,155 620,155 620,185 700,185"/>
<!-- d_ready: low 5, high 6 -->
<polyline points="140,230 540,230 540,200 620,200 620,230 700,230"/>
</g>
<!-- highlight handshake cycles -->
<g fill="#2b8a3e" opacity="0.18"><rect x="300" y="40" width="80" height="95"/><rect x="540" y="145" width="80" height="95"/></g>
<g fill="#d9480f" opacity="0.18"><rect x="460" y="145" width="80" height="95"/></g>
<g fill="currentColor">
<text x="20" y="70">a_valid</text>
<text x="20" y="115">a_ready</text>
<text x="20" y="175">d_valid</text>
<text x="20" y="220">d_ready</text>
<text x="340" y="150" text-anchor="middle" font-size="11" fill="#2b8a3e">request taken</text>
<text x="580" y="255" text-anchor="middle" font-size="11" fill="#2b8a3e">response taken</text>
<text x="500" y="270" text-anchor="middle" font-size="11" fill="#d9480f">must hold</text>
</g>
</svg>
<figcaption>Cycle 3: the request transfers. Cycle 5: the device offers a response but the receiver is not ready, so the device must keep offering it. Cycle 6: it transfers. The platform cache did not hold — it dropped the response after cycle 5 — and that single omission is the "d_ready defect" the accelerator's retry timer exists to survive.</figcaption>
</figure>

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

<figure>
<svg viewBox="0 0 860 330" role="img" aria-label="Gantt-style timeline of one inference: weight load over JTAG about 66 seconds, patch embedding under a second, twelve transformer blocks about five seconds each, DPT head about eight seconds. Within each block, GEMMs run on the accelerator and everything else on the CPU." style="max-width:100%;height:auto;font-family:system-ui,sans-serif;font-size:12px">
<g fill="none" stroke="currentColor" stroke-width="1.2">
<line x1="60" y1="70" x2="840" y2="70"/>
<line x1="60" y1="66" x2="60" y2="74"/><line x1="357" y1="66" x2="357" y2="74"/>
<line x1="365" y1="66" x2="365" y2="74"/><line x1="770" y1="66" x2="770" y2="74"/>
<line x1="840" y1="66" x2="840" y2="74"/>
</g>
<g fill="currentColor" font-size="11">
<text x="60" y="58" text-anchor="middle">0 s</text>
<text x="357" y="58" text-anchor="middle">66 s</text>
<text x="770" y="58" text-anchor="middle">156 s</text>
<text x="840" y="58" text-anchor="middle">160 s</text>
</g>
<!-- phase bars -->
<rect x="60" y="85" width="297" height="26" rx="3" fill="currentColor" opacity="0.15"/>
<rect x="357" y="85" width="8" height="26" rx="3" fill="#d9480f"/>
<rect x="365" y="85" width="405" height="26" rx="3" fill="#1c7ed6" opacity="0.6"/>
<rect x="770" y="85" width="70" height="26" rx="3" fill="#2b8a3e" opacity="0.6"/>
<g fill="currentColor" font-size="11">
<text x="208" y="102" text-anchor="middle">weight load, JTAG, 0.37 MB/s</text>
<text x="567" y="102" text-anchor="middle" fill="#fff">12 transformer blocks</text>
<text x="805" y="102" text-anchor="middle" fill="#fff">DPT head</text>
<text x="361" y="130" text-anchor="middle" fill="#d9480f">patch embed</text>
</g>
<!-- zoom into one block -->
<g fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3">
<line x1="400" y1="111" x2="60" y2="170"/><line x1="434" y1="111" x2="840" y2="170"/>
</g>
<g fill="currentColor" font-size="11"><text x="450" y="160" text-anchor="middle">one block, ≈5 s</text></g>
<g fill="none" stroke="currentColor" stroke-width="1.2"><line x1="60" y1="185" x2="840" y2="185"/></g>
<!-- block internals: attention then MLP; each has GEMMs (accel) and CPU work -->
<rect x="60" y="195" width="60" height="22" rx="3" fill="currentColor" opacity="0.2"/>
<rect x="120" y="195" width="110" height="22" rx="3" fill="#1c7ed6" opacity="0.7"/>
<rect x="230" y="195" width="140" height="22" rx="3" fill="currentColor" opacity="0.2"/>
<rect x="370" y="195" width="60" height="22" rx="3" fill="#1c7ed6" opacity="0.7"/>
<rect x="430" y="195" width="60" height="22" rx="3" fill="currentColor" opacity="0.2"/>
<rect x="490" y="195" width="130" height="22" rx="3" fill="#1c7ed6" opacity="0.7"/>
<rect x="620" y="195" width="80" height="22" rx="3" fill="currentColor" opacity="0.2"/>
<rect x="700" y="195" width="140" height="22" rx="3" fill="#1c7ed6" opacity="0.7"/>
<g fill="currentColor" font-size="10">
<text x="90" y="210" text-anchor="middle">norm</text>
<text x="175" y="210" text-anchor="middle" fill="#fff">qkv GEMM</text>
<text x="300" y="210" text-anchor="middle">attention (CPU)</text>
<text x="400" y="210" text-anchor="middle" fill="#fff">proj</text>
<text x="460" y="210" text-anchor="middle">norm</text>
<text x="555" y="210" text-anchor="middle" fill="#fff">fc1 GEMM</text>
<text x="660" y="210" text-anchor="middle">GELU</text>
<text x="770" y="210" text-anchor="middle" fill="#fff">fc2 GEMM</text>
</g>
<g fill="currentColor" font-size="11">
<text x="60" y="250">Blue = matrix multiply on the accelerator. Grey = element-wise work on the CPU: LayerNorm, the</text>
<text x="60" y="266">attention softmax, GELU, and the requantisation that turns each GEMM's int32 output back into int16.</text>
<text x="60" y="292">The accelerator has removed the multiplications from the critical path; what remains is CPU work</text>
<text x="60" y="308">and the 25 MB of weights crossing DDR3 once per frame. Note the weight load itself is not part of a</text>
<text x="60" y="324">frame — it happens once per session — so the model's own time is 94 s, of which ~60 s is the 12 blocks.</text>
</g>
</svg>
<figcaption>The JTAG weight load dominates the wall clock but is a one-time setup cost. Inside the model, every block alternates accelerator GEMMs with CPU element-wise work; the CPU work is now the larger share.</figcaption>
</figure>

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
  run on a scalar CPU with software floating point.

---

## 3. The write-back bug, cycle by cycle

This is the sequence that cost the most to find, drawn at the level where it
becomes obvious. Two back-to-back writes to addresses that collide in the
cache (same line index, different tags). The first write misses, is brought
in, and lands in the cache RAM. The second write misses the same line the
very next cycle and must evict the first — writing its contents back to DDR3.

<figure>
<svg viewBox="0 0 860 420" role="img" aria-label="Cycle diagram of the cache write-back bug. In cycle N the first write lands in the data RAM. In cycle N+1 the second write to the same line misses and the cache issues the write-back using data_rdata_raw, which is the RAM's output from before the cycle-N write landed. The forwarded signal data_rdata already had the correct value and was the fix." style="max-width:100%;height:auto;font-family:ui-monospace,monospace;font-size:12px">
<g stroke="currentColor" stroke-width="1" opacity="0.25">
<line x1="200" y1="30" x2="200" y2="330"/><line x1="330" y1="30" x2="330" y2="330"/>
<line x1="460" y1="30" x2="460" y2="330"/><line x1="590" y1="30" x2="590" y2="330"/>
<line x1="720" y1="30" x2="720" y2="330"/>
</g>
<g fill="currentColor" font-size="11">
<text x="265" y="24" text-anchor="middle">N−1</text><text x="395" y="24" text-anchor="middle">N</text>
<text x="525" y="24" text-anchor="middle">N+1</text><text x="655" y="24" text-anchor="middle">N+2</text>
</g>
<g fill="currentColor" font-size="12">
<text x="20" y="70">front-end request</text>
<text x="20" y="120">data RAM write</text>
<text x="20" y="170">data_rdata_raw</text>
<text x="20" y="220">data_rdata (fwd)</text>
<text x="20" y="270">write-back to DDR3</text>
</g>
<!-- request row -->
<rect x="205" y="52" width="120" height="26" rx="3" fill="currentColor" opacity="0.15"/>
<rect x="335" y="52" width="120" height="26" rx="3" fill="#1c7ed6" opacity="0.5"/>
<rect x="465" y="52" width="120" height="26" rx="3" fill="#d9480f" opacity="0.5"/>
<g fill="currentColor" font-size="11">
<text x="265" y="69" text-anchor="middle">write A (miss)</text>
<text x="395" y="69" text-anchor="middle">A refilled</text>
<text x="525" y="69" text-anchor="middle">write B: evict A</text>
</g>
<!-- RAM write row -->
<rect x="335" y="102" width="120" height="26" rx="3" fill="#1c7ed6" opacity="0.5"/>
<g fill="currentColor" font-size="11"><text x="395" y="119" text-anchor="middle">A's new data lands</text></g>
<!-- raw row: shows OLD value in N+1 -->
<rect x="335" y="152" width="120" height="26" rx="3" fill="currentColor" opacity="0.15"/>
<rect x="465" y="152" width="120" height="26" rx="3" fill="#d9480f" opacity="0.5"/>
<rect x="595" y="152" width="120" height="26" rx="3" fill="#1c7ed6" opacity="0.5"/>
<g fill="currentColor" font-size="11">
<text x="395" y="169" text-anchor="middle">old line</text>
<text x="525" y="169" text-anchor="middle">STILL old line</text>
<text x="655" y="169" text-anchor="middle">A's data (too late)</text>
</g>
<!-- fwd row: correct in N+1 -->
<rect x="465" y="202" width="120" height="26" rx="3" fill="#2b8a3e" opacity="0.5"/>
<g fill="currentColor" font-size="11"><text x="525" y="219" text-anchor="middle">A.s data ✓</text></g>
<!-- write-back row -->
<rect x="465" y="252" width="120" height="26" rx="3" fill="#d9480f" opacity="0.5"/>
<g fill="currentColor" font-size="11"><text x="525" y="269" text-anchor="middle">sends data_rdata_raw</text></g>
<!-- annotation -->
<g fill="currentColor" font-size="11">
<text x="20" y="360">The RAM is read one cycle behind: in N+1, data_rdata_raw still shows what the line held BEFORE</text>
<text x="20" y="376">cycle N's write. The cache already had a forwarding path (data_rdata) that patches in last cycle's</text>
<text x="20" y="392">write for exactly this case — it used it for the front-end response but not for the write-back.</text>
<text x="20" y="410">Fix: one line — the write-back reads data_rdata instead of data_rdata_raw.</text>
</g>
</svg>
<figcaption>Write A, then evict A one cycle later. The eviction reads the RAM before A's write is visible and sends the previous contents to DDR3. A's data is lost. Every lost word on the board was a tile's final write, because that is the write followed immediately by the next tile's first miss to the same line.</figcaption>
</figure>

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

## 4. How a stalled bus was diagnosed without halting the CPU

Not a timing diagram but a data-flow one, because the mechanism is the
point: when the CPU is wedged on a bus access it can never retire, the
debugger cannot halt it, so the usual "read the program counter" is
impossible. A hardware register that watches the bus was the way in.

<figure>
<svg viewBox="0 0 760 240" role="img" aria-label="Data flow of the watchdog diagnosis: the CPU is wedged and cannot be halted, but the debug module reads memory over the bus directly; the watchdog register on the DDR3 port has latched the oldest unanswered request, and reading it over JTAG names the stalled address, opcode and master." style="max-width:100%;height:auto;font-family:system-ui,sans-serif;font-size:12px">
<defs><marker id="ah4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>
<g fill="none" stroke="currentColor" stroke-width="1.4">
<rect x="20" y="40" width="120" height="60" rx="6" stroke-dasharray="5 3"/>
<rect x="20" y="140" width="120" height="60" rx="6"/>
<rect x="260" y="90" width="120" height="60" rx="6"/>
<rect x="480" y="90" width="150" height="60" rx="6"/>
<rect x="660" y="90" width="80" height="60" rx="6"/>
<line x1="140" y1="70" x2="260" y2="110" stroke-dasharray="5 3"/>
<line x1="140" y1="170" x2="260" y2="130" marker-end="url(#ah4)" stroke="#2b8a3e" stroke-width="2"/>
<line x1="380" y1="120" x2="480" y2="120" marker-end="url(#ah4)" stroke="#2b8a3e" stroke-width="2"/>
<line x1="630" y1="120" x2="660" y2="120" marker-end="url(#ah4)"/>
</g>
<g fill="currentColor">
<text x="80" y="65" text-anchor="middle" font-weight="600">CPU</text>
<text x="80" y="83" text-anchor="middle" font-size="11">wedged; halt ignored</text>
<text x="80" y="165" text-anchor="middle" font-weight="600">Debug module</text>
<text x="80" y="183" text-anchor="middle" font-size="11">sysbus read, via JTAG</text>
<text x="320" y="115" text-anchor="middle" font-weight="600">crossbar</text>
<text x="320" y="133" text-anchor="middle" font-size="11">still routing</text>
<text x="555" y="112" text-anchor="middle" font-weight="600">watchdog reg</text>
<text x="555" y="128" text-anchor="middle" font-size="11">latched: addr, opcode,</text>
<text x="555" y="142" text-anchor="middle" font-size="11">master id, stall cycles</text>
<text x="700" y="115" text-anchor="middle" font-size="11">DDR3</text>
<text x="700" y="130" text-anchor="middle" font-size="11">port</text>
<text x="200" y="78" text-anchor="middle" font-size="11">no answer</text>
<text x="380" y="220" text-anchor="middle" font-size="11" fill="#2b8a3e">a read the CPU never has to execute</text>
</g>
</svg>
<figcaption>The CPU's stalled access blocks the CPU, not the bus. The debug module's own bus port still works, so a register that has been watching the DDR3 port can be read out and says which request never came back — from which master, to which address, for how long.</figcaption>
</figure>

This is the diagnostic that broke the longest impasse in the project. The
register (`student_tl_watch.sv`, exposed at `DDR_CTRL0 + 0x4/0x8`) reported
in a single line an accelerator write, thirty transactions outstanding on the
port, and a saturated stall counter — enough to go straight to the request
mux and find bug 1.
