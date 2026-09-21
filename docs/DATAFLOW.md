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
<svg viewBox="0 0 860 330" role="img" aria-label="Gantt-style timeline of one inference: weight load over JTAG about 66 seconds, patch embedding under a second, twelve transformer blocks about 0.7 seconds each, DPT head about five seconds. Within each block, GEMMs and requantisation run on the accelerator and the element-wise work on the CPU." style="max-width:100%;height:auto;font-family:system-ui,sans-serif;font-size:12px">
<g fill="none" stroke="currentColor" stroke-width="1.2">
<line x1="60" y1="70" x2="840" y2="70"/>
<line x1="60" y1="66" x2="60" y2="74"/><line x1="703" y1="66" x2="703" y2="74"/>
<line x1="708" y1="66" x2="708" y2="74"/><line x1="793" y1="66" x2="793" y2="74"/>
<line x1="840" y1="66" x2="840" y2="74"/>
</g>
<g fill="currentColor" font-size="11">
<text x="60" y="58" text-anchor="middle">0 s</text>
<text x="703" y="58" text-anchor="middle">66 s</text>
<text x="793" y="58" text-anchor="middle">75 s</text>
<text x="840" y="58" text-anchor="middle">80 s</text>
</g>
<!-- phase bars -->
<rect x="60" y="85" width="643" height="26" rx="3" fill="currentColor" opacity="0.15"/>
<rect x="703" y="85" width="5" height="26" rx="3" fill="#d9480f"/>
<rect x="708" y="85" width="85" height="26" rx="3" fill="#1c7ed6" opacity="0.6"/>
<rect x="793" y="85" width="47" height="26" rx="3" fill="#2b8a3e" opacity="0.6"/>
<g fill="currentColor" font-size="11">
<text x="380" y="102" text-anchor="middle">weight load, JTAG, 0.37 MB/s — once per session</text>
<text x="750" y="102" text-anchor="middle" fill="#fff">12 blocks</text>
<text x="816" y="102" text-anchor="middle" fill="#fff" font-size="10">head</text>
<text x="705" y="130" text-anchor="middle" fill="#d9480f">patch embed</text>
</g>
<!-- zoom into one block -->
<g fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3">
<line x1="708" y1="111" x2="60" y2="170"/><line x1="715" y1="111" x2="840" y2="170"/>
</g>
<g fill="currentColor" font-size="11"><text x="450" y="160" text-anchor="middle">one block, ≈0.7 s (was ≈5 s)</text></g>
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
<text x="300" y="210" text-anchor="middle">attention (CPU + accel)</text>
<text x="400" y="210" text-anchor="middle" fill="#fff">proj</text>
<text x="460" y="210" text-anchor="middle">norm</text>
<text x="555" y="210" text-anchor="middle" fill="#fff">fc1 GEMM</text>
<text x="660" y="210" text-anchor="middle">GELU</text>
<text x="770" y="210" text-anchor="middle" fill="#fff">fc2 GEMM</text>
</g>
<g fill="currentColor" font-size="11">
<text x="60" y="250">Blue = on the accelerator: every matrix multiply, and the requantisation that turns its int32 output back</text>
<text x="60" y="266">into int16. Grey = element-wise work on the CPU: LayerNorm, the attention softmax and gathers, GELU, adds.</text>
<text x="60" y="292">The model's own time is 14.2 s per frame (it was 93.6 s — see PERFORMANCE.md), of which 8.7 s is the</text>
<text x="60" y="308">12 blocks and 5 s the DPT head. The weight load is not part of a frame: it happens once per session,</text>
<text x="60" y="324">after which images go in over JTAG in 0.3 s and depth maps come out in 0.7 s.</text>
</g>
</svg>
<figcaption>The JTAG weight load dominates the wall clock but is a one-time setup cost. Inside the model, every block alternates accelerator jobs (GEMM, requantisation) with CPU element-wise work; after the speed-up the two are of similar size.</figcaption>
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
  run on a scalar CPU that retires about one instruction per cycle; they are
  in fixed point now, the per-row statistics excepted.

---

## 3. The whole computation as a flow

Section 2 showed *when*; this shows *what happens, in what order, by whom,
and on which memory*. Four lanes: the PC (a Python script talking over
JTAG), the CPU, the accelerator, and memory. Numbers give the order. Every
step carries tags for the memory it **R**eads and **W**rites, colour-coded to
the regions in the right-hand lane. Two things are expanded at the bottom
because every "GEMM" and "requant" above means them: **A**, one matrix
product on the accelerator, and **B**, the requantisation that follows it.

Where things happen at the same time:

- **Inside the accelerator** (bracket under A): read requests are in flight
  (up to eight), the 64 multipliers accumulate, and up to eight writes await
  their acknowledgement — all at once. That overlap is what took it from
  8.3 to 3.2 cycles per beat.
- **Inside the requantisation job** (bracket under B): the element pipeline,
  the output FIFO and the write engine run concurrently.
- **Not yet overlapped:** the CPU and the accelerator. While a job runs, the
  CPU polls its status register. The CPU work of the next step (say, the
  gathers for the next attention head) could start meanwhile; nothing does
  today. `PERFORMANCE.md` §5 lists it.
- **Between frames:** the PC's readout of one result and its upload of the
  next image are sequential with the board; a double-buffered image address
  would let the upload overlap inference.

<figure>
<svg viewBox="0 0 1000 1636" role="img" aria-label="The complete computation of one frame as a flow: PC, CPU, accelerator and memory in four lanes; 14 numbered steps from the weight load to the result readout, with the transformer block and the attention head expanded; every step tagged with the memory it reads and writes; concurrent activity marked." style="max-width:100%;height:auto;font-family:system-ui,sans-serif;font-size:11px">
<defs><marker id="fa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker><marker id="fd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#2b8a3e"/></marker></defs>
<rect x="20" y="20" width="150" height="1596" rx="8" fill="#868e96" opacity="0.06"/>
<rect x="20" y="20" width="150" height="26" rx="8" fill="#868e96" opacity="0.85"/>
<text x="95.0" y="38" text-anchor="middle" fill="#fff" font-weight="600" font-size="12">PC (Python, JTAG)</text>
<rect x="185" y="20" width="420" height="1596" rx="8" fill="#1c7ed6" opacity="0.06"/>
<rect x="185" y="20" width="420" height="26" rx="8" fill="#1c7ed6" opacity="0.85"/>
<text x="395.0" y="38" text-anchor="middle" fill="#fff" font-weight="600" font-size="12">CPU  CV32E40P, 50 MHz</text>
<rect x="620" y="20" width="200" height="1596" rx="8" fill="#e8590c" opacity="0.06"/>
<rect x="620" y="20" width="200" height="26" rx="8" fill="#e8590c" opacity="0.85"/>
<text x="720.0" y="38" text-anchor="middle" fill="#fff" font-weight="600" font-size="12">Accelerator  student_gemm</text>
<rect x="835" y="20" width="145" height="1596" rx="8" fill="#2b8a3e" opacity="0.06"/>
<rect x="835" y="20" width="145" height="26" rx="8" fill="#2b8a3e" opacity="0.85"/>
<text x="907.5" y="38" text-anchor="middle" fill="#fff" font-weight="600" font-size="12">Memory</text>
<rect x="845" y="70" width="125" height="40" rx="5" fill="#ffc9c9" stroke="#2b8a3e" stroke-width="1"/>
<text x="907" y="86" text-anchor="middle" font-size="10">dav2_go flag</text>
<text x="907" y="101" text-anchor="middle" font-size="10">(BRAM)</text>
<rect x="845" y="120" width="125" height="40" rx="5" fill="#fff3bf" stroke="#2b8a3e" stroke-width="1"/>
<text x="907" y="136" text-anchor="middle" font-size="10">on-chip scratch</text>
<text x="907" y="151" text-anchor="middle" font-size="10">(BRAM, 40 kB)</text>
<rect x="845" y="200" width="125" height="40" rx="5" fill="#c3fae8" stroke="#2b8a3e" stroke-width="1"/>
<text x="907" y="216" text-anchor="middle" font-size="10">DDR3 weights</text>
<text x="907" y="231" text-anchor="middle" font-size="10">0x8000_0000</text>
<rect x="845" y="260" width="125" height="40" rx="5" fill="#d0ebff" stroke="#2b8a3e" stroke-width="1"/>
<text x="907" y="276" text-anchor="middle" font-size="10">DDR3 image</text>
<text x="907" y="291" text-anchor="middle" font-size="10">0x81E0_0000</text>
<rect x="845" y="320" width="125" height="40" rx="5" fill="#ffe8cc" stroke="#2b8a3e" stroke-width="1"/>
<text x="907" y="336" text-anchor="middle" font-size="10">DDR3 result</text>
<text x="907" y="351" text-anchor="middle" font-size="10">0x81F1_0000</text>
<rect x="845" y="380" width="125" height="40" rx="5" fill="#e5dbff" stroke="#2b8a3e" stroke-width="1"/>
<text x="907" y="396" text-anchor="middle" font-size="10">DDR3 arena</text>
<text x="907" y="411" text-anchor="middle" font-size="10">0x8200_0000</text>
<rect x="845" y="470" width="125" height="40" rx="5" fill="#ffd8a8" stroke="#2b8a3e" stroke-width="1"/>
<text x="907" y="486" text-anchor="middle" font-size="10">accelerator tile RAM</text>
<text x="907" y="501" text-anchor="middle" font-size="10">(256 kB)</text>
<text x="907" y="540" text-anchor="middle" font-size="10" font-style="italic">tags on each step:</text>
<text x="907" y="554" text-anchor="middle" font-size="10" font-style="italic">R = reads, W = writes</text>
<text x="907" y="580" text-anchor="middle" font-size="10">weights: 25 MB int8</text>
<text x="907" y="594" text-anchor="middle" font-size="10">image: 95 kB int16</text>
<text x="907" y="608" text-anchor="middle" font-size="10">result: 63 kB float32</text>
<text x="907" y="622" text-anchor="middle" font-size="10">arena: activations,</text>
<text x="907" y="636" text-anchor="middle" font-size="10">accumulators, tables</text>
<text x="907" y="650" text-anchor="middle" font-size="10">(peak 15 MB of 64)</text>
<rect x="30" y="70" width="130" height="83" rx="6" fill="#fff" stroke="#868e96" stroke-width="1.6"/>
<circle cx="30" cy="70" r="11" fill="#868e96"/>
<text x="30" y="74" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">1</text>
<text x="44" y="86" font-weight="600" font-size="11.5">load weights</text>
<text x="40" y="100" font-size="10.5">25 MB blob over JTAG,</text>
<text x="40" y="113" font-size="10.5">once per session (~66 s);</text>
<text x="40" y="126" font-size="10.5">verify sampled words</text>
<rect x="40" y="132" width="46" height="13" rx="3" fill="#c3fae8" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="44" y="142" font-size="9.5">W blob</text>
<rect x="195" y="70" width="170" height="83" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="195" cy="70" r="11" fill="#1c7ed6"/>
<text x="195" y="74" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">2</text>
<text x="209" y="86" font-weight="600" font-size="11.5">boot, self-test</text>
<text x="205" y="100" font-size="10.5">DDR3 init; accelerator</text>
<text x="205" y="113" font-size="10.5">self-test 70×64×6 vs CPU;</text>
<text x="205" y="126" font-size="10.5">wait for the go flag</text>
<rect x="205" y="132" width="46" height="13" rx="3" fill="#ffc9c9" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="209" y="142" font-size="9.5">R flag</text>
<line x1="160" y1="100" x2="195" y2="100" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<line x1="365" y1="100" x2="400" y2="100" stroke="currentColor" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fa)"/>
<text x="383" y="92" text-anchor="middle" font-size="10" fill="currentColor">GO_MAGIC</text>
<text x="410" y="104" font-size="10">→ prints DAV2_READY</text>
<rect x="25" y="162" width="800" height="1011" rx="10" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="8 4"/>
<text x="35" y="178" font-size="11" font-weight="600">per frame (14.2 s on the board)</text>
<rect x="30" y="195" width="130" height="83" rx="6" fill="#fff" stroke="#868e96" stroke-width="1.6"/>
<circle cx="30" cy="195" r="11" fill="#868e96"/>
<text x="30" y="199" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">3</text>
<text x="44" y="211" font-weight="600" font-size="11.5">send image</text>
<text x="40" y="225" font-size="10.5">126×126×3 int16 + scale</text>
<text x="40" y="238" font-size="10.5">→ 0x81E0_0000 (0.3 s);</text>
<text x="40" y="251" font-size="10.5">write dav2_go = GO_FRAME</text>
<rect x="40" y="257" width="52" height="13" rx="3" fill="#d0ebff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="44" y="267" font-size="9.5">W image</text>
<rect x="96.4" y="257" width="46" height="13" rx="3" fill="#ffc9c9" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="100.4" y="267" font-size="9.5">W flag</text>
<rect x="195" y="195" width="210" height="83" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="195" cy="195" r="11" fill="#1c7ed6"/>
<text x="195" y="199" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">4</text>
<text x="209" y="211" font-weight="600" font-size="11.5">patch embedding</text>
<text x="205" y="225" font-size="10.5">im2col: 81 patches × 588</text>
<text x="205" y="238" font-size="10.5">then GEMM + requant (→ steps A, B)</text>
<text x="205" y="251" font-size="10.5">+ position embedding, class token</text>
<rect x="205" y="257" width="52" height="13" rx="3" fill="#d0ebff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="209" y="267" font-size="9.5">R image</text>
<rect x="261.4" y="257" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="265.4" y="267" font-size="9.5">W arena</text>
<line x1="160" y1="232" x2="195" y2="232" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<rect x="190" y="290" width="630" height="636" rx="8" fill="none" stroke="#1c7ed6" stroke-width="1.2"/>
<text x="200" y="306" font-size="11" font-weight="600" fill="#1c7ed6">5  transformer block, × 12  (0.72 s each) — x is the 82 × 384 token matrix</text>
<rect x="200" y="318" width="190" height="86" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="200" cy="318" r="11" fill="#1c7ed6"/>
<text x="200" y="322" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">5a</text>
<text x="214" y="334" font-weight="600" font-size="11.5">LayerNorm 1 (CPU)</text>
<text x="210" y="348" font-size="10.5">row stats in float, elements</text>
<text x="210" y="361" font-size="10.5">fixed point; x → n</text>
<rect x="210" y="367" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="214" y="377" font-size="9.5">R arena</text>
<rect x="266.4" y="367" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="270.4" y="377" font-size="9.5">W arena</text>
<rect x="210" y="383" width="65" height="13" rx="3" fill="#fff3bf" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="214" y="393" font-size="9.5">R scratch</text>
<rect x="405" y="318" width="190" height="70" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="405" cy="318" r="11" fill="#e8590c"/>
<text x="405" y="322" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">5b</text>
<text x="419" y="334" font-weight="600" font-size="11.5">qkv = n · Wqkvᵀ</text>
<text x="415" y="348" font-size="10.5">GEMM 82×384×1152 (→ A)</text>
<text x="415" y="361" font-size="10.5">requant (→ B)</text>
<rect x="415" y="367" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="419" y="377" font-size="9.5">R arena</text>
<rect x="471.4" y="367" width="46" height="13" rx="3" fill="#c3fae8" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="475.4" y="377" font-size="9.5">R blob</text>
<rect x="521.6" y="367" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="525.6" y="377" font-size="9.5">W arena</text>
<rect x="200" y="404" width="610" height="270" rx="6" fill="none" stroke="#e8590c" stroke-width="1" stroke-dasharray="4 3"/>
<text x="210" y="418" font-size="10.5" font-weight="600" fill="#e8590c">5c  attention, × 6 heads</text>
<rect x="210" y="428" width="180" height="99" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="210" cy="428" r="11" fill="#1c7ed6"/>
<text x="210" y="432" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">i</text>
<text x="224" y="444" font-weight="600" font-size="11.5">gather + split (CPU)</text>
<text x="220" y="458" font-size="10.5">q,k,v of the head → scratch;</text>
<text x="220" y="471" font-size="10.5">k16, q>>4, q&amp;15 → arena;</text>
<text x="220" y="484" font-size="10.5">vᵀ>>6, vᵀ&amp;63 → arena</text>
<rect x="220" y="490" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="224" y="500" font-size="9.5">R arena</text>
<rect x="276.4" y="490" width="65" height="13" rx="3" fill="#fff3bf" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="280.4" y="500" font-size="9.5">W scratch</text>
<rect x="220" y="506" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="224" y="516" font-size="9.5">W arena</text>
<rect x="405" y="428" width="190" height="70" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="405" cy="428" r="11" fill="#e8590c"/>
<text x="405" y="432" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">ii</text>
<text x="419" y="444" font-weight="600" font-size="11.5">S = k · qᵀ  (2 jobs)</text>
<text x="415" y="458" font-size="10.5">16·(k·q_hi) + (k·q_lo), exact</text>
<text x="415" y="471" font-size="10.5">int32 scores 82×82</text>
<rect x="415" y="477" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="419" y="487" font-size="9.5">R arena</text>
<rect x="471.4" y="477" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="475.4" y="487" font-size="9.5">W arena</text>
<rect x="610" y="428" width="190" height="86" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="610" cy="428" r="11" fill="#1c7ed6"/>
<text x="610" y="432" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">iii</text>
<text x="624" y="444" font-weight="600" font-size="11.5">softmax (CPU)</text>
<text x="620" y="458" font-size="10.5">Q15 probabilities per row,</text>
<text x="620" y="471" font-size="10.5">row sums → P table</text>
<rect x="620" y="477" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="624" y="487" font-size="9.5">R arena</text>
<rect x="676.4" y="477" width="65" height="13" rx="3" fill="#fff3bf" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="680.4" y="487" font-size="9.5">R scratch</text>
<rect x="620" y="493" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="624" y="503" font-size="9.5">W arena</text>
<rect x="405" y="532" width="190" height="70" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="405" cy="532" r="11" fill="#e8590c"/>
<text x="405" y="536" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">iv</text>
<text x="419" y="548" font-weight="600" font-size="11.5">C = P · vᵀ  (2 jobs)</text>
<text x="415" y="562" font-size="10.5">64·(P·v_hi) + (P·v_lo)</text>
<text x="415" y="575" font-size="10.5">int32 context 64×82</text>
<rect x="415" y="581" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="419" y="591" font-size="9.5">R arena</text>
<rect x="471.4" y="581" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="475.4" y="591" font-size="9.5">W arena</text>
<rect x="610" y="532" width="190" height="86" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="610" cy="532" r="11" fill="#1c7ed6"/>
<text x="610" y="536" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">v</text>
<text x="624" y="548" font-weight="600" font-size="11.5">normalise (CPU)</text>
<text x="620" y="562" font-size="10.5">÷ row sum via reciprocal,</text>
<text x="620" y="575" font-size="10.5">exact; → ctx head slice</text>
<rect x="620" y="581" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="624" y="591" font-size="9.5">R arena</text>
<rect x="676.4" y="581" width="65" height="13" rx="3" fill="#fff3bf" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="680.4" y="591" font-size="9.5">W scratch</text>
<rect x="620" y="597" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="624" y="607" font-size="9.5">W arena</text>
<line x1="390" y1="459" x2="405" y2="459" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<line x1="595" y1="459" x2="610" y2="459" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<path d="M705 514 L705 524 L500 524 L500 532" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<line x1="595" y1="563" x2="610" y2="563" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<rect x="200" y="684" width="190" height="70" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="200" cy="684" r="11" fill="#e8590c"/>
<text x="200" y="688" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">5d</text>
<text x="214" y="700" font-weight="600" font-size="11.5">attn = ctx · Wprojᵀ</text>
<text x="210" y="714" font-size="10.5">GEMM 82×384×384 (→ A)</text>
<text x="210" y="727" font-size="10.5">requant (→ B)</text>
<rect x="210" y="733" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="214" y="743" font-size="9.5">R arena</text>
<rect x="266.4" y="733" width="46" height="13" rx="3" fill="#c3fae8" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="270.4" y="743" font-size="9.5">R blob</text>
<rect x="316.59999999999997" y="733" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="320.59999999999997" y="743" font-size="9.5">W arena</text>
<rect x="405" y="684" width="190" height="70" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="405" cy="684" r="11" fill="#1c7ed6"/>
<text x="405" y="688" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">5e</text>
<text x="419" y="700" font-weight="600" font-size="11.5">x = x + attn  (CPU)</text>
<text x="415" y="714" font-size="10.5">range scan, two multipliers,</text>
<text x="415" y="727" font-size="10.5">two int16 per word</text>
<rect x="415" y="733" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="419" y="743" font-size="9.5">R arena</text>
<rect x="471.4" y="733" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="475.4" y="743" font-size="9.5">W arena</text>
<rect x="200" y="760" width="190" height="57" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="200" cy="760" r="11" fill="#1c7ed6"/>
<text x="200" y="764" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">5f</text>
<text x="214" y="776" font-weight="600" font-size="11.5">LayerNorm 2 (CPU)</text>
<text x="210" y="790" font-size="10.5">x → n2</text>
<rect x="210" y="796" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="214" y="806" font-size="9.5">R arena</text>
<rect x="266.4" y="796" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="270.4" y="806" font-size="9.5">W arena</text>
<rect x="405" y="760" width="190" height="70" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="405" cy="760" r="11" fill="#e8590c"/>
<text x="405" y="764" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">5g</text>
<text x="419" y="776" font-weight="600" font-size="11.5">h = n2 · Wfc1ᵀ</text>
<text x="415" y="790" font-size="10.5">GEMM 82×384×1536 (→ A)</text>
<text x="415" y="803" font-size="10.5">requant (→ B)</text>
<rect x="415" y="809" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="419" y="819" font-size="9.5">R arena</text>
<rect x="471.4" y="809" width="46" height="13" rx="3" fill="#c3fae8" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="475.4" y="819" font-size="9.5">R blob</text>
<rect x="521.6" y="809" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="525.6" y="819" font-size="9.5">W arena</text>
<rect x="610" y="760" width="190" height="70" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="610" cy="760" r="11" fill="#1c7ed6"/>
<text x="610" y="764" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">5h</text>
<text x="624" y="776" font-weight="600" font-size="11.5">GELU (CPU)</text>
<text x="620" y="790" font-size="10.5">257-entry table, linear</text>
<text x="620" y="803" font-size="10.5">interpolation, in place</text>
<rect x="620" y="809" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="624" y="819" font-size="9.5">R arena</text>
<rect x="676.4" y="809" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="680.4" y="819" font-size="9.5">W arena</text>
<rect x="200" y="836" width="190" height="70" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="200" cy="836" r="11" fill="#e8590c"/>
<text x="200" y="840" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">5i</text>
<text x="214" y="852" font-weight="600" font-size="11.5">h2 = h · Wfc2ᵀ</text>
<text x="210" y="866" font-size="10.5">GEMM 82×1536×384 (→ A)</text>
<text x="210" y="879" font-size="10.5">requant (→ B)</text>
<rect x="210" y="885" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="214" y="895" font-size="9.5">R arena</text>
<rect x="266.4" y="885" width="46" height="13" rx="3" fill="#c3fae8" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="270.4" y="895" font-size="9.5">R blob</text>
<rect x="316.59999999999997" y="885" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="320.59999999999997" y="895" font-size="9.5">W arena</text>
<rect x="405" y="836" width="190" height="57" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="405" cy="836" r="11" fill="#1c7ed6"/>
<text x="405" y="840" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">5j</text>
<text x="419" y="852" font-weight="600" font-size="11.5">x = x + h2  (CPU)</text>
<text x="415" y="866" font-size="10.5">as 5e</text>
<rect x="415" y="872" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="419" y="882" font-size="9.5">R arena</text>
<rect x="471.4" y="872" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="475.4" y="882" font-size="9.5">W arena</text>
<rect x="610" y="836" width="190" height="70" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="610" cy="836" r="11" fill="#1c7ed6"/>
<text x="610" y="840" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">5k</text>
<text x="624" y="852" font-weight="600" font-size="11.5">feature tap (CPU)</text>
<text x="620" y="866" font-size="10.5">after blocks 4, 7, 10, 12:</text>
<text x="620" y="879" font-size="10.5">LayerNorm → feats[i]</text>
<rect x="620" y="885" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="624" y="895" font-size="9.5">R arena</text>
<rect x="676.4" y="885" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="680.4" y="895" font-size="9.5">W arena</text>
<text x="200" y="922" font-size="10" fill="#1c7ed6">order: 5a → 5b → 5c → 5d → 5e → 5f → 5g → 5h → 5i → 5j (→ 5k). The CPU polls STATUS during every accelerator job.</text>
<rect x="195" y="954" width="610" height="100" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="195" cy="954" r="11" fill="#1c7ed6"/>
<text x="195" y="958" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">6</text>
<text x="209" y="970" font-weight="600" font-size="11.5">DPT head  (5 s) — the decoder, four feature levels → one depth map</text>
<text x="205" y="984" font-size="10.5">6a  per level: 1×1 conv (GEMM) → resize: conv-transpose ×4 / ×2 (GEMM + scatter), identity, or 3×3 stride-2 conv (K = 3456: two</text>
<text x="205" y="997" font-size="10.5">      column-slice GEMMs, partial sums added on the CPU) → 3×3 conv (im2col on the CPU, then GEMM)</text>
<text x="205" y="1010" font-size="10.5">6b  fusion, coarse to fine: ResidualConvUnit = ReLU → 3×3 conv → ReLU → 3×3 conv → add; bilinear ×2 (CPU); 1×1 conv</text>
<text x="205" y="1023" font-size="10.5">6c  output: 3×3 conv → bilinear to 126×126 (CPU) → 3×3 conv → ReLU → 1×1 conv → ReLU → float depth</text>
<text x="205" y="1046" font-size="10">every conv is im2col (CPU, gathers patches) + GEMM + requant (→ A, B); 2 275 accelerator jobs per frame in total</text>
<rect x="195" y="1071" width="210" height="83" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="195" cy="1071" r="11" fill="#1c7ed6"/>
<text x="195" y="1075" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">7</text>
<text x="209" y="1087" font-weight="600" font-size="11.5">publish result</text>
<text x="205" y="1101" font-size="10.5">depth 126×126 float32 →</text>
<text x="205" y="1114" font-size="10.5">0x81F1_0000; print profile;</text>
<text x="205" y="1127" font-size="10.5">dav2_go = GO_DONE</text>
<rect x="205" y="1133" width="59" height="13" rx="3" fill="#ffe8cc" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="209" y="1143" font-size="9.5">W result</text>
<rect x="267.6" y="1133" width="46" height="13" rx="3" fill="#ffc9c9" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="271.6" y="1143" font-size="9.5">W flag</text>
<rect x="30" y="1071" width="130" height="83" rx="6" fill="#fff" stroke="#868e96" stroke-width="1.6"/>
<circle cx="30" cy="1071" r="11" fill="#868e96"/>
<text x="30" y="1075" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">8</text>
<text x="44" y="1087" font-weight="600" font-size="11.5">read result</text>
<text x="40" y="1101" font-size="10.5">dump_image 63 kB over</text>
<text x="40" y="1114" font-size="10.5">JTAG (0.7 s) → .npy;</text>
<text x="40" y="1127" font-size="10.5">compare with host build</text>
<rect x="40" y="1133" width="59" height="13" rx="3" fill="#ffe8cc" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="44" y="1143" font-size="9.5">R result</text>
<rect x="102.6" y="1133" width="46" height="13" rx="3" fill="#ffc9c9" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="106.6" y="1143" font-size="9.5">R flag</text>
<line x1="195" y1="1102" x2="160" y2="1102" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<path d="M95 1149 L95 1164 L27 1164 L27 232 L30 232" fill="none" stroke="currentColor" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fa)"/>
<text x="105" y="1161" font-size="10">next image → step 3</text>
<rect x="190" y="1186" width="630" height="420" rx="8" fill="none" stroke="#e8590c" stroke-width="1.2"/>
<text x="200" y="1202" font-size="11" font-weight="600" fill="#e8590c">A  one GEMM on the accelerator — what "GEMM" means in every step above</text>
<rect x="200" y="1214" width="180" height="70" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="200" cy="1214" r="11" fill="#1c7ed6"/>
<text x="200" y="1218" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">A1</text>
<text x="214" y="1230" font-weight="600" font-size="11.5">CPU programs the job</text>
<text x="210" y="1244" font-size="10.5">A, W, C addresses + strides;</text>
<text x="210" y="1257" font-size="10.5">K, M, rows; S_ADDR for stats;</text>
<text x="210" y="1270" font-size="10.5">CTRL.start — then polls STATUS</text>
<rect x="405" y="1214" width="190" height="83" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="405" cy="1214" r="11" fill="#e8590c"/>
<text x="405" y="1218" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">A2</text>
<text x="419" y="1230" font-weight="600" font-size="11.5">load A tile</text>
<text x="415" y="1244" font-size="10.5">64 rows × K int16 → tile RAM;</text>
<text x="415" y="1257" font-size="10.5">8 reads in flight, reorder buffer;</text>
<text x="415" y="1270" font-size="10.5">a lost response is re-issued</text>
<rect x="415" y="1276" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="419" y="1286" font-size="9.5">R arena</text>
<rect x="471.4" y="1276" width="46" height="13" rx="3" fill="#ffd8a8" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="475.4" y="1286" font-size="9.5">W tile</text>
<rect x="610" y="1214" width="195" height="83" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="610" cy="1214" r="11" fill="#e8590c"/>
<text x="610" y="1218" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">A3</text>
<text x="624" y="1230" font-weight="600" font-size="11.5">stream W, MAC</text>
<text x="620" y="1244" font-size="10.5">each word = 4 int8 weights,</text>
<text x="620" y="1257" font-size="10.5">broadcast to 64 multipliers;</text>
<text x="620" y="1270" font-size="10.5">64 MACs per cycle, K cycles per row</text>
<rect x="620" y="1276" width="46" height="13" rx="3" fill="#c3fae8" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="624" y="1286" font-size="9.5">R blob</text>
<rect x="670.2" y="1276" width="46" height="13" rx="3" fill="#ffd8a8" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="674.2" y="1286" font-size="9.5">R tile</text>
<line x1="380" y1="1249" x2="405" y2="1249" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<line x1="595" y1="1249" x2="610" y2="1249" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<rect x="405" y="1300" width="190" height="83" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="405" cy="1300" r="11" fill="#e8590c"/>
<text x="405" y="1304" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">A4</text>
<text x="419" y="1316" font-weight="600" font-size="11.5">drain row m</text>
<text x="415" y="1330" font-size="10.5">64 int32 → C column m;</text>
<text x="415" y="1343" font-size="10.5">then {max, min} of the row</text>
<text x="415" y="1356" font-size="10.5">→ S_ADDR + 8m</text>
<rect x="415" y="1362" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="419" y="1372" font-size="9.5">W arena</text>
<rect x="610" y="1300" width="195" height="62" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="610" cy="1300" r="11" fill="#e8590c"/>
<text x="610" y="1304" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">A5</text>
<text x="624" y="1316" font-weight="600" font-size="11.5">next row / next tile</text>
<text x="620" y="1330" font-size="10.5">m + 1 → A3 (accumulators clear)</text>
<text x="620" y="1343" font-size="10.5">all M rows done → status DONE;</text>
<text x="620" y="1356" font-size="10.5">CPU starts the next 64-row tile → A2</text>
<path d="M707 1284 L707 1300" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<line x1="595" y1="1331" x2="610" y2="1331" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<path d="M805 1320 L815 1320 L815 1240 L805 1240" fill="none" stroke="currentColor" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fa)"/>
<path d="M392 1370 h6 v48 h-6" fill="none" stroke="#e8590c" stroke-width="1.6"/>
<text x="402" y="1398.0" font-size="10" fill="#e8590c">concurrent inside the block: read requests in flight ∥ multiply-accumulate ∥ up to 8 writes awaiting ack</text>
<text x="200" y="1310" font-size="10" fill="#1c7ed6">meanwhile the CPU only</text>
<text x="200" y="1323" font-size="10" fill="#1c7ed6">polls STATUS (nothing</text>
<text x="200" y="1336" font-size="10" fill="#1c7ed6">is overlapped with it yet)</text>
<text x="200" y="1440" font-size="11" font-weight="600" fill="#e8590c">B  requantisation — int32 sums back to int16 activations, after every GEMM</text>
<rect x="200" y="1452" width="180" height="83" rx="6" fill="#fff" stroke="#1c7ed6" stroke-width="1.6"/>
<circle cx="200" cy="1452" r="11" fill="#1c7ed6"/>
<text x="200" y="1456" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">B1</text>
<text x="214" y="1468" font-weight="600" font-size="11.5">per-row parameters (CPU)</text>
<text x="210" y="1482" font-size="10.5">from the {max,min} per row: output</text>
<text x="210" y="1495" font-size="10.5">scale (float, one per tensor), then</text>
<text x="210" y="1508" font-size="10.5">{mult, shift, bias} per row → table</text>
<rect x="210" y="1514" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="214" y="1524" font-size="9.5">R arena</text>
<rect x="266.4" y="1514" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="270.4" y="1524" font-size="9.5">W arena</text>
<rect x="405" y="1452" width="190" height="83" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="405" cy="1452" r="11" fill="#e8590c"/>
<text x="405" y="1456" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">B2</text>
<text x="419" y="1468" font-weight="600" font-size="11.5">load chunk, transposed</text>
<text x="415" y="1482" font-size="10.5">≤1024 rows × 64 cols of acc[m][n]</text>
<text x="415" y="1495" font-size="10.5">→ tile RAM row n, word m;</text>
<text x="415" y="1508" font-size="10.5">parameter table → param RAM</text>
<rect x="415" y="1514" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="419" y="1524" font-size="9.5">R arena</text>
<rect x="471.4" y="1514" width="46" height="13" rx="3" fill="#ffd8a8" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="475.4" y="1524" font-size="9.5">W tile</text>
<rect x="610" y="1452" width="195" height="83" rx="6" fill="#fff" stroke="#e8590c" stroke-width="1.6"/>
<circle cx="610" cy="1452" r="11" fill="#e8590c"/>
<text x="610" y="1456" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">B3</text>
<text x="624" y="1468" font-weight="600" font-size="11.5">stream out</text>
<text x="620" y="1482" font-size="10.5">7-stage pipeline: ×mult, round,</text>
<text x="620" y="1495" font-size="10.5">>>shift, +bias, saturate ±8191;</text>
<text x="620" y="1508" font-size="10.5">2 int16 per word → out[n][m]</text>
<rect x="620" y="1514" width="46" height="13" rx="3" fill="#ffd8a8" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="624" y="1524" font-size="9.5">R tile</text>
<rect x="670.2" y="1514" width="52" height="13" rx="3" fill="#e5dbff" stroke="#2b8a3e" stroke-width="0.6"/>
<text x="674.2" y="1524" font-size="9.5">W arena</text>
<line x1="380" y1="1487" x2="405" y2="1487" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<line x1="595" y1="1487" x2="610" y2="1487" stroke="currentColor" stroke-width="1.4" marker-end="url(#fa)"/>
<path d="M392 1532 h6 v20 h-6" fill="none" stroke="#e8590c" stroke-width="1.6"/>
<text x="402" y="1546.0" font-size="10" fill="#e8590c">B3: element pipeline ∥ 16-entry FIFO ∥ write engine</text>
</svg>
<figcaption>One frame, start to finish. Steps 1–2 happen once per session; 3–8 once per image. The transformer block (5) runs twelve times, the attention sub-flow (5c) six times per block. The A and B expansions are what every "GEMM" and "requant" in the upper part stands for.</figcaption>
</figure>

Reading it in simple terms: the PC puts the picture in memory and raises a
flag (3). The CPU cuts the picture into 81 patches and turns each into a
384-number vector (4). Twelve times over (5), the CPU normalises those
vectors, the accelerator multiplies them by learned weights, the CPU lets
every patch look at every other (attention, with the two big products on the
accelerator), and so on through the block's four multiplications. The decoder
(6) turns the transformer's features back into a 126×126 picture of depth,
with the same tools: convolutions become matrix products on the accelerator,
the CPU gathers patches and resizes. The CPU writes the depth map to a fixed
address and lowers the flag (7); the PC reads it back (8).

A PNG of the same figure is at `img/flow.png`.

---

## 4. The write-back bug, cycle by cycle

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

## 5. How a stalled bus was diagnosed without halting the CPU

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
