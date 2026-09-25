// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// GEMM accelerator for the Depth-Anything V2 engine.
// -------------------------------------------------------------------------
//
// The network spends >98% of its time in dav2_qgemm(), a matrix multiply:
//
//   C[m][t] = sum_k A[t][k] * W[m][k]      A: int16, W: int8, C: int32
//
// The CPU (CV32E40P, a plain scalar core, no SIMD) would need several cycles
// per multiply-accumulate. This block instead does NROWS of them per cycle:
// it keeps NROWS rows of A resident on-chip and streams the much larger W
// past them once.
//
// One job (one NROWS-row tile of A against all of W) has three phases:
//
//   1. LOAD_A  Read the A tile (NROWS rows x K int16) from memory into
//              NROWS private block RAMs, one per row.
//   2. MAC     Stream W as a flat byte sequence. Each 32-bit beat carries
//              four int8 weights; each weight is broadcast to all NROWS
//              multipliers, which each pair it with their own A row. One
//              weight row m takes K cycles and yields NROWS int32 sums.
//   3. DRAIN   Write row m's NROWS accumulators to memory as one run, then
//              start the next m.
//
// A, W and C are all read/written sequentially, which suits the direct-
// mapped DDR3 cache in front of them.
//
// Two extra features ride on that same loop:
//
//   * Row statistics. If S_ADDR is set, DRAIN also writes each row's
//     {max, min} to S_ADDR + m*8. Requantisation (below) needs exactly that
//     range; without this it had to re-read the whole result matrix to find
//     it.
//
//   * Gather mode (CTRL.gather). A convolution is a GEMM whose A matrix is
//     the "im2col" patch matrix: for every output pixel, the k x k input
//     pixels under the kernel laid end to end, zeros where the kernel hangs
//     over the image edge. Software used to build that matrix in DDR3 (36 MB
//     of copies per frame). In gather mode the A-load walks the output
//     pixels and kernel positions itself, reads each in-bounds run of C
//     channels straight from the NHWC image, and writes zeros into the tile
//     RAM for the out-of-bounds ones -- so the tile RAM ends up holding
//     exactly the patch matrix rows, and nothing was copied.
//
//   * Requantisation job (CTRL.requant). A second mode that turns the int32
//     result acc[m][n] into int16 activations out[n][m], applying a
//     per-row multiplier/shift/bias — the same arithmetic the C code uses,
//     bit for bit. This used to cost ~70 CPU cycles per element (a third of
//     the frame); doing it here reads a chunk of the result transposed into
//     the tile RAM (RAM row = n, word = m, which produces n-major output for
//     free) and streams two int16 out per cycle.
//
// The bus, not the multiplier array, is the bottleneck: one 32-bit beat
// feeds 4*NROWS MACs, so at NROWS=64 (the board setting, see student.sv)
// the array is starved unless reads are pipelined. To hide DDR3 latency the
// read side keeps up to OUTSTANDING requests in flight and reassembles
// their responses in order with a small reorder buffer indexed by the bus
// source ID.
//
// The register map and software contract are in
// src/design/reggen/student_gemm.hjson.
//
// Where this fits
// ---------------
//
//   src/rtl/student/student.sv           instantiates this block: register
//                                        window + host bus port
//   src/design/reggen/student_gemm.hjson the register map (generates
//                                        student_gemm_reg_top / _reg_pkg)
//   src/sw/project/dav2_accel.c          the driver: splits a GEMM into
//                                        NROWS-row jobs, polls STATUS
//   src/tb/student_gemm_tb.sv            module test, ideal memory
//   src/tb/student_gemm_ddrpath_tb.sv    same, through the real DDR3 cache
//   src/tb/student_gemm_droprsp_tb.sv    proves the lost-response retry works
//
// Vocabulary: a "beat" is one 32-bit TL-UL transfer; a "tile" is the NROWS
// rows of A a job processes; "MAC" is one multiply-accumulate;
// "requantisation" scales int32 accumulators back down to int16 activations.

module student_gemm #(
  // Activation rows held in the tile == multipliers == MACs per cycle.
  parameter int unsigned NROWS       = 16,
  // Largest supported reduction length K (the model needs 1536).
  parameter int unsigned KMAX        = 2048,
  // Read requests in flight (also the write-ack credit). Power of two.
  parameter int unsigned OUTSTANDING = 8,
  // Reads actually allowed in flight at once (<= OUTSTANDING). The rvlab
  // DDR3 cache pulses d_valid for one cycle without checking d_ready, so a
  // response arriving while this block is busy elsewhere is simply lost.
  // RETRY_CYCLES below recovers from that, so student.sv now runs this at
  // the full OUTSTANDING depth (8) instead of forcing 1 as an earlier
  // version did. Against ideal BRAM (no such loss) it also just wants the
  // full depth.
  parameter int unsigned MAX_INFLIGHT = OUTSTANDING,
  // How long an outstanding read may go unanswered before it is re-issued.
  // Exists because of the same dropped-response behaviour: measured on
  // hardware as accepted=444, responses=443 for one job. Zero disables the
  // recovery.
  // Must comfortably outlast real memory latency so it never fires on a
  // transaction that is merely slow. 2048 cycles = 41 us at 50 MHz; hardware
  // shows 8.3 cycles/beat in normal operation, so this is a ~250x margin.
  // (The original guess of 65536, made before latency could be measured,
  // cost 25.1M cycles per tile -- 98% of tile time -- idling before re-issue.)
  parameter int unsigned RETRY_CYCLES = 32'd2048,
  // Never enable: forcing reads and writes to alternate (never both
  // outstanding) deadlocks in ST_DRAIN. The read engine prefetches weight
  // beats that only ST_MAC consumes, so with no read in flight rb_cnt never
  // reaches zero, so the drain's write can never issue. Kept only so the
  // dead end stays documented instead of being rediscovered.
  parameter bit          STRICT_SERIAL = 1'b0
) (
  input logic clk_i,   // system clock; everything below is synchronous to this edge
  input logic rst_ni,  // active-low asynchronous reset

  // Register interface (device): the CPU's writes/reads of this block's own
  // registers (addresses, strides, K/M, CTRL, STATUS, ...) land here.
  input  tlul_pkg::tl_h2d_t tl_i,    // CPU -> block: register requests
  output tlul_pkg::tl_d2h_t tl_o,    // block -> CPU: register responses

  // Memory interface (host): this block acting as a bus master, reading A
  // and W and writing C directly, without the CPU's involvement.
  input  tlul_pkg::tl_d2h_t tl_host_i,  // DDR3 path -> block: read data / write acks
  output tlul_pkg::tl_h2d_t tl_host_o   // block -> DDR3 path: read/write requests
);
  import student_gemm_reg_pkg::*;

  // Derived widths
  localparam int unsigned KWORDS = KMAX / 2;              // 32-bit words per row
  localparam int unsigned AW     = $clog2(KWORDS);        // A-tile word address
  localparam int unsigned KCW    = $clog2(KMAX + 1);      // k counter
  localparam int unsigned RW     = $clog2(NROWS);         // row index
  localparam int unsigned NRW    = RW + 1;                // row count (0..NROWS)
  localparam int unsigned SW     = $clog2(OUTSTANDING);   // reorder slot index
  localparam int unsigned CW     = SW + 1;                // credit counters

  // ---------------------------------------------------------------- registers

  student_gemm_reg2hw_t reg2hw;  // register file -> this block: what the CPU wrote
  student_gemm_hw2reg_t hw2reg;  // this block -> register file: what the CPU reads back

  student_gemm_reg_top reg_top_i (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .devmode_i('1)
  );

  logic busy_q, done_q, err_q;  // STATUS bits: job running / job finished / bus error seen
  logic [31:0] cycles_q;        // cycle counter, running while busy_q, for CYCLES

  assign hw2reg.status.d = {err_q, done_q, busy_q};        // STATUS register readback
  assign hw2reg.caps.d   = {8'd63, 16'(KMAX), 8'(NROWS)};   // CAPS: what this instance supports
  assign hw2reg.cycles.d = cycles_q;                       // CYCLES register readback

  logic start_strobe, start_requant, start_gather;
  // One-cycle pulse the instant the CPU writes CTRL.start=1: the signal that
  // kicks the FSM out of ST_IDLE.
  assign start_strobe  = reg2hw.ctrl.start.qe & reg2hw.ctrl.start.q;
  // Which job to run: 1 = requantisation, 0 = a plain GEMM tile. Only
  // meaningful the same cycle as start_strobe, so it is latched below.
  assign start_requant = reg2hw.ctrl.requant.q;   // sampled with start
  assign start_gather  = reg2hw.ctrl.gather.q;
  logic start_greuse;
  assign start_greuse  = reg2hw.ctrl.greuse.q;    // gather: reuse the left taps    // GEMM job: A tile via the gather walkers
  logic start_add, start_relu;
  assign start_add     = reg2hw.ctrl.add.q;       // requant job: add a residual
  assign start_relu    = reg2hw.ctrl.relu.q;      // requant job: clamp the output at 0
  logic start_lut, start_lut_load;
  assign start_lut      = reg2hw.ctrl.lut.q;      // requant job: map the output through the LUT
  assign start_lut_load = reg2hw.ctrl.lut_load.q; // requant job: load the LUT first
  logic start_a16;
  assign start_a16      = reg2hw.ctrl.a16.q;      // requant job: int16 input
  logic start_w16;
  assign start_w16      = reg2hw.ctrl.w16.q;      // GEMM job: int16 weights
  logic start_ostats;
  assign start_ostats   = reg2hw.ctrl.ostats.q;   // requant job: output row statistics
  logic start_onchip;
  assign start_onchip   = reg2hw.ctrl.onchip.q;   // results in the on-chip RAM

  // Configuration snapshot, taken when a job starts so software may reprogram
  // the registers for the next tile while this one runs.
  logic [31:0]    w_addr_q;                  // GEMM: base address of the W matrix
  logic [31:0]    c_stride_q;                // bytes from one output row of C to the next
  logic [31:0]    w_stride_q;                // bytes between W rows, never 0 here
  logic [31:0]    s_addr_q;                  // per-row stats stream (0 = off)
  logic [31:0]    a_addr_q, a_stride_q;      // requant job: acc chunk
  logic [KCW-1:0] k_len_q;                   // GEMM: reduction length K, this job
  logic [15:0]    m_len_q;                   // number of W rows (GEMM) / chunk rows (requant)
  logic [NRW-1:0] n_rows_q;                  // number of A-tile rows actually in use (<= NROWS)
  logic           stats_en_q;                // s_addr_q != 0, GEMM jobs
  logic           gather_q;                  // A tile comes from the gather walkers
  // Requant epilogue (snapshot at start): residual add and ReLU.
  logic           add_q, relu_q;
  logic [31:0]    x_addr_q;                  // residual chunk, rows C_STRIDE apart
  logic [31:0]    add_mx_q, add_mh_q;        // multipliers for residual and value
  logic [5:0]     add_sx_q, add_sh_q;        // their shifts
  // Requant lookup table (CTRL.lut, CTRL.lut_load): out = LUT[v + 8192].
  logic           lut_q;                     // map the output through the table
  logic [31:0]    p_addr_q;                  // parameter table, read after a table load
  logic [12:0]    lut_cnt_q;                 // table load: word being filled
  logic           lut_ld_done;               // table load: the last word has arrived
  logic           a16_q;                     // requant input is int16, two per word
  logic           w16_q;                     // GEMM weights are int16, two per word
  logic           ostats_q;                  // requant: write each output row's {max, min}
  logic           onchip_q;                  // GEMM: drain to / requant: load from the result RAM
  localparam int unsigned CRW = 131072;      // result RAM words (128 BRAM36)
  localparam int unsigned CRA = $clog2(CRW);
  logic [31:0]    cr_rdata_q;                // result RAM: the word read for the load
  logic           cr_v_q;                    // cr_rdata_q holds the next input word
  logic           acc_valid;                 // requant load: an input word arrives
  // Gather parameters (snapshot at start) and the walker signals the read
  // engine and the A-load writer use; the walkers themselves are further down.
  logic [31:0]    g_addr_q;
  logic [15:0]    g_h_q, g_w_q, g_ow_q, g_c_q;
  logic [3:0]     g_k_q, g_stride_q, g_pad_q, g_ky0_q, g_kx0_q;
  logic [7:0]     g_kpos_q;
  logic [15:0]    g_cwords;                  // beats per kernel position
  logic           greuse_q;                  // CTRL.greuse, sampled at start
  logic           g_reuse_ok;                // tap reuse applies to this job
  logic           gw_inb;                    // writer: current position in bounds
  logic           gi_handover;               // issuer: a run is ready for the read engine
  logic [31:0]    gi_addr_q;                 // its address

  // ------------------------------------------------------------- control FSM

  typedef enum logic [3:0] {
    ST_IDLE,       // waiting for CTRL.start
    ST_LOAD_A,     // reading the A tile into on-chip RAM
    ST_MAC,        // streaming one row of W, multiply-accumulating into acc_q
    ST_MAC_TAIL,   // last W beat consumed; draining the MAC pipeline before DRAIN
    ST_DRAIN,      // writing this row's NROWS accumulators (+ stats) to DDR3
    ST_FINISH,     // job done; waiting for outstanding writes to be acked
    // requantisation job
    RQ_LOAD_P,     // parameter table -> param RAM
    RQ_LOAD_ACC,   // acc chunk -> tile RAM, transposed
    RQ_LOAD_X,     // residual chunk -> upper half of tile RAM (CTRL.add)
    RQ_OUT,         // stream int16 pairs out
    RQ_LOAD_L,     // lookup table -> LUT RAM (CTRL.lut_load)
    RQ_STATS      // output row statistics -> S_ADDR (CTRL.ostats)
  } state_e;

  state_e state_q, state_d;  // state_q: current state (registered); state_d: next state

  // ---------------------------------------------------------------- bus side
  //
  // One registered request is presented on the A channel at a time. A new one
  // is loaded whenever the current one has been accepted (or none is pending),
  // which keeps a_valid/a_* stable as TL-UL requires. Writes win arbitration:
  // they are rare (NROWS words per weight row) and the read stream has the
  // reorder buffer to absorb the resulting bubble.

  tlul_pkg::tl_h2d_t areq_q;      // the one A-channel request currently presented on the bus
  logic              areq_valid_q; // areq_q's a_valid bit (kept separate so areq_q itself can hold still)
  logic              load_next;    // this cycle, replace areq_q with the next request to issue
  logic              req_accepted; // this cycle's request was accepted (a_valid & a_ready)

  // Drive the host (master) port from the one registered request, always
  // ready to accept a response since every issued request already has a
  // reserved reorder-buffer or write-credit slot waiting for it.
  always_comb begin
    tl_host_o          = areq_q;
    tl_host_o.a_valid  = areq_valid_q;
    tl_host_o.d_ready  = 1'b1;   // a slot is reserved for every request issued
  end

  // Read stream ------------------------------------------------------------
  //
  // Both streams (A tile, then W) are rows of rd_row_beats_q beats each; rows
  // start rd_stride_q bytes apart. With the stride equal to the row length
  // this is one contiguous run, which is the common case.
  logic [31:0]  rd_addr_q;       // address of the next read beat to issue
  logic [31:0]  rd_left_q;      // beats not yet requested, all rows
  logic [31:0]  rd_row_base_q;  // first address of the current row
  logic [31:0]  rd_row_beats_q; // beats per row
  logic [31:0]  rd_row_left_q;  // beats left in the current row
  logic [31:0]  rd_stride_q;    // bytes from one row start to the next
  logic         rd_can_issue;   // room in the reorder buffer for one more read, and reads are wanted

  logic [31:0]  rb_data_q [OUTSTANDING]; // reorder buffer: response data, one slot per outstanding read
  logic         rb_val_q  [OUTSTANDING]; // reorder buffer: slot holds an answered (not yet popped) beat
  logic [SW-1:0] rb_wr_q;       // slot a newly issued read will land in (advances on issue_rd)
  logic [SW-1:0] rb_rd_q;       // slot the consumer reads next (advances on rd_pop)
  logic [CW-1:0] rb_cnt_q;      // issued but not yet consumed

  logic         rd_valid;       // next beat, in order, is available
  logic [31:0]  rd_data;        // that beat's data
  logic         rd_pop;         // this cycle, the consumer takes rd_data and advances rb_rd_q

  assign rd_valid     = (rb_cnt_q != '0) & rb_val_q[rb_rd_q];
  assign rd_data      = rb_data_q[rb_rd_q];

  // Write stream -----------------------------------------------------------
  // Two producers (the GEMM drain and the requant output stage) share one
  // write path; only one is active in any given state, so a plain mux picks
  // between them.
  logic         wr_req;                 // the active producer has a word ready to write
  logic [31:0]  wr_addr, wr_data;       // that word's address and data
  logic         drain_wr_req, rq_wr_req;
  logic [31:0]  drain_wr_addr, drain_wr_data, rq_wr_addr, rq_wr_data;
  logic         os_wr_req;
  logic [31:0]  os_wr_addr, os_wr_data;   // RQ_STATS: one word per output row
  assign wr_req  = (state_q == RQ_OUT) ? rq_wr_req : (state_q == RQ_STATS) ? os_wr_req : drain_wr_req;
  assign wr_addr = (state_q == RQ_OUT) ? rq_wr_addr : (state_q == RQ_STATS) ? os_wr_addr : drain_wr_addr;
  assign wr_data = (state_q == RQ_OUT) ? rq_wr_data : (state_q == RQ_STATS) ? os_wr_data : drain_wr_data;
  logic [CW-1:0] wr_out_q;      // writes issued without an ack yet
  logic [SW-1:0] wr_src_q;      // a_source to tag the next write with (cycles through OUTSTANDING slots)

  logic sel_wr, sel_rd, issue_wr, issue_rd;
  // sel_wr/sel_rd: this cycle's arbitration winner (write beats read).
  // issue_wr/issue_rd: sel_* actually turned into a bus request this cycle.

  // Lost-response recovery. Every issued read remembers its address in its
  // reorder slot; when the head slot stays empty for RETRY_CYCLES after the
  // bus has gone silent, that one request is issued again. This works for
  // any MAX_INFLIGHT: with several in flight the later responses keep
  // arriving, the head stays empty, issue stops once the slots are full, and
  // the ensuing silence trips the timer.
  logic [31:0]   rb_addr_q [OUTSTANDING]; // reorder buffer: the address each outstanding read was sent to
  logic [31:0]   retry_addr;    // address to re-issue: the oldest outstanding read's
  logic [SW-1:0] retry_slot;    // its reorder-buffer slot (same as rb_rd_q)
  assign retry_addr = rb_addr_q[rb_rd_q];
  assign retry_slot = rb_rd_q;
  logic [31:0]   retry_cnt_q;      // cycles since the last sign of life (response, issue, or empty)
  logic          retry_pending_q;  // the oldest read has been silent for RETRY_CYCLES; re-issue it
  logic          sel_retry;        // this cycle's arbitration winner is the retry (highest priority)
  logic          issue_retry;      // the retry actually went out on the bus this cycle
  logic          retry_expired;    // retry_cnt_q has reached RETRY_CYCLES
  logic [31:0]   retry_n_q;   // retries performed, for diagnostics

  assign retry_expired = (RETRY_CYCLES != 0) & (retry_cnt_q == RETRY_CYCLES);
  assign sel_retry     = retry_pending_q;
  assign issue_retry   = load_next & sel_retry;

  assign rd_can_issue = (rd_left_q != 32'd0) & (rb_cnt_q != CW'(MAX_INFLIGHT))
                        & (STRICT_SERIAL ? (wr_out_q == '0) : 1'b1);

  // STRICT_SERIAL: never have a read and a write outstanding at the same time.
  //
  // rvlab_ddr_block_cache asserts fe_rsp_o.d_valid for exactly one cycle on a
  // hit and never looks at its front-end d_ready (grep: the only d_ready uses
  // in that file are back-end). A response therefore evaporates if a FIFO
  // between the cache and this block is full that cycle -- losing exactly one,
  // which is what the board shows (accepted=444, responses=443). Keeping a
  // single transaction in flight means never two responses converging at once.
  assign sel_wr   = ~sel_retry & wr_req & (wr_out_q != CW'(MAX_INFLIGHT))
                           & (STRICT_SERIAL ? (rb_cnt_q == '0) : 1'b1);
  assign sel_rd   = ~sel_retry & ~sel_wr & rd_can_issue;
  // Only load a new request when one is actually being issued. The previous
  // formulation (~areq_valid_q | a_ready) reloaded areq_q speculatively
  // whenever nothing was pending, which changed a_address while a transaction
  // was still outstanding.
  //
  // rvlab_ddr_block_cache cannot tolerate that: its tag lookup is stall-gated
  //     tag_rdata   <= tag_mem[stall ? access_idx_q : access_idx];
  // but its dirty bit is not
  //     dirty_rdata <= dirty_mem[access_idx];        // live bus index
  // so once the address drifts mid-transaction the cache reloads the dirty
  // flag from an unrelated set and evicts the same line forever. Holding the
  // request stable until the next issue is what the CPU does, and costs
  // nothing at MAX_INFLIGHT=1.
  assign req_accepted = areq_valid_q & tl_host_i.a_ready;
  assign load_next    = (~areq_valid_q | tl_host_i.a_ready)
                        & (sel_wr | sel_rd | sel_retry);

  assign issue_wr = load_next & sel_wr;
  assign issue_rd = load_next & sel_rd;

  // Responses --------------------------------------------------------------
  logic          rsp_rd;    // this cycle's D-channel response is read data
  logic          rsp_wr;    // this cycle's D-channel response is a write acknowledgement
  logic [SW-1:0] rsp_slot;  // which reorder-buffer/write-credit slot it answers (from d_source)

  assign rsp_rd   = tl_host_i.d_valid & (tl_host_i.d_opcode == tlul_pkg::AccessAckData);
  assign rsp_wr   = tl_host_i.d_valid & (tl_host_i.d_opcode == tlul_pkg::AccessAck);
  assign rsp_slot = tl_host_i.d_source[SW-1:0];

  // Registers and issues the next A-channel request (read, write or retry),
  // whichever arbitration picked; see load_next/sel_wr/sel_rd/sel_retry above.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      areq_q       <= '0;
      areq_valid_q <= 1'b0;
    end else begin
      // Accepted and nothing new to issue: drop a_valid but keep areq_q, so
      // the address stays put until the next real request.
      if (req_accepted) areq_valid_q <= 1'b0;

      if (load_next) begin
      areq_q <= '{
        a_opcode:  sel_wr ? tlul_pkg::PutFullData : tlul_pkg::Get,
        a_param:   3'h0,
        a_size:    top_pkg::TL_SZW'(2),   // 2^2 = 4 bytes
        a_source:  sel_wr    ? top_pkg::TL_AIW'({1'b1, wr_src_q})
                 : sel_retry ? top_pkg::TL_AIW'({1'b0, retry_slot})
                             : top_pkg::TL_AIW'({1'b0, rb_wr_q}),
        a_address: sel_wr    ? wr_addr
                 : sel_retry ? retry_addr
                             : rd_addr_q,
        a_mask:    4'hf,
        a_data:    sel_wr ? wr_data : 32'd0,
        a_user:    '0,
        a_valid:   1'b0,                  // driven separately
        d_ready:   1'b1
      };
      areq_valid_q <= 1'b1;
      end
    end
  end

  // Reorder buffer. A slot can never be re-issued before its response has been
  // consumed (rb_cnt_q gates issue), so the three writers below are mutually
  // exclusive per slot.
  for (genvar s = 0; s < int'(OUTSTANDING); s++) begin : gen_rb
    // Per-slot valid bit: set when that slot's response arrives, cleared
    // when a new read is issued into it or its data is consumed.
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        rb_val_q[s] <= 1'b0;
      end else if (issue_rd && (rb_wr_q == SW'(s))) begin
        rb_val_q[s] <= 1'b0;
      end else if (rsp_rd && (rsp_slot == SW'(s))) begin
        rb_val_q[s] <= 1'b1;
      end else if (rd_pop && (rb_rd_q == SW'(s))) begin
        rb_val_q[s] <= 1'b0;
      end
    end
    // Per-slot payload: the response data, and the address it was for
    // (needed only for a retry).
    always_ff @(posedge clk_i) begin
      if (rsp_rd && (rsp_slot == SW'(s))) rb_data_q[s] <= tl_host_i.d_data;
      if (issue_rd && (rb_wr_q == SW'(s))) rb_addr_q[s] <= rd_addr_q;
    end
  end

  // --------------------------------------------------------------- A tile RAM

  logic [AW-1:0]    a_wr_addr, a_rd_addr;  // word address, shared by all NROWS row-RAMs
  logic [31:0]      a_wr_data;             // data written to whichever row a_we selects
  logic [NROWS-1:0] a_we;                  // one write-enable bit per row RAM
  logic [31:0]      a_q [NROWS];           // read output of each row RAM, one cycle after a_rd_addr

  // NROWS independent single-port RAMs, one per activation row of the tile,
  // so all of them can be read together every cycle during the MAC phase.
  for (genvar r = 0; r < int'(NROWS); r++) begin : gen_arow
    (* ram_style = "block" *) logic [31:0] mem [KWORDS];
    always_ff @(posedge clk_i) begin
      if (a_we[r]) mem[a_wr_addr] <= a_wr_data;
      a_q[r] <= mem[a_rd_addr];
    end
  end

  // ------------------------------------------------------------- A load phase

  logic [AW-1:0]  a_ld_word_q;  // word index inside the current row
  logic [NRW-1:0] a_ld_row_q;   // which tile row is currently being filled
  logic [AW-1:0] k_words;       // 32-bit words per activation row

  assign k_words = AW'(k_len_q >> 1);

  // Requantisation: the acc chunk arrives row m by row m (n inner), and goes
  // into RAM row n at word m, so that afterwards RAM row n holds out row n.
  logic [AW-1:0]  rq_m_q;       // load: current acc row m; out: current word
  logic [NRW-1:0] rq_n_q;       // load: current column n;  out: current row
  logic [AW-1:0]  rq_mlen;      // rows m in this chunk (M_LEN)
  assign rq_mlen = AW'(m_len_q);
  // Residual chunk: in add mode an acc chunk is at most KWORDS/2 rows, so the
  // upper half of every tile-RAM row is free for that row's residual words.
  localparam logic [AW-1:0] XOFF = AW'(KWORDS / 2);
  logic [AW-1:0]  rq_xw_q;      // residual load: word within the row
  logic [AW-1:0]  rq_xwords;    // residual words per row = M_LEN/2
  logic           rq_x_done;    // residual chunk fully loaded
  assign rq_xwords = AW'(m_len_q >> 1);
  logic [1:0]     rq_p_sel_q;   // param load: which of the three words is arriving
  logic [AW-1:0]  rq_p_cnt_q;   // param load: row m being filled
  logic           rq_p_done;    // the {mult,shift,bias} table has been fully loaded
  logic           rq_acc_done;  // the acc chunk has been fully loaded, transposed, into tile RAM
  logic           rq_out_done;  // every output word has been streamed out and acknowledged
  logic           rq_adv;       // this cycle, the output pipeline consumes one more element

  // A-tile RAM write side: shared between loading a GEMM's A tile
  // (ST_LOAD_A, rows in order) and loading a requant chunk transposed
  // (RQ_LOAD_ACC, row = n instead of the arrival order m).
  // The word being written into the tile during LOAD_A: a bus beat, or a
  // zero when the gather walker is over an out-of-bounds kernel position.
  logic        ld_valid;
  logic [31:0] ld_data;
  // Gather: a word comes from the bus (through the FIFO gf_*) for an
  // in-bounds position the issuer read, is a zero for an out-of-bounds one,
  // and is a copy from the previous tile row for a reused one (gw_reuse).
  logic        gw_reuse, gw_bus;
  logic        gf_valid, gf_full, gf_push, gf_pop;
  logic [31:0] gf_data;
  assign gw_bus   = gw_inb & ~gw_reuse;
  assign ld_valid = gather_q ? (gw_bus ? gf_valid : 1'b1) : rd_valid;
  assign ld_data  = gather_q ? (gw_bus ? gf_data : 32'd0) : rd_data;

  // Gather read FIFO: the reorder buffer drains into it whenever it has
  // room, so reads for new taps continue while the writer copies.
  localparam int unsigned GFD = 128;
  localparam int unsigned GFA = $clog2(GFD);
  (* ram_style = "distributed" *) logic [31:0] gf_mem [GFD];
  logic [GFA-1:0] gf_wp_q, gf_rp_q;
  logic [GFA:0]   gf_cnt_q;
  assign gf_valid = gf_cnt_q != '0;
  assign gf_full  = gf_cnt_q == (GFA+1)'(GFD);
  assign gf_data  = gf_mem[gf_rp_q];
  assign gf_push  = (state_q == ST_LOAD_A) & gather_q & rd_valid & ~gf_full;
  assign gf_pop   = (state_q == ST_LOAD_A) & gather_q & gw_bus & gf_valid;
  always_ff @(posedge clk_i) begin
    if (gf_push) gf_mem[gf_wp_q] <= rd_data;
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      gf_wp_q <= '0; gf_rp_q <= '0; gf_cnt_q <= '0;
    end else if (state_q == ST_IDLE) begin
      gf_wp_q <= '0; gf_rp_q <= '0; gf_cnt_q <= '0;
    end else begin
      if (gf_push) gf_wp_q <= gf_wp_q + 1'b1;
      if (gf_pop)  gf_rp_q <= gf_rp_q + 1'b1;
      gf_cnt_q <= gf_cnt_q + (gf_push ? 1'b1 : 1'b0) - (gf_pop ? 1'b1 : 1'b0);
    end
  end

  // Gather writes one cycle late (stage g1): a copy reads word
  // a_ld_word_q + C/2 of every row this cycle (a_rd_addr), and row t-1's
  // word is in a_q the next.
  logic           g1_we_q, g1_copy_q;
  logic [RW-1:0]  g1_row_q;
  logic [AW-1:0]  g1_addr_q;
  logic [31:0]    g1_data_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) g1_we_q <= 1'b0;
    else         g1_we_q <= (state_q == ST_LOAD_A) & gather_q & ld_valid;
  end
  always_ff @(posedge clk_i) begin
    g1_copy_q <= gw_reuse;
    g1_row_q  <= a_ld_row_q[RW-1:0];
    g1_addr_q <= a_ld_word_q;
    g1_data_q <= ld_data;
  end
  logic [31:0] g1_prev;                  // row t-1's word, for a copy
  assign g1_prev = a_q[RW'(g1_row_q - 1'b1)];

  assign a_wr_data = (onchip_q && state_q == RQ_LOAD_ACC) ? cr_rdata_q
                   : g1_we_q ? (g1_copy_q ? g1_prev : g1_data_q) : ld_data;
  always_comb begin
    a_we      = '0;
    a_wr_addr = g1_we_q ? g1_addr_q : a_ld_word_q;
    if ((state_q == ST_LOAD_A) && ld_valid && !gather_q) a_we[a_ld_row_q[RW-1:0]] = 1'b1;
    if (g1_we_q) a_we[g1_row_q] = 1'b1;
    if (state_q == RQ_LOAD_ACC) begin
      a_wr_addr = rq_m_q;
      if (acc_valid) begin
        a_we[rq_n_q[RW-1:0]] = 1'b1;
        if (a16_q) a_we[RW'(rq_n_q[RW-1:0] + 1'b1)] = 1'b1;  // A16: the pair's 2nd column
      end
    end
    if (state_q == RQ_LOAD_X) begin
      a_wr_addr = XOFF + rq_xw_q;
      if (rd_valid) a_we[rq_n_q[RW-1:0]] = 1'b1;
    end
  end

  // ---------------------------------------------------------------- MAC phase

  logic [31:0]    wbuf_q;      // latest 32-bit W beat (4 packed int8 weights)
  logic           wbuf_val_q;  // wbuf_q holds a beat not yet fully consumed
  logic [1:0]     wsel_q;      // which of the 4 bytes in wbuf_q is next
  logic [KCW-1:0] kcnt_q;      // reduction index k, 0..K-1, within the current W row
  logic [15:0]    m_q;         // which W row (0..M-1) is currently being processed

  logic adv, wlast, klast;
  // adv:   this cycle a weight byte is consumed and one MAC happens
  // wlast: wsel_q is on the buffer's last (4th) byte
  // klast: this is the last k of the current W row

  assign adv   = (state_q == ST_MAC) & wbuf_val_q;
  assign wlast = (w16_q | (wsel_q == 2'd1));
  assign klast = adv & (kcnt_q == (k_len_q - KCW'(2)));

  // A-tile read address: the requant output stage reads row rq_m_q (one
  // word = one output row m); the MAC phase reads word k/2 of every row.
  // In add mode the output pipeline reads the tile RAM twice per element:
  // the accumulator in the cycle the element enters, the residual word the
  // cycle after (rq_xcyc_q); no element enters on that second cycle.
  logic          rq_xcyc_q;
  logic [AW-1:0] rq_xaddr_q;
  assign a_rd_addr = (state_q == RQ_OUT)    ? (rq_xcyc_q ? rq_xaddr_q : rq_m_q)
                   : (state_q == ST_LOAD_A) ? a_ld_word_q + AW'(g_cwords)   // gather copy source
                                            : AW'(kcnt_q >> 1);

  // When to consume the next word from the read stream (rd_pop -> rb_rd_q
  // advances): every load phase takes one word at a time as it arrives;
  // ST_MAC only needs a new word once the current one is exhausted.
  always_comb begin
    rd_pop = 1'b0;
    if (state_q == ST_LOAD_A) begin
      rd_pop = gather_q ? gf_push : rd_valid;
    end else if (state_q == RQ_LOAD_P || state_q == RQ_LOAD_ACC || state_q == RQ_LOAD_X
                 || state_q == RQ_LOAD_L) begin
      rd_pop = rd_valid;
    end else if (state_q == ST_MAC) begin
      // Refill an empty buffer, or replace the buffer as its last byte is
      // consumed, so that a beat is never wasted waiting a cycle.
      rd_pop = rd_valid & (~wbuf_val_q | wlast);
    end
  end

  // The pair of weights wsel_q selects: int8 bytes 0,1 or 2,3 of wbuf_q, or
  // with int16 weights (CTRL.w16) the word's two halves.
  logic signed [15:0] wbyte, wbyte1;   // weights k and k+1
  always_comb begin
    if (w16_q) begin
      wbyte  = $signed(wbuf_q[15:0]);
      wbyte1 = $signed(wbuf_q[31:16]);
    end else if (wsel_q == 2'd0) begin
      wbyte  = 16'($signed(wbuf_q[7:0]));
      wbyte1 = 16'($signed(wbuf_q[15:8]));
    end else begin
      wbyte  = 16'($signed(wbuf_q[23:16]));
      wbyte1 = 16'($signed(wbuf_q[31:24]));
    end
  end
  // MAC pipeline, two weights (k, k+1) per cycle:
  //   s0  present the A-tile read address (combinational from kcnt_q)
  //   s1  the A word a_q arrives: a[k] in bits 15:0, a[k+1] in bits 31:16
  //   s2  a_s2 / a1_s2: the two halves, in fabric registers
  //   s3  DSP input registers (AREG, BREG)
  //   s4  DSP multiplier registers (MREG)
  //   s5  DSP output registers (PREG): the two products
  //   then the accumulator adds both products (in logic)
  logic              v_s1, v_s2, v_s3, v_s4;   // valid bit per stage (adv delayed by 1..4 cycles)
  // v_s5 enables all NROWS*32 accumulator bits. max_fanout makes synthesis
  // copy the register, so no single net spans the whole lane array.
  (* max_fanout = 64 *) logic v_s5;
  logic signed [15:0] w_s1, w_s2, w_s3, w1_s1, w1_s2, w1_s3;   // weights k and k+1, pipelined to s3
  // keep: without it the DSP48 takes both a_s2 and the RAM read register
  // a_q as its input registers, and the row RAMs become LUT RAM.
  (* keep = "true" *) logic signed [15:0] a_s2 [NROWS], a1_s2 [NROWS];
  // s3..s5 map onto one DSP48 per product with all its registers. The
  // accumulator adds the two products in logic: left to itself, synthesis
  // tried to fold the three-input add into the DSPs and did not finish in
  // an hour.
  (* use_dsp = "yes" *) logic signed [15:0] a_s3 [NROWS], a1_s3 [NROWS];
  (* use_dsp = "yes" *) logic signed [31:0] m0_s4 [NROWS], m1_s4 [NROWS];
  (* use_dsp = "yes" *) logic signed [31:0] p0_s5 [NROWS], p1_s5 [NROWS];
  (* use_dsp = "no" *) logic signed [31:0] acc_q [NROWS]; // the running sum for each of the NROWS rows (this is C, in progress)
  logic              acc_clr;        // synchronously clear all NROWS accumulators this cycle

  // Only the valid bits are reset. The data registers (weight byte, A
  // operand, accumulator) deliberately have no asynchronous reset: DSP48E1
  // internal registers have synchronous reset only, and Vivado will not pull
  // a register with an async reset into the DSP -- 64 x 32 accumulators
  // worth of methodology warnings (DPIR-1) and a longer path. The
  // accumulators are cleared synchronously at job start and after every
  // drain, so they never hold anything a result depends on before then.
  // Shifts the "a MAC is happening" bit down the pipeline: adv (s0) -> v_s1 -> ... -> v_s5.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v_s1 <= 1'b0;
      v_s2 <= 1'b0;
      v_s3 <= 1'b0;
      v_s4 <= 1'b0;
      v_s5 <= 1'b0;
    end else begin
      v_s1 <= adv;
      v_s2 <= v_s1;
      v_s3 <= v_s2;
      v_s4 <= v_s3;
      v_s5 <= v_s4;
    end
  end
  // Carries the two weights alongside v_s1 .. v_s3.
  always_ff @(posedge clk_i) begin
    w_s1    <= wbyte;
    w1_s1   <= wbyte1;
    w_s2    <= w_s1;
    w1_s2   <= w1_s1;
    w_s3    <= w_s2;
    w1_s3   <= w1_s2;
  end

  // One multiply-accumulate lane per tile row, replicated NROWS times; each
  // uses two DSP48E1 slices, one per product (A/B/M/P registers).
  for (genvar r = 0; r < int'(NROWS); r++) begin : gen_pe
    always_ff @(posedge clk_i) begin
      a_s2[r]  <= $signed(a_q[r][15:0]);     // a[k]
      a1_s2[r] <= $signed(a_q[r][31:16]);    // a[k+1]
      a_s3[r]  <= a_s2[r];
      a1_s3[r] <= a1_s2[r];
      m0_s4[r] <= a_s3[r] * w_s3;
      m1_s4[r] <= a1_s3[r] * w1_s3;
      p0_s5[r] <= m0_s4[r];
      p1_s5[r] <= m1_s4[r];
      if (acc_clr)   acc_q[r] <= '0;
      else if (v_s5) acc_q[r] <= acc_q[r] + p0_s5[r] + p1_s5[r];
    end
  end

  // -------------------------------------------------------------- drain phase
  //
  // n_rows_q words of accumulators, then -- when statistics are on -- two
  // more: the max and the min of what was just drained, to s_ptr_q. The
  // extremes are tracked as the words go out; the stats words are issued at
  // least one cycle after the last accumulator, so they see it.

  logic [NRW:0]   t_q;          // output word inside the current run
  logic [31:0]    c_ptr_q;      // base of the current run
  logic [31:0]    s_ptr_q;      // where this row's {max, min} go
  logic [NRW:0]   n_wr;         // words to write this drain
  logic signed [31:0] acc_max_q, acc_min_q;  // running max/min of the accumulators drained so far
  logic           t_is_acc;     // t_q still indexes an accumulator (vs. the trailing stats words)

  assign n_wr     = {1'b0, n_rows_q} + (stats_en_q ? 2 : 0);
  assign t_is_acc = t_q < {1'b0, n_rows_q};
  assign drain_wr_req  = (state_q == ST_DRAIN) & (t_q != n_wr) & ~(onchip_q & t_is_acc);
  assign drain_wr_addr = t_is_acc ? c_ptr_q + 32'(t_q) * 32'd4
                                  : s_ptr_q + (t_q[0] ^ n_rows_q[0] ? 32'd4 : 32'd0);
  assign drain_wr_data = t_is_acc            ? acc_q[t_q[RW-1:0]]
                       : (t_q == {1'b0, n_rows_q}) ? acc_max_q : acc_min_q;

  // ------------------------------------------------------------ state machine

  logic pipe_idle;  // the MAC pipeline has fully drained (safe to start DRAIN)
  assign pipe_idle = ~v_s1 & ~v_s2 & ~v_s3 & ~v_s4 & ~v_s5;

  logic a_load_done;  // the A tile's last word (last row, last word) has just arrived
  assign a_load_done = ld_valid & (a_ld_row_q == (n_rows_q - 1'b1))
                                & (a_ld_word_q == (k_words - 1'b1));

  // Next-state logic: each state advances only on its own completion signal,
  // computed above (a_load_done, klast, pipe_idle, t_q==n_wr, ...).
  always_comb begin
    state_d = state_q;
    unique case (state_q)
      ST_IDLE:     if (start_strobe)              // CPU asked for a job
                     state_d = !start_requant ? ST_LOAD_A
                             : start_lut_load ? RQ_LOAD_L : RQ_LOAD_P;
      RQ_LOAD_L:   if (lut_ld_done)      state_d = RQ_LOAD_P;
      ST_LOAD_A:   if (a_load_done)              state_d = ST_MAC;       // tile fully loaded
      ST_MAC:      if (klast)                    state_d = ST_MAC_TAIL; // W row fully streamed
      ST_MAC_TAIL: if (pipe_idle)                state_d = ST_DRAIN;    // pipeline flushed
      ST_DRAIN:    if (t_q == n_wr)                                     // row's outputs all issued
                     state_d = (m_q == (m_len_q - 1'b1)) ? ST_FINISH : ST_MAC; // last W row? else next row
      ST_FINISH:   if (wr_out_q == '0)           state_d = ST_IDLE;     // all writes acked
      RQ_LOAD_P:   if (rq_p_done)                state_d = RQ_LOAD_ACC; // param table loaded
      RQ_LOAD_ACC: if (rq_acc_done)              state_d = add_q ? RQ_LOAD_X : RQ_OUT;
      RQ_LOAD_X:   if (rq_x_done)                state_d = RQ_OUT;      // residual loaded
      RQ_OUT:      if (rq_out_done)      state_d = ostats_q ? RQ_STATS : ST_FINISH;
      RQ_STATS:    if (t_q == {1'b0, n_rows_q}) state_d = ST_FINISH;   // every output word written
      default:                                   state_d = ST_IDLE;
    endcase
  end

  // Clear the accumulators at job start and for the next weight row while
  // the drain finishes. (Job start matters: they have no reset.)
  assign acc_clr = ((state_q == ST_DRAIN) & (t_q == n_wr))
                 | ((state_q == ST_IDLE) & start_strobe);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q     <= ST_IDLE;
      busy_q      <= 1'b0;
      done_q      <= 1'b0;
      err_q       <= 1'b0;
      cycles_q    <= '0;
      w_addr_q    <= '0;
      c_stride_q  <= '0;
      k_len_q     <= '0;
      m_len_q     <= '0;
      n_rows_q    <= '0;
      rd_addr_q   <= '0;
      rd_left_q   <= '0;
      rd_row_base_q  <= '0;
      rd_row_beats_q <= '0;
      rd_row_left_q  <= '0;
      rd_stride_q    <= '0;
      w_stride_q  <= '0;
      s_addr_q    <= '0;
      gather_q    <= 1'b0;
      add_q <= 1'b0; relu_q <= 1'b0; x_addr_q <= '0;
      lut_q <= 1'b0; p_addr_q <= '0;
      a16_q <= 1'b0;
      w16_q <= 1'b0;
      ostats_q <= 1'b0;
      onchip_q <= 1'b0;
      greuse_q <= 1'b0;
      // rq_m_q, rq_p_cnt_q and lut_cnt_q have no reset: they address block
      // RAMs (an asynchronous reset there is DRC REQP-1840), and every job
      // sets them before use.
      rq_xw_q <= '0;
      g_addr_q <= '0; g_h_q <= '0; g_w_q <= '0; g_ow_q <= '0; g_c_q <= '0;
      g_k_q <= '0; g_stride_q <= '0; g_pad_q <= '0; g_ky0_q <= '0; g_kx0_q <= '0; g_kpos_q <= '0;
      a_addr_q    <= '0;
      a_stride_q  <= '0;
      stats_en_q  <= 1'b0;
      s_ptr_q     <= '0;
      acc_max_q   <= '0;
      acc_min_q   <= '0;
      rq_n_q      <= '0;
      rq_p_sel_q  <= '0;
      rb_wr_q     <= '0;
      rb_rd_q     <= '0;
      retry_cnt_q     <= '0;
      retry_pending_q <= 1'b0;
      retry_n_q       <= '0;
      rb_cnt_q    <= '0;
      wr_out_q    <= '0;
      wr_src_q    <= '0;
      a_ld_word_q <= '0;
      a_ld_row_q  <= '0;
      wbuf_q      <= '0;
      wbuf_val_q  <= 1'b0;
      wsel_q      <= '0;
      kcnt_q      <= '0;
      m_q         <= '0;
      t_q         <= '0;
      c_ptr_q     <= '0;
    end else begin
      state_q <= state_d;

      // Bus bookkeeping --------------------------------------------------
      if (issue_rd) begin
        rd_left_q <= rd_left_q - 32'd1;
        rb_wr_q   <= rb_wr_q + 1'b1;
        if (rd_row_left_q == 32'd1) begin
          // last beat of this row: jump to the start of the next one
          rd_addr_q      <= rd_row_base_q + rd_stride_q;
          rd_row_base_q  <= rd_row_base_q + rd_stride_q;
          rd_row_left_q  <= rd_row_beats_q;
        end else begin
          rd_addr_q      <= rd_addr_q + 32'd4;
          rd_row_left_q  <= rd_row_left_q - 32'd1;
        end
      end
      if (issue_wr) begin
        wr_src_q <= wr_src_q + 1'b1;
      end
      if (rd_pop) begin
        rb_rd_q <= rb_rd_q + 1'b1;
      end

      // Lost-response recovery ------------------------------------------
      // Remember the read in flight, and watch for its answer. Any response
      // or any issue restarts the clock; only genuine silence trips it.
      if (issue_retry) begin
        retry_pending_q <= 1'b0;
        retry_cnt_q     <= '0;
        retry_n_q       <= retry_n_q + 32'd1;
      end else if (retry_expired && (rb_cnt_q != '0) && !rb_val_q[rb_rd_q]) begin
        retry_pending_q <= 1'b1;
        retry_cnt_q     <= '0;
      end else if (issue_rd || rsp_rd || (rb_cnt_q == '0)) begin
        retry_cnt_q <= '0;
      end else begin
        retry_cnt_q <= retry_cnt_q + 32'd1;
      end
      unique case ({issue_rd, rd_pop})
        2'b10:   rb_cnt_q <= rb_cnt_q + 1'b1;
        2'b01:   rb_cnt_q <= rb_cnt_q - 1'b1;
        default: ;
      endcase
      unique case ({issue_wr, rsp_wr})
        2'b10:   wr_out_q <= wr_out_q + 1'b1;
        2'b01:   wr_out_q <= wr_out_q - 1'b1;
        default: ;
      endcase
      if (tl_host_i.d_valid & tl_host_i.d_error) err_q <= 1'b1;

      // Datapath ---------------------------------------------------------
      if (busy_q) cycles_q <= cycles_q + 32'd1;

      // Per-state datapath actions: this case implements what each state in
      // the FSM above actually does (as opposed to when it exits).
      unique case (state_q)
        ST_IDLE: begin
          // Latch every register the job needs (so software is free to
          // reprogram them for the *next* job as soon as this one starts),
          // and program the read engine to fetch the first stream: the A
          // tile for a GEMM, or the parameter table for a requant job.
          if (start_strobe) begin
            w_addr_q   <= reg2hw.w_addr.q;
            c_stride_q <= reg2hw.c_stride.q;
            k_len_q    <= KCW'(reg2hw.k_len.q);
            m_len_q    <= reg2hw.m_len.q;
            n_rows_q   <= NRW'(reg2hw.n_rows.q);
            c_ptr_q    <= reg2hw.c_addr.q;

            // A stride of 0 means "contiguous": one row length apart.
            w_stride_q <= (reg2hw.w_stride.q != 32'd0) ? reg2hw.w_stride.q
                                                       : start_w16 ? 32'(reg2hw.k_len.q) * 32'd2 : 32'(reg2hw.k_len.q);
            s_addr_q   <= reg2hw.s_addr.q;
            s_ptr_q    <= reg2hw.s_addr.q;
            a_addr_q   <= reg2hw.a_addr.q;
            a_stride_q <= reg2hw.a_stride.q;
            stats_en_q <= (reg2hw.s_addr.q != 32'd0) & ~start_requant;
            gather_q   <= start_gather & ~start_requant;
            add_q      <= start_add & start_requant;
            relu_q     <= start_relu & start_requant;
            lut_q      <= start_lut & start_requant;
            a16_q      <= start_a16 & start_requant;
            w16_q      <= start_w16 & ~start_requant;
            ostats_q   <= start_ostats & start_requant;
            onchip_q   <= start_onchip;
            greuse_q   <= start_greuse & start_gather & ~start_requant;
            p_addr_q   <= reg2hw.p_addr.q;
            lut_cnt_q  <= '0;
            x_addr_q   <= reg2hw.x_addr.q;
            g_addr_q   <= reg2hw.g_addr.q;
            g_h_q      <= reg2hw.g_geom.q[31:16];
            g_w_q      <= reg2hw.g_geom.q[15:0];
            g_ow_q     <= reg2hw.g_chan.q[31:16];
            g_c_q      <= reg2hw.g_chan.q[15:0];
            g_k_q      <= reg2hw.g_conv.q[3:0];
            g_stride_q <= reg2hw.g_conv.q[7:4];
            g_pad_q    <= reg2hw.g_conv.q[11:8];
            g_ky0_q    <= reg2hw.g_conv.q[15:12];
            g_kx0_q    <= reg2hw.g_conv.q[19:16];
            g_kpos_q   <= reg2hw.g_conv.q[27:20];
            rq_m_q     <= '0;
            rq_n_q     <= '0;
            rq_p_cnt_q <= '0;
            rq_p_sel_q <= '0;

            // A tile: n_rows rows of k/2 beats, starting at a_addr.
            rd_addr_q      <= reg2hw.a_addr.q;
            rd_row_base_q  <= reg2hw.a_addr.q;
            rd_row_beats_q <= 32'(reg2hw.k_len.q >> 1);
            rd_row_left_q  <= 32'(reg2hw.k_len.q >> 1);
            rd_stride_q    <= (reg2hw.a_stride.q != 32'd0) ? reg2hw.a_stride.q
                                                           : 32'(reg2hw.k_len.q) * 32'd2;
            // In gather mode nothing is issued until the walker hands over
            // the first run.
            rd_left_q      <= (start_gather & ~start_requant) ? 32'd0
                            : 32'(reg2hw.n_rows.q) * 32'(reg2hw.k_len.q >> 1);

            if (start_requant) begin
              // Parameter table first: 3 words per row m, contiguous.
              rd_addr_q      <= reg2hw.p_addr.q;
              rd_row_base_q  <= reg2hw.p_addr.q;
              rd_row_beats_q <= 32'(reg2hw.m_len.q) * 32'd3;
              rd_row_left_q  <= 32'(reg2hw.m_len.q) * 32'd3;
              rd_stride_q    <= 32'(reg2hw.m_len.q) * 32'd12;
              rd_left_q      <= 32'(reg2hw.m_len.q) * 32'd3;
              if (start_lut_load) begin
                // Lookup table first: 8192 contiguous words.
                rd_addr_q      <= reg2hw.lut_addr.q;
                rd_row_base_q  <= reg2hw.lut_addr.q;
                rd_row_beats_q <= 32'd8192;
                rd_row_left_q  <= 32'd8192;
                rd_stride_q    <= 32'd32768;
                rd_left_q      <= 32'd8192;
              end
            end

            a_ld_word_q <= '0;
            a_ld_row_q  <= '0;
            kcnt_q      <= '0;
            m_q         <= '0;
            t_q         <= '0;
            wbuf_val_q  <= 1'b0;
            wsel_q      <= '0;
            busy_q      <= 1'b1;
            done_q      <= 1'b0;
            err_q       <= 1'b0;
            cycles_q    <= '0;
          end
        end

        ST_LOAD_A: begin
          // Walk a_ld_row_q/a_ld_word_q over the tile as words arrive; once
          // the last one lands, reprogram the read engine for the W stream.
          if (ld_valid) begin
            if (a_ld_word_q == (k_words - 1'b1)) begin
              a_ld_word_q <= '0;
              a_ld_row_q  <= a_ld_row_q + 1'b1;
            end else begin
              a_ld_word_q <= a_ld_word_q + 1'b1;
            end
          end
          if (gather_q && gi_handover) begin
            // next in-bounds run of the gather: C/2 beats, contiguous
            rd_addr_q      <= gi_addr_q;
            rd_row_base_q  <= gi_addr_q;
            rd_row_beats_q <= 32'(g_cwords);
            rd_row_left_q  <= 32'(g_cwords);
            rd_stride_q    <= 32'(g_cwords) * 32'd4;
            rd_left_q      <= 32'(g_cwords);
          end
          if (a_load_done) begin
            // Reprogram the read engine for the weight stream: M rows of
            // K/4 beats, read exactly once for the whole tile.
            rd_addr_q      <= w_addr_q;
            rd_row_base_q  <= w_addr_q;
            rd_row_beats_q <= (w16_q ? 32'(k_len_q >> 1) : 32'(k_len_q >> 2));
            rd_row_left_q  <= (w16_q ? 32'(k_len_q >> 1) : 32'(k_len_q >> 2));
            rd_stride_q    <= w_stride_q;
            rd_left_q      <= 32'(m_len_q) * (w16_q ? 32'(k_len_q >> 1) : 32'(k_len_q >> 2));
          end
        end

        ST_MAC: begin
          // Advance the byte selector through wbuf_q, refilling it from the
          // read stream every 4th byte; reset k and the drain index when
          // the row's last k has been consumed.
          if (adv) begin
            kcnt_q <= kcnt_q + KCW'(2);
            if (wlast) begin
              wsel_q <= '0;
              if (rd_valid) wbuf_q     <= rd_data;
              else          wbuf_val_q <= 1'b0;
            end else begin
              wsel_q <= wsel_q + 1'b1;
            end
          end else if (rd_valid) begin
            wbuf_q     <= rd_data;
            wbuf_val_q <= 1'b1;
            wsel_q     <= '0;
          end
          if (klast) begin
            kcnt_q <= '0;
            t_q    <= '0;
          end
        end

        ST_DRAIN: begin
          // Track max/min as accumulators go out (for the trailing stats
          // words), and advance to the next W row once this run is done.
          if (issue_wr | (onchip_q & t_is_acc & (t_q != n_wr))) begin
            t_q <= t_q + 1'b1;
            if (t_is_acc) begin
              if (t_q == '0 || $signed(drain_wr_data) > acc_max_q) acc_max_q <= drain_wr_data;
              if (t_q == '0 || $signed(drain_wr_data) < acc_min_q) acc_min_q <= drain_wr_data;
            end
          end
          if (t_q == n_wr) begin
            m_q     <= m_q + 1'b1;
            c_ptr_q <= c_ptr_q + c_stride_q;
            s_ptr_q <= s_ptr_q + 32'd8;
            // Leaving t_q at n_rows_q would index an accumulator whose A-tile
            // row was never loaded when the tile is partial (n_rows < NROWS).
            t_q     <= '0;
          end
        end

        // ---- requantisation job -------------------------------------------
        RQ_STATS: begin
          // one statistics word per output row, t_q = row
          if (issue_wr) t_q <= t_q + 1'b1;
        end
        RQ_LOAD_L: begin
          // One table word per beat; then the parameter table, as a job
          // without a table load starts with.
          if (rd_valid) lut_cnt_q <= lut_cnt_q + 1'b1;
          if (lut_ld_done) begin
            rd_addr_q      <= p_addr_q;
            rd_row_base_q  <= p_addr_q;
            rd_row_beats_q <= 32'(m_len_q) * 32'd3;
            rd_row_left_q  <= 32'(m_len_q) * 32'd3;
            rd_stride_q    <= 32'(m_len_q) * 32'd12;
            rd_left_q      <= 32'(m_len_q) * 32'd3;
          end
        end
        RQ_LOAD_P: begin
          // Walk rq_p_sel_q/rq_p_cnt_q over the incoming {mult,shift,bias}
          // words; once the last row's triple has arrived, reprogram the
          // read engine to fetch the acc chunk next.
          if (rd_valid) begin
            // words arrive mult, shift, bias for row 0, then row 1, ...
            if (rq_p_sel_q == 2'd2) begin
              rq_p_sel_q <= 2'd0;
              rq_p_cnt_q <= rq_p_cnt_q + 1'b1;
            end else begin
              rq_p_sel_q <= rq_p_sel_q + 1'b1;
            end
          end
          if (rq_p_done) begin
            // acc chunk: M_LEN rows of N_ROWS words, rows A_STRIDE apart
            rd_addr_q      <= a_addr_q;
            rd_row_base_q  <= a_addr_q;
            rd_row_beats_q <= (a16_q ? 32'(n_rows_q >> 1) : 32'(n_rows_q));
            rd_row_left_q  <= (a16_q ? 32'(n_rows_q >> 1) : 32'(n_rows_q));
            rd_stride_q    <= a_stride_q;
            rd_left_q      <= onchip_q ? 32'd0 : 32'(m_len_q) * (a16_q ? 32'(n_rows_q >> 1) : 32'(n_rows_q));
            rq_m_q <= '0;
            rq_n_q <= '0;
          end
        end

        RQ_LOAD_ACC: begin
          // Walk rq_m_q (row, outer) / rq_n_q (column, inner) over the
          // incoming acc words, matching the write-side transpose above.
          if (acc_valid) begin
            if (rq_n_q == n_rows_q - (a16_q ? NRW'(2) : NRW'(1))) begin
              rq_n_q <= '0;
              rq_m_q <= rq_m_q + 1'b1;
            end else begin
              rq_n_q <= rq_n_q + (a16_q ? NRW'(2) : NRW'(1));
            end
          end
          if (rq_acc_done) begin
            rq_m_q <= '0;
            rq_n_q <= '0;
            rq_xw_q <= '0;
            if (add_q) begin
              // residual chunk: N_ROWS rows of M_LEN/2 words, C_STRIDE apart
              rd_addr_q      <= x_addr_q;
              rd_row_base_q  <= x_addr_q;
              rd_row_beats_q <= 32'(rq_xwords);
              rd_row_left_q  <= 32'(rq_xwords);
              rd_stride_q    <= c_stride_q;
              rd_left_q      <= 32'(n_rows_q) * 32'(rq_xwords);
            end
          end
        end

        RQ_LOAD_X: begin
          // Walk rq_n_q (row, outer) / rq_xw_q (word, inner) over the
          // residual words: they are already [n][m], so no transpose.
          if (rd_valid) begin
            if (rq_xw_q == rq_xwords - 1'b1) begin
              rq_xw_q <= '0;
              rq_n_q  <= rq_n_q + 1'b1;
            end else begin
              rq_xw_q <= rq_xw_q + 1'b1;
            end
          end
          if (rq_x_done) begin
            rq_m_q <= '0;
            rq_n_q <= '0;
          end
        end

        RQ_OUT: begin
          // Walk rq_m_q (inner) / rq_n_q (outer) over the output elements,
          // one per cycle while the output path has room (rq_adv).
          if (rq_adv) begin
            if (rq_m_q == rq_mlen - 1'b1) begin
              rq_m_q <= '0;
              rq_n_q <= rq_n_q + 1'b1;
            end else begin
              rq_m_q <= rq_m_q + 1'b1;
            end
          end
        end

        ST_FINISH: begin
          // Wait for the last outstanding write to be acked, then report done.
          if (wr_out_q == '0) begin
            busy_q <= 1'b0;
            done_q <= 1'b1;
          end
        end

        default: ;
      endcase
    end
  end

  // ------------------------------------------------------------------ gather
  //
  // Two walkers step through the same sequence of (tile row t, kernel row ky,
  // kernel column kx): the ISSUER on the bus side, which skips out-of-bounds
  // positions and hands each in-bounds one to the read engine as a run of
  // C/2 beats; and the WRITER on the RAM side, which consumes the beats in
  // order and inserts C/2 zero words for the out-of-bounds positions. Both
  // derive "in bounds" from the same registers, so they agree.
  //
  //   iy = oy*stride + ky - pad,  ix = ox*stride + kx - pad
  //   in bounds  <=>  0 <= iy < h  and  0 <= ix < w
  //   address    =    G_ADDR + ((iy*w + ix) * C) * 2
  //
  // Products oy*stride and ox*stride are kept incrementally.

  assign g_cwords = g_c_q >> 1;
  // Reuse needs stride 1 and one chunk holding all k*k positions from
  // (0, 0); then tile row t's tap (ky, kx) is row t-1's (ky, kx+1),
  // zeros included, whenever both pixels are on one output row.
  assign g_reuse_ok = greuse_q & (g_stride_q == 4'd1) & (g_k_q >= 4'd2)
                    & (g_ky0_q == '0) & (g_kx0_q == '0)
                    & (g_kpos_q == 8'(g_k_q) * 8'(g_k_q));

  // ---- issuer
  typedef enum logic [1:0] { GI_IDLE, GI_SCAN, GI_CALC, GI_LOAD } gi_e;
  gi_e gi_q;
  logic [NRW-1:0] gi_t_q;
  logic [15:0]    gi_oy_q, gi_ox_q, gi_oys_q, gi_oxs_q;   // pixel, and pixel*stride
  logic [3:0]     gi_ky_q, gi_kx_q;
  logic [7:0]     gi_pos_q;
  logic signed [17:0] gi_iy, gi_ix;
  logic           gi_inb;
  logic [31:0]    gi_pix_q;
  logic           gi_step, gi_last_pos, gi_last_row;

  assign gi_iy  = $signed({2'b0, gi_oys_q}) + $signed({14'b0, gi_ky_q}) - $signed({14'b0, g_pad_q});
  assign gi_ix  = $signed({2'b0, gi_oxs_q}) + $signed({14'b0, gi_kx_q}) - $signed({14'b0, g_pad_q});
  assign gi_inb = (gi_iy >= 0) & (gi_iy < $signed({2'b0, g_h_q}))
                & (gi_ix >= 0) & (gi_ix < $signed({2'b0, g_w_q}));

  assign gi_last_pos = (gi_pos_q == g_kpos_q - 1'b1);
  assign gi_last_row = (gi_t_q == n_rows_q - 1'b1);
  // Hand the computed run to the read engine once the previous run has been
  // fully issued (the main always_ff loads rd_addr_q/rd_left_q on this).
  assign gi_handover = (gi_q == GI_LOAD) & (rd_left_q == 32'd0) & (state_q == ST_LOAD_A);
  // Advance one kernel position: after skipping an out-of-bounds one, or
  // after handing an in-bounds one over.
  logic gi_reuse;   // this position is copied by the writer, not read
  assign gi_reuse = g_reuse_ok & (gi_t_q != '0) & (gi_ox_q != '0)
                  & (gi_kx_q != g_k_q - 1'b1);
  assign gi_step = ((gi_q == GI_SCAN) & (~gi_inb | gi_reuse)) | gi_handover;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      gi_q <= GI_IDLE;
      gi_t_q <= '0; gi_oy_q <= '0; gi_ox_q <= '0; gi_oys_q <= '0; gi_oxs_q <= '0;
      gi_ky_q <= '0; gi_kx_q <= '0; gi_pos_q <= '0;
    end else if (state_q == ST_IDLE && start_strobe && start_gather && !start_requant) begin
      gi_q     <= GI_SCAN;
      gi_t_q   <= '0;
      gi_oy_q  <= reg2hw.g_start.q[31:16];
      gi_ox_q  <= reg2hw.g_start.q[15:0];
      gi_oys_q <= reg2hw.g_start.q[31:16] * 16'(reg2hw.g_conv.q[7:4]);
      gi_oxs_q <= reg2hw.g_start.q[15:0]  * 16'(reg2hw.g_conv.q[7:4]);
      gi_ky_q  <= reg2hw.g_conv.q[15:12];
      gi_kx_q  <= reg2hw.g_conv.q[19:16];
      gi_pos_q <= '0;
    end else begin
      if (gi_q == GI_SCAN && gi_inb && !gi_reuse) gi_q <= GI_CALC;
      if (gi_q == GI_CALC)           gi_q <= GI_LOAD;
      if (gi_step) begin
        gi_q <= GI_SCAN;
        if (gi_last_pos) begin
          gi_pos_q <= '0;
          gi_ky_q  <= g_ky0_q;
          gi_kx_q  <= g_kx0_q;
          if (gi_last_row) begin
            gi_q <= GI_IDLE;
          end else begin
            gi_t_q <= gi_t_q + 1'b1;
            if (gi_ox_q == g_ow_q - 1'b1) begin
              gi_ox_q  <= '0;
              gi_oxs_q <= '0;
              gi_oy_q  <= gi_oy_q + 1'b1;
              gi_oys_q <= gi_oys_q + 16'(g_stride_q);
            end else begin
              gi_ox_q  <= gi_ox_q + 1'b1;
              gi_oxs_q <= gi_oxs_q + 16'(g_stride_q);
            end
          end
        end else begin
          gi_pos_q <= gi_pos_q + 1'b1;
          if (gi_kx_q == g_k_q - 1'b1) begin
            gi_kx_q <= '0;
            gi_ky_q <= gi_ky_q + 1'b1;
          end else begin
            gi_kx_q <= gi_kx_q + 1'b1;
          end
        end
      end
    end
  end

  // Address arithmetic for the run, without reset: written in SCAN/CALC
  // before it is read in LOAD, and a register with an asynchronous reset
  // cannot be folded into the DSP48 that does the multiply (DPIR-1).
  always_ff @(posedge clk_i) begin
    if (gi_q == GI_SCAN) gi_pix_q  <= 32'(gi_iy) * 32'(g_w_q) + 32'(gi_ix);
    if (gi_q == GI_CALC) gi_addr_q <= g_addr_q + gi_pix_q * 32'(g_c_q) * 32'd2;
  end

  // ---- writer
  logic [NRW-1:0] gw_t_q;
  logic [15:0]    gw_oys_q, gw_oxs_q, gw_ox_q;
  logic [3:0]     gw_ky_q, gw_kx_q;
  logic [7:0]     gw_pos_q;
  logic [15:0]    gw_cw_q;                 // word within the position
  logic signed [17:0] gw_iy, gw_ix;

  assign gw_iy  = $signed({2'b0, gw_oys_q}) + $signed({14'b0, gw_ky_q}) - $signed({14'b0, g_pad_q});
  assign gw_ix  = $signed({2'b0, gw_oxs_q}) + $signed({14'b0, gw_kx_q}) - $signed({14'b0, g_pad_q});
  assign gw_inb = (gw_iy >= 0) & (gw_iy < $signed({2'b0, g_h_q}))
                & (gw_ix >= 0) & (gw_ix < $signed({2'b0, g_w_q}));
  assign gw_reuse = g_reuse_ok & (gw_t_q != '0) & (gw_ox_q != '0)
                  & (gw_kx_q != g_k_q - 1'b1);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      gw_t_q <= '0; gw_oys_q <= '0; gw_oxs_q <= '0; gw_ox_q <= '0;
      gw_ky_q <= '0; gw_kx_q <= '0; gw_pos_q <= '0; gw_cw_q <= '0;
    end else if (state_q == ST_IDLE && start_strobe && start_gather && !start_requant) begin
      gw_t_q   <= '0;
      gw_ox_q  <= reg2hw.g_start.q[15:0];
      gw_oys_q <= reg2hw.g_start.q[31:16] * 16'(reg2hw.g_conv.q[7:4]);
      gw_oxs_q <= reg2hw.g_start.q[15:0]  * 16'(reg2hw.g_conv.q[7:4]);
      gw_ky_q  <= reg2hw.g_conv.q[15:12];
      gw_kx_q  <= reg2hw.g_conv.q[19:16];
      gw_pos_q <= '0;
      gw_cw_q  <= '0;
    end else if (state_q == ST_LOAD_A && gather_q && ld_valid) begin
      if (gw_cw_q == g_cwords - 1'b1) begin
        gw_cw_q <= '0;
        if (gw_pos_q == g_kpos_q - 1'b1) begin
          gw_pos_q <= '0;
          gw_ky_q  <= g_ky0_q;
          gw_kx_q  <= g_kx0_q;
          gw_t_q   <= gw_t_q + 1'b1;
          if (gw_ox_q == g_ow_q - 1'b1) begin
            gw_ox_q  <= '0;
            gw_oxs_q <= '0;
            gw_oys_q <= gw_oys_q + 16'(g_stride_q);
          end else begin
            gw_ox_q  <= gw_ox_q + 1'b1;
            gw_oxs_q <= gw_oxs_q + 16'(g_stride_q);
          end
        end else begin
          gw_pos_q <= gw_pos_q + 1'b1;
          if (gw_kx_q == g_k_q - 1'b1) begin
            gw_kx_q <= '0;
            gw_ky_q <= gw_ky_q + 1'b1;
          end else begin
            gw_kx_q <= gw_kx_q + 1'b1;
          end
        end
      end else begin
        gw_cw_q <= gw_cw_q + 1'b1;
      end
    end
  end

  // ------------------------------------------------------- requantisation
  //
  // Parameter RAM: {mult, shift, bias} per row m of the chunk, filled by
  // RQ_LOAD_P from the read stream. KWORDS entries, matching the tile RAM's
  // words per row, which is what bounds a chunk's M_LEN.
  logic [31:0]    pm_mult [KWORDS];   // per-row multiplier, indexed by output row m
  logic [5:0]     pm_shift[KWORDS];   // per-row right-shift amount
  logic [31:0]    pm_bias [KWORDS];   // per-row bias, added after the shift

  // Demultiplex the incoming words into the three parameter RAMs, in the
  // order they arrive: mult, shift, bias, for row 0, then row 1, ...
  always_ff @(posedge clk_i) begin
    if ((state_q == RQ_LOAD_P) && rd_valid) begin
      unique case (rq_p_sel_q)
        2'd0:    pm_mult [rq_p_cnt_q] <= rd_data;
        2'd1:    pm_shift[rq_p_cnt_q] <= rd_data[5:0];
        default: pm_bias [rq_p_cnt_q] <= rd_data;
      endcase
    end
  end

  // rq_p_done: the last row's bias word has just arrived.
  assign rq_p_done   = (state_q == RQ_LOAD_P) & rd_valid
                     & (rq_p_sel_q == 2'd2) & (rq_p_cnt_q == rq_mlen - 1'b1);
  // rq_acc_done: the acc chunk's last word (last row, last column) has just arrived.
  assign rq_x_done   = (state_q == RQ_LOAD_X) & rd_valid
                     & (rq_n_q == n_rows_q - 1'b1) & (rq_xw_q == rq_xwords - 1'b1);
  assign rq_acc_done = (state_q == RQ_LOAD_ACC) & acc_valid
                     & (rq_n_q == n_rows_q - (a16_q ? NRW'(2) : NRW'(1))) & (rq_m_q == rq_mlen - 1'b1);

  // Output pipeline. One element per cycle:
  //   q0  RAM read address = m (a_rd_addr), param RAM read address = m
  //   q1  acc = tile RAM row n; mult/shift/bias arrive
  //   q2  64-bit product (DSP)
  //   q3  product registered again (DSP output register)
  //   q4  + rounding constant
  //   q5  arithmetic shift right
  //   q6  + bias, saturate to 14 bits
  //   q7  epilogue, add mode: residual * ADD_MULT_X and value * ADD_MULT_H
  //   q8  products registered again
  //   q9  + rounding constants
  //   q10 arithmetic shifts
  //   q11 sum, saturate to 14 bits (add mode); ReLU; pair with the previous
  //       element. Without CTRL.add the q6 value passes through unchanged.
  // The producer only advances while the small output FIFO has room for
  // everything already in the pipeline.
  localparam int unsigned RQ_STAGES = 13;
  localparam int unsigned RQ_FIFO_D = 32;

  // Naming convention: a signal suffixed N holds the value valid at pipeline
  // stage qN (see the stage table above) — it is the same quantity shifted
  // one register further along each cycle, so only its first appearance is
  // commented below.
  logic               rq_v1, rq_v2, rq_v3, rq_v4, rq_v5, rq_v6;  // "an element is live here"
  logic               rq_last1, rq_last2, rq_last3, rq_last4, rq_last5, rq_last6;  // "this is the very last element"
  logic               rq_odd1, rq_odd2, rq_odd3, rq_odd4, rq_odd5, rq_odd6;  // "this element is the odd (2nd) one of its output pair"
  logic [NRW-1:0]     rq_row1;      // tile-RAM row n this element reads (q1)
  logic signed [31:0] rq_acc1;      // the raw int32 accumulator value read for this element
  logic signed [31:0] rq_mult1, rq_bias1, rq_bias2, rq_bias3, rq_bias4, rq_bias5;  // this row's multiplier / bias, carried alongside
  logic [5:0]         rq_sh1, rq_sh2, rq_sh3, rq_sh4;  // this row's shift amount, carried alongside
  logic signed [63:0] rq_prod2;     // acc * mult (q2, the DSP product)
  logic signed [63:0] rq_prod3;     // rq_prod2 re-registered (keeps the DSP's own output register)
  logic signed [63:0] rq_rnd4;      // rq_prod3 plus the rounding constant (q4)
  logic signed [63:0] rq_shf5;      // rq_rnd4 shifted right by the row's shift amount (q5)
  logic signed [31:0] rq_val6;      // final int16-ish value after bias + saturation (q6)
  logic [15:0]        rq_prev_q;          // even element, waiting for its pair
  logic [31:0]        rq_out_ptr_q;       // address of the next output word
  logic [31:0]        rq_row_ptr_q;       // start of the current output row

  // stage 0 -> 1
  logic rq_last_elem;  // the element about to enter the pipeline (q0) is the very last one
  assign rq_last_elem = (rq_n_q == n_rows_q - 1'b1) & (rq_m_q == rq_mlen - 1'b1);

  // Shifts "an element is live" down the 6-cycle pipeline, one stage per cycle.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rq_v1 <= 1'b0; rq_v2 <= 1'b0; rq_v3 <= 1'b0;
      rq_v4 <= 1'b0; rq_v5 <= 1'b0; rq_v6 <= 1'b0;
    end else begin
      rq_v1 <= rq_adv;
      rq_v2 <= rq_v1; rq_v3 <= rq_v2; rq_v4 <= rq_v3; rq_v5 <= rq_v4; rq_v6 <= rq_v5;
    end
  end
  always_ff @(posedge clk_i) begin
    // q1: the tile RAM and the parameter RAM answer this cycle
    rq_row1   <= rq_n_q;
    rq_odd1   <= rq_m_q[0];
    rq_last1  <= rq_last_elem;
    rq_mult1  <= pm_mult [rq_m_q];
    rq_sh1    <= pm_shift[rq_m_q];
    rq_bias1  <= pm_bias [rq_m_q];
    // q2: product
    rq_prod2  <= $signed(rq_acc1) * $signed(rq_mult1);
    rq_sh2    <= rq_sh1;  rq_bias2 <= rq_bias1;  rq_odd2 <= rq_odd1;  rq_last2 <= rq_last1;
    // q3: register (lets the DSP keep its output register)
    rq_prod3  <= rq_prod2;
    rq_sh3    <= rq_sh2;  rq_bias3 <= rq_bias2;  rq_odd3 <= rq_odd2;  rq_last3 <= rq_last2;
    // q4: rounding
    rq_rnd4   <= rq_prod3 + ((rq_sh3 != 6'd0) ? (64'sd1 <<< (rq_sh3 - 6'd1)) : 64'sd0);
    rq_sh4    <= rq_sh3;  rq_bias4 <= rq_bias3;  rq_odd4 <= rq_odd3;  rq_last4 <= rq_last3;
    // q5: shift
    rq_shf5   <= rq_rnd4 >>> rq_sh4;
    rq_bias5  <= rq_bias4;  rq_odd5 <= rq_odd4;  rq_last5 <= rq_last4;
    // q6: bias and saturation
    begin
      logic signed [32:0] sum;
      // part-selects are unsigned: cast before widening or negatives break
      sum = 33'($signed(rq_shf5[31:0])) + 33'(rq_bias5);
      rq_val6 <= (sum >  33'sd8191) ?  32'sd8191
               : (sum < -33'sd8191) ? -32'sd8191 : 32'(sum);
    end
    rq_odd6 <= rq_odd5;  rq_last6 <= rq_last5;
  end
  // Epilogue multipliers and shifts, latched at job start. No reset: they
  // are always written before a job uses them, and a register with an
  // asynchronous reset cannot be folded into the DSP48 it feeds (DPIR-1).
  always_ff @(posedge clk_i) begin
    if (state_q == ST_IDLE && start_strobe) begin
      add_mx_q <= reg2hw.add_mult_x.q;
      add_mh_q <= reg2hw.add_mult_h.q;
      add_sx_q <= reg2hw.add_shift.q[5:0];
      add_sh_q <= reg2hw.add_shift.q[13:8];
    end
  end

  // ---- epilogue (q2 .. q11)
  logic [NRW-1:0]     rq_row2;
  logic signed [15:0] rq_x3, rq_x4, rq_x5, rq_x6;       // residual element
  logic               rq_v7, rq_v8, rq_v9, rq_v10, rq_v11;
  logic               rq_odd7, rq_odd8, rq_odd9, rq_odd10, rq_odd11;
  logic               rq_last7, rq_last8, rq_last9, rq_last10, rq_last11;
  logic signed [31:0] rq_h7, rq_h8, rq_h9, rq_h10;      // q6 value carried along
  logic signed [63:0] rq_px7, rq_ph7, rq_px8, rq_ph8, rq_px9, rq_ph9, rq_px10, rq_ph10;
  logic signed [31:0] rq_val11;                         // final value

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rq_xcyc_q <= 1'b0;
      rq_v7 <= 1'b0; rq_v8 <= 1'b0; rq_v9 <= 1'b0; rq_v10 <= 1'b0; rq_v11 <= 1'b0;
    end else begin
      rq_xcyc_q <= rq_adv & add_q;
      rq_v7 <= rq_v6; rq_v8 <= rq_v7; rq_v9 <= rq_v8; rq_v10 <= rq_v9; rq_v11 <= rq_v10;
    end
  end
  always_ff @(posedge clk_i) begin
    if (rq_adv) rq_xaddr_q <= XOFF + (rq_m_q >> 1);
    rq_row2 <= rq_row1;
    // q2 -> q3: the residual word read the cycle after the accumulator
    rq_x3 <= rq_odd2 ? $signed(a_q[rq_row2[RW-1:0]][31:16])
                     : $signed(a_q[rq_row2[RW-1:0]][15:0]);
    rq_x4 <= rq_x3; rq_x5 <= rq_x4; rq_x6 <= rq_x5;
    // q7: the two products
    rq_px7 <= $signed(rq_x6)  * $signed({1'b0, add_mx_q});
    rq_ph7 <= $signed(rq_val6) * $signed({1'b0, add_mh_q});
    rq_h7  <= rq_val6;
    rq_odd7 <= rq_odd6; rq_last7 <= rq_last6;
    // q8
    rq_px8 <= rq_px7; rq_ph8 <= rq_ph7; rq_h8 <= rq_h7;
    rq_odd8 <= rq_odd7; rq_last8 <= rq_last7;
    // q9: rounding
    rq_px9 <= rq_px8 + ((add_sx_q != 6'd0) ? (64'sd1 <<< (add_sx_q - 6'd1)) : 64'sd0);
    rq_ph9 <= rq_ph8 + ((add_sh_q != 6'd0) ? (64'sd1 <<< (add_sh_q - 6'd1)) : 64'sd0);
    rq_h9  <= rq_h8;
    rq_odd9 <= rq_odd8; rq_last9 <= rq_last8;
    // q10: shifts
    rq_px10 <= rq_px9 >>> add_sx_q;
    rq_ph10 <= rq_ph9 >>> add_sh_q;
    rq_h10  <= rq_h9;
    rq_odd10 <= rq_odd9; rq_last10 <= rq_last9;
    // q11: sum and saturate (add mode), ReLU
    begin
      logic signed [32:0] sum;
      logic signed [31:0] v;
      sum = 33'($signed(rq_px10[31:0])) + 33'($signed(rq_ph10[31:0]));
      v = !add_q ? rq_h10
        : (sum >  33'sd8191) ?  32'sd8191
        : (sum < -33'sd8191) ? -32'sd8191 : 32'(sum);
      rq_val11 <= (relu_q && v < 0) ? 32'sd0 : v;
    end
    rq_odd11 <= rq_odd10; rq_last11 <= rq_last10;
  end

  // acc value: tile RAM read happens on the a_rd_addr presented at q0; a_q
  // is registered, so it is valid in q1 -- select the row there.
  // q12: the lookup table. 8192 words of two int16 entries (8 BRAM36),
  // written by RQ_LOAD_L and read here at v + 8192: the word (v + 8192) >> 1,
  // the half v[0] (8192 is even). The RAM's registered read is the q12
  // register; without CTRL.lut the q11 value passes through.
  (* ram_style = "block" *) logic [31:0] lut_ram [8192];
  logic [31:0]        lut_rd12;
  logic               rq_v12, rq_odd12, rq_last12, rq_hi12;
  logic signed [31:0] rq_pass12, rq_val12;
  logic [13:0]        lut_idx11;
  assign lut_idx11   = 14'(rq_val11 + 32'sd8192);
  assign lut_ld_done = (state_q == RQ_LOAD_L) & rd_valid & (lut_cnt_q == 13'd8191);
  always_ff @(posedge clk_i) begin
    if ((state_q == RQ_LOAD_L) && rd_valid) lut_ram[lut_cnt_q] <= rd_data;
    lut_rd12 <= lut_ram[lut_idx11[13:1]];
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) rq_v12 <= 1'b0;
    else         rq_v12 <= rq_v11;
  end
  always_ff @(posedge clk_i) begin
    rq_odd12  <= rq_odd11; rq_last12 <= rq_last11;
    rq_hi12   <= lut_idx11[0];
    rq_pass12 <= rq_val11;
  end
  assign rq_val12 = !lut_q ? rq_pass12
                  : rq_hi12 ? 32'($signed(lut_rd12[31:16])) : 32'($signed(lut_rd12[15:0]));
  assign rq_acc1 = !a16_q ? a_q[rq_row1[RW-1:0]]
                 : rq_row1[0] ? 32'($signed(a_q[rq_row1[RW-1:0]][31:16]))
                              : 32'($signed(a_q[rq_row1[RW-1:0]][15:0]));

  // "last pair of this output row" travels with the element
  logic rq_row_end0, rq_row_end1, rq_row_end2, rq_row_end3, rq_row_end4, rq_row_end5, rq_row_end6;
  logic rq_row_end7, rq_row_end8, rq_row_end9, rq_row_end10, rq_row_end11, rq_row_end12;
  assign rq_row_end0 = (rq_m_q == rq_mlen - 1'b1);
  always_ff @(posedge clk_i) begin
    rq_row_end1 <= rq_row_end0; rq_row_end2 <= rq_row_end1; rq_row_end3 <= rq_row_end2;
    rq_row_end4 <= rq_row_end3; rq_row_end5 <= rq_row_end4; rq_row_end6 <= rq_row_end5;
    rq_row_end7 <= rq_row_end6; rq_row_end8 <= rq_row_end7; rq_row_end9 <= rq_row_end8;
    rq_row_end10 <= rq_row_end9; rq_row_end11 <= rq_row_end10;
    rq_row_end12 <= rq_row_end11;
  end

  // Largest |out| of the job, for the consumer of the result (the residual
  // add needs the range of its operands and would otherwise scan them).
  // Output row statistics (CTRL.ostats): the rows reach q12 in order, so a
  // row counter and a running {max, min} suffice; at each row's end the
  // pair goes into a small LUT RAM, which RQ_STATS writes out.
  logic [31:0]        os_ram [NROWS];
  logic [NRW-1:0]     os_row_q;
  logic               os_first_q;
  logic signed [15:0] os_max_q, os_min_q;
  always_ff @(posedge clk_i) begin
    if (state_q != RQ_OUT) begin
      os_row_q   <= '0;
      os_first_q <= 1'b1;
    end else if (rq_v12) begin
      logic signed [15:0] v, mx, mn;
      v  = rq_val12[15:0];
      mx = (os_first_q || v > os_max_q) ? v : os_max_q;
      mn = (os_first_q || v < os_min_q) ? v : os_min_q;
      os_max_q <= mx;
      os_min_q <= mn;
      if (rq_last12 || rq_row_end12) begin
        os_ram[os_row_q[RW-1:0]] <= {mx, mn};
        os_row_q   <= os_row_q + 1'b1;
        os_first_q <= 1'b1;
      end else begin
        os_first_q <= 1'b0;
      end
    end
  end
  assign os_wr_req  = (state_q == RQ_STATS) & (t_q != {1'b0, n_rows_q});
  assign os_wr_addr = s_ptr_q + 32'(t_q) * 32'd4;
  assign os_wr_data = os_ram[t_q[RW-1:0]];
  // Result RAM (CTRL.onchip). The drain writes accumulator t of weight row
  // m at C_ADDR + m*C_STRIDE + t (words); a requantisation job reads its
  // chunk at A_ADDR + m*A_STRIDE + n, one word per cycle, into the tile
  // RAM as a bus load would (acc_valid / cr_rdata_q replace rd_valid /
  // rd_data). Saves the int32 results' trip to DDR3 and back.
  (* ram_style = "block" *) logic [31:0] cr_ram [CRW];
  logic [CRA-1:0] cr_base_q;             // request side: current row's first word
  logic [NRW-1:0] cr_n_q;                // request side: column
  logic [AW-1:0]  cr_m_q;                // request side: row
  logic           cr_act_q;              // request side: words left to read
  assign acc_valid = onchip_q ? cr_v_q : rd_valid;
  always_ff @(posedge clk_i) begin
    if ((state_q == ST_DRAIN) && onchip_q && t_is_acc && (t_q != n_wr))
      cr_ram[CRA'(c_ptr_q) + CRA'(t_q)] <= acc_q[t_q[RW-1:0]];
    cr_rdata_q <= cr_ram[cr_base_q + CRA'(cr_n_q)];
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cr_v_q   <= 1'b0;
      cr_act_q <= 1'b0;
    end else begin
      cr_v_q <= (state_q == RQ_LOAD_ACC) & onchip_q & cr_act_q;
      if (rq_p_done & onchip_q) begin
        cr_act_q <= 1'b1;
      end else if ((state_q == RQ_LOAD_ACC) & cr_act_q
                   & (cr_n_q == n_rows_q - 1'b1) & (cr_m_q == rq_mlen - 1'b1)) begin
        cr_act_q <= 1'b0;
      end
    end
  end
  always_ff @(posedge clk_i) begin
    if (rq_p_done) begin
      cr_base_q <= CRA'(a_addr_q);
      cr_n_q    <= '0;
      cr_m_q    <= '0;
    end else if ((state_q == RQ_LOAD_ACC) & cr_act_q) begin
      if (cr_n_q == n_rows_q - 1'b1) begin
        cr_n_q    <= '0;
        cr_m_q    <= cr_m_q + 1'b1;
        cr_base_q <= cr_base_q + CRA'(a_stride_q);
      end else begin
        cr_n_q <= cr_n_q + 1'b1;
      end
    end
  end
  logic [31:0] rq_amax_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rq_amax_q <= '0;
    end else if (start_strobe && start_requant) begin
      rq_amax_q <= '0;
    end else if (rq_v12) begin
      rq_amax_q <= (rq_val12 < 0) ? (32'(-rq_val12) > rq_amax_q ? 32'(-rq_val12) : rq_amax_q)
                                  : (32'(rq_val12)  > rq_amax_q ? 32'(rq_val12)  : rq_amax_q);
    end
  end

  // Pair the even element with the odd one and enqueue the word.
  logic        rq_fifo_push;
  logic [63:0] rq_fifo_wdata;   // {addr, data}
  assign rq_fifo_push  = rq_v12 & rq_odd12;
  assign rq_fifo_wdata = {rq_out_ptr_q, rq_val12[15:0], rq_prev_q};

  always_ff @(posedge clk_i) begin
    if (rq_v12 & ~rq_odd12) rq_prev_q <= rq_val12[15:0];
  end

  // Output FIFO and write issue. rq_out_ptr_q walks the output row: +4 per
  // pair, and jumps to the next row (C_STRIDE further) after the last pair.
  logic [63:0] rq_fifo [RQ_FIFO_D];    // queued {address, data} output words, awaiting the bus
  logic [$clog2(RQ_FIFO_D)-1:0] rq_fifo_wr_q;  // next slot to fill
  logic [$clog2(RQ_FIFO_D)-1:0] rq_fifo_rd_q;  // next slot to write out
  logic [$clog2(RQ_FIFO_D):0]   rq_fifo_cnt_q; // words currently queued
  logic        rq_fifo_pop;    // this cycle, the queued word at rq_fifo_rd_q is issued on the bus
  logic        rq_drained_q;    // every element has been enqueued

  assign rq_wr_req  = (rq_fifo_cnt_q != '0);
  assign rq_wr_addr = rq_fifo[rq_fifo_rd_q][63:32];
  assign rq_wr_data = rq_fifo[rq_fifo_rd_q][31:0];
  assign rq_fifo_pop = issue_wr & (state_q == RQ_OUT);

  // Produce while there is room for what is in flight plus a margin.
  logic rq_last_seen_q;        // the last element has entered the pipeline
  assign rq_adv = (state_q == RQ_OUT) & ~rq_drained_q & ~rq_last_seen_q & ~rq_xcyc_q
                & (rq_fifo_cnt_q < ($clog2(RQ_FIFO_D)+1)'(RQ_FIFO_D - RQ_STAGES - 1));
  assign rq_out_done = (state_q == RQ_OUT) & rq_drained_q & (rq_fifo_cnt_q == '0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rq_fifo_wr_q  <= '0;
      rq_fifo_rd_q  <= '0;
      rq_fifo_cnt_q <= '0;
      rq_drained_q  <= 1'b0;
      rq_last_seen_q <= 1'b0;
      rq_out_ptr_q  <= '0;
      rq_row_ptr_q  <= '0;
    end else begin
      if (state_q != RQ_OUT) begin
        rq_drained_q   <= 1'b0;
        rq_last_seen_q <= 1'b0;
        rq_out_ptr_q   <= c_ptr_q;
        rq_row_ptr_q   <= c_ptr_q;
      end
      if (rq_adv & rq_last_elem) rq_last_seen_q <= 1'b1;
      if (rq_v12 & rq_last12)    rq_drained_q   <= 1'b1;
      if (rq_fifo_push) begin
        rq_fifo[rq_fifo_wr_q] <= rq_fifo_wdata;
        rq_fifo_wr_q <= rq_fifo_wr_q + 1'b1;
        // next word: along the row, or the next row after the row's last pair
        if (rq_last12 || rq_row_end12) begin
          rq_out_ptr_q <= rq_row_ptr_q + c_stride_q;
          rq_row_ptr_q <= rq_row_ptr_q + c_stride_q;
        end else begin
          rq_out_ptr_q <= rq_out_ptr_q + 32'd4;
        end
      end
      if (rq_fifo_pop) rq_fifo_rd_q <= rq_fifo_rd_q + 1'b1;
      unique case ({rq_fifo_push, rq_fifo_pop})
        2'b10:   rq_fifo_cnt_q <= rq_fifo_cnt_q + 1'b1;
        2'b01:   rq_fifo_cnt_q <= rq_fifo_cnt_q - 1'b1;
        default: ;
      endcase
    end
  end

  // Diagnostics: what the FSM is waiting on. Declared here, after the signals
  // it samples, so the testbench compiler accepts it.
  // Per-job bus counters. The question a stalled job has to answer is whether
  // its outstanding read was never accepted, or accepted and never answered;
  // rb_cnt alone cannot distinguish those.
  //
  // wr_issue_cnt_q / wr_ack_cnt_q count writes specifically, exposed on dbg2
  // as {issued, acked} for the job just finished. Software compares them
  // against n_rows * m_len, the number of writes a job owes.
  //
  // History: these were added to test the theory that an accumulator word
  // lost on hardware (always a tile's final row) was never issued by this
  // block. They showed issued == acked == owed on every job, which
  // exonerated the accelerator: the write reached the bus and was acked, and
  // the data was lost downstream. The real fault was rvlab_ddr_block_cache
  // writing back a line with data one cycle stale (fixed there). The counters
  // stay because "did the block issue every write it owed" is a question
  // worth being able to answer in one register read.
  logic [31:0] acc_cnt_q;  // A-channel requests accepted this job (reads + writes + retries)
  logic [31:0] rsp_cnt_q;  // D-channel responses seen this job (reads + write acks)
  logic [15:0] wr_issue_cnt_q;  // writes issued this job
  logic [15:0] wr_ack_cnt_q;    // write acks received this job
  // All four reset to 0 at the start of every job, so they read out as
  // "this job's" counts rather than a running total.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      acc_cnt_q      <= '0;
      rsp_cnt_q      <= '0;
      wr_issue_cnt_q <= '0;
      wr_ack_cnt_q   <= '0;
    end else if (start_strobe) begin
      acc_cnt_q      <= '0;
      rsp_cnt_q      <= '0;
      wr_issue_cnt_q <= '0;
      wr_ack_cnt_q   <= '0;
    end else begin
      if (areq_valid_q & tl_host_i.a_ready) acc_cnt_q <= acc_cnt_q + 32'd1;
      if (tl_host_i.d_valid)                rsp_cnt_q <= rsp_cnt_q + 32'd1;
      if (issue_wr)                         wr_issue_cnt_q <= wr_issue_cnt_q + 16'd1;
      if (rsp_wr)                           wr_ack_cnt_q   <= wr_ack_cnt_q + 16'd1;
    end
  end

  // DBG: a live snapshot of what the FSM is doing/waiting on right now.
  assign hw2reg.dbg.d  = {rd_left_q[15:0], err_q, tl_host_i.d_valid,
                          tl_host_i.a_ready, areq_valid_q,
                          4'(wr_out_q), 4'(rb_cnt_q), 4'(state_q)};
  // DBG2: writes {issued, acked} for the job just finished.
  // dbg2 used to expose areq_q.a_address; the write counters are worth more.
  assign hw2reg.dbg2.d = {wr_issue_cnt_q, wr_ack_cnt_q};
  // DBG3: A-channel requests accepted this job.
  assign hw2reg.dbg3.d = acc_cnt_q;
  // DBG4: {retries performed (all-time), D-channel responses this job}.
  assign hw2reg.dbg4.d = {retry_n_q[15:0], rsp_cnt_q[15:0]};
  assign hw2reg.rq_amax.d = rq_amax_q;

`ifndef SYNTHESIS
  // Simulation-only stall watchdog. If the block is busy but nothing has
  // moved for a long time, dump what it is waiting on.
  logic [OUTSTANDING-1:0] rb_val_packed;  // rb_val_q as one vector, for a single %b in $display
  always_comb for (int i = 0; i < int'(OUTSTANDING); i++) rb_val_packed[i] = rb_val_q[i];

  int stall_cnt;  // cycles since anything last moved; dumps state once it crosses the threshold
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      stall_cnt <= 0;
    end else if (state_q == ST_IDLE || issue_rd || issue_wr || rsp_rd || rsp_wr || adv) begin
      stall_cnt <= 0;
    end else begin
      stall_cnt <= stall_cnt + 1;
      if (stall_cnt == 3000) begin
        $display("student_gemm STALL @%0t state=%0d m=%0d t=%0d kcnt=%0d n_rows=%0d",
                 $time, state_q, m_q, t_q, kcnt_q, n_rows_q);
        $display("  rd_left=%0d rb_cnt=%0d rb_wr=%0d rb_rd=%0d rb_val=%b wr_out=%0d",
                 rd_left_q, rb_cnt_q, rb_wr_q, rb_rd_q, rb_val_packed, wr_out_q);
        $display("  rd_valid=%b wbuf_val=%b areq_valid=%b a_ready=%b d_valid=%b",
                 rd_valid, wbuf_val_q, areq_valid_q, tl_host_i.a_ready, tl_host_i.d_valid);
        $display("  sel_wr=%b sel_rd=%b rd_can_issue=%b wr_req=%b pipe_idle=%b",
                 sel_wr, sel_rd, rd_can_issue, wr_req, pipe_idle);
      end
    end
  end

  // Simulation-only. The TL-UL sockets push the whole request struct through a
  // prim_fifo_sync whose DataKnown_A assertion fails on any X, but by then the
  // offending field is no longer identifiable. Catch it at the source.
  logic x_seen_a, x_seen_d;  // latch so each channel reports its first X only once
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      x_seen_a <= 1'b0;
      x_seen_d <= 1'b0;
    end else begin
      if (!x_seen_d && tl_host_i.d_valid &&
          $isunknown({tl_host_i.d_opcode, tl_host_i.d_size, tl_host_i.d_source,
                      tl_host_i.d_data, tl_host_i.d_error})) begin
        x_seen_d <= 1'b1;
        $display("student_gemm: X on D channel @%0t state=%0d opcode=%b size=%b source=%b error=%b data=%h",
                 $time, state_q, tl_host_i.d_opcode, tl_host_i.d_size,
                 tl_host_i.d_source, tl_host_i.d_error, tl_host_i.d_data);
      end
      if (!x_seen_a && tl_host_o.a_valid &&
          $isunknown({tl_host_o.a_opcode, tl_host_o.a_param, tl_host_o.a_size,
                      tl_host_o.a_source, tl_host_o.a_address, tl_host_o.a_mask,
                      tl_host_o.a_data, tl_host_o.a_user})) begin
        x_seen_a <= 1'b1;
        $display("student_gemm: X on A channel @%0t state=%0d opcode=%b size=%b source=%b addr=%h mask=%b data=%h",
                 $time, state_q, tl_host_o.a_opcode, tl_host_o.a_size,
                 tl_host_o.a_source, tl_host_o.a_address, tl_host_o.a_mask,
                 tl_host_o.a_data);
      end
    end
  end
`endif

endmodule
