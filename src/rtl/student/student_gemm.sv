// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// int8 x int16 GEMM accelerator for the Depth-Anything V2 engine.
// -------------------------------------------------------------------------
//
// The whole network spends >98% of its time in dav2_qgemm(), which computes
//
//   C[m][t] = sum_{k} A[t][k] * W[m][k]      A: int16, W: int8, C: int32
//
// The CV32E40P is a scalar in-order core without SIMD, so it needs several
// cycles per multiply-accumulate. This block does NROWS MACs per cycle by
// keeping a tile of NROWS activation rows resident in block RAM and streaming
// the (much larger) weight matrix past it exactly once.
//
// Dataflow
// --------
//
//   1. LOAD_A  A tile (NROWS x K int16) is read from memory into NROWS
//              private block RAMs, one per row of the tile.
//   2. MAC     The weight matrix is read as one contiguous byte stream. Each
//              32-bit beat carries four int8 weights; each weight is
//              broadcast to all NROWS multipliers, which each pair it with
//              their own A element. One weight row m therefore takes K cycles
//              and produces NROWS int32 accumulators.
//   3. DRAIN   The NROWS accumulators of row m are written back as one
//              contiguous run of NROWS words, then the next m starts.
//
// All three streams are purely sequential in memory, which is what the
// direct-mapped DDR3 last-level cache wants.
//
// The bus is the limiting resource: one 32-bit TL-UL beat feeds 4*NROWS MACs,
// so NROWS = 64 (the board setting, see student.sv) needs under 0.1
// beats/cycle to stay compute-bound -- and the weight matrix is streamed once
// per tile, so a bigger tile also means fewer passes over it. To get
// there the read side keeps up to OUTSTANDING requests in flight and
// reassembles the responses in order via a small reorder buffer indexed by
// a_source, so it does not pay the full memory latency per beat.
//
// Everything the software contract needs is documented in
// src/design/reggen/student_gemm.hjson.
//
// Where this fits
// ---------------
//
//   src/rtl/student/student.sv          instantiates this block, gives it a
//                                       register window and a host port
//   src/design/reggen/student_gemm.hjson the register map (generated into
//                                       student_gemm_reg_top / _reg_pkg)
//   src/sw/project/dav2_accel.c         the driver: tiles a GEMM into jobs
//                                       of NROWS rows and polls STATUS
//   src/tb/student_gemm_tb.sv           module test against ideal memory
//   src/tb/student_gemm_ddrpath_tb.sv   same, through the real DDR3 cache
//   src/tb/student_gemm_droprsp_tb.sv   proves the lost-response retry works
//
// Terms used below: a "beat" is one 32-bit transfer on the TL-UL bus; a
// "tile" is the NROWS activation rows a job processes; "MAC" is one
// multiply-accumulate; "requantisation" (done in software, not here) scales
// the int32 accumulators back to int16 activations.

module student_gemm #(
  // Activation rows held in the tile == multipliers == MACs per cycle.
  parameter int unsigned NROWS       = 16,
  // Largest supported reduction length K (the model needs 1536).
  parameter int unsigned KMAX        = 2048,
  // Read requests in flight (also the write-ack credit). Power of two.
  parameter int unsigned OUTSTANDING = 8,
  // Requests actually allowed in flight at once. student.sv sets this to 1
  // for the DDR3-facing instance. The reason is the rvlab cache's response
  // handshake: it pulses d_valid for a single cycle without consulting
  // d_ready, so a response that arrives while this block is busy with
  // another is lost. With one request in flight the response can always be
  // taken the cycle it appears. (An earlier version of this comment blamed
  // rvlab_ddr_prefetch; that block is now bypassed for an unrelated aliasing
  // defect, and the cache constraint above still applies.) Against ideal
  // BRAM the full OUTSTANDING depth works.
  parameter int unsigned MAX_INFLIGHT = OUTSTANDING,
  // Cycles an outstanding read may go unanswered before it is re-issued. The
  // rvlab DDR3 cache pulses d_valid for one cycle without consulting d_ready,
  // so a response can be lost outright -- measured on hardware as
  // accepted=444, responses=443. Zero disables recovery. Must comfortably
  // exceed worst-case memory latency so it never fires on a slow-but-live
  // transaction.
  // 2048 cycles = 41 us at 50 MHz. Measured on hardware: real work costs
  // 8.3 cycles/beat, so this leaves a ~250x margin over typical latency and
  // still far exceeds a miss-plus-refill. The original 65536 was a guess made
  // before latency could be measured, and cost 383 x 65536 = 25.1M cycles per
  // tile -- 98% of the tile time -- purely idling before each re-issue.
  parameter int unsigned RETRY_CYCLES = 32'd2048,
  // DEADLOCKS -- do not enable. Blocking writes while reads are outstanding
  // hangs in ST_DRAIN: the read engine prefetches weight beats that are only
  // consumed in ST_MAC, so rb_cnt never falls to zero, so no write can issue,
  // so ST_DRAIN never completes. Kept only to document the dead end.
  parameter bit          STRICT_SERIAL = 1'b0
) (
  input logic clk_i,
  input logic rst_ni,

  // Register interface (device)
  input  tlul_pkg::tl_h2d_t tl_i,
  output tlul_pkg::tl_d2h_t tl_o,

  // Memory interface (host)
  input  tlul_pkg::tl_d2h_t tl_host_i,
  output tlul_pkg::tl_h2d_t tl_host_o
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

  student_gemm_reg2hw_t reg2hw;
  student_gemm_hw2reg_t hw2reg;

  student_gemm_reg_top reg_top_i (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .devmode_i('1)
  );

  logic busy_q, done_q, err_q;
  logic [31:0] cycles_q;

  assign hw2reg.status.d = {err_q, done_q, busy_q};
  assign hw2reg.caps.d   = {8'd0, 16'(KMAX), 8'(NROWS)};
  assign hw2reg.cycles.d = cycles_q;

  logic start_strobe;
  assign start_strobe = reg2hw.ctrl.qe & reg2hw.ctrl.q;

  // Configuration snapshot, taken when a job starts so software may reprogram
  // the registers for the next tile while this one runs.
  logic [31:0]    w_addr_q, c_stride_q;
  logic [31:0]    w_stride_q;                // bytes between W rows, never 0 here
  logic [KCW-1:0] k_len_q;
  logic [15:0]    m_len_q;
  logic [NRW-1:0] n_rows_q;

  // ------------------------------------------------------------- control FSM

  typedef enum logic [2:0] {
    ST_IDLE,
    ST_LOAD_A,
    ST_MAC,
    ST_MAC_TAIL,
    ST_DRAIN,
    ST_FINISH
  } state_e;

  state_e state_q, state_d;

  // ---------------------------------------------------------------- bus side
  //
  // One registered request is presented on the A channel at a time. A new one
  // is loaded whenever the current one has been accepted (or none is pending),
  // which keeps a_valid/a_* stable as TL-UL requires. Writes win arbitration:
  // they are rare (NROWS words per weight row) and the read stream has the
  // reorder buffer to absorb the resulting bubble.

  tlul_pkg::tl_h2d_t areq_q;
  logic              areq_valid_q;
  logic              load_next;
  logic              req_accepted;


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
  logic [31:0]  rd_addr_q;
  logic [31:0]  rd_left_q;      // beats not yet requested, all rows
  logic [31:0]  rd_row_base_q;  // first address of the current row
  logic [31:0]  rd_row_beats_q; // beats per row
  logic [31:0]  rd_row_left_q;  // beats left in the current row
  logic [31:0]  rd_stride_q;    // bytes from one row start to the next
  logic         rd_can_issue;

  logic [31:0]  rb_data_q [OUTSTANDING];
  logic         rb_val_q  [OUTSTANDING];
  logic [SW-1:0] rb_wr_q, rb_rd_q;
  logic [CW-1:0] rb_cnt_q;      // issued but not yet consumed

  logic         rd_valid;       // next beat, in order, is available
  logic [31:0]  rd_data;
  logic         rd_pop;

  assign rd_valid     = (rb_cnt_q != '0) & rb_val_q[rb_rd_q];
  assign rd_data      = rb_data_q[rb_rd_q];

  // Write stream -----------------------------------------------------------
  logic         wr_req;
  logic [31:0]  wr_addr, wr_data;
  logic [CW-1:0] wr_out_q;      // writes issued without an ack yet
  logic [SW-1:0] wr_src_q;

  logic sel_wr, sel_rd, issue_wr, issue_rd;

  // Lost-response recovery. Every issued read remembers its address in its
  // reorder slot; when the head slot stays empty for RETRY_CYCLES after the
  // bus has gone silent, that one request is issued again. This works for
  // any MAX_INFLIGHT: with several in flight the later responses keep
  // arriving, the head stays empty, issue stops once the slots are full, and
  // the ensuing silence trips the timer.
  logic [31:0]   rb_addr_q [OUTSTANDING];
  logic [31:0]   retry_addr;
  logic [SW-1:0] retry_slot;
  assign retry_addr = rb_addr_q[rb_rd_q];
  assign retry_slot = rb_rd_q;
  logic [31:0]   retry_cnt_q;
  logic          retry_pending_q;
  logic          sel_retry, issue_retry, retry_expired;
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
  logic          rsp_rd, rsp_wr;
  logic [SW-1:0] rsp_slot;

  assign rsp_rd   = tl_host_i.d_valid & (tl_host_i.d_opcode == tlul_pkg::AccessAckData);
  assign rsp_wr   = tl_host_i.d_valid & (tl_host_i.d_opcode == tlul_pkg::AccessAck);
  assign rsp_slot = tl_host_i.d_source[SW-1:0];

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
    always_ff @(posedge clk_i) begin
      if (rsp_rd && (rsp_slot == SW'(s))) rb_data_q[s] <= tl_host_i.d_data;
      if (issue_rd && (rb_wr_q == SW'(s))) rb_addr_q[s] <= rd_addr_q;
    end
  end

  // --------------------------------------------------------------- A tile RAM

  logic [AW-1:0]    a_wr_addr, a_rd_addr;
  logic [31:0]      a_wr_data;
  logic [NROWS-1:0] a_we;
  logic [31:0]      a_q [NROWS];

  for (genvar r = 0; r < int'(NROWS); r++) begin : gen_arow
    logic [31:0] mem [KWORDS];
    always_ff @(posedge clk_i) begin
      if (a_we[r]) mem[a_wr_addr] <= a_wr_data;
      a_q[r] <= mem[a_rd_addr];
    end
  end

  // ------------------------------------------------------------- A load phase

  logic [AW-1:0]  a_ld_word_q;  // word index inside the current row
  logic [NRW-1:0] a_ld_row_q;
  logic [AW-1:0] k_words;       // 32-bit words per activation row

  assign k_words = AW'(k_len_q >> 1);

  assign a_wr_data = rd_data;
  always_comb begin
    a_we      = '0;
    a_wr_addr = a_ld_word_q;
    if ((state_q == ST_LOAD_A) && rd_valid) a_we[a_ld_row_q[RW-1:0]] = 1'b1;
  end

  // ---------------------------------------------------------------- MAC phase

  logic [31:0]    wbuf_q;
  logic           wbuf_val_q;
  logic [1:0]     wsel_q;
  logic [KCW-1:0] kcnt_q;
  logic [15:0]    m_q;

  logic adv, wlast, klast;

  assign adv   = (state_q == ST_MAC) & wbuf_val_q;
  assign wlast = (wsel_q == 2'd3);
  assign klast = adv & (kcnt_q == (k_len_q - 1'b1));

  assign a_rd_addr = AW'(kcnt_q >> 1);

  always_comb begin
    rd_pop = 1'b0;
    if (state_q == ST_LOAD_A) begin
      rd_pop = rd_valid;
    end else if (state_q == ST_MAC) begin
      // Refill an empty buffer, or replace the buffer as its last byte is
      // consumed, so that a beat is never wasted waiting a cycle.
      rd_pop = rd_valid & (~wbuf_val_q | wlast);
    end
  end

  logic signed [7:0] wbyte;
  always_comb begin
    case (wsel_q)
      2'd0:    wbyte = $signed(wbuf_q[7:0]);
      2'd1:    wbyte = $signed(wbuf_q[15:8]);
      2'd2:    wbyte = $signed(wbuf_q[23:16]);
      default: wbyte = $signed(wbuf_q[31:24]);
    endcase
  end

  // Three-stage MAC pipeline:
  //   s0  present the A-tile read address (combinational from kcnt_q)
  //   s1  A word arrives; select the half addressed by k[0]
  //   s2  multiply-accumulate (maps onto a DSP48E1 with A/B/P registers)
  logic              v_s1, v_s2;
  logic              ksel_s1;
  logic signed [7:0] w_s1, w_s2;
  logic signed [15:0] a_s2 [NROWS];
  logic signed [31:0] acc_q [NROWS];
  logic              acc_clr;

  // Only the valid bits are reset. The data registers (weight byte, A
  // operand, accumulator) deliberately have no asynchronous reset: DSP48E1
  // internal registers have synchronous reset only, and Vivado will not pull
  // a register with an async reset into the DSP -- 64 x 32 accumulators
  // worth of methodology warnings (DPIR-1) and a longer path. The
  // accumulators are cleared synchronously at job start and after every
  // drain, so they never hold anything a result depends on before then.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v_s1 <= 1'b0;
      v_s2 <= 1'b0;
    end else begin
      v_s1 <= adv;
      v_s2 <= v_s1;
    end
  end
  always_ff @(posedge clk_i) begin
    ksel_s1 <= kcnt_q[0];
    w_s1    <= wbyte;
    w_s2    <= w_s1;
  end

  for (genvar r = 0; r < int'(NROWS); r++) begin : gen_pe
    always_ff @(posedge clk_i) begin
      a_s2[r] <= ksel_s1 ? $signed(a_q[r][31:16]) : $signed(a_q[r][15:0]);
      if (acc_clr)   acc_q[r] <= '0;
      else if (v_s2) acc_q[r] <= acc_q[r] + (a_s2[r] * w_s2);
    end
  end

  // -------------------------------------------------------------- drain phase

  logic [NRW-1:0] t_q;          // output word inside the current run
  logic [31:0]  c_ptr_q;        // base of the current run

  assign wr_req  = (state_q == ST_DRAIN) & (t_q != n_rows_q);
  assign wr_addr = c_ptr_q + 32'(t_q) * 32'd4;
  assign wr_data = acc_q[t_q[RW-1:0]];

  // ------------------------------------------------------------ state machine

  logic pipe_idle;
  assign pipe_idle = ~v_s1 & ~v_s2;

  logic a_load_done;
  assign a_load_done = rd_valid & (a_ld_row_q == (n_rows_q - 1'b1))
                                & (a_ld_word_q == (k_words - 1'b1));

  always_comb begin
    state_d = state_q;
    unique case (state_q)
      ST_IDLE:     if (start_strobe)             state_d = ST_LOAD_A;
      ST_LOAD_A:   if (a_load_done)              state_d = ST_MAC;
      ST_MAC:      if (klast)                    state_d = ST_MAC_TAIL;
      ST_MAC_TAIL: if (pipe_idle)                state_d = ST_DRAIN;
      ST_DRAIN:    if (t_q == n_rows_q)
                     state_d = (m_q == (m_len_q - 1'b1)) ? ST_FINISH : ST_MAC;
      ST_FINISH:   if (wr_out_q == '0)           state_d = ST_IDLE;
      default:                                   state_d = ST_IDLE;
    endcase
  end

  // Clear the accumulators at job start and for the next weight row while
  // the drain finishes. (Job start matters: they have no reset.)
  assign acc_clr = ((state_q == ST_DRAIN) & (t_q == n_rows_q))
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

      unique case (state_q)
        ST_IDLE: begin
          if (start_strobe) begin
            w_addr_q   <= reg2hw.w_addr.q;
            c_stride_q <= reg2hw.c_stride.q;
            k_len_q    <= KCW'(reg2hw.k_len.q);
            m_len_q    <= reg2hw.m_len.q;
            n_rows_q   <= NRW'(reg2hw.n_rows.q);
            c_ptr_q    <= reg2hw.c_addr.q;

            // A stride of 0 means "contiguous": one row length apart.
            w_stride_q <= (reg2hw.w_stride.q != 32'd0) ? reg2hw.w_stride.q
                                                       : 32'(reg2hw.k_len.q);

            // A tile: n_rows rows of k/2 beats, starting at a_addr.
            rd_addr_q      <= reg2hw.a_addr.q;
            rd_row_base_q  <= reg2hw.a_addr.q;
            rd_row_beats_q <= 32'(reg2hw.k_len.q >> 1);
            rd_row_left_q  <= 32'(reg2hw.k_len.q >> 1);
            rd_stride_q    <= (reg2hw.a_stride.q != 32'd0) ? reg2hw.a_stride.q
                                                           : 32'(reg2hw.k_len.q) * 32'd2;
            rd_left_q      <= 32'(reg2hw.n_rows.q) * 32'(reg2hw.k_len.q >> 1);

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
          if (rd_valid) begin
            if (a_ld_word_q == (k_words - 1'b1)) begin
              a_ld_word_q <= '0;
              a_ld_row_q  <= a_ld_row_q + 1'b1;
            end else begin
              a_ld_word_q <= a_ld_word_q + 1'b1;
            end
          end
          if (a_load_done) begin
            // Reprogram the read engine for the weight stream: M rows of
            // K/4 beats, read exactly once for the whole tile.
            rd_addr_q      <= w_addr_q;
            rd_row_base_q  <= w_addr_q;
            rd_row_beats_q <= 32'(k_len_q >> 2);
            rd_row_left_q  <= 32'(k_len_q >> 2);
            rd_stride_q    <= w_stride_q;
            rd_left_q      <= 32'(m_len_q) * 32'(k_len_q >> 2);
          end
        end

        ST_MAC: begin
          if (adv) begin
            kcnt_q <= kcnt_q + 1'b1;
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
          if (issue_wr) t_q <= t_q + 1'b1;
          if (t_q == n_rows_q) begin
            m_q     <= m_q + 1'b1;
            c_ptr_q <= c_ptr_q + c_stride_q;
            // Leaving t_q at n_rows_q would index an accumulator whose A-tile
            // row was never loaded when the tile is partial (n_rows < NROWS).
            t_q     <= '0;
          end
        end

        ST_FINISH: begin
          if (wr_out_q == '0) begin
            busy_q <= 1'b0;
            done_q <= 1'b1;
          end
        end

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
  logic [31:0] acc_cnt_q, rsp_cnt_q;
  logic [15:0] wr_issue_cnt_q, wr_ack_cnt_q;
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

  assign hw2reg.dbg.d  = {rd_left_q[15:0], err_q, tl_host_i.d_valid,
                          tl_host_i.a_ready, areq_valid_q,
                          4'(wr_out_q), 4'(rb_cnt_q), 1'b0, 3'(state_q)};
  // dbg2 used to expose areq_q.a_address; the write counters are worth more.
  assign hw2reg.dbg2.d = {wr_issue_cnt_q, wr_ack_cnt_q};
  assign hw2reg.dbg3.d = acc_cnt_q;
  assign hw2reg.dbg4.d = {retry_n_q[15:0], rsp_cnt_q[15:0]};

`ifndef SYNTHESIS
  // Simulation-only stall watchdog. If the block is busy but nothing has
  // moved for a long time, dump what it is waiting on.
  logic [OUTSTANDING-1:0] rb_val_packed;
  always_comb for (int i = 0; i < int'(OUTSTANDING); i++) rb_val_packed[i] = rb_val_q[i];

  int stall_cnt;
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
  logic x_seen_a, x_seen_d;
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
