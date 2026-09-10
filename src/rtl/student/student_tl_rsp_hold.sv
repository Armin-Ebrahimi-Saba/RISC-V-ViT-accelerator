// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// TL-UL response skid buffer for the DDR3 cache.
//
// rvlab_ddr_block_cache drives fe_rsp_o.d_valid straight from its tag-hit
// signal and never looks at fe_req_i.d_ready, and rvlab_ddr_cache passes both
// through to TL-UL unchanged.  TL-UL requires a response to be held until the
// cycle where d_valid and d_ready are both high, so any host that happens to
// be unready on the single cycle d_valid is asserted loses that response for
// good.  tlul_socket_m1 computes the DDR3 port's d_ready combinationally from
// the routed host's d_ready, so this is reachable in the real system.
//
// student_gemm survives it with a retry timer.  The CPU cannot: a lost
// response leaves cv32e40p stalled on a load forever, and a core that cannot
// retire an instruction cannot enter debug mode either.
//
// This module sits between the bus and the cache and makes the cache's own
// d_ready permanently high, so a response is always taken on the cycle it is
// offered.  It then holds that response until the real host accepts it, and
// throttles the A channel so no more responses can be in flight than it has
// room to hold.  The cache's behaviour is unchanged; what changes is that it
// is no longer asked to do something it cannot do.
//
// DEPTH is the number of responses that may be outstanding at once.  Two is
// enough to keep a single-outstanding host from ever stalling on the A
// channel while a response is still being accepted.

module student_tl_rsp_hold #(
    parameter int unsigned DEPTH = 2
) (
    input  logic clk_i,
    input  logic rst_ni,

    // Bus side (this module looks like the device).
    input  tlul_pkg::tl_h2d_t tl_h_i,
    output tlul_pkg::tl_d2h_t tl_h_o,

    // Cache side (this module looks like the host, and is always ready).
    output tlul_pkg::tl_h2d_t tl_d_o,
    input  tlul_pkg::tl_d2h_t tl_d_i
);

  localparam int unsigned PTRW = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
  localparam int unsigned CNTW = $clog2(DEPTH + 1);

  // ------------------------------------------------------------ response fifo
  // Only the payload is stored.  d_valid and a_ready are recomputed from the
  // fifo occupancy and the cache's own a_ready, so storing them would leave
  // flops with no load for synthesis to strip.
  typedef struct packed {
    tlul_pkg::tl_d_op_e                d_opcode;
    logic                       [2:0]  d_param;
    logic  [top_pkg::TL_SZW-1:0]       d_size;
    logic  [top_pkg::TL_AIW-1:0]       d_source;
    logic  [top_pkg::TL_DIW-1:0]       d_sink;
    logic   [top_pkg::TL_DW-1:0]       d_data;
    logic  [top_pkg::TL_DUW-1:0]       d_user;
    logic                              d_error;
  } rsp_payload_t;

  function automatic rsp_payload_t to_payload(input tlul_pkg::tl_d2h_t r);
    to_payload = '{d_opcode: r.d_opcode, d_param: r.d_param, d_size: r.d_size,
                   d_source: r.d_source, d_sink: r.d_sink, d_data: r.d_data,
                   d_user: r.d_user, d_error: r.d_error};
  endfunction

  rsp_payload_t fifo_q [DEPTH];
  logic [PTRW-1:0]   rd_ptr_q, wr_ptr_q;
  logic [CNTW-1:0]   occ_q;

  wire fifo_push = tl_d_i.d_valid;                    // always accepted
  wire fifo_pop  = tl_h_o.d_valid & tl_h_i.d_ready;

  // ------------------------------------------------------------ a-channel credit
  // pend_q counts responses promised but not yet handed to the host: one per
  // request the cache has accepted, cleared when the host takes the response.
  // Holding it at DEPTH is what guarantees the fifo can never overflow, which
  // is what lets d_ready be tied high above.
  logic [CNTW-1:0] pend_q;
  wire credit_ok = (pend_q < CNTW'(DEPTH));

  wire a_beat = tl_d_o.a_valid & tl_d_i.a_ready;

  always_comb begin
    tl_d_o         = tl_h_i;
    tl_d_o.a_valid = tl_h_i.a_valid & credit_ok;
    tl_d_o.d_ready = 1'b1;               // the whole point of the module

    tl_h_o          = '0;
    tl_h_o.d_opcode = fifo_q[rd_ptr_q].d_opcode;
    tl_h_o.d_param  = fifo_q[rd_ptr_q].d_param;
    tl_h_o.d_size   = fifo_q[rd_ptr_q].d_size;
    tl_h_o.d_source = fifo_q[rd_ptr_q].d_source;
    tl_h_o.d_sink   = fifo_q[rd_ptr_q].d_sink;
    tl_h_o.d_data   = fifo_q[rd_ptr_q].d_data;
    tl_h_o.d_user   = fifo_q[rd_ptr_q].d_user;
    tl_h_o.d_error  = fifo_q[rd_ptr_q].d_error;
    tl_h_o.d_valid  = (occ_q != '0);
    tl_h_o.a_ready  = tl_d_i.a_ready & credit_ok;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rd_ptr_q <= '0;
      wr_ptr_q <= '0;
      occ_q    <= '0;
      pend_q   <= '0;
      for (int i = 0; i < int'(DEPTH); i++) fifo_q[i] <= '0;
    end else begin
      if (fifo_push) begin
        fifo_q[wr_ptr_q] <= to_payload(tl_d_i);
        wr_ptr_q <= (DEPTH == 1) ? '0
                  : (wr_ptr_q == PTRW'(DEPTH - 1)) ? '0 : wr_ptr_q + 1'b1;
      end
      if (fifo_pop) begin
        rd_ptr_q <= (DEPTH == 1) ? '0
                  : (rd_ptr_q == PTRW'(DEPTH - 1)) ? '0 : rd_ptr_q + 1'b1;
      end

      unique case ({fifo_push, fifo_pop})
        2'b10:   occ_q <= occ_q + 1'b1;
        2'b01:   occ_q <= occ_q - 1'b1;
        default: ;
      endcase

      unique case ({a_beat, fifo_pop})
        2'b10:   pend_q <= pend_q + 1'b1;
        2'b01:   pend_q <= pend_q - 1'b1;
        default: ;
      endcase
    end
  end

`ifndef SYNTHESIS
  // If either of these ever fires the throttle is wrong, and a response is
  // being dropped again -- silently, which is exactly what made the original
  // defect so expensive to find.
  assert property (@(posedge clk_i) disable iff (!rst_ni)
                   fifo_push |-> (occ_q < CNTW'(DEPTH)) || fifo_pop)
    else $error("student_tl_rsp_hold: response fifo overflow");
  assert property (@(posedge clk_i) disable iff (!rst_ni)
                   fifo_pop |-> (pend_q != '0))
    else $error("student_tl_rsp_hold: pending counter underflow");
`endif

endmodule
