// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// Behavioural TL-UL memory device for unit testbenches.
//
// It deliberately does two things a simple model would not:
//
//   * it accepts up to DEPTH requests before answering any of them, so a host
//     that pipelines its reads is actually exercised, and
//   * it answers them out of order (the arbitration pointer rotates every
//     cycle), so a host that assumes responses come back in issue order fails
//     here rather than on the FPGA.
//
// Simulation only.

module tlul_test_mem #(
    parameter int unsigned SIZE_WORDS = 65536,
    parameter logic [31:0] BASE_ADDR  = 32'h8000_0000,
    parameter int unsigned DEPTH      = 8,     // requests accepted in flight
    parameter int unsigned MIN_LAT    = 2,
    parameter int unsigned MAX_LAT    = 12
) (
    input  logic              clk_i,
    input  logic              rst_ni,
    input  tlul_pkg::tl_h2d_t tl_i,
    output tlul_pkg::tl_d2h_t tl_o
);

  logic [31:0] mem [SIZE_WORDS];

  logic                     p_val  [DEPTH];
  int                       p_cnt  [DEPTH];
  logic [top_pkg::TL_AIW-1:0] p_src[DEPTH];
  logic [top_pkg::TL_SZW-1:0] p_sz [DEPTH];
  logic                     p_read [DEPTH];
  logic [31:0]              p_data [DEPTH];

  int rot;
  int free_slot;
  int rsp_slot;

  // Pick any free slot for a new request, and (starting from a rotating
  // index) any finished slot to answer.
  always_comb begin
    free_slot = -1;
    for (int i = DEPTH - 1; i >= 0; i--)
      if (!p_val[i]) free_slot = i;

    rsp_slot = -1;
    for (int j = DEPTH - 1; j >= 0; j--) begin
      int i = (rot + j) % DEPTH;
      if (p_val[i] && (p_cnt[i] <= 0)) rsp_slot = i;
    end
  end

  always_comb begin
    tl_o          = '0;
    tl_o.a_ready  = (free_slot >= 0);
    if (rsp_slot >= 0) begin
      tl_o.d_valid  = 1'b1;
      tl_o.d_opcode = p_read[rsp_slot] ? tlul_pkg::AccessAckData : tlul_pkg::AccessAck;
      tl_o.d_size   = p_sz[rsp_slot];
      tl_o.d_source = p_src[rsp_slot];
      tl_o.d_data   = p_data[rsp_slot];
      tl_o.d_error  = 1'b0;
    end
  end

  function automatic int word_index(input logic [31:0] addr);
    return int'((addr - BASE_ADDR) >> 2);
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < DEPTH; i++) p_val[i] <= 1'b0;
      rot <= 0;
    end else begin
      rot <= (rot + 1) % DEPTH;

      for (int i = 0; i < DEPTH; i++)
        if (p_val[i] && (p_cnt[i] > 0)) p_cnt[i] <= p_cnt[i] - 1;

      if (tl_i.a_valid && (free_slot >= 0)) begin
        int idx = word_index(tl_i.a_address);
        if ((idx < 0) || (idx >= int'(SIZE_WORDS)))
          $fatal(1, "tlul_test_mem: address 0x%08x out of range", tl_i.a_address);

        p_val [free_slot] <= 1'b1;
        p_cnt [free_slot] <= MIN_LAT + ($urandom % (MAX_LAT - MIN_LAT + 1));
        p_src [free_slot] <= tl_i.a_source;
        p_sz  [free_slot] <= tl_i.a_size;
        p_read[free_slot] <= (tl_i.a_opcode == tlul_pkg::Get);

        if (tl_i.a_opcode == tlul_pkg::Get) begin
          p_data[free_slot] <= mem[idx];
        end else begin
          p_data[free_slot] <= '0;
          for (int b = 0; b < 4; b++)
            if (tl_i.a_mask[b]) mem[idx][b*8+:8] <= tl_i.a_data[b*8+:8];
        end
      end

      if ((rsp_slot >= 0) && tl_i.d_ready) p_val[rsp_slot] <= 1'b0;
    end
  end

endmodule
