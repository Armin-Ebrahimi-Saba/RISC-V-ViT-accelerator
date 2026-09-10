// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// Behavioural stand-in for rvlab_ddr_blkmgr plus the DDR3 controller and PHY.
//
// It speaks the rvlab_ddr_pkg block protocol: 256-bit blocks, ancillary data
// echoed back verbatim, and -- as that package requires -- responses strictly
// in order. Replacing the real back end lets a testbench exercise the actual
// rvlab_ddr_cache and rvlab_ddr_prefetch RTL without paying the ~90 minutes of
// DDR3 calibration a full system simulation needs.
//
// Simulation only.

module ddr3_blk_model #(
    parameter int unsigned SIZE_BLOCKS = 8192,   // 256 kB of 32-byte blocks
    parameter int unsigned DEPTH       = 16,     // matches blkmgr REQBUF_SIZE
    parameter int unsigned MIN_LAT     = 6,
    parameter int unsigned MAX_LAT     = 40
) (
    input  logic                      clk_i,
    input  logic                      rst_ni,
    input  rvlab_ddr_pkg::ddr3_h2d_t  req_i,
    output rvlab_ddr_pkg::ddr3_d2h_t  rsp_o
);

  import rvlab_ddr_pkg::*;
  import tlul_pkg::*;

  localparam int unsigned IDXW = $clog2(SIZE_BLOCKS);

  logic [255:0] mem [SIZE_BLOCKS];

  // Real DRAM powers up with garbage, not X. Leaving these unknown makes the
  // cache pull X into whole 32-byte blocks whenever a single word is touched,
  // which trips the TL-UL DataKnown assertions for reasons that have nothing
  // to do with the design under test.
  // Backdoor image load.
  //
  // The blob reaches DDR3 over JTAG on hardware, which is not simulatable
  // (24.87 MB at the JTAG bit rate would dwarf the run). Reading the same file
  // straight into the backing array puts the design in the state it is in on
  // the board at the moment inference starts, which is the state under
  // investigation. Byte 0 of the file is byte 0 of block 0, matching the
  // little-endian word order the cache expects.
  string blob_file;
  int    blob_fd, blob_n, blob_blk;
  logic [7:0] blob_buf [32];

  initial begin
    for (int i = 0; i < int'(SIZE_BLOCKS); i++) mem[i] = '0;

    if ($value$plusargs("ddr_blob=%s", blob_file)) begin
      blob_fd = $fopen(blob_file, "rb");
      if (blob_fd == 0) $fatal(1, "ddr3_blk_model: cannot open %s", blob_file);
      blob_blk = 0;
      forever begin
        blob_n = $fread(blob_buf, blob_fd);
        if (blob_n <= 0) break;
        if (blob_blk >= int'(SIZE_BLOCKS))
          $fatal(1, "ddr3_blk_model: image larger than SIZE_BLOCKS");
        for (int b = 0; b < 32; b++)
          mem[blob_blk][8*b +: 8] = (b < blob_n) ? blob_buf[b] : 8'h00;
        blob_blk++;
      end
      $fclose(blob_fd);
      $display("ddr3_blk_model: loaded %0d blocks (%0d bytes) from %s",
               blob_blk, blob_blk * 32, blob_file);
    end
  end

  // Strictly in-order queue: head answers first, as the protocol demands.
  logic                 q_val  [DEPTH];
  int                   q_cnt  [DEPTH];
  logic [255:0]         q_data [DEPTH];
  logic [DDR_ANCW-1:0]  q_anc  [DEPTH];
  logic                 q_read [DEPTH];

  int wr_ptr, rd_ptr, count;

  wire accept = req_i.a_valid && (count < DEPTH);
  wire head_ready = (count > 0) && q_val[rd_ptr] && (q_cnt[rd_ptr] <= 0);

  always_comb begin
    rsp_o          = '0;
    rsp_o.a_ready  = (count < DEPTH);
    if (head_ready) begin
      rsp_o.d_valid  = 1'b1;
      rsp_o.d_opcode = q_read[rd_ptr] ? AccessAckData : AccessAck;
      rsp_o.d_data   = q_data[rd_ptr];
      rsp_o.d_anc    = q_anc[rd_ptr];
    end
  end

  function automatic int blk_index(input logic [DDR_AW-1:0] a);
    return int'(a[IDXW-1:0]);
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < DEPTH; i++) q_val[i] <= 1'b0;
      wr_ptr <= 0;
      rd_ptr <= 0;
      count  <= 0;
    end else begin
      for (int i = 0; i < DEPTH; i++)
        if (q_val[i] && (q_cnt[i] > 0)) q_cnt[i] <= q_cnt[i] - 1;

      if (accept) begin
        int idx = blk_index(req_i.a_address);
        q_val [wr_ptr] <= 1'b1;
        q_cnt [wr_ptr] <= MIN_LAT + ($urandom % (MAX_LAT - MIN_LAT + 1));
        q_anc [wr_ptr] <= req_i.a_anc;
        q_read[wr_ptr] <= (req_i.a_opcode == Get);

        if (req_i.a_opcode == Get) begin
          q_data[wr_ptr] <= mem[idx];
        end else begin
          q_data[wr_ptr] <= '0;
          for (int b = 0; b < 32; b++)
            if (req_i.a_mask[b]) mem[idx][b*8+:8] <= req_i.a_data[b*8+:8];
        end

        wr_ptr <= (wr_ptr + 1) % DEPTH;
      end

      if (head_ready && req_i.d_ready) begin
        q_val[rd_ptr] <= 1'b0;
        rd_ptr <= (rd_ptr + 1) % DEPTH;
      end

      case ({accept, head_ready && req_i.d_ready})
        2'b10:   count <= count + 1;
        2'b01:   count <= count - 1;
        default: ;
      endcase
    end
  end

endmodule
