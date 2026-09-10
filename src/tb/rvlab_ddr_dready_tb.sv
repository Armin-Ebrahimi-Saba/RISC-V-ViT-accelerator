// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// Does rvlab_ddr_cache honour TL-UL d_ready on its front end?
//
// rvlab_ddr_block_cache drives fe_rsp_o.d_valid straight from the tag-hit
// signal and never looks at fe_req_i.d_ready, and rvlab_ddr_cache passes both
// through to TL-UL unchanged.  TL-UL requires a response to be held until the
// cycle where d_valid and d_ready are both high, so if the host happens to be
// unready on the single cycle d_valid is asserted, the response is gone.
//
// student_gemm already carries a retry timer to survive this.  The CPU cannot:
// a lost response leaves cv32e40p stalled on a load forever, and a core that
// cannot retire an instruction cannot enter debug mode either, which is
// exactly what the board shows -- running, silent, and refusing to halt.
//
// This testbench issues one Get with d_ready held high (the control) and one
// with d_ready deasserted on the cycle the response appears (the test).
// No DDR3 controller or PHY is involved, so it runs in seconds.

module rvlab_ddr_dready_tb;

  localparam logic [31:0] ADDR_A = 32'h8000_0040;
  localparam logic [31:0] ADDR_B = 32'h8000_1080;   // different set, no reuse

  logic clk, rst_n;
  always begin clk = '1; #10000; clk = '0; #10000; end

  tlul_pkg::tl_h2d_t h2d, h2d_drv;
  tlul_pkg::tl_d2h_t d2h;   // driven by the skid buffer or straight by the cache

  // d_ready is driven separately from the request fields so the two are not
  // two drivers on one variable, and it only ever changes on a negedge, so a
  // cycle in which the device sees d_ready low is unambiguous.
  logic dready_drv;
  always_comb begin
    h2d         = h2d_drv;
    h2d.d_ready = dready_drv;
  end

  rvlab_ddr_pkg::ddr3_h2d_t llc_req, pf_req;
  rvlab_ddr_pkg::ddr3_d2h_t llc_rsp, pf_rsp;

  // USE_SKID mirrors rvlab_tlul_ddr.sv.  Set it to 0 to see the raw defect the
  // buffer exists to cover: the same stimulus then loses the response.
  localparam bit USE_SKID = 1'b1;

  tlul_pkg::tl_h2d_t llc_h2d;
  tlul_pkg::tl_d2h_t llc_d2h;

  if (USE_SKID) begin : gen_skid
    student_tl_rsp_hold #(.DEPTH(2)) hold_i (
        .clk_i (clk),
        .rst_ni(rst_n),
        .tl_h_i(h2d),
        .tl_h_o(d2h),
        .tl_d_o(llc_h2d),
        .tl_d_i(llc_d2h)
    );
  end else begin : gen_direct
    assign llc_h2d = h2d;
    assign d2h     = llc_d2h;
  end

  rvlab_ddr_cache #(.IDX_BITS(9)) cache_i (
      .clk_i      (clk),
      .rst_ni     (rst_n),
      .tl_i       (llc_h2d),
      .tl_o       (llc_d2h),
      .block_req_o(llc_req),
      .block_rsp_i(llc_rsp)
  );

  rvlab_ddr_prefetch prefetch_i (
      .clk_i   (clk),
      .rst_ni  (rst_n),
      .fe_req_i(llc_req),
      .fe_rsp_o(llc_rsp),
      .be_req_o(pf_req),
      .be_rsp_i(pf_rsp)
  );

  ddr3_blk_model #(
      .SIZE_BLOCKS(8192), .DEPTH(1), .MIN_LAT(6), .MAX_LAT(40)
  ) memory (
      .clk_i (clk), .rst_ni(rst_n), .req_i (pf_req), .rsp_o (pf_rsp)
  );

  // ---------------------------------------------------------------- driver
  // TL-UL requires the device to hold a response until the cycle where
  // d_valid and d_ready are both high.  So the test simply refuses the
  // response for a fixed window and then accepts: a compliant device delivers
  // it late, a device that pulses d_valid for one cycle has already lost it.
  logic [31:0] got_data;

  task automatic do_get(input logic [31:0] addr, input int stall_cycles,
                        output bit ok, output int latency);
    begin
      ok      = 1'b0;
      latency = 0;

      @(negedge clk);
      dready_drv        = (stall_cycles == 0);
      h2d_drv.a_valid   = 1'b1;
      h2d_drv.a_opcode  = tlul_pkg::Get;
      h2d_drv.a_size    = 3'd2;
      h2d_drv.a_mask    = 4'hF;
      h2d_drv.a_source  = 8'h01;
      h2d_drv.a_address = addr;
      h2d_drv.a_data    = 32'h0;

      while (!d2h.a_ready) @(negedge clk);
      @(negedge clk);
      h2d_drv.a_valid = 1'b0;

      // Hold the host unready across the whole window in which the response
      // can appear.
      for (int i = 0; i < stall_cycles; i++) begin
        if (d2h.d_valid)
          $display("    device asserted d_valid at stall cycle %0d while d_ready was low", i);
        @(negedge clk);
      end
      dready_drv = 1'b1;

      // Now accept.  A held response arrives immediately; a lost one never
      // arrives at all.
      for (latency = 0; latency < 5000; latency++) begin
        if (d2h.d_valid && dready_drv) begin
          got_data = d2h.d_data;
          ok = 1'b1;
          @(negedge clk);
          break;
        end
        @(negedge clk);
      end
    end
  endtask

  bit ok_ctrl, ok_test;
  int lat_ctrl, lat_test;

  initial begin
    h2d_drv    = '0;
    dready_drv = 1'b1;
    rst_n     = 1'b0;
    repeat (20) @(negedge clk);
    rst_n = 1'b1;
    repeat (20) @(negedge clk);

    $display("--- control: d_ready held high -------------------------------");
    do_get(ADDR_A, 0, ok_ctrl, lat_ctrl);
    $display("    response %s after %0d cycles", ok_ctrl ? "ARRIVED" : "LOST",
             lat_ctrl);

    $display("--- test: d_ready held low for 200 cycles, then raised --------");
    do_get(ADDR_B, 200, ok_test, lat_test);
    $display("    response %s after %0d cycles", ok_test ? "ARRIVED" : "LOST",
             lat_test);

    $display("");
    if (!ok_ctrl) begin
      $display("RESULT: FAIL -- the control case did not complete at all.");
    end else if (ok_test && USE_SKID) begin
      $display("RESULT: PASS -- with student_tl_rsp_hold in the path the");
      $display("        response is held across 200 unready cycles and");
      $display("        delivered when the host finally accepts it.");
    end else if (ok_test && !USE_SKID) begin
      $display("RESULT: the bare cache honoured d_ready; the buffer is moot.");
    end else if (!ok_test && !USE_SKID) begin
      $display("RESULT: as expected without the buffer -- rvlab_ddr_cache");
      $display("        DROPS a response when the host is not ready on the");
      $display("        single cycle d_valid is asserted. A CPU load that");
      $display("        loses its response stalls the core permanently and");
      $display("        makes it unhaltable.");
    end else begin
      $display("RESULT: FAIL -- the buffer is in the path and the response");
      $display("        was still lost.");
    end
    $finish;
  end

  initial begin
    #500_000_000;
    $display("RESULT: testbench timeout");
    $finish;
  end

endmodule
