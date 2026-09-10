// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// Does rvlab_ddr_cache return the right line when two regions alias?
//
// The cache is direct-mapped with IDX_BITS(9): 512 sets of 32 bytes, 16 kB
// total, index = addr[13:5], tag = addr[31:14]. The blob lives at 0x80000000
// and the activation arena at 0x82000000, so the two differ only in tag and
// collide in every set.
//
// On hardware the model reports tensors "not found in blob" that are demonstrably
// present: DDR3 holds the correct bytes (300/300 random words verified over
// JTAG) but the CPU's cached reads return something else. The blob directory
// is 21 kB and is scanned linearly on every lookup, so each lookup evicts the
// whole cache while the accelerator writes the aliasing arena.
//
// This walks that pattern directly: write distinct patterns to both regions,
// then read them back interleaved so every access evicts the other region's
// line and forces a dirty write-back. Any address that returns its alias
// partner's data -- or stale data -- is the defect.

module rvlab_ddr_alias_tb;

  localparam logic [31:0] BLOB_BASE  = 32'h8000_0000;
  localparam logic [31:0] ARENA_BASE = 32'h8200_0000;
  localparam int unsigned NSETS      = 64;    // of 512; enough to thrash
  localparam int unsigned SET_BYTES  = 32;

  logic clk;
  wire  rst_n;
  always begin clk = '1; #10000; clk = '0; #10000; end

  tlul_pkg::tl_h2d_t h2d;
  tlul_pkg::tl_d2h_t d2h;

  rvlab_ddr_pkg::ddr3_h2d_t llc_req, pf_req;
  rvlab_ddr_pkg::ddr3_d2h_t llc_rsp, pf_rsp;

  tlul_pkg::tl_h2d_t llc_h2d;
  tlul_pkg::tl_d2h_t llc_d2h;

  // Mirrors rvlab_tlul_ddr.sv.
  student_tl_rsp_hold #(.DEPTH(2)) hold_i (
      .clk_i (clk), .rst_ni(rst_n),
      .tl_h_i(h2d), .tl_h_o(d2h),
      .tl_d_o(llc_h2d), .tl_d_i(llc_d2h)
  );

  rvlab_ddr_cache #(.IDX_BITS(9)) cache_i (
      .clk_i      (clk),
      .rst_ni     (rst_n),
      .tl_i       (llc_h2d),
      .tl_o       (llc_d2h),
      .block_req_o(llc_req),
      .block_rsp_i(llc_rsp)
  );

  // Set BYPASS_PREFETCH to wire the cache straight to the backend, which
  // says whether the wrong-line reads come from the cache or the prefetcher.
  localparam bit BYPASS_PREFETCH = 1'b1;

  if (BYPASS_PREFETCH) begin : gen_bypass
    assign pf_req  = llc_req;
    assign llc_rsp = pf_rsp;
  end else begin : gen_prefetch
    rvlab_ddr_prefetch prefetch_i (
        .clk_i   (clk),
        .rst_ni  (rst_n),
        .fe_req_i(llc_req),
        .fe_rsp_o(llc_rsp),
        .be_req_o(pf_req),
        .be_rsp_i(pf_rsp)
    );
  end

  ddr3_blk_model #(
      .SIZE_BLOCKS(2**21), .DEPTH(16), .MIN_LAT(6), .MAX_LAT(40)
  ) memory (
      .clk_i (clk), .rst_ni(rst_n), .req_i (pf_req), .rsp_o (pf_rsp)
  );

  tlul_test_host bus (
      .clk_i (clk),
      .rst_no(rst_n),     // the host drives reset
      .tl_o  (h2d),
      .tl_i  (d2h)
  );

  function automatic logic [31:0] pat(input logic [31:0] base,
                                      input int s, input int w);
    pat = {base[31:24], 8'(s), 8'(w), 8'hA5};
  endfunction

  int errors;
  logic [31:0] rd;

  initial begin
    errors = 0;
    bus.reset();
    repeat (20) @(negedge clk);

    $display("--- writing both aliasing regions ---");
    for (int s = 0; s < int'(NSETS); s++)
      for (int w = 0; w < 2; w++) begin
        bus.put_word(BLOB_BASE  + s*SET_BYTES + w*4, pat(BLOB_BASE,  s, w));
        bus.put_word(ARENA_BASE + s*SET_BYTES + w*4, pat(ARENA_BASE, s, w));
      end

    $display("--- reading back, interleaved so every access evicts the other ---");
    for (int s = 0; s < int'(NSETS); s++)
      for (int w = 0; w < 2; w++) begin
        bus.get_word(BLOB_BASE + s*SET_BYTES + w*4, rd);
        if (rd !== pat(BLOB_BASE, s, w)) begin
          errors++;
          if (errors <= 10)
            $display("  MISMATCH blob  set=%0d w=%0d: got %08x want %08x%s",
                     s, w, rd, pat(BLOB_BASE, s, w),
                     (rd === pat(ARENA_BASE, s, w)) ? "  <-- ARENA'S DATA" : "");
        end

        bus.get_word(ARENA_BASE + s*SET_BYTES + w*4, rd);
        if (rd !== pat(ARENA_BASE, s, w)) begin
          errors++;
          if (errors <= 10)
            $display("  MISMATCH arena set=%0d w=%0d: got %08x want %08x%s",
                     s, w, rd, pat(ARENA_BASE, s, w),
                     (rd === pat(BLOB_BASE, s, w)) ? "  <-- BLOB'S DATA" : "");
        end
      end

    $display("");
    if (errors == 0)
      $display("RESULT: PASS -- %0d aliasing accesses all returned their own data.",
               NSETS * 4);
    else
      $display("RESULT: FAIL -- %0d of %0d aliasing accesses returned wrong data.",
               errors, NSETS * 4);
    $finish;
  end

  initial begin
    #2_000_000_000;
    $display("RESULT: testbench timeout");
    $finish;
  end

endmodule
