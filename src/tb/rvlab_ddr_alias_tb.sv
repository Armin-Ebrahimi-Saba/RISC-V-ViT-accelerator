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
//
// Two defects were found with this bench, and it covers both:
//
//  1. rvlab_ddr_prefetch returns another region's line under aliasing.
//     The interleaved section above fails 65/256 with the prefetcher in the
//     path and passes 256/256 with BYPASS_PREFETCH=1. (Fix: bypassed in
//     rvlab_tlul_ddr.sv.)
//
//  2. rvlab_ddr_block_cache wrote back a dirty line with data one cycle
//     stale when the line had been written on the immediately preceding
//     access. The "random pipelined" section at the end found it -- but only
//     after the driver was changed to present the next request the moment
//     the previous is accepted, the way a CPU does. tlul_test_host waits for
//     each response first and can never open that window, so the same test
//     PASSED on unfixed RTL until the driver was pipelined. (Fix: write-back
//     now uses the forwarded data_rdata, in rvlab_ddr_block_cache.sv.)
//
// Terms: a cache "set" is the slot an address maps to; "tag" is the part of
// the address that identifies which of the aliasing addresses currently
// occupies it; a line is "dirty" when it holds writes not yet in DRAM;
// "eviction" is throwing a line out to make room, which for a dirty line
// means writing it back first.
//
// Run:  flow rvlab_ddr_alias_tb.sim_rtl_xsim

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
  localparam bit BYPASS_PREFETCH = 1'b0;   // the repaired prefetcher is in the path

  if (BYPASS_PREFETCH) begin : gen_bypass
    assign pf_req  = llc_req;
    assign llc_rsp = pf_rsp;
  end else begin : gen_prefetch
    // PF_SIZE: prefetch table entries. 4 on the board. The testbench can use
    // 2 (+define+PF_SIZE=2): then losing two entries -- what the first
    // phase does to a prefetcher with the lost-entry defect -- stops the
    // port completely within this short test instead of after minutes.
    rvlab_ddr_prefetch #(
`ifdef PF_SIZE
        .SIZE(`PF_SIZE)
`else
        .SIZE(4)
`endif
    ) prefetch_i (
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

    // Random write-then-read across many sets, the CPU pattern that loses
    // words on hardware. The test host issues requests back to back, so the
    // next request is on the bus while the previous one is stalled on a miss
    // -- which is what exposed the cache reading data_mem and dirty_mem at
    // the live index instead of the stalled one.
    $display("--- random access across 4096 sets with dirty evictions ---");
    // tlul_test_host serialises transactions -- it waits for each response
    // before presenting the next request -- so a stalled miss never has a
    // second request behind it, and that is the exact condition the defect
    // needs. Drive the bus directly instead, the way a CPU does: as soon as
    // one request is accepted, present the next, and collect responses as
    // they come. The bus interface belongs to the test host, so this borrows
    // its tl_o for the duration.
    begin
      int rerr = 0;
      int seed = 32'h1234_5678;
      logic [31:0] addrs [1024];
      logic [31:0] vals  [1024];
      int  issued, got;
      logic [31:0] rsp_data [1024];

      for (int i = 0; i < 1024; i++) begin
        seed     = seed * 1103515245 + 12345;
        addrs[i] = BLOB_BASE + ((seed >> 4) & 32'h0001_FFFC);
        seed     = seed * 1103515245 + 12345;
        vals[i]  = seed;
      end

      // Pipelined writes.
      issued = 0; got = 0;
      bus.tl_o.d_ready <= 1'b1;
      while (got < 1024) begin
        @(posedge clk);
        if (issued < 1024) begin
          bus.tl_o.a_valid   <= 1'b1;
          bus.tl_o.a_opcode  <= tlul_pkg::PutFullData;
          bus.tl_o.a_size    <= 2;
          bus.tl_o.a_mask    <= 4'hF;
          bus.tl_o.a_address <= addrs[issued];
          bus.tl_o.a_data    <= vals[issued];
        end else bus.tl_o.a_valid <= 1'b0;
        #1;
        if (bus.tl_o.a_valid && d2h.a_ready) issued++;
        if (d2h.d_valid) got++;
      end
      bus.tl_o.a_valid <= 1'b0;
      repeat (4) @(posedge clk);

      // Pipelined reads, responses in order (single outstanding source).
      issued = 0; got = 0;
      while (got < 1024) begin
        @(posedge clk);
        if (issued < 1024) begin
          bus.tl_o.a_valid   <= 1'b1;
          bus.tl_o.a_opcode  <= tlul_pkg::Get;
          bus.tl_o.a_address <= addrs[issued];
        end else bus.tl_o.a_valid <= 1'b0;
        #1;
        if (bus.tl_o.a_valid && d2h.a_ready) issued++;
        if (d2h.d_valid) begin rsp_data[got] = d2h.d_data; got++; end
      end
      bus.tl_o.a_valid <= 1'b0;

      for (int i = 0; i < 1024; i++) begin
        logic [31:0] want = vals[i];
        for (int j = i + 1; j < 1024; j++)
          if (addrs[j] == addrs[i]) want = vals[j];
        if (rsp_data[i] !== want) begin
          rerr++;
          if (rerr <= 6)
            $display("  MISMATCH random %08x: got %08x want %08x (xor %08x)",
                     addrs[i], rsp_data[i], want, rsp_data[i] ^ want);
        end
      end
      $display("    random pipelined: %0d of 1024 wrong", rerr);
      errors += rerr;
    end

    // Mixed stress. Reads and writes interleaved at random, pipelined, over
    // both aliasing regions, for several rounds. This is the traffic that
    // makes a DRAM response arrive in the same cycle as the prefetcher
    // drops or invalidates the entry it belongs to. An entry lost that way
    // can never be allocated again; after all of them are lost, no read miss
    // reaches DRAM and the port stops answering. The stall detector turns
    // that into a failure instead of a testbench timeout.
    $display("--- mixed read/write stress, 8 rounds x 2048 pipelined accesses ---");
    begin
      int serr = 0, hang = 0;
      int seed = 32'h0bad_cafe;
      logic [31:0] shadow [logic [31:0]];
      for (int round = 0; round < 8 && !hang; round++) begin
        logic [31:0] q_addr [2048];
        logic [31:0] q_data [2048];
        bit          q_wr   [2048];
        logic [31:0] q_want [2048];
        bit          q_chk  [2048];
        int issued = 0, got = 0, idle = 0;
        for (int i = 0; i < 2048; i++) begin
          seed = seed * 1103515245 + 12345;
          q_addr[i] = ((seed >> 20) & 1 ? ARENA_BASE : BLOB_BASE)
                    + ((seed >> 3) & 32'h0000_7FFC);        // 32 kB: every set, twice
          seed = seed * 1103515245 + 12345;
          q_wr[i]   = (seed >> 16) & 1;
          seed = seed * 1103515245 + 12345;
          q_data[i] = seed;
        end
        bus.tl_o.d_ready <= 1'b1;
        while (got < 2048) begin
          @(posedge clk);
          if (issued < 2048) begin
            bus.tl_o.a_valid   <= 1'b1;
            bus.tl_o.a_opcode  <= q_wr[issued] ? tlul_pkg::PutFullData : tlul_pkg::Get;
            bus.tl_o.a_size    <= 2;
            bus.tl_o.a_mask    <= 4'hF;
            bus.tl_o.a_address <= q_addr[issued];
            bus.tl_o.a_data    <= q_data[issued];
          end else bus.tl_o.a_valid <= 1'b0;
          #1;
          if (bus.tl_o.a_valid && d2h.a_ready) begin
            // requests are served in order, so the expected value of a read
            // is the last write to its address issued before it
            if (q_wr[issued]) begin
              shadow[q_addr[issued]] = q_data[issued];
              q_chk[issued] = 0;
            end else begin
              q_chk[issued]  = shadow.exists(q_addr[issued]);
              q_want[issued] = q_chk[issued] ? shadow[q_addr[issued]] : '0;
            end
            issued++;
          end
          if (d2h.d_valid) begin
            if (!q_wr[got] && q_chk[got] && d2h.d_data !== q_want[got]) begin
              serr++;
              if (serr <= 6)
                $display("  MISMATCH stress round %0d %08x: got %08x want %08x",
                         round, q_addr[got], d2h.d_data, q_want[got]);
            end
            got++;
            idle = 0;
          end else if (++idle == 20000) begin
            $display("  HANG in round %0d: no response for 20000 cycles (%0d of 2048 issued, %0d answered)",
                     round, issued, got);
            hang = 1;
            break;
          end
        end
        bus.tl_o.a_valid <= 1'b0;
        repeat (4) @(posedge clk);
        $display("    round %0d done @%0t: %0d wrong so far", round, $time, serr);
      end
      $display("    mixed stress: %0d wrong reads, %s", serr, hang ? "HANG" : "no hang");
      errors += serr + (hang ? 1 : 0);
    end

    // Repeated aliasing writes. The first phase above is the one that made
    // the prefetcher lose entries (a DRAM response arriving in the same
    // cycle the entry was marked out of date). It loses about two per pass
    // over 64 sets; an entry lost that way is never reused, and with all
    // four lost no read miss reaches DRAM. Four passes over 256 sets with
    // different data make that certain on a prefetcher with the defect,
    // which then hangs in bus.put_word and ends in the testbench timeout.
    $display("--- repeated aliasing writes, 4 passes over 256 sets ---");
    for (int pass = 0; pass < 4; pass++) begin
      int perr = 0;
      for (int s = 0; s < 256; s++)
        for (int w = 0; w < 2; w++) begin
          bus.put_word(BLOB_BASE  + s*SET_BYTES + w*4, pat(BLOB_BASE,  s, w) ^ pass);
          bus.put_word(ARENA_BASE + s*SET_BYTES + w*4, pat(ARENA_BASE, s, w) ^ pass);
        end
      for (int s = 0; s < 256; s++)
        for (int w = 0; w < 2; w++) begin
          bus.get_word(BLOB_BASE + s*SET_BYTES + w*4, rd);
          if (rd !== (pat(BLOB_BASE, s, w) ^ pass)) perr++;
          bus.get_word(ARENA_BASE + s*SET_BYTES + w*4, rd);
          if (rd !== (pat(ARENA_BASE, s, w) ^ pass)) perr++;
        end
      $display("    pass %0d done @%0t: %0d wrong", pass, $time, perr);
      errors += perr;
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
    // 40 ms: the mixed stress phase is long. In steps of 1 ms, because a
    // single delay literal above 2^32 time units overflows.
    repeat (40) #1_000_000_000;
    $display("RESULT: testbench timeout");
    $finish;
  end

endmodule
