// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// student_gemm against the REAL rvlab DDR3 front end.
//
//   student_gemm ---+
//                   |-- tlul_socket_m1 -- rvlab_ddr_cache -- rvlab_ddr_prefetch -- ddr3_blk_model
//   testbench host -+        (real)            (real)              (real)           (behavioural)
//
// Only the block manager, DDR3 controller and PHY are replaced, so this
// exercises the cache and prefetch logic the board actually runs without
// paying the ~90 minutes of DDR3 calibration a system simulation needs.
//
// The testbench reaches memory through its own port on the same socket rather
// than poking the backend array. rvlab_ddr_cache is write-back: data written
// by the accelerator sits in the cache until eviction, so a backend peek would
// report stale values and stale cache lines would feed the next test case.

module student_gemm_ddrpath_tb;

  localparam int unsigned NROWS = 16;
  localparam int unsigned KMAX  = 2048;

  localparam logic [31:0] A_BASE = 32'h8000_0000;
  localparam logic [31:0] W_BASE = 32'h8002_0000;
  localparam logic [31:0] C_BASE = 32'h8009_0000;

  localparam logic [31:0] R_STATUS   = 32'h00;
  localparam logic [31:0] R_CTRL     = 32'h04;
  localparam logic [31:0] R_A_ADDR   = 32'h08;
  localparam logic [31:0] R_W_ADDR   = 32'h0c;
  localparam logic [31:0] R_C_ADDR   = 32'h10;
  localparam logic [31:0] R_C_STRIDE = 32'h14;
  localparam logic [31:0] R_K_LEN    = 32'h18;
  localparam logic [31:0] R_M_LEN    = 32'h1c;
  localparam logic [31:0] R_N_ROWS   = 32'h20;
  localparam logic [31:0] R_CAPS     = 32'h24;
  localparam logic [31:0] R_CYCLES   = 32'h28;

  logic clk, rst_n;

  always begin
    clk = '1; #10000; clk = '0; #10000;
  end

  tlul_pkg::tl_h2d_t reg_h2d;
  tlul_pkg::tl_d2h_t reg_d2h;
  tlul_pkg::tl_h2d_t acc_h2d;
  tlul_pkg::tl_d2h_t acc_d2h;
  tlul_pkg::tl_h2d_t tbh_h2d;
  tlul_pkg::tl_d2h_t tbh_d2h;
  tlul_pkg::tl_h2d_t cache_h2d;
  tlul_pkg::tl_d2h_t cache_d2h;

  student_gemm #(
      .NROWS       (NROWS),
      .KMAX        (KMAX),
      .OUTSTANDING (8),
      .MAX_INFLIGHT(1)      // matches student.sv on the board
  ) dut (
      .clk_i    (clk),
      .rst_ni   (rst_n),
      .tl_i     (reg_h2d),
      .tl_o     (reg_d2h),
      .tl_host_i(acc_d2h),
      .tl_host_o(acc_h2d)
  );

  tlul_pkg::tl_h2d_t host_h2d [2];
  tlul_pkg::tl_d2h_t host_d2h [2];

  assign host_h2d[0] = acc_h2d;
  assign acc_d2h     = host_d2h[0];
  assign host_h2d[1] = tbh_h2d;
  assign tbh_d2h     = host_d2h[1];

  tlul_socket_m1 #(.M(2)) merge_i (
      .clk_i (clk),
      .rst_ni(rst_n),
      .tl_h_i(host_h2d),
      .tl_h_o(host_d2h),
      .tl_d_o(cache_h2d),
      .tl_d_i(cache_d2h)
  );

  rvlab_ddr_pkg::ddr3_h2d_t llc_req, pf_req;
  rvlab_ddr_pkg::ddr3_d2h_t llc_rsp, pf_rsp;

  rvlab_ddr_cache #(.IDX_BITS(9)) cache_i (
      .clk_i      (clk),
      .rst_ni     (rst_n),
      .tl_i       (cache_h2d),
      .tl_o       (cache_d2h),
      .block_req_o(llc_req),
      .block_rsp_i(llc_rsp)
  );

  // Set BYPASS_PREFETCH to take rvlab_ddr_prefetch out of the path and wire the
  // cache straight to the backend. A plain write hangs waiting for a response
  // on the first dirty eviction; this says whether the prefetcher is involved.
  localparam bit BYPASS_PREFETCH = 1'b0;

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
      .SIZE_BLOCKS(8192),
      .DEPTH      (1),
      .MIN_LAT    (6),
      .MAX_LAT    (40)
  ) memory (
      .clk_i (clk),
      .rst_ni(rst_n),
      .req_i (pf_req),
      .rsp_o (pf_rsp)
  );

  tlul_test_host bus (
      .clk_i (clk),
      .rst_no(rst_n),
      .tl_i  (reg_d2h),
      .tl_o  (reg_h2d)
  );

  // Memory host, written here rather than reused from tlul_test_host, because
  // that one deasserts a_valid as soon as a_ready arrives. rvlab_ddr_block_cache
  // reloads its dirty bit from dirty_mem[access_idx] using the LIVE bus address
  // while stalled (the tag path correctly uses access_idx_q), so a released
  // address makes it evict the same line forever. Holding the request until the
  // response is what the CPU does, and what keeps that cache working.
  initial tbh_h2d = '{a_opcode: tlul_pkg::PutFullData, default: '0};

  task automatic mem_write(input logic [31:0] addr, input logic [31:0] data);
    @(posedge clk);
    tbh_h2d.a_valid   <= 1'b1;
    tbh_h2d.a_opcode  <= tlul_pkg::PutFullData;
    tbh_h2d.a_size    <= 2;
    tbh_h2d.a_address <= addr;
    tbh_h2d.a_data    <= data;
    tbh_h2d.a_mask    <= 4'hf;
    tbh_h2d.a_source  <= '0;
    tbh_h2d.d_ready   <= 1'b1;
    // Wait for acceptance first: without it we can sample the tail of the
    // previous transaction's response, which a cache hit makes likely.
    // a_valid must stay asserted until the response: releasing it -- even with
    // the address still driven -- makes rvlab_ddr_block_cache re-evict the same
    // line forever (measured: 164 accepts in a 4000-cycle window).
    do @(posedge clk); while (!tbh_d2h.d_valid);
    tbh_h2d.a_valid <= 1'b0;
    // Drain the response before the next request, or a fast cache hit lets us
    // sample this one's d_valid again.
    do @(posedge clk); while (tbh_d2h.d_valid);
  endtask

  task automatic mem_read(input logic [31:0] addr, output logic [31:0] data);
    @(posedge clk);
    tbh_h2d.a_valid   <= 1'b1;
    tbh_h2d.a_opcode  <= tlul_pkg::Get;
    tbh_h2d.a_size    <= 2;
    tbh_h2d.a_address <= addr;
    tbh_h2d.a_mask    <= 4'hf;
    tbh_h2d.a_source  <= '0;
    tbh_h2d.d_ready   <= 1'b1;
    do @(posedge clk); while (!tbh_d2h.d_valid);
    data = tbh_d2h.d_data;
    tbh_h2d.a_valid <= 1'b0;
    do @(posedge clk); while (tbh_d2h.d_valid);
  endtask

  // Watchdog. Trigger on "no response reached the TL-UL host", not on bus
  // idleness: if the cache loops issuing requests the backend keeps accepting,
  // an idleness-based watchdog never fires. Counting backend acceptances in
  // the window separates a livelock from a deadlock.
  int mem_idle, be_accepts, be_rsps;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      mem_idle <= 0; be_accepts <= 0; be_rsps <= 0;
    end else begin
      if (llc_req.a_valid && llc_rsp.a_ready) be_accepts <= be_accepts + 1;
      if (llc_rsp.d_valid && llc_req.d_ready) be_rsps <= be_rsps + 1;
      if (cache_d2h.d_valid) begin
        mem_idle <= 0; be_accepts <= 0; be_rsps <= 0;
      end else begin
        mem_idle <= mem_idle + 1;
        if (mem_idle == 4000) begin
          $display("MEMSTALL @%0t  backend accepts=%0d rsps=%0d in window",
                   $time, be_accepts, be_rsps);
          $display("  TL  h2d: a_valid=%b a_opcode=%0d a_address=%08x",
                   cache_h2d.a_valid, cache_h2d.a_opcode, cache_h2d.a_address);
          $display("  TL  d2h: a_ready=%b d_valid=%b", cache_d2h.a_ready, cache_d2h.d_valid);
          $display("  BLK req: a_valid=%b a_opcode=%0d a_address=%06x",
                   llc_req.a_valid, llc_req.a_opcode, llc_req.a_address);
          $display("  BLK rsp: a_ready=%b d_valid=%b d_opcode=%0d",
                   llc_rsp.a_ready, llc_rsp.d_valid, llc_rsp.d_opcode);
          $display("  model  : count=%0d q_val0=%b q_cnt0=%0d",
                   memory.count, memory.q_val[0], memory.q_cnt[0]);
          $display("  cache  : stall=%b miss=%b hit=%b dirty=%b access_q=%b tag_rd=%0h tag_q=%0h",
                   cache_i.cache_i.stall, cache_i.cache_i.miss, cache_i.cache_i.hit,
                   cache_i.cache_i.dirty_rdata, cache_i.cache_i.access_q,
                   cache_i.cache_i.tag_rdata, cache_i.cache_i.access_tag_q);
        end
      end
    end
  end

  // ------------------------------------------------------------- reference

  int errors, checks;

  localparam int MAXN = 20;
  localparam int MAXK = 640;
  localparam int MAXM = 8;

  logic signed [15:0] a_ref [MAXN*MAXK];
  logic signed [ 7:0] w_ref [MAXM*MAXK];

  // Harness self-test. Before trusting this bench as an oracle for the DUT,
  // prove the path testbench -> socket -> cache -> prefetch -> backend stores
  // and returns data correctly on its own. A failure here is a harness bug,
  // not an accelerator bug.
  task automatic mem_selftest();
    logic [31:0] rd;
    int bad;
    bad = 0;

    // Span several cache sets and both dirty-eviction and clean-fill paths.
    for (int i = 0; i < 128; i++)
      mem_write(A_BASE + 32'(i*4), 32'hA000_0000 + 32'(i));
    for (int i = 0; i < 128; i++)
      mem_write(W_BASE + 32'(i*4), 32'hB000_0000 + 32'(i));
    for (int i = 0; i < 128; i++)
      mem_write(C_BASE + 32'(i*4), 32'hC000_0000 + 32'(i));

    for (int i = 0; i < 128; i++) begin
      mem_read(A_BASE + 32'(i*4), rd);
      if (rd !== 32'hA000_0000 + 32'(i)) begin
        if (bad < 5) $display("  MEMFAIL A[%0d]: got %08x want %08x",
                              i, rd, 32'hA000_0000 + 32'(i));
        bad++;
      end
    end
    for (int i = 0; i < 128; i++) begin
      mem_read(W_BASE + 32'(i*4), rd);
      if (rd !== 32'hB000_0000 + 32'(i)) begin
        if (bad < 5) $display("  MEMFAIL W[%0d]: got %08x want %08x",
                              i, rd, 32'hB000_0000 + 32'(i));
        bad++;
      end
    end
    for (int i = 0; i < 128; i++) begin
      mem_read(C_BASE + 32'(i*4), rd);
      if (rd !== 32'hC000_0000 + 32'(i)) begin
        if (bad < 5) $display("  MEMFAIL C[%0d]: got %08x want %08x",
                              i, rd, 32'hC000_0000 + 32'(i));
        bad++;
      end
    end

    if (bad) begin
      $display("HARNESS BROKEN: %0d/384 words wrong through cache+prefetch", bad);
      errors += bad;
    end else begin
      $display("harness ok: 384 words survive cache+prefetch+backend");
    end
  endtask

  // ---------------------------------------------------------------- driver

  task automatic run_job(input logic [31:0] a_addr, input logic [31:0] c_addr,
                         input int c_stride, input int kdim, input int mdim,
                         input int nt);
    logic [31:0] st;
    int guard;
    bus.put_word(R_A_ADDR,   a_addr);
    bus.put_word(R_W_ADDR,   W_BASE);
    bus.put_word(R_C_ADDR,   c_addr);
    bus.put_word(R_C_STRIDE, c_stride);
    bus.put_word(R_K_LEN,    kdim);
    bus.put_word(R_M_LEN,    mdim);
    bus.put_word(R_N_ROWS,   nt);
    bus.put_word(R_CTRL,     32'h1);

    guard = 0;
    forever begin
      bus.get_word(R_STATUS, st);
      if (!(st & 32'h1)) break;
      if (++guard > 2000000) begin
        $display("FAIL: job did not finish (status=0x%08x)", st);
        errors++;
        return;
      end
    end
    if (st & 32'h4) begin
      $display("FAIL: bus error latched (status=0x%08x)", st);
      errors++;
    end
  endtask

  task automatic run_gemm(input int ndim, input int kdim, input int mdim);
    int nt, expected, got, mismatches;
    logic [31:0] word, rd;
    logic [31:0] cyc;

    $display("--- GEMM N=%0d K=%0d M=%0d", ndim, kdim, mdim);

    for (int n = 0; n < ndim; n++)
      for (int k = 0; k < kdim; k++)
        a_ref[n*kdim + k] = 16'($signed($urandom % 16383) - 8191);
    for (int m = 0; m < mdim; m++)
      for (int k = 0; k < kdim; k++)
        w_ref[m*kdim + k] = 8'($signed($urandom % 255) - 127);

    // Everything goes through the cache, so the DUT and the testbench always
    // agree on what memory holds.
    for (int i = 0; i < (ndim*kdim)/2; i++) begin
      word = {a_ref[i*2 + 1], a_ref[i*2]};
      mem_write(A_BASE + 32'(i*4), word);
    end
    for (int i = 0; i < (mdim*kdim)/4; i++) begin
      word = {w_ref[i*4 + 3], w_ref[i*4 + 2], w_ref[i*4 + 1], w_ref[i*4]};
      mem_write(W_BASE + 32'(i*4), word);
    end
    for (int i = 0; i < mdim*ndim; i++)
      mem_write(C_BASE + 32'(i*4), 32'hdead_beef);

    for (int n0 = 0; n0 < ndim; n0 += NROWS) begin
      nt = ndim - n0;
      if (nt > int'(NROWS)) nt = NROWS;
      run_job(A_BASE + 32'(n0 * kdim * 2), C_BASE + 32'(n0 * 4),
              ndim * 4, kdim, mdim, nt);
    end

    bus.get_word(R_CYCLES, cyc);

    mismatches = 0;
    for (int m = 0; m < mdim; m++) begin
      for (int n = 0; n < ndim; n++) begin
        expected = 0;
        for (int k = 0; k < kdim; k++)
          expected += int'(a_ref[n*kdim + k]) * int'(w_ref[m*kdim + k]);
        mem_read(C_BASE + 32'((m*ndim + n)*4), rd);
        got = int'(rd);
        checks++;
        if (got !== expected) begin
          if (mismatches < 5)
            $display("  FAIL m=%0d n=%0d: got %0d expected %0d", m, n, got, expected);
          mismatches++;
        end
      end
    end

    if (mismatches) begin
      $display("  %0d/%0d words wrong", mismatches, mdim*ndim);
      errors += mismatches;
    end else begin
      $display("  ok, %0d words, last tile took %0d cycles (%0d MACs)",
               mdim*ndim, cyc, nt*kdim*mdim);
    end
  endtask

  // ------------------------------------------------------------------ main

  initial begin
    logic [31:0] caps;
    errors = 0;
    checks = 0;

    bus.reset();

    bus.get_word(R_CAPS, caps);
    $display("caps = 0x%08x (nrows=%0d kmax=%0d)", caps, caps[7:0], caps[23:8]);

    mem_selftest();

    // Actually drive the DUT. Without these the testbench compiled the
    // accelerator against the real cache and prefetcher and then checked
    // nothing, reporting "PASSED (0 words checked)".
    run_gemm(20, 64,  6);    // two tiles, second partial
    run_gemm(17, 128, 4);    // final tile of a single row
    run_gemm(16, 384, 12);   // a shape the model issues

    $display("=======================================");
    if (checks == 0) begin
      $display("student_gemm_ddrpath_tb FAILED (no words checked -- the testbench drove nothing)");
      errors++;
    end
    if (errors == 0)
      $display("student_gemm_ddrpath_tb PASSED (%0d words checked)", checks);
    else
      $display("student_gemm_ddrpath_tb FAILED (%0d errors, %0d words checked)",
               errors, checks);
    $display("=======================================");
    $finish;
  end

endmodule
