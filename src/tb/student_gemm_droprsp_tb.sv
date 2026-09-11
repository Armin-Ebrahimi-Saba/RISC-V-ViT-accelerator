// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// Regression test for the accelerator's lost-response recovery.
//
// The rvlab DDR3 cache can lose a read response outright (it pulses d_valid
// for one cycle without honouring d_ready). student_gemm survives that with a
// retry timer: a read unanswered for RETRY_CYCLES is re-issued. This bench
// proves the timer works by using tlul_test_mem's DROP_NTH_RD parameter to
// accept the 200th read and never answer it.
//
// Expected: the job still completes and every output word is correct.
// Without the retry (RETRY_CYCLES=0) the job hangs with STATUS=0x1 forever --
// that failing case was run deliberately before the retry was trusted, since
// a test that cannot fail proves nothing.
//
// Otherwise identical to student_gemm_tb: same register sequence as
// dav2_accel.c, same behavioural model for the expected values.
//
// Run:  flow student_gemm_droprsp_tb.sim_rtl_xsim

module student_gemm_droprsp_tb;

  localparam int unsigned NROWS       = 16;
  localparam int unsigned KMAX        = 2048;
  localparam int unsigned OUTSTANDING = 8;

  localparam logic [31:0] MEM_BASE = 32'h8000_0000;
  localparam logic [31:0] A_BASE   = 32'h8000_0000;
  localparam logic [31:0] W_BASE   = 32'h8001_0000;
  localparam logic [31:0] C_BASE   = 32'h8002_0000;

  // Register offsets, from student_gemm.hjson.
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

  logic clk;
  logic rst_n;

  // 50 MHz, matching sys_clk.
  always begin
    clk = '1;
    #10000;
    clk = '0;
    #10000;
  end

  tlul_pkg::tl_h2d_t reg_h2d;
  tlul_pkg::tl_d2h_t reg_d2h;
  tlul_pkg::tl_h2d_t mem_h2d;
  tlul_pkg::tl_d2h_t mem_d2h;

  student_gemm #(
      .NROWS      (NROWS),
      .KMAX       (KMAX),
      .OUTSTANDING(OUTSTANDING),
      .MAX_INFLIGHT(1)
  ) dut (
      .clk_i     (clk),
      .rst_ni    (rst_n),
      .tl_i      (reg_h2d),
      .tl_o      (reg_d2h),
      .tl_host_i (mem_d2h),
      .tl_host_o (mem_h2d)
  );

  tlul_test_mem #(
      .SIZE_WORDS(65536),
      .BASE_ADDR (MEM_BASE),
      .DEPTH     (1),
      .DROP_NTH_RD(200)
  ) memory (
      .clk_i (clk),
      .rst_ni(rst_n),
      .tl_i  (mem_h2d),
      .tl_o  (mem_d2h)
  );

  tlul_test_host bus (
      .clk_i (clk),
      .rst_no(rst_n),
      .tl_i  (reg_d2h),
      .tl_o  (reg_h2d)
  );

  // --------------------------------------------------- host A-channel X check
  //
  // The real TL-UL sockets push the whole request struct through a
  // prim_fifo_sync whose DataKnown_A assertion fails on any X, so an unknown
  // in a "don't care" field (a_data on a read, say) is a real bug even though
  // a permissive memory model would never notice.

  int x_errors;

  always @(posedge clk) begin
    if (rst_n && mem_h2d.a_valid) begin
      if ($isunknown({mem_h2d.a_opcode, mem_h2d.a_param, mem_h2d.a_size,
                      mem_h2d.a_source, mem_h2d.a_address, mem_h2d.a_mask,
                      mem_h2d.a_data, mem_h2d.a_user})) begin
        if (x_errors < 5)
          $display("  FAIL: X on host A channel at %0t: op=%0d addr=%h data=%h src=%h mask=%h",
                   $time, mem_h2d.a_opcode, mem_h2d.a_address, mem_h2d.a_data,
                   mem_h2d.a_source, mem_h2d.a_mask);
        x_errors++;
      end
    end
  end

  // ------------------------------------------------------------- reference

  int errors;
  int checks;

  function automatic int mem_word(input logic [31:0] addr);
    return int'((addr - MEM_BASE) >> 2);
  endfunction

  // Activations are int16, two per memory word.
  task automatic poke_a(input int n, input int k, input int kdim,
                        input logic signed [15:0] val);
    int elem = n * kdim + k;
    int widx = mem_word(A_BASE) + (elem >> 1);
    if (elem[0]) memory.mem[widx][31:16] = val;
    else         memory.mem[widx][15:0]  = val;
  endtask

  // Weights are int8, four per memory word.
  task automatic poke_w(input int m, input int k, input int kdim,
                        input logic signed [7:0] val);
    int elem = m * kdim + k;
    int widx = mem_word(W_BASE) + (elem >> 2);
    memory.mem[widx][(elem % 4) * 8 +: 8] = val;
  endtask

  function automatic logic signed [15:0] peek_a(input int n, input int k, input int kdim);
    int elem = n * kdim + k;
    int widx = mem_word(A_BASE) + (elem >> 1);
    return elem[0] ? memory.mem[widx][31:16] : memory.mem[widx][15:0];
  endfunction

  function automatic logic signed [7:0] peek_w(input int m, input int k, input int kdim);
    int elem = m * kdim + k;
    int widx = mem_word(W_BASE) + (elem >> 2);
    return memory.mem[widx][(elem % 4) * 8 +: 8];
  endfunction

  // ---------------------------------------------------------------- driver

  // One accelerator job, i.e. one tile of nt activation rows.
  task automatic run_job(input logic [31:0] a_addr,
                         input logic [31:0] c_addr,
                         input int          c_stride,
                         input int          kdim,
                         input int          mdim,
                         input int          nt);
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
      if (++guard > 100000) begin
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

  // Full GEMM, tiled the way dav2_accel_qgemm() tiles it.
  task automatic run_gemm(input int ndim, input int kdim, input int mdim);
    int nt;
    int expected;
    int got;
    int mismatches;
    logic [31:0] cyc;

    $display("--- GEMM N=%0d K=%0d M=%0d", ndim, kdim, mdim);

    for (int n = 0; n < ndim; n++)
      for (int k = 0; k < kdim; k++)
        poke_a(n, k, kdim, 16'($signed($urandom % 16383) - 8191));
    for (int m = 0; m < mdim; m++)
      for (int k = 0; k < kdim; k++)
        poke_w(m, k, kdim, 8'($signed($urandom % 255) - 127));

    // Poison the output region so a job that writes nothing is not mistaken
    // for a job that writes the right thing.
    for (int i = 0; i < mdim * ndim; i++)
      memory.mem[mem_word(C_BASE) + i] = 32'hdead_beef;

    for (int n0 = 0; n0 < ndim; n0 += NROWS) begin
      nt = ndim - n0;
      if (nt > int'(NROWS)) nt = NROWS;
      run_job(A_BASE + 32'(n0 * kdim * 2),
              C_BASE + 32'(n0 * 4),
              ndim * 4, kdim, mdim, nt);
    end

    bus.get_word(R_CYCLES, cyc);

    mismatches = 0;
    for (int m = 0; m < mdim; m++) begin
      for (int n = 0; n < ndim; n++) begin
        expected = 0;
        for (int k = 0; k < kdim; k++)
          expected += int'(peek_a(n, k, kdim)) * int'(peek_w(m, k, kdim));
        got = int'(memory.mem[mem_word(C_BASE) + m * ndim + n]);
        checks++;
        if (got !== expected) begin
          if (mismatches < 5)
            $display("  FAIL m=%0d n=%0d: got %0d expected %0d", m, n, got, expected);
          mismatches++;
        end
      end
    end

    if (mismatches) begin
      $display("  %0d/%0d words wrong", mismatches, mdim * ndim);
      errors += mismatches;
    end else begin
      $display("  ok, %0d words, last tile took %0d cycles (%0d MACs)",
               mdim * ndim, cyc, nt * kdim * mdim);
    end
  endtask

  // ------------------------------------------------------------------ main

  initial begin
    logic [31:0] caps;

    errors = 0;
    checks = 0;
    x_errors = 0;

    bus.reset();

    bus.get_word(R_CAPS, caps);
    $display("caps = 0x%08x (nrows=%0d kmax=%0d)", caps, caps[7:0], caps[23:8]);
    if (caps[7:0] !== NROWS[7:0] || caps[23:8] !== KMAX[15:0]) begin
      $display("FAIL: caps does not match the parameters");
      errors++;
    end

    // Reproduces the board failure: the memory accepts a read and never
    // answers it. Without recovery the block waits forever, exactly as
    // hardware showed (accepted=444, responses=443).
    run_gemm(16, 384, 12);

    if (x_errors) begin
      $display("X on the host A channel in %0d cycles", x_errors);
      errors += x_errors;
    end

    $display("=======================================");
    if (errors == 0)
      $display("student_gemm_droprsp_tb PASSED (%0d words checked)", checks);
    else
      $display("student_gemm_droprsp_tb FAILED (%0d errors, %0d words checked)", errors, checks);
    $display("=======================================");
    $finish;
  end

endmodule
