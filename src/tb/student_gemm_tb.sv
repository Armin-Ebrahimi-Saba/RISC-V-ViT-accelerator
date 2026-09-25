// SPDX-License-Identifier: CC0-1.0
// SPDX-FileCopyrightText: 2026 RVLab Student Project
//
// Unit testbench for the int8 GEMM accelerator.
//
// It drives the register interface exactly the way dav2_accel.c does -- same
// tiling loop, same register order -- against a TL-UL memory that answers out
// of order, and compares every output word against a behavioural model.
//
// Run:  flow student_gemm_tb.sim_rtl_xsim        (seconds)
//
// Or standalone:
//   xvlog -sv <pkgs> <tlul sources> src/rtl/student/student_gemm.sv \
//              src/tb/tlul_test_mem.sv src/tb/student_gemm_tb.sv
//   xelab work.student_gemm_tb -s gemm && xsim gemm -runall
//
// This is the fast, ideal-memory test. Its siblings cover what it cannot:
//   student_gemm_droprsp_tb   a read response deliberately dropped
//   student_gemm_ddrpath_tb   the real DDR3 cache in the path
// A shape that passes here but fails on the board (as 81x588x384 once did)
// is therefore a memory-path fault, not an accelerator fault -- that
// distinction is exactly what this bench is for.

module student_gemm_tb;

  localparam int unsigned NROWS       = 128;   // the board setting (student.sv)
  localparam int unsigned KMAX        = 2048;
  localparam int unsigned OUTSTANDING = 8;

  localparam logic [31:0] MEM_BASE = 32'h8000_0000;
  localparam logic [31:0] A_BASE   = 32'h8000_0000;
  localparam logic [31:0] W_BASE   = 32'h8002_0000;
  localparam logic [31:0] C_BASE   = 32'h8006_0000;

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
  localparam logic [31:0] R_A_STRIDE = 32'h3c;
  localparam logic [31:0] R_W_STRIDE = 32'h40;
  localparam logic [31:0] R_S_ADDR   = 32'h44;
  localparam logic [31:0] R_P_ADDR   = 32'h48;
  localparam logic [31:0] R_RQ_AMAX  = 32'h4c;
  localparam logic [31:0] R_G_ADDR   = 32'h50;
  localparam logic [31:0] R_G_GEOM   = 32'h54;
  localparam logic [31:0] R_G_CHAN   = 32'h58;
  localparam logic [31:0] R_G_CONV   = 32'h5c;
  localparam logic [31:0] R_G_START  = 32'h60;
  localparam logic [31:0] I_BASE     = 32'h8012_0000;   // gather: input image
  localparam logic [31:0] R_X_ADDR   = 32'h64;
  localparam logic [31:0] R_ADD_MX   = 32'h68;
  localparam logic [31:0] R_ADD_MH   = 32'h6c;
  localparam logic [31:0] R_ADD_SH   = 32'h70;
  localparam logic [31:0] R_LUT_ADDR = 32'h74;
  localparam logic [31:0] L_BASE     = 32'h801C_0000;   // requant: lookup table
  localparam logic [31:0] X_BASE     = 32'h8016_0000;   // requant: residual
  localparam logic [31:0] S_BASE     = 32'h800C_0000;   // per-row {max,min}
  localparam logic [31:0] P_BASE     = 32'h800D_0000;   // requant params
  localparam logic [31:0] O_BASE     = 32'h800E_0000;   // requant output

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
      .OUTSTANDING(OUTSTANDING)
  ) dut (
      .clk_i     (clk),
      .rst_ni    (rst_n),
      .tl_i      (reg_h2d),
      .tl_o      (reg_d2h),
      .tl_host_i (mem_d2h),
      .tl_host_o (mem_h2d)
  );

  tlul_test_mem #(
      .SIZE_WORDS(524288),
      .BASE_ADDR (MEM_BASE),
      .DEPTH     (OUTSTANDING)
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

  logic [31:0] stats_addr = 32'h0;    // S_ADDR for the next GEMM jobs (0 = off)
  logic [31:0] ctrl_extra = 32'h0;    // extra CTRL bits for the next GEMM jobs (CTRL.w16)

  // One accelerator job, i.e. one tile of nt activation rows.
  task automatic run_job(input logic [31:0] a_addr,
                         input logic [31:0] w_addr,
                         input logic [31:0] c_addr,
                         input int          c_stride,
                         input int          kdim,
                         input int          mdim,
                         input int          nt,
                         input int          a_stride,
                         input int          w_stride);
    logic [31:0] st;
    int guard;

    bus.put_word(R_A_ADDR,   a_addr);
    bus.put_word(R_W_ADDR,   w_addr);
    bus.put_word(R_A_STRIDE, a_stride);
    bus.put_word(R_W_STRIDE, w_stride);
    bus.put_word(R_S_ADDR,   stats_addr);
    bus.put_word(R_C_ADDR,   c_addr);
    bus.put_word(R_C_STRIDE, c_stride);
    bus.put_word(R_K_LEN,    kdim);
    bus.put_word(R_M_LEN,    mdim);
    bus.put_word(R_N_ROWS,   nt);
    bus.put_word(R_CTRL,     32'h1 | ctrl_extra);

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
  //
  // kfull > kdim stores A and W with rows kfull long and multiplies only the
  // kdim columns starting at koff: the strided mode software uses to split a
  // long reduction, and to read one attention head out of a qkv tensor.
  // A stride of 0 is sent whenever the rows are contiguous, so the default
  // path is exercised too.
  // GEMM with int16 weights (CTRL.w16), two per word. |A| <= 8191 and
  // |W| <= 2047 keep K <= 128 sums within int32. wstride_mode: 0 = W_STRIDE
  // 0 (contiguous, K*2), 1 = explicit rows twice as wide.
  task automatic run_gemm_w16(input int ndim, input int kdim, input int mdim,
                              input int wstride_mode);
    int mismatches = 0;
    int wrow = wstride_mode ? kdim * 2 : kdim;     // int16 per W row in memory
    $display("--- GEMM W16 N=%0d K=%0d M=%0d%s", ndim, kdim, mdim,
             wstride_mode ? " (strided W rows)" : "");
    for (int n = 0; n < ndim; n++)
      for (int k = 0; k < kdim; k++)
        poke_a(n, k, kdim, 16'($signed($urandom % 16383) - 8191));
    for (int i = 0; i < mdim * wrow / 2; i++)
      memory.mem[mem_word(W_BASE) + i] = {16'($signed($urandom % 4095) - 2047),
                                          16'($signed($urandom % 4095) - 2047)};
    for (int i = 0; i < mdim * ndim; i++)
      memory.mem[mem_word(C_BASE) + i] = 32'hdead_beef;
    ctrl_extra = 32'h100;
    for (int n0 = 0; n0 < ndim; n0 += NROWS) begin
      int nt = (ndim - n0 > int'(NROWS)) ? NROWS : ndim - n0;
      run_job(A_BASE + 32'(n0 * kdim * 2), W_BASE, C_BASE + 32'(n0 * 4),
              ndim * 4, kdim, mdim, nt, 0, wstride_mode ? wrow * 2 : 0);
    end
    ctrl_extra = 32'h0;
    for (int m = 0; m < mdim; m++)
      for (int n = 0; n < ndim; n++) begin
        int expected = 0, got;
        for (int k = 0; k < kdim; k++) begin
          int wi = m * wrow + k;
          logic [31:0] ww = memory.mem[mem_word(W_BASE) + (wi >> 1)];
          int w = wi[0] ? int'($signed(ww[31:16])) : int'($signed(ww[15:0]));
          expected += int'(peek_a(n, k, kdim)) * w;
        end
        got = int'(memory.mem[mem_word(C_BASE) + m * ndim + n]);
        checks++;
        if (got !== expected) begin
          if (mismatches < 5)
            $display("  FAIL m=%0d n=%0d: got %0d expected %0d", m, n, got, expected);
          mismatches++;
        end
      end
    if (mismatches) begin
      $display("  %0d/%0d words wrong", mismatches, mdim * ndim);
      errors += mismatches;
    end else
      $display("  ok, %0d words", mdim * ndim);
  endtask

  task automatic run_gemm(input int ndim, input int kdim, input int mdim,
                          input int kfull = 0, input int koff = 0);
    int nt;
    int expected;
    int got;
    int mismatches;
    int a_stride, w_stride;
    logic [31:0] cyc;

    if (kfull == 0) kfull = kdim;
    a_stride = (kfull == kdim) ? 0 : kfull * 2;
    w_stride = (kfull == kdim) ? 0 : kfull;

    if (kfull == kdim)
      $display("--- GEMM N=%0d K=%0d M=%0d", ndim, kdim, mdim);
    else
      $display("--- GEMM N=%0d K=%0d M=%0d  (columns %0d..%0d of %0d-wide rows)",
               ndim, kdim, mdim, koff, koff + kdim - 1, kfull);

    for (int n = 0; n < ndim; n++)
      for (int k = 0; k < kfull; k++)
        poke_a(n, k, kfull, 16'($signed($urandom % 16383) - 8191));
    for (int m = 0; m < mdim; m++)
      for (int k = 0; k < kfull; k++)
        poke_w(m, k, kfull, 8'($signed($urandom % 255) - 127));

    // Poison the output region so a job that writes nothing is not mistaken
    // for a job that writes the right thing.
    for (int i = 0; i < mdim * ndim; i++)
      memory.mem[mem_word(C_BASE) + i] = 32'hdead_beef;

    for (int n0 = 0; n0 < ndim; n0 += NROWS) begin
      nt = ndim - n0;
      if (nt > int'(NROWS)) nt = NROWS;
      run_job(A_BASE + 32'(n0 * kfull * 2 + koff * 2),
              W_BASE + 32'(koff),
              C_BASE + 32'(n0 * 4),
              ndim * 4, kdim, mdim, nt, a_stride, w_stride);
    end

    bus.get_word(R_CYCLES, cyc);

    mismatches = 0;
    for (int m = 0; m < mdim; m++) begin
      for (int n = 0; n < ndim; n++) begin
        expected = 0;
        for (int k = koff; k < koff + kdim; k++)
          expected += int'(peek_a(n, k, kfull)) * int'(peek_w(m, k, kfull));
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

    // Per-row statistics of the LAST tile (the job overwrites S_BASE each
    // tile in this tb; software gives every tile its own area).
    if (stats_addr != 0) begin
      int n0_last = ((ndim - 1) / NROWS) * NROWS;
      int sbad = 0;
      for (int m = 0; m < mdim; m++) begin
        int mx = -2147483647 - 1, mn = 2147483647, v;
        for (int n = n0_last; n < ndim; n++) begin
          v = int'(memory.mem[mem_word(C_BASE) + m * ndim + n]);
          if (v > mx) mx = v;
          if (v < mn) mn = v;
        end
        checks += 2;
        if (int'(memory.mem[mem_word(S_BASE) + 2*m]) !== mx ||
            int'(memory.mem[mem_word(S_BASE) + 2*m + 1]) !== mn) begin
          if (sbad < 3)
            $display("  FAIL stats m=%0d: got {%0d,%0d} expected {%0d,%0d}", m,
                     int'(memory.mem[mem_word(S_BASE) + 2*m]),
                     int'(memory.mem[mem_word(S_BASE) + 2*m + 1]), mx, mn);
          sbad++;
        end
      end
      if (sbad) errors += sbad;
      else $display("  stats ok for %0d rows", mdim);
    end
  endtask

  // Requantisation job(s) over the acc matrix left in C_BASE by run_gemm:
  // out[n][m] = sat14((acc[m][n] * mult[m] + 2^(sh[m]-1)) >> sh[m] + bias[m]),
  // chunked as the driver chunks it (<= NROWS columns, <= KWORDS rows).
  // Convolution through gather mode: an h x w x C int16 image (NHWC) at
  // I_BASE, a k x k conv with the given stride and padding, M output channels
  // with weights W[m][pos*C + c] (pos = ky*k + kx). The reference builds the
  // im2col matrix in the testbench and multiplies. Kernel positions are split
  // into jobs of kchunk positions each (K per job = kchunk*C), partial sums
  // added here as the driver does.
  task automatic run_conv(input int h, input int w, input int C, input int k,
                          input int stride, input int pad, input int mdim,
                          input int kchunk);
    int oh = (h + 2*pad - k) / stride + 1;
    int ow = (w + 2*pad - k) / stride + 1;
    int ndim = oh * ow;
    int kfull = k * k * C;
    int mismatches = 0;
    logic [31:0] st;
    int guard;
    $display("--- CONV %0dx%0dx%0d k=%0d s=%0d p=%0d -> %0dx%0d, M=%0d, K=%0d in chunks of %0d positions",
             h, w, C, k, stride, pad, oh, ow, mdim, kfull, kchunk);

    // image and weights
    for (int i = 0; i < h*w*C/2; i++)
      memory.mem[mem_word(I_BASE) + i] = {16'($signed($urandom % 16383) - 8191),
                                          16'($signed($urandom % 16383) - 8191)};
    for (int m = 0; m < mdim; m++)
      for (int kk = 0; kk < kfull; kk++)
        poke_w(m, kk, kfull, 8'($signed($urandom % 255) - 127));
    for (int i = 0; i < mdim * ndim; i++)
      memory.mem[mem_word(C_BASE) + i] = 32'hdead_beef;

    // accumulate over position chunks: first chunk to C_BASE, later ones
    // to O_BASE and added into C_BASE by the tb (int32 words)
    for (int p0 = 0; p0 < k*k; p0 += kchunk) begin
      int pc = (k*k - p0 > kchunk) ? kchunk : k*k - p0;
      logic [31:0] cbase = (p0 == 0) ? C_BASE : O_BASE;
      for (int n0 = 0; n0 < ndim; n0 += NROWS) begin
        int nt = (ndim - n0 > int'(NROWS)) ? NROWS : ndim - n0;
        bus.put_word(R_G_ADDR,   I_BASE);
        bus.put_word(R_G_GEOM,   {16'(h), 16'(w)});
        bus.put_word(R_G_CHAN,   {16'(ow), 16'(C)});
        bus.put_word(R_G_CONV,   {4'd0, 8'(pc), 4'((p0 % k)), 4'((p0 / k)), 4'(pad), 4'(stride), 4'(k)});
        bus.put_word(R_G_START,  {16'(n0 / ow), 16'(n0 % ow)});
        bus.put_word(R_W_ADDR,   W_BASE + 32'(p0 * C));
        bus.put_word(R_W_STRIDE, kfull);
        bus.put_word(R_A_STRIDE, 0);
        bus.put_word(R_S_ADDR,   0);
        bus.put_word(R_C_ADDR,   cbase + 32'(n0 * 4));
        bus.put_word(R_C_STRIDE, ndim * 4);
        bus.put_word(R_K_LEN,    pc * C);
        bus.put_word(R_M_LEN,    mdim);
        bus.put_word(R_N_ROWS,   nt);
        bus.put_word(R_CTRL,     32'h5);      // start | gather
        guard = 0;
        forever begin
          bus.get_word(R_STATUS, st);
          if (!(st & 32'h1)) break;
          if (++guard > 400000) begin
            $display("FAIL: gather job did not finish (status=0x%08x)", st);
            errors++;
            return;
          end
        end
      end
      if (p0 != 0)
        for (int i = 0; i < mdim * ndim; i++)
          memory.mem[mem_word(C_BASE) + i] = memory.mem[mem_word(C_BASE) + i]
                                            + memory.mem[mem_word(O_BASE) + i];
    end

    // reference
    for (int m = 0; m < mdim; m++) begin
      for (int n = 0; n < ndim; n++) begin
        int oy = n / ow, ox = n % ow;
        int expected = 0;
        for (int ky = 0; ky < k; ky++)
          for (int kx = 0; kx < k; kx++) begin
            int iy = oy*stride + ky - pad, ix = ox*stride + kx - pad;
            if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
            for (int c = 0; c < C; c++) begin
              int e = (iy*w + ix)*C + c;
              logic [31:0] wd = memory.mem[mem_word(I_BASE) + (e >> 1)];
              int a = e[0] ? int'($signed(wd[31:16])) : int'($signed(wd[15:0]));
              expected += a * int'(peek_w(m, (ky*k + kx)*C + c, kfull));
            end
          end
        checks++;
        if (int'(memory.mem[mem_word(C_BASE) + m*ndim + n]) !== expected) begin
          if (mismatches < 5)
            $display("  FAIL m=%0d n=%0d: got %0d expected %0d", m, n,
                     int'(memory.mem[mem_word(C_BASE) + m*ndim + n]), expected);
          mismatches++;
        end
      end
    end
    if (mismatches) begin
      $display("  %0d/%0d words wrong", mismatches, mdim * ndim);
      errors += mismatches;
    end else
      $display("  ok, %0d words", mdim * ndim);
  endtask

  // apply_multiplier as the C code does it: (v*mult + 2^(sh-1)) >> sh
  function automatic longint apply_mult(input longint v, input longint mult, input int sh);
    longint r = v * mult;
    if (sh > 0) r += 64'sd1 <<< (sh - 1);
    return r >>> sh;
  endfunction
  function automatic int sat14(input longint v);
    return (v > 8191) ? 8191 : (v < -8191) ? -8191 : int'(v);
  endfunction

  // Requantisation with the epilogue: out = sat14(apply(x, mx, sx) +
  // apply(h, mh, sh)) with h the plain requantised value, optionally ReLU.
  // mode: 0 = requant only + relu, 1 = add, 2 = add + relu. Chunks of at
  // most 512 rows, as the add mode requires.
  task automatic run_requant_epi(input int ndim, input int mdim, input int mode);
    int mismatches = 0;
    logic [31:0] st;
    int guard;
    longint mx = 32'h4000_0000 + ($urandom % 32'h3fff_ffff);
    longint mh = 32'h4000_0000 + ($urandom % 32'h3fff_ffff);
    int sx = 31 + ($urandom % 2), shh = 31 + ($urandom % 2);   // factors near 1/2..1
    int add = (mode != 0), relu = (mode != 1);
    $display("--- REQUANT+%s%s N=%0d M=%0d", add ? "ADD" : "", relu ? "+RELU" : "", ndim, mdim);

    for (int m = 0; m < mdim; m++) begin
      memory.mem[mem_word(P_BASE) + 3*m]     = 32'h4000_0000 + ($urandom % 32'h3fff_ffff);
      memory.mem[mem_word(P_BASE) + 3*m + 1] = 36 + ($urandom % 6);
      memory.mem[mem_word(P_BASE) + 3*m + 2] = 32'($signed($urandom % 2001) - 1000);
    end
    for (int i = 0; i < ndim * mdim / 2; i++) begin
      memory.mem[mem_word(X_BASE) + i] = {16'($signed($urandom % 16383) - 8191),
                                          16'($signed($urandom % 16383) - 8191)};
      memory.mem[mem_word(O_BASE) + i] = 32'hdead_beef;
    end

    for (int n0 = 0; n0 < ndim; n0 += NROWS) begin
      int nc = (ndim - n0 > int'(NROWS)) ? NROWS : ndim - n0;
      for (int m0 = 0; m0 < mdim; m0 += 512) begin
        int mc = (mdim - m0 > 512) ? 512 : mdim - m0;
        bus.put_word(R_A_ADDR,   C_BASE + 32'((m0 * ndim + n0) * 4));
        bus.put_word(R_A_STRIDE, ndim * 4);
        bus.put_word(R_P_ADDR,   P_BASE + 32'(m0 * 12));
        bus.put_word(R_C_ADDR,   O_BASE + 32'(n0 * mdim * 2 + m0 * 2));
        bus.put_word(R_X_ADDR,   X_BASE + 32'(n0 * mdim * 2 + m0 * 2));
        bus.put_word(R_C_STRIDE, mdim * 2);
        bus.put_word(R_ADD_MX,   32'(mx));
        bus.put_word(R_ADD_MH,   32'(mh));
        bus.put_word(R_ADD_SH,   {18'd0, 6'(shh), 2'd0, 6'(sx)});
        bus.put_word(R_M_LEN,    mc);
        bus.put_word(R_N_ROWS,   nc);
        bus.put_word(R_CTRL,     32'h3 | (add ? 32'h8 : 0) | (relu ? 32'h10 : 0));
        guard = 0;
        forever begin
          bus.get_word(R_STATUS, st);
          if (!(st & 32'h1)) break;
          if (++guard > 200000) begin
            $display("FAIL: epilogue job did not finish (status=0x%08x)", st);
            errors++;
            return;
          end
        end
      end
    end

    for (int n = 0; n < ndim; n++) begin
      for (int m = 0; m < mdim; m++) begin
        longint acc  = longint'(int'(memory.mem[mem_word(C_BASE) + m * ndim + n]));
        longint mult = longint'(int'(memory.mem[mem_word(P_BASE) + 3*m]));
        int     sh   = int'(memory.mem[mem_word(P_BASE) + 3*m + 1]);
        longint bias = longint'(int'(memory.mem[mem_word(P_BASE) + 3*m + 2]));
        int     idx  = n * mdim + m;
        logic [31:0] xw = memory.mem[mem_word(X_BASE) + (idx >> 1)];
        longint x    = idx[0] ? longint'($signed(xw[31:16])) : longint'($signed(xw[15:0]));
        int     h    = sat14(apply_mult(acc, mult, sh) + bias);
        int     e    = add ? sat14(longint'(int'(apply_mult(x, mx, sx))) +
                               longint'(int'(apply_mult(h, mh, shh)))) : h;
        logic [31:0] w;
        int     got;
        if (relu && e < 0) e = 0;
        w   = memory.mem[mem_word(O_BASE) + (idx >> 1)];
        got = idx[0] ? int'($signed(w[31:16])) : int'($signed(w[15:0]));
        checks++;
        if (got !== e) begin
          if (mismatches < 5)
            $display("  FAIL n=%0d m=%0d: got %0d expected %0d (h=%0d x=%0d)", n, m, got, e, h, x);
          mismatches++;
        end
      end
    end
    if (mismatches) begin
      $display("  %0d/%0d outputs wrong", mismatches, ndim * mdim);
      errors += mismatches;
    end else
      $display("  ok, %0d outputs", ndim * mdim);
  endtask

  // Requantisation with the lookup table: out = LUT[e + 8192], e the value
  // after the optional add and ReLU. The table is random; the first job
  // loads it (CTRL.lut_load), the later chunks reuse it. Chunks of 256 rows
  // so the reuse is exercised.
  task automatic run_requant_lut(input int ndim, input int mdim, input int add, input int relu);
    int mismatches = 0;
    logic [31:0] st;
    int guard;
    int first = 1;
    longint mx = 32'h4000_0000 + ($urandom % 32'h3fff_ffff);
    longint mh = 32'h4000_0000 + ($urandom % 32'h3fff_ffff);
    int sx = 31 + ($urandom % 2), shh = 31 + ($urandom % 2);
    $display("--- REQUANT+LUT%s%s N=%0d M=%0d", add ? "+ADD" : "", relu ? "+RELU" : "", ndim, mdim);

    for (int i = 0; i < 8192; i++)
      memory.mem[mem_word(L_BASE) + i] = $urandom;
    for (int m = 0; m < mdim; m++) begin
      memory.mem[mem_word(P_BASE) + 3*m]     = 32'h4000_0000 + ($urandom % 32'h3fff_ffff);
      memory.mem[mem_word(P_BASE) + 3*m + 1] = 36 + ($urandom % 6);
      memory.mem[mem_word(P_BASE) + 3*m + 2] = 32'($signed($urandom % 2001) - 1000);
    end
    for (int i = 0; i < ndim * mdim / 2; i++) begin
      memory.mem[mem_word(X_BASE) + i] = {16'($signed($urandom % 16383) - 8191),
                                          16'($signed($urandom % 16383) - 8191)};
      memory.mem[mem_word(O_BASE) + i] = 32'hdead_beef;
    end

    for (int n0 = 0; n0 < ndim; n0 += NROWS) begin
      int nc = (ndim - n0 > int'(NROWS)) ? NROWS : ndim - n0;
      for (int m0 = 0; m0 < mdim; m0 += 256) begin
        int mc = (mdim - m0 > 256) ? 256 : mdim - m0;
        bus.put_word(R_A_ADDR,   C_BASE + 32'((m0 * ndim + n0) * 4));
        bus.put_word(R_A_STRIDE, ndim * 4);
        bus.put_word(R_P_ADDR,   P_BASE + 32'(m0 * 12));
        bus.put_word(R_C_ADDR,   O_BASE + 32'(n0 * mdim * 2 + m0 * 2));
        bus.put_word(R_X_ADDR,   X_BASE + 32'(n0 * mdim * 2 + m0 * 2));
        bus.put_word(R_C_STRIDE, mdim * 2);
        bus.put_word(R_ADD_MX,   32'(mx));
        bus.put_word(R_ADD_MH,   32'(mh));
        bus.put_word(R_ADD_SH,   {18'd0, 6'(shh), 2'd0, 6'(sx)});
        bus.put_word(R_LUT_ADDR, L_BASE);
        bus.put_word(R_M_LEN,    mc);
        bus.put_word(R_N_ROWS,   nc);
        bus.put_word(R_CTRL,     32'h3 | (add ? 32'h8 : 0) | (relu ? 32'h10 : 0)
                               | 32'h20 | (first ? 32'h40 : 0));
        first = 0;
        guard = 0;
        forever begin
          bus.get_word(R_STATUS, st);
          if (!(st & 32'h1)) break;
          if (++guard > 400000) begin
            $display("FAIL: lookup-table job did not finish (status=0x%08x)", st);
            errors++;
            return;
          end
        end
      end
    end

    for (int n = 0; n < ndim; n++) begin
      for (int m = 0; m < mdim; m++) begin
        longint acc  = longint'(int'(memory.mem[mem_word(C_BASE) + m * ndim + n]));
        longint mult = longint'(int'(memory.mem[mem_word(P_BASE) + 3*m]));
        int     sh   = int'(memory.mem[mem_word(P_BASE) + 3*m + 1]);
        longint bias = longint'(int'(memory.mem[mem_word(P_BASE) + 3*m + 2]));
        int     idx  = n * mdim + m;
        logic [31:0] xw = memory.mem[mem_word(X_BASE) + (idx >> 1)];
        longint x    = idx[0] ? longint'($signed(xw[31:16])) : longint'($signed(xw[15:0]));
        int     h    = sat14(apply_mult(acc, mult, sh) + bias);
        int     e    = add ? sat14(longint'(int'(apply_mult(x, mx, sx))) +
                               longint'(int'(apply_mult(h, mh, shh)))) : h;
        int     li;
        logic [31:0] lw, w;
        int     got;
        if (relu && e < 0) e = 0;
        li  = e + 8192;
        lw  = memory.mem[mem_word(L_BASE) + (li >> 1)];
        e   = li[0] ? int'($signed(lw[31:16])) : int'($signed(lw[15:0]));
        w   = memory.mem[mem_word(O_BASE) + (idx >> 1)];
        got = idx[0] ? int'($signed(w[31:16])) : int'($signed(w[15:0]));
        checks++;
        if (got !== e) begin
          if (mismatches < 5)
            $display("  FAIL n=%0d m=%0d: got %0d expected %0d", n, m, got, e);
          mismatches++;
        end
      end
    end
    if (mismatches) begin
      $display("  %0d/%0d outputs wrong", mismatches, ndim * mdim);
      errors += mismatches;
    end else
      $display("  ok, %0d outputs", ndim * mdim);
  endtask

  // Requantisation over int16 input (CTRL.a16): acc[m][n] int16, rows of
  // ndim int16 (ndim even), optionally with the lookup table too.
  task automatic run_requant_a16(input int ndim, input int mdim, input int lut);
    int mismatches = 0;
    logic [31:0] st;
    int guard;
    localparam logic [31:0] H_BASE = 32'h8018_0000;   // int16 input
    $display("--- REQUANT A16%s N=%0d M=%0d", lut ? "+LUT" : "", ndim, mdim);
    for (int i = 0; i < ndim * mdim / 2; i++) begin
      memory.mem[mem_word(H_BASE) + i] = {16'($signed($urandom % 16383) - 8191),
                                          16'($signed($urandom % 16383) - 8191)};
      memory.mem[mem_word(O_BASE) + i] = 32'hdead_beef;
    end
    if (lut)
      for (int i = 0; i < 8192; i++)
        memory.mem[mem_word(L_BASE) + i] = $urandom;
    for (int m = 0; m < mdim; m++) begin
      memory.mem[mem_word(P_BASE) + 3*m]     = ($urandom % 2) ? 32'h4000_0000 + ($urandom % 32'h3fff_ffff)
                                                             : -(32'h4000_0000 + ($urandom % 32'h3fff_ffff));
      memory.mem[mem_word(P_BASE) + 3*m + 1] = 30 + ($urandom % 6);
      memory.mem[mem_word(P_BASE) + 3*m + 2] = 32'($signed($urandom % 2001) - 1000);
    end
    for (int n0 = 0; n0 < ndim; n0 += NROWS) begin
      int nc = (ndim - n0 > int'(NROWS)) ? NROWS : ndim - n0;
      for (int m0 = 0; m0 < mdim; m0 += 256) begin
        int mc = (mdim - m0 > 256) ? 256 : mdim - m0;
        bus.put_word(R_A_ADDR,   H_BASE + 32'((m0 * ndim + n0) * 2));
        bus.put_word(R_A_STRIDE, ndim * 2);
        bus.put_word(R_P_ADDR,   P_BASE + 32'(m0 * 12));
        bus.put_word(R_C_ADDR,   O_BASE + 32'(n0 * mdim * 2 + m0 * 2));
        bus.put_word(R_C_STRIDE, mdim * 2);
        bus.put_word(R_LUT_ADDR, L_BASE);
        bus.put_word(R_M_LEN,    mc);
        bus.put_word(R_N_ROWS,   nc);
        bus.put_word(R_CTRL,     32'h83 | (lut ? 32'h60 : 0));
        guard = 0;
        forever begin
          bus.get_word(R_STATUS, st);
          if (!(st & 32'h1)) break;
          if (++guard > 400000) begin
            $display("FAIL: A16 job did not finish (status=0x%08x)", st);
            errors++;
            return;
          end
        end
      end
    end
    for (int n = 0; n < ndim; n++) begin
      for (int m = 0; m < mdim; m++) begin
        int     ai   = m * ndim + n;
        logic [31:0] aw = memory.mem[mem_word(H_BASE) + (ai >> 1)];
        longint acc  = ai[0] ? longint'($signed(aw[31:16])) : longint'($signed(aw[15:0]));
        longint mult = longint'(int'(memory.mem[mem_word(P_BASE) + 3*m]));
        int     sh   = int'(memory.mem[mem_word(P_BASE) + 3*m + 1]);
        longint bias = longint'(int'(memory.mem[mem_word(P_BASE) + 3*m + 2]));
        int     idx  = n * mdim + m;
        int     e    = sat14(apply_mult(acc, mult, sh) + bias);
        logic [31:0] w, lw;
        int     got;
        if (lut) begin
          lw = memory.mem[mem_word(L_BASE) + ((e + 8192) >> 1)];
          e  = ((e + 8192) & 1) ? int'($signed(lw[31:16])) : int'($signed(lw[15:0]));
        end
        w   = memory.mem[mem_word(O_BASE) + (idx >> 1)];
        got = idx[0] ? int'($signed(w[31:16])) : int'($signed(w[15:0]));
        checks++;
        if (got !== e) begin
          if (mismatches < 5)
            $display("  FAIL n=%0d m=%0d: got %0d expected %0d", n, m, got, e);
          mismatches++;
        end
      end
    end
    if (mismatches) begin
      $display("  %0d/%0d outputs wrong", mismatches, ndim * mdim);
      errors += mismatches;
    end else
      $display("  ok, %0d outputs", ndim * mdim);
  endtask

  // Output row statistics (CTRL.ostats) on an int16-input job, optionally
  // with ReLU and the table: each output row n's {max, min} at S_BASE + n*4
  // for every n-tile (checked per tile, since each tile rewrites S_BASE).
  task automatic run_requant_ostats(input int ndim, input int mdim, input int relu, input int lut);
    int mismatches = 0;
    logic [31:0] st;
    int guard;
    localparam logic [31:0] H_BASE = 32'h8018_0000;
    $display("--- REQUANT OSTATS%s%s N=%0d M=%0d", relu ? "+RELU" : "", lut ? "+LUT" : "", ndim, mdim);
    for (int i = 0; i < ndim * mdim / 2; i++)
      memory.mem[mem_word(H_BASE) + i] = {16'($signed($urandom % 16383) - 8191),
                                          16'($signed($urandom % 16383) - 8191)};
    if (lut)
      for (int i = 0; i < 8192; i++)
        memory.mem[mem_word(L_BASE) + i] = $urandom;
    for (int m = 0; m < mdim; m++) begin
      memory.mem[mem_word(P_BASE) + 3*m]     = 32'h4000_0000 + ($urandom % 32'h3fff_ffff);
      memory.mem[mem_word(P_BASE) + 3*m + 1] = 30 + ($urandom % 4);
      memory.mem[mem_word(P_BASE) + 3*m + 2] = 32'($signed($urandom % 2001) - 1000);
    end
    for (int n0 = 0; n0 < ndim; n0 += NROWS) begin
      int nc = (ndim - n0 > int'(NROWS)) ? NROWS : ndim - n0;
      for (int i = 0; i < nc; i++) memory.mem[mem_word(S_BASE) + i] = 32'hdead_beef;
      bus.put_word(R_A_ADDR,   H_BASE + 32'(n0 * 2));
      bus.put_word(R_A_STRIDE, ndim * 2);
      bus.put_word(R_P_ADDR,   P_BASE);
      bus.put_word(R_C_ADDR,   O_BASE + 32'(n0 * mdim * 2));
      bus.put_word(R_C_STRIDE, mdim * 2);
      bus.put_word(R_S_ADDR,   S_BASE);
      bus.put_word(R_LUT_ADDR, L_BASE);
      bus.put_word(R_M_LEN,    mdim);
      bus.put_word(R_N_ROWS,   nc);
      bus.put_word(R_CTRL,     32'h283 | (relu ? 32'h10 : 0) | (lut ? 32'h60 : 0));
      guard = 0;
      forever begin
        bus.get_word(R_STATUS, st);
        if (!(st & 32'h1)) break;
        if (++guard > 400000) begin
          $display("FAIL: statistics job did not finish (status=0x%08x)", st);
          errors++;
          return;
        end
      end
      for (int n = n0; n < n0 + nc; n++) begin
        int mx = -32768, mn = 32767;
        logic [31:0] sw;
        for (int m = 0; m < mdim; m++) begin
          int idx = n * mdim + m;
          logic [31:0] w = memory.mem[mem_word(O_BASE) + (idx >> 1)];
          int v = idx[0] ? int'($signed(w[31:16])) : int'($signed(w[15:0]));
          if (v > mx) mx = v;
          if (v < mn) mn = v;
        end
        sw = memory.mem[mem_word(S_BASE) + (n - n0)];
        checks++;
        if (int'($signed(sw[31:16])) !== mx || int'($signed(sw[15:0])) !== mn) begin
          if (mismatches < 5)
            $display("  FAIL row %0d: got {%0d,%0d} expected {%0d,%0d}", n,
                     int'($signed(sw[31:16])), int'($signed(sw[15:0])), mx, mn);
          mismatches++;
        end
      end
    end
    bus.put_word(R_S_ADDR, 0);
    if (mismatches) begin
      $display("  %0d rows wrong", mismatches);
      errors += mismatches;
    end else
      $display("  ok, %0d rows", ndim);
  endtask

  // GEMM into the result RAM (CTRL.onchip), then its requantisation from
  // there: nothing of the int32 result goes through memory, only the
  // statistics. Chunks of 256 rows, one n-tile (ndim <= NROWS).
  task automatic run_onchip(input int ndim, input int kdim, input int mdim);
    int mismatches = 0;
    logic [31:0] st;
    int guard;
    $display("--- ONCHIP GEMM+REQUANT N=%0d K=%0d M=%0d", ndim, kdim, mdim);
    for (int n = 0; n < ndim; n++)
      for (int k = 0; k < kdim; k++)
        poke_a(n, k, kdim, 16'($signed($urandom % 16383) - 8191));
    for (int m = 0; m < mdim; m++)
      for (int k = 0; k < kdim; k++)
        poke_w(m, k, kdim, 8'($signed($urandom % 255) - 127));
    for (int m = 0; m < mdim; m++) begin
      memory.mem[mem_word(P_BASE) + 3*m]     = 32'h4000_0000 + ($urandom % 32'h3fff_ffff);
      memory.mem[mem_word(P_BASE) + 3*m + 1] = 44 + ($urandom % 4);
      memory.mem[mem_word(P_BASE) + 3*m + 2] = 32'($signed($urandom % 2001) - 1000);
      memory.mem[mem_word(S_BASE) + 2*m]     = 32'hdead_beef;
    end
    for (int i = 0; i < ndim * mdim / 2; i++)
      memory.mem[mem_word(O_BASE) + i] = 32'hdead_beef;
    for (int i = 0; i < ndim * mdim; i++)
      memory.mem[mem_word(C_BASE) + i] = 32'h1234_5678;      // must stay untouched
    // the GEMM: C_ADDR/C_STRIDE in words of the result RAM
    stats_addr = S_BASE;
    ctrl_extra = 32'h400;
    run_job(A_BASE, W_BASE, 0, ndim, kdim, mdim, ndim, 0, 0);
    ctrl_extra = 32'h0;
    stats_addr = 0;
    // the requantisation: A_ADDR/A_STRIDE in words of the result RAM
    for (int m0 = 0; m0 < mdim; m0 += 256) begin
      int mc = (mdim - m0 > 256) ? 256 : mdim - m0;
      bus.put_word(R_A_ADDR,   m0 * ndim);
      bus.put_word(R_A_STRIDE, ndim);
      bus.put_word(R_P_ADDR,   P_BASE + 32'(m0 * 12));
      bus.put_word(R_C_ADDR,   O_BASE + 32'(m0 * 2));
      bus.put_word(R_C_STRIDE, mdim * 2);
      bus.put_word(R_S_ADDR,   0);
      bus.put_word(R_M_LEN,    mc);
      bus.put_word(R_N_ROWS,   ndim);
      bus.put_word(R_CTRL,     32'h403);
      guard = 0;
      forever begin
        bus.get_word(R_STATUS, st);
        if (!(st & 32'h1)) break;
        if (++guard > 400000) begin
          $display("FAIL: on-chip requant did not finish (status=0x%08x)", st);
          errors++;
          return;
        end
      end
    end
    for (int m = 0; m < mdim; m++) begin
      int mx = -2147483647 - 1, mn = 2147483647;
      for (int n = 0; n < ndim; n++) begin
        int acc = 0, idx = n * mdim + m, e, got;
        logic [31:0] w;
        for (int k = 0; k < kdim; k++)
          acc += int'(peek_a(n, k, kdim)) * int'(peek_w(m, k, kdim));
        if (acc > mx) mx = acc;
        if (acc < mn) mn = acc;
        e = sat14(apply_mult(longint'(acc), longint'(int'(memory.mem[mem_word(P_BASE) + 3*m])),
                             int'(memory.mem[mem_word(P_BASE) + 3*m + 1]))
                  + longint'(int'(memory.mem[mem_word(P_BASE) + 3*m + 2])));
        w = memory.mem[mem_word(O_BASE) + (idx >> 1)];
        got = idx[0] ? int'($signed(w[31:16])) : int'($signed(w[15:0]));
        checks++;
        if (got !== e) begin
          if (mismatches < 5)
            $display("  FAIL n=%0d m=%0d: got %0d expected %0d (acc %0d)", n, m, got, e, acc);
          mismatches++;
        end
      end
      checks += 2;
      if (int'(memory.mem[mem_word(S_BASE) + 2*m]) !== mx || int'(memory.mem[mem_word(S_BASE) + 2*m + 1]) !== mn)
        mismatches++;
    end
    for (int i = 0; i < ndim * mdim; i++)
      if (memory.mem[mem_word(C_BASE) + i] !== 32'h1234_5678) begin
        mismatches++;
        break;
      end
    if (mismatches) begin
      $display("  %0d wrong", mismatches);
      errors += mismatches;
    end else
      $display("  ok, %0d outputs, statistics, memory untouched", ndim * mdim);
  endtask

  task automatic run_requant(input int ndim, input int mdim, input int sh_min = 33);
    int mismatches = 0;
    logic [31:0] st;
    int guard;
    $display("--- REQUANT N=%0d M=%0d", ndim, mdim);

    // parameter table: mult in [2^30, 2^31), shift 33..44, small bias
    for (int m = 0; m < mdim; m++) begin
      memory.mem[mem_word(P_BASE) + 3*m]     = 32'h4000_0000 + ($urandom % 32'h3fff_ffff);
      memory.mem[mem_word(P_BASE) + 3*m + 1] = sh_min + ($urandom % 12);
      memory.mem[mem_word(P_BASE) + 3*m + 2] = 32'($signed($urandom % 2001) - 1000);
    end
    for (int i = 0; i < ndim * mdim / 2; i++)
      memory.mem[mem_word(O_BASE) + i] = 32'hdead_beef;

    for (int n0 = 0; n0 < ndim; n0 += NROWS) begin
      int nc = (ndim - n0 > int'(NROWS)) ? NROWS : ndim - n0;
      for (int m0 = 0; m0 < mdim; m0 += 1024) begin
        int mc = (mdim - m0 > 1024) ? 1024 : mdim - m0;
        bus.put_word(R_A_ADDR,   C_BASE + 32'((m0 * ndim + n0) * 4));
        bus.put_word(R_A_STRIDE, ndim * 4);
        bus.put_word(R_P_ADDR,   P_BASE + 32'(m0 * 12));
        bus.put_word(R_C_ADDR,   O_BASE + 32'(n0 * mdim * 2 + m0 * 2));
        bus.put_word(R_C_STRIDE, mdim * 2);
        bus.put_word(R_M_LEN,    mc);
        bus.put_word(R_N_ROWS,   nc);
        bus.put_word(R_CTRL,     32'h3);
        guard = 0;
        forever begin
          bus.get_word(R_STATUS, st);
          if (!(st & 32'h1)) break;
          if (++guard > 200000) begin
            $display("FAIL: requant job did not finish (status=0x%08x)", st);
            errors++;
            return;
          end
        end
      end
    end

    for (int n = 0; n < ndim; n++) begin
      for (int m = 0; m < mdim; m++) begin
        longint acc  = longint'(int'(memory.mem[mem_word(C_BASE) + m * ndim + n]));
        longint mult = longint'(int'(memory.mem[mem_word(P_BASE) + 3*m]));
        int     sh   = int'(memory.mem[mem_word(P_BASE) + 3*m + 1]);
        longint bias = longint'(int'(memory.mem[mem_word(P_BASE) + 3*m + 2]));
        longint r    = ((acc * mult + (64'sd1 << (sh - 1))) >>> sh) + bias;
        int     e    = (r > 8191) ? 8191 : (r < -8191) ? -8191 : int'(r);
        int     idx  = n * mdim + m;
        logic [31:0] w = memory.mem[mem_word(O_BASE) + (idx >> 1)];
        int     got  = idx[0] ? int'($signed(w[31:16])) : int'($signed(w[15:0]));
        checks++;
        if (got !== e) begin
          if (mismatches < 5)
            $display("  FAIL n=%0d m=%0d: got %0d expected %0d (acc=%0d)", n, m, got, e, acc);
          mismatches++;
        end
      end
    end
    if (mismatches) begin
      $display("  %0d/%0d outputs wrong", mismatches, ndim * mdim);
      errors += mismatches;
    end else begin
      $display("  ok, %0d outputs", ndim * mdim);
    end

    // rq_amax holds the last job's largest |out|; check it when the whole
    // matrix was one job.
    if (ndim <= int'(NROWS) && mdim <= 1024) begin
      logic [31:0] got;
      int emax = 0;
      for (int i = 0; i < ndim * mdim; i++) begin
        logic [31:0] w = memory.mem[mem_word(O_BASE) + (i >> 1)];
        int v = i[0] ? int'($signed(w[31:16])) : int'($signed(w[15:0]));
        if (v < 0) v = -v;
        if (v > emax) emax = v;
      end
      bus.get_word(R_RQ_AMAX, got);
      checks++;
      if (int'(got) !== emax) begin
        $display("  FAIL rq_amax: got %0d expected %0d", got, emax);
        errors++;
      end else
        $display("  rq_amax ok (%0d)", emax);
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

    run_gemm(4,  8,   3);    // smaller than one tile
    // Partial FIRST tile with a weight stream long enough that reads are
    // still being issued after the first drain. This is the shape the
    // software self-test uses, and the one that exposed a stale t_q
    // indexing an accumulator whose A-tile row was never loaded.
    run_gemm(8,  32,  16);
    run_gemm(16, 64,  8);    // exactly one tile
    run_gemm(20, 64,  6);    // two tiles, second one partial
    run_gemm(17, 128, 4);    // partial tile of a single row
    run_gemm(16, 384, 12);   // a shape the model actually uses
    run_gemm(16, 384, 64);   // enough weight rows that the MAC array dominates

    // Patch embedding, the exact shape the model issues. On hardware this is
    // the only GEMM that disagrees with the CPU kernel, and it does so
    // deterministically: one accumulator word of 31104 is never written
    // (m=233, n=31). K=588 is unique to this shape, and N=81 leaves a final
    // tile of a single row.
    run_gemm(81, 588, 384);

    // NROWS = 64 shapes: one exact tile, and the encoder's 82 tokens as a
    // full tile plus a partial one of 18.
    run_gemm(64, 128, 8);
    run_gemm(82, 384, 12);
    run_gemm(130, 64, 8);    // NROWS = 128: a full tile plus 2
    run_gemm(128, 32, 4);    //                 exactly one tile

    // Strided rows. First a reduction split in two halves (rows 192 wide,
    // columns 96..191); then the attention shape: a 64-wide head out of
    // 576-wide rows, unaligned to the tile.
    run_gemm(70, 96, 8,  192, 96);
    run_gemm(66, 64, 20, 576, 64);
    run_gemm(66, 64, 20, 576, 0);

    // Statistics stream on, then requantise what the GEMM left behind.
    stats_addr = S_BASE;
    run_gemm(82, 128, 12);   run_requant(82, 12);     // two tiles
    run_gemm(20, 64, 30);    run_requant(20, 30);     // one partial tile
    run_requant(20, 30, 46);                          // small outputs: rq_amax < 8191
    // epilogue: ReLU alone, add, add + ReLU; one chunk and chunks of 512
    run_gemm(20, 64, 30);    run_requant_epi(20, 30, 0);
    run_requant_epi(20, 30, 1);
    run_requant_epi(20, 30, 2);
    run_gemm(70, 96, 1100);  run_requant_epi(70, 1100, 1);  // 512 + 512 + 76 rows
    run_gemm(130, 64, 8);    run_requant_epi(130, 8, 2);    // two n-tiles
    run_gemm(70, 96, 1100);  run_requant(70, 1100);   // M over one param chunk
    // lookup table (GELU): plain, with ReLU, with add; several chunks
    run_gemm(82, 96, 600);   run_requant_lut(82, 600, 0, 0);   // 3 chunks, table reused
    run_requant_lut(82, 600, 0, 1);
    run_gemm(20, 64, 30);    run_requant_lut(20, 30, 1, 0);
    run_gemm(130, 64, 8);    run_requant_lut(130, 8, 1, 1);    // two n-tiles
    // int16 weights: attention's shapes (K = 64 scores, K = 84 context),
    // two tiles, strided rows, with statistics on; then an int8 GEMM again
    run_gemm_w16(82, 64, 82, 0);
    run_gemm_w16(82, 84, 64, 1);
    run_gemm_w16(130, 64, 8, 0);
    run_gemm(20, 64, 30);
    // int16 input: LayerNorm's two shapes (tokens as rows, channels as
    // rows), n-tiles, negative multipliers, and with the table
    run_requant_a16(384, 82, 0);                     // 3 n-tiles of 128
    run_requant_a16(82, 384, 0);                     // 2 m-chunks
    run_requant_a16(20, 30, 1);
    // output row statistics: LayerNorm's first job (3 tiles of 128 rows),
    // with ReLU, with the table; then a plain job
    run_requant_ostats(384, 82, 0, 0);
    run_requant_ostats(82, 30, 1, 0);
    run_requant_ostats(20, 30, 0, 1);
    // results kept on chip: encoder shapes (82 rows), several chunks
    run_onchip(82, 64, 600);
    run_onchip(20, 128, 30);
    // and a plain job afterwards must not use the table
    run_gemm(20, 64, 30);    run_requant_epi(20, 30, 1);
    stats_addr = 0;

    // Convolutions through gather mode.
    run_conv(9, 9, 64, 3, 1, 1, 12, 9);        // 3x3 pad 1: N=81, K=576
    run_conv(9, 9, 32, 3, 2, 1, 8, 9);         // stride 2: N=25
    run_conv(9, 9, 384, 3, 2, 1, 6, 5);        // K=3456 split 5+4 positions
    run_conv(10, 7, 16, 3, 1, 1, 4, 9);        // non-square, N=70
    run_conv(12, 12, 16, 3, 1, 1, 4, 9);       // N=144: a full 128-row tile + 16

    if (x_errors) begin
      $display("X on the host A channel in %0d cycles", x_errors);
      errors += x_errors;
    end

    $display("=======================================");
    if (errors == 0)
      $display("student_gemm_tb PASSED (%0d words checked)", checks);
    else
      $display("student_gemm_tb FAILED (%0d errors, %0d words checked)", errors, checks);
    $display("=======================================");
    $finish;
  end

endmodule
