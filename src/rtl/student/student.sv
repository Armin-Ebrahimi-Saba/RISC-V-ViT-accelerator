// SPDX-License-Identifier: SHL-2.1
// SPDX-FileCopyrightText: 2024 RVLab Contributors

module student (
  input logic clk_i,
  input logic rst_ni,

  input  top_pkg::userio_board2fpga_t userio_i,
  output top_pkg::userio_fpga2board_t userio_o,

  output logic irq_o,

  input  tlul_pkg::tl_h2d_t tl_device_peri_i,
  output tlul_pkg::tl_d2h_t tl_device_peri_o,
  input  tlul_pkg::tl_h2d_t tl_device_fast_i,
  output tlul_pkg::tl_d2h_t tl_device_fast_o,

  input  tlul_pkg::tl_d2h_t tl_host_i,
  output tlul_pkg::tl_h2d_t tl_host_o
);

  logic [7:0] led;
  assign userio_o = '{
    led: led,
    default: '0
  };

  assign irq_o         = '0;

  student_rlight rlight_i (
    .clk_i,
    .rst_ni,
    .tl_o (tl_device_peri_o),
    .tl_i (tl_device_peri_i),
    .led_o(led)
  );

  // The fast device window (0x2000_0000, 256 MB) is shared by two masters,
  // each of which also drives a host port:
  //
  //   0x2000_0000  student_dma
  //   0x2001_0000  student_gemm   (int8 GEMM accelerator, see student_gemm.sv)
  //
  // Requests are steered by address on the way down and merged by the
  // standard TL-UL sockets on the way up.

  tlul_pkg::tl_h2d_t fast_h2d [2];
  tlul_pkg::tl_d2h_t fast_d2h [2];
  logic [1:0]        fast_sel;

  always_comb begin
    unique case (tl_device_fast_i.a_address[27:16])
      12'h000: fast_sel = 2'd0;   // student_dma
      12'h001: fast_sel = 2'd1;   // student_gemm
      default: fast_sel = 2'd3;   // out of range -> socket returns an error
    endcase
  end

  // The register adapters' back-pressure would otherwise reach the crossbar
  // (and from there the CPU's instruction fetch) combinationally through this
  // socket; post-PnR that path missed 20 ns by 77 ps. Registering the host
  // side of the socket cuts it, at the cost of one cycle each way on fast
  // window accesses -- which are register pokes, so the latency is irrelevant.
  tlul_socket_1n #(
    .N(2),
    .HReqPass(1'b0),
    .HRspPass(1'b0)
  ) fast_split_i (
    .clk_i,
    .rst_ni,
    .tl_h_i     (tl_device_fast_i),
    .tl_h_o     (tl_device_fast_o),
    .tl_d_o     (fast_h2d),
    .tl_d_i     (fast_d2h),
    .dev_select (fast_sel)
  );

  tlul_pkg::tl_h2d_t host_h2d [2];
  tlul_pkg::tl_d2h_t host_d2h [2];

  tlul_socket_m1 #(
    .M(2)
  ) host_merge_i (
    .clk_i,
    .rst_ni,
    .tl_h_i (host_h2d),
    .tl_h_o (host_d2h),
    .tl_d_o (tl_host_o),
    .tl_d_i (tl_host_i)
  );

  student_dma dma_i (
    .clk_i,
    .rst_ni,
    .tl_o      (fast_d2h[0]),
    .tl_i      (fast_h2d[0]),
    .tl_host_o (host_h2d[0]),
    .tl_host_i (host_d2h[0])
  );

  // MAX_INFLIGHT(1): the DDR3 path cannot hold a transaction whose response
  // comes after the request is withdrawn. See student_gemm.sv.
  student_gemm #(
    .MAX_INFLIGHT(1)
  ) gemm_i (
    .clk_i,
    .rst_ni,
    .tl_o      (fast_d2h[1]),
    .tl_i      (fast_h2d[1]),
    .tl_host_o (host_h2d[1]),
    .tl_host_i (host_d2h[1])
  );

endmodule
