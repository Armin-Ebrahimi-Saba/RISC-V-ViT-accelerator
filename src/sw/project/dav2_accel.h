/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Driver for the student_gemm int8 GEMM accelerator (src/rtl/student/).
 *
 * The accelerator is optional: every entry point degrades to "not available"
 * so the same sources build and run unchanged on the host, in an RTL
 * simulation without the block, and on the FPGA with it.
 */

#ifndef DAV2_ACCEL_H
#define DAV2_ACCEL_H

#include <stdint.h>
#include "dav2.h"

/* Probe the hardware and latch its capabilities. Safe to call more than once.
 * Returns non-zero when a usable accelerator was found. */
int dav2_accel_init(void);

/* Non-zero once dav2_accel_init() has found a usable accelerator and no bus
 * error has disabled it since. */
int dav2_accel_present(void);

/* Human-readable one-line summary for the boot log. */
void dav2_accel_report(void);

/* Computes acc[m*N + n] = sum_k a->v[n*K+k] * wt->w[m*K+k] in hardware.
 *
 * Returns 1 when the whole product was produced by the accelerator, 0 when it
 * declined (absent, or the shape/alignment is outside what it supports, or a
 * bus error occurred). On 0 the contents of acc are undefined and the caller
 * must run the software kernel instead. */
/* Per-weight-row accumulator statistics the GEMM job writes as it drains:
 * v[(tile*M + m)*2] = max, v[..+1] = min over that tile's rows. tiles = 0 when
 * the block produced none (absent, or a K-split job), in which case the
 * caller scans acc itself. Allocated from the arena inside the caller's
 * mark/release. */
typedef struct {
    const int32_t *v;
    int            tiles;
} dav2_accel_stats_t;

/* A statistics slot not yet written by a running job (see *_async below). */
#define DAV2_STATS_EMPTY_MAX ((int32_t)0x80000000)
#define DAV2_STATS_EMPTY_MIN ((int32_t)0x7fffffff)

int dav2_accel_qgemm(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc,
                     dav2_accel_stats_t *st);

/* Convolution with the A matrix gathered in hardware from the NHWC int16
 * image img (h x w x C): acc[m][n], n = output pixel, as dav2_accel_qgemm
 * would produce for the im2col matrix. wt rows are k*k*C long, kernel
 * position major (ky, kx, c). C must be a multiple of 4. Statistics as for
 * dav2_accel_qgemm (none when K is split). 0 = declined, use im2col. */
int dav2_accel_conv(const int16_t *img, int h, int w, int C, int k, int stride,
                    int pad, const int8_t *wt, int M, int32_t *acc,
                    dav2_accel_stats_t *st);

/* Requantisation job: out[n][m] = sat14((acc[m][n]*mult + 2^(sh-1)) >> sh + bias)
 * with params[3m] = mult, [3m+1] = shift, [3m+2] = bias. M must be even.
 * *amax_out receives the largest |out| written (the block tracks it), or -1
 * if it is not available. Returns 1 when done in hardware, 0 when the
 * caller must do it. */
int dav2_accel_requant(const int32_t *acc, int N, int M, const int32_t *params,
                       int16_t *out, int32_t *amax_out);

/* The same product on raw pointers with explicit row strides (bytes; 0 =
 * contiguous), for operands that are slices of larger tensors -- attention
 * reads q, k and v out of the qkv tensor this way. K must be a multiple of 4
 * and at most CAPS.KMAX; all pointers 4-byte aligned. Same return contract. */
int dav2_accel_gemm_raw(const int16_t *a, uint32_t a_stride,
                        const int8_t *w, uint32_t w_stride,
                        int32_t *acc, int N, int K, int M);

/* Software reference for the same product and the same acc layout. Exposed so
 * dav2_accel_check() can compare the two. */
int dav2_accel_bigcheck(int N, int K, int M);
void dav2_qgemm_cpu(const int16_t *av, const int8_t *w, int32_t *acc,
                    int n, int k, int m);

/* Runs a small GEMM through both paths and compares them element by element.
 * Returns 0 on match, or the number of mismatching elements. Called once at
 * start-up so a broken accelerator is caught before the model runs, rather
 * than showing up as a subtly wrong depth map. */
int dav2_accel_check(void);

/* ---- asynchronous use ---------------------------------------------------
 *
 * The *_async variants behave like their synchronous twins, except that the
 * LAST job of the operation may be left running: they then return 2
 * instead of 1. Until dav2_accel_finish() has returned 1, the outputs of
 * that last job are not all there -- but a GEMM job writes each weight
 * row's accumulators and then its {max, min} statistics in row order, so a
 * caller may consume rows whose statistics have appeared (see dav2_qgemm).
 * dav2_accel_finish() returns 0 if the job failed; the caller then redoes
 * the work on the CPU. Starting any other accelerator operation settles a
 * pending job first, so at most one is ever outstanding. */
int dav2_accel_qgemm_async(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc,
                           dav2_accel_stats_t *st);
/* in_relu: the block applies ReLU to the image while it gathers it
 * (CTRL.grelu); 0 (declined) when it cannot. */
int dav2_accel_conv_async(const int16_t *img, int h, int w, int C, int k, int stride,
                          int pad, const int8_t *wt, int M, int32_t *acc,
                          dav2_accel_stats_t *st, int in_relu);
int dav2_accel_gemm_raw_async(const int16_t *a, uint32_t a_stride,
                              const int8_t *w, uint32_t w_stride,
                              int32_t *acc, int N, int K, int M);
/* Rows m0..m0+mc of a requantisation (mc even, <= CAPS.KMAX/2); *amax is
 * raised to the largest |out| once the job has been collected. */
/* Optional epilogue of a requantisation job (CTRL.add / CTRL.relu):
 *   add:  out = sat14(apply(x[n][m], mx, sx) + apply(h, mh, sh)), x laid out
 *         like out (rows M int16 apart); h the plain requantised value
 *   relu: out = max(out, 0)
 * apply(v, mult, shift) is apply_multiplier in dav2_ops.c. With add, a chunk
 * is at most CAPS.KMAX/4 rows. */
typedef struct {
    const int16_t *x;
    int32_t        mx, mh;
    int            sx, sh;
    int            add, relu;
    /* CTRL.lut: out = lut[v + 8192] after add and ReLU (GELU). lut holds
     * 16384 int16 (32 kB, 4-byte aligned). lut_load: the block reads the
     * table first; set it on the first job of a GEMM only, the table is
     * kept for later jobs. */
    const int16_t *lut;
    int            lut_load;
    /* CTRL.ostats: each output row n's {max, min} of this job's columns,
     * as one word (max in bits 31:16), at ostats[n]. */
    uint32_t      *ostats;
    /* with ostats, CTRL.osums: four words per row at ostats[4n]: {max, min},
     * the row's sum, its sum of squares (bits 31:0, then 63:32) */
    int            osums;
    /* CTRL.onchip: the input is the result the last GEMM left in the
     * block's result RAM (dav2_accel_qgemm_onchip_async), not acc */
    int            onchip;
} dav2_rq_epi_t;
/* A GEMM whose int32 result stays in the block's result RAM (CTRL.onchip,
 * CAPS bit 28): N <= CAPS.NROWS, N*M <= DAV2_ACCEL_CR_WORDS. Only the row
 * statistics come back (st, as for dav2_accel_qgemm_async); the next
 * requantisation must read it with epi.onchip. 0 when not possible. */
#define DAV2_ACCEL_CR_WORDS 131072
int dav2_accel_onchip_ok(void);
int dav2_accel_qgemm_onchip_async(const dav2_tensor_t *a, const dav2_qw_t *wt,
                                  dav2_accel_stats_t *st);
/* A convolution (as dav2_accel_conv_async) whose result stays in the
 * result RAM: acc[m][n] at word m*N + n. Only when all k*k positions fit one
 * job and N*M <= DAV2_ACCEL_CR_WORDS; the statistics come back in st.
 * 0 when not possible. */
int dav2_accel_conv_onchip_async(const int16_t *img, int h, int w, int C, int k, int stride,
                                 int pad, const int8_t *wt, int M, dav2_accel_stats_t *st,
                                 int in_relu);
/* Non-zero if gather jobs can apply ReLU to the image (CAPS bit 31). */
int dav2_accel_grelu_ok(void);
/* Non-zero if the block writes output row statistics (CAPS bit 27). */
int dav2_accel_ostats_ok(void);
int dav2_accel_osums_ok(void);
/* Non-zero if the block has the lookup table (CAPS bit 24) and its boot
 * self-test passed. */
int dav2_accel_lut_ok(void);
/* A requantisation over int16 input (CTRL.a16): out[n][m] = sat14(round(
 * in[m][n] * mult_m / 2^shift_m) + bias_m), in rows of N int16 (N even),
 * out rows of M int16 (M even). Synchronous. Returns 0 when the block
 * lacks the mode (CAPS bit 25) or the shape; *amax gets the largest |out|. */
/* GEMM with int16 weights (CTRL.w16): acc[m][n] = sum_k a[n][k] w[m][k],
 * strides in bytes (0 = contiguous). stats, if given, receives each row's
 * {max, min} (N <= CAPS.NROWS). Left running like the other _async calls;
 * 0 when the block lacks the mode (CAPS bit 26) or the shape. */
int dav2_accel_w16_ok(void);
int dav2_accel_gemm16_async(const int16_t *a, uint32_t a_stride,
                            const int16_t *w, uint32_t w_stride,
                            int32_t *acc, int N, int K, int M, int32_t *stats);
/* dav2_accel_requant with output rows out_stride int16 apart (instead of
 * M): writes into a slice of a wider tensor. */
int dav2_accel_requant_stride(const int32_t *acc, int N, int M, const int32_t *params,
                              int16_t *out, int out_stride, int32_t *amax);
int dav2_accel_requant16(const int16_t *in, int N, int M, const int32_t *params,
                         int16_t *out, int32_t *amax, uint32_t *ostats);

int dav2_accel_requant_rows_async(const int32_t *acc, int N, int M, int m0, int mc,
                                  const int32_t *params, int16_t *out, int32_t *amax,
                                  const dav2_rq_epi_t *epi);
int dav2_accel_finish(void);
int dav2_accel_busy(void);      /* a job is pending and still running */

#endif /* DAV2_ACCEL_H */
