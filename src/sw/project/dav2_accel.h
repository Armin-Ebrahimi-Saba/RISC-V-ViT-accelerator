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

#endif /* DAV2_ACCEL_H */
