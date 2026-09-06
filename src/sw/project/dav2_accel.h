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
int dav2_accel_qgemm(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc);

/* Software reference for the same product and the same acc layout. Exposed so
 * dav2_accel_check() can compare the two. */
void dav2_qgemm_cpu(const int16_t *av, const int8_t *w, int32_t *acc,
                    int n, int k, int m);

/* Runs a small GEMM through both paths and compares them element by element.
 * Returns 0 on match, or the number of mismatching elements. Called once at
 * start-up so a broken accelerator is caught before the model runs, rather
 * than showing up as a subtly wrong depth map. */
int dav2_accel_check(void);

#endif /* DAV2_ACCEL_H */
