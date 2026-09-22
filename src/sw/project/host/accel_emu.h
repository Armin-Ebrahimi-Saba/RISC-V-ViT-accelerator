/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Register-level C model of student_gemm, for the host build.
 *
 * With -DDAV2_ACCEL_EMU the driver (dav2_accel.c) is compiled for the host
 * and REG32() resolves here instead of to a bus address. The model follows
 * the register descriptions in src/design/reggen/student_gemm.hjson -- not
 * the RTL -- so running the engine through it checks the driver's register
 * programming, the job semantics it relies on and every asynchronous path,
 * bit for bit against the plain host build, without a board.
 *
 * Addresses the driver passes are 32-bit, so every buffer the model is
 * handed must lie below 4 GB: the emulated build is linked non-PIE and the
 * host allocates its big buffers with MAP_32BIT (see host_main.c).
 */
#ifndef ACCEL_EMU_H
#define ACCEL_EMU_H

#include <stdint.h>

volatile uint32_t *dav2_emu_reg(uint32_t addr);
uint32_t           dav2_emu_mcycle(void);

#define REG32(a) (*dav2_emu_reg((uint32_t)(a)))

#endif
