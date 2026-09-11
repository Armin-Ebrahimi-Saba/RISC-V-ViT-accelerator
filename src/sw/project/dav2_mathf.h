/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * The float functions the engine needs that libm would normally provide.
 * The target links -nostdlib, so there is no libm; these are hand-written
 * and used on the host build too, which keeps the two bit-identical.
 * See dav2_mathf.c for the algorithms and their accuracy. */

#ifndef DAV2_MATHF_H
#define DAV2_MATHF_H

float dav2_fabsf(float x);
float dav2_frexpf(float x, int *e);
float dav2_ldexpf(float x, int e);
float dav2_sqrtf(float x);
float dav2_expf(float x);
float dav2_erff(float x);
float dav2_gelu_f(float x);

#endif
