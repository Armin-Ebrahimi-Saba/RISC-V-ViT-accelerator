/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Minimal float math for the inference engine.
 *
 * The rvlab software build links with -nostdlib and only -lgcc, so libm is not
 * available: libgcc supplies soft-float add/mul/div/compare but no
 * transcendentals. These implementations are used on both the RISC-V target
 * and the host verification build, which keeps the two bit-identical.
 *
 * All of these are called O(channels) times per layer (never per activation),
 * so clarity beats speed.
 */

#pragma GCC optimize ("O2")

#include "dav2_mathf.h"

#include <stdint.h>

typedef union { float f; uint32_t u; } fu_t;

float dav2_fabsf(float x)
{
    fu_t v; v.f = x; v.u &= 0x7fffffffu; return v.f;
}

float dav2_frexpf(float x, int *e)
{
    fu_t v; v.f = x;
    uint32_t exp = (v.u >> 23) & 0xffu;
    if (exp == 0) {                 /* zero or subnormal */
        if ((v.u & 0x7fffffffu) == 0) { *e = 0; return x; }
        v.f = x * 16777216.0f;      /* scale by 2^24 and retry */
        uint32_t u2 = v.u;
        exp = (u2 >> 23) & 0xffu;
        *e = (int)exp - 126 - 24;
        v.u = (u2 & 0x807fffffu) | (126u << 23);
        return v.f;
    }
    if (exp == 0xffu) { *e = 0; return x; }   /* inf / nan */
    *e = (int)exp - 126;
    v.u = (v.u & 0x807fffffu) | (126u << 23); /* force exponent to 2^-1 */
    return v.f;
}

float dav2_ldexpf(float x, int e)
{
    /* Repeated multiplication keeps this exact and avoids exponent overflow
     * corner cases; e is small in this engine. */
    float r = x;
    while (e > 30)  { r *= 1073741824.0f; e -= 30; }
    while (e < -30) { r *= (1.0f / 1073741824.0f); e += 30; }
    fu_t s;
    s.u = (uint32_t)(127 + e) << 23;
    return r * s.f;
}

float dav2_sqrtf(float x)
{
    if (x <= 0.0f) return 0.0f;
    int e;
    float m = dav2_frexpf(x, &e);      /* x = m * 2^e, m in [0.5,1) */
    if (e & 1) { m *= 0.5f; e += 1; }  /* make the exponent even */
    /* Newton-Raphson on m in [0.25,1); 4 iterations reach float precision. */
    float r = 0.5f * (m + 1.0f) * 0.75f + 0.25f;
    r = 0.5f * (r + m / r);
    r = 0.5f * (r + m / r);
    r = 0.5f * (r + m / r);
    r = 0.5f * (r + m / r);
    return dav2_ldexpf(r, e / 2);
}

/* exp2 for the fractional range [0,1) via a degree-5 minimax-ish polynomial
 * (Taylor of 2^f with enough terms for float accuracy). */
static float exp2_frac(float f)
{
    const float c1 = 0.6931471805599453f;
    float t = f * c1;
    /* e^t, t in [0, 0.6931] */
    return 1.0f + t * (1.0f + t * (0.5f + t * (0.16666667f +
           t * (0.041666667f + t * (0.008333333f + t * 0.001388889f)))));
}

float dav2_expf(float x)
{
    if (x > 88.0f)  return 3.4028235e38f;
    if (x < -88.0f) return 0.0f;
    const float log2e = 1.4426950408889634f;
    float y = x * log2e;
    int i = (int)(y >= 0.0f ? y + 0.5f : y - 0.5f);  /* nearest integer */
    float f = y - (float)i;                          /* f in [-0.5, 0.5] */
    float m = (f >= 0.0f) ? exp2_frac(f) : (1.0f / exp2_frac(-f));
    return dav2_ldexpf(m, i);
}

/* Abramowitz & Stegun 7.1.26; |error| < 1.5e-7, far below int8 weight noise. */
float dav2_erff(float x)
{
    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    float a = dav2_fabsf(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t)
                        + 1.421413741f) * t - 0.284496736f) * t
                      + 0.254829592f) * t * dav2_expf(-a * a);
    return sign * y;
}

float dav2_gelu_f(float x)
{
    /* Exact erf GELU, matching torch.nn.GELU's default. */
    return 0.5f * x * (1.0f + dav2_erff(x * 0.70710678118654752f));
}
