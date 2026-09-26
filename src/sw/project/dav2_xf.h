/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Real numbers without floating point.
 *
 * The CV32E40P has no FPU, and every float operation in C is a libgcc
 * call of 35 to 200 cycles. The engine therefore keeps every scale as a
 * pair of integers:
 *
 *     value = m * 2^-sh,   2^30 <= |m| < 2^31, or m = 0.
 *
 * This is the same (multiplier, shift) pair that apply_multiplier in
 * dav2_ops.c uses to scale an integer, so a scale can be applied directly.
 * The routines below are short integer code: one or two 64-bit multiplies
 * and a normalisation. The mantissa has 31 bits, so they are more precise
 * than float32 (24 bits). The results therefore differ from float32
 * arithmetic in the last bits, but host and board run this same code and
 * agree exactly.
 */
#ifndef DAV2_XF_H
#define DAV2_XF_H

#include <stdint.h>

typedef struct {
    int32_t m;      /* signed mantissa, 2^30 <= |m| < 2^31, or 0 */
    int32_t sh;     /* value = m * 2^-sh                          */
} dav2_xf_t;

#define XF_ZERO ((dav2_xf_t){ 0, 0 })
#define XF_ONE  ((dav2_xf_t){ 1 << 30, 30 })

/* Leading zeros of a != 0. The core has no clz instruction, and
 * __builtin_clzll is a libgcc call there; a binary search inline costs
 * less. Elsewhere the builtin. Both give the same number. */
static inline int xf_clz64(uint64_t a)
{
#if defined(__riscv) && !defined(__riscv_zbb)
    uint32_t x = (uint32_t)(a >> 32);
    int n = 0;
    if (!x) { x = (uint32_t)a; n = 32; }
    if (!(x >> 16)) { n += 16; x <<= 16; }
    if (!(x >> 24)) { n += 8;  x <<= 8; }
    if (!(x >> 28)) { n += 4;  x <<= 4; }
    if (!(x >> 30)) { n += 2;  x <<= 2; }
    return n + (int)(!(x >> 31));
#else
    return __builtin_clzll(a);
#endif
}

/* Normalise v * 2^-sh, rounding to nearest (ties away from zero). */
static inline dav2_xf_t xf_norm(int64_t v, int32_t sh)
{
    if (v == 0)
        return XF_ZERO;
    uint64_t a = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v;
    int s = 33 - xf_clz64(a);                 /* bits(a) - 31 */
    if (s > 0) {
        a = (a + ((uint64_t)1 << (s - 1))) >> s;
        if (a >> 31) {                        /* rounding carried to 2^31 */
            a >>= 1;
            s++;
        }
    } else if (s < 0) {
        a <<= -s;
    }
    dav2_xf_t r = { v < 0 ? -(int32_t)a : (int32_t)a, sh - s };
    return r;
}

static inline dav2_xf_t xf_from_int(int64_t v) { return xf_norm(v, 0); }

/* An IEEE float32, given as its bit pattern, converted exactly. Used for
 * the one float the board still receives: the image scale in the .dav2img
 * file. Only bit operations; denormals are treated as zero. */
static inline dav2_xf_t xf_from_f32_bits(uint32_t b)
{
    int32_t e = (int32_t)((b >> 23) & 0xffu);
    if (e == 0)
        return XF_ZERO;
    int32_t m = (int32_t)(((b & 0x7fffffu) | 0x800000u) << 7);
    dav2_xf_t r = { (b >> 31) ? -m : m, 157 - e };
    return r;
}

static inline dav2_xf_t xf_mul(dav2_xf_t a, dav2_xf_t b)
{
    if (a.m == 0 || b.m == 0)
        return XF_ZERO;
    return xf_norm((int64_t)a.m * (int64_t)b.m, a.sh + b.sh);
}

static inline dav2_xf_t xf_add(dav2_xf_t a, dav2_xf_t b)
{
    if (a.m == 0) return b;
    if (b.m == 0) return a;
    /* hi has the larger magnitude scale (smaller sh) */
    dav2_xf_t hi = a.sh <= b.sh ? a : b, lo = a.sh <= b.sh ? b : a;
    int64_t A = (int64_t)hi.m * ((int64_t)1 << 31);
    int32_t d = lo.sh - hi.sh;
    int64_t B;
    if (d <= 31) {
        B = (int64_t)lo.m * ((int64_t)1 << (31 - d));
    } else if (d - 31 < 40) {
        int s = d - 31;
        int64_t am = lo.m < 0 ? -(int64_t)lo.m : lo.m;
        am = (am + ((int64_t)1 << (s - 1))) >> s;
        B = lo.m < 0 ? -am : am;
    } else {
        B = 0;
    }
    return xf_norm(A + B, hi.sh + 31);
}

static inline dav2_xf_t xf_neg(dav2_xf_t a) { a.m = -a.m; return a; }
static inline dav2_xf_t xf_abs(dav2_xf_t a) { if (a.m < 0) a.m = -a.m; return a; }

/* Compare |a| with |b|: > 0 if |a| > |b|, 0 if equal, < 0 if smaller. */
static inline int xf_cmp_abs(dav2_xf_t a, dav2_xf_t b)
{
    if (a.m == 0) return b.m == 0 ? 0 : -1;
    if (b.m == 0) return 1;
    if (a.sh != b.sh) return a.sh < b.sh ? 1 : -1;
    int32_t x = a.m < 0 ? -a.m : a.m, y = b.m < 0 ? -b.m : b.m;
    return (x > y) - (x < y);
}

/* 1 / a, for a != 0. One 64-bit integer division. */
static inline dav2_xf_t xf_recip(dav2_xf_t a)
{
    uint64_t x = a.m < 0 ? (uint64_t)(-(int64_t)a.m) : (uint64_t)a.m;
    uint64_t q = (((uint64_t)1 << 61) + x / 2u) / x;      /* (2^30, 2^31] */
    dav2_xf_t r = xf_norm((int64_t)q, 61 - a.sh);
    if (a.m < 0) r.m = -r.m;
    return r;
}

static inline dav2_xf_t xf_div(dav2_xf_t a, dav2_xf_t b) { return xf_mul(a, xf_recip(b)); }

/* round(a * 2^k), ties away from zero, as a 64-bit integer. */
static inline int64_t xf_round(dav2_xf_t a, int k)
{
    int32_t s = a.sh - k;
    if (a.m == 0 || s > 62)
        return 0;
    if (s <= 0)
        return (int64_t)a.m * ((int64_t)1 << (-s > 32 ? 32 : -s));
    int64_t am = a.m < 0 ? -(int64_t)a.m : a.m;
    am = (am + ((int64_t)1 << (s - 1))) >> s;
    return a.m < 0 ? -am : am;
}

/* 1 / sqrt(a), for a > 0: five Newton steps in Q30, from the chord of
 * 1/sqrt(x) over [1, 4]. The chord lies above the curve, so every step
 * moves down towards the root; five steps reach the Q30 resolution. */
static inline dav2_xf_t xf_rsqrt(dav2_xf_t a)
{
    /* All in 32-bit operands with 64-bit products (no 64-bit division or
     * 64 x 64 multiply, which are library calls on this core): X < 2^32;
     * y starts in [2^29, 2^30] and stays below 2^30; x y^2 <= 1.41 at the
     * start and <= 1 after the first step, so 3 - x y^2 is in (0, 3). */
    uint32_t X = (uint32_t)a.m;             /* a = X/2^30 * 2^E, X/2^30 in [1,2) */
    int32_t E = 30 - a.sh;
    if (E & 1) { X <<= 1; E -= 1; }         /* X/2^30 in [1,4), E even */
    const uint32_t one = 1u << 30;
    uint32_t y = one - (X - one) / 6u;
    for (int i = 0; i < 5; i++) {
        uint32_t y2  = (uint32_t)(((uint64_t)y * y) >> 30);
        uint32_t xy2 = (uint32_t)(((uint64_t)X * y2) >> 30);
        y = (uint32_t)(((uint64_t)y * (3u * one - xy2)) >> 31);
    }
    return xf_norm((int64_t)y, 30 + E / 2); /* y/2^30 * 2^(-E/2) */
}

/* The (multiplier, shift) pair for apply_multiplier, for a > 0. A factor
 * below 2^-32 becomes 0; a factor of 2^31 or more (never needed) is
 * clamped. */
static inline void xf_to_mult(dav2_xf_t a, int32_t *mult, int *shift)
{
    if (a.m <= 0 || a.sh > 62) { *mult = 0; *shift = 0; return; }
    if (a.sh < 0)              { *mult = 2147483647; *shift = 0; return; }
    *mult = a.m;
    *shift = a.sh;
}

#endif
