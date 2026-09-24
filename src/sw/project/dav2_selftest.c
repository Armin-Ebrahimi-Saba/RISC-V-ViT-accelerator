/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Deterministic kernel self-test.
 *
 * Purpose: prove that the RISC-V build of the engine computes bit-identically
 * to the host build. The engine is all integer, scales included (dav2_xf.h),
 * so the checksum must match exactly. A mismatch points at the toolchain or
 * a 32-bit assumption -- not at the model.
 *
 * It is deliberately small (a few hundred thousand cycles, a 32 KB arena in
 * BRAM) so it can run inside an RTL simulation, where a full 126x126 inference
 * would take billions of cycles and is not feasible.
 */

#include "dav2.h"

#include <stdio.h>

static uint8_t selftest_arena[32768];

/* xorshift32: same sequence everywhere, no library dependency */
static uint32_t rng_state;
static uint32_t rng_next(void)
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static int16_t rng_act(void)
{
    return (int16_t)((int32_t)(rng_next() % (2u * DAV2_ACT_QMAX + 1u)) - DAV2_ACT_QMAX);
}

/* FNV-1a over raw bytes */
static uint32_t hash_update(uint32_t h, const void *data, int nbytes)
{
    const uint8_t *p = (const uint8_t *)data;
    for (int i = 0; i < nbytes; i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t hash_tensor(uint32_t h, const dav2_tensor_t *t)
{
    h = hash_update(h, t->v, t->n * t->c * (int)sizeof(int16_t));
    h = hash_update(h, &t->scale, (int)sizeof(dav2_xf_t));
    return h;
}

static void fill_qw(dav2_qw_t *w, int8_t *wbuf, dav2_xf_t *sbuf, dav2_xf_t *bbuf,
                    int m, int k, int with_bias)
{
    for (int i = 0; i < m * k; i++)
        wbuf[i] = (int8_t)((int32_t)(rng_next() % 255u) - 127);
    for (int i = 0; i < m; i++) {
        sbuf[i] = xf_norm(1 + (rng_next() % 1000u), 17);               /* ~1e-5..1e-2 */
        if (bbuf)
            bbuf[i] = xf_norm((int32_t)(rng_next() % 2000u) - 1000, 10);  /* ~+-1 */
    }
    w->w = wbuf;
    w->s = sbuf;
    w->b = with_bias ? bbuf : 0;
    w->m = m;
    w->k = k;
}

uint32_t dav2_selftest(void)
{
    uint32_t h = 2166136261u;
    rng_state = 0x12345678u;
    dav2_arena_init(selftest_arena, sizeof(selftest_arena));

    static int8_t wbuf[64 * 72];
    static dav2_xf_t sbuf[64], bbuf[64];

    /* --- GEMM: the kernel that dominates the whole network ------------- */
    {
        dav2_tensor_t a = dav2_tensor_new(8, 32);
        for (int i = 0; i < 8 * 32; i++) a.v[i] = rng_act();
        a.scale = xf_norm(1294, 20);              /* ~1.234e-3 */

        dav2_qw_t w;
        fill_qw(&w, wbuf, sbuf, bbuf, 16, 32, 1);
        dav2_tensor_t out = dav2_tensor_new(8, 16);
        dav2_qgemm(&a, &w, &out);
        h = hash_tensor(h, &out);

        /* the no-bias path (used by the scratch layer_rn convolutions) */
        dav2_qw_t w2;
        fill_qw(&w2, wbuf, sbuf, bbuf, 16, 32, 0);
        dav2_qgemm(&a, &w2, &out);
        h = hash_tensor(h, &out);
    }

    /* --- LayerNorm: exercises int64 accumulation and xf_rsqrt ---------- */
    {
        dav2_tensor_t a = dav2_tensor_new(6, 64);
        for (int i = 0; i < 6 * 64; i++) a.v[i] = rng_act();
        a.scale = xf_norm(577, 20);               /* ~5.5e-4 */
        static int32_t g[64], b[64];              /* Q15 and Q16 */
        for (int i = 0; i < 64; i++) {
            g[i] = 16384 + (int32_t)(rng_next() % 100u) * 328;         /* 0.5..1.5 */
            b[i] = ((int32_t)(rng_next() % 200u) - 100) * 66;          /* +-0.1 */
        }
        dav2_tensor_t out = dav2_tensor_new(6, 64);
        dav2_layernorm(&a, g, b, &out);
        h = hash_tensor(h, &out);
    }

    /* --- GELU: exercises the LUT build (Phi table) and interpolation --- */
    {
        dav2_tensor_t a = dav2_tensor_new(4, 64);
        for (int i = 0; i < 4 * 64; i++) a.v[i] = rng_act();
        a.scale = xf_norm(2097, 20);              /* ~2e-3 */
        dav2_gelu(&a);
        h = hash_tensor(h, &a);
    }

    /* --- residual add with mismatched scales --------------------------- */
    {
        dav2_tensor_t a = dav2_tensor_new(4, 32);
        dav2_tensor_t b = dav2_tensor_new(4, 32);
        for (int i = 0; i < 4 * 32; i++) { a.v[i] = rng_act(); b.v[i] = rng_act(); }
        a.scale = xf_norm(1049, 20);              /* ~1e-3 */
        b.scale = xf_norm(77, 20);                /* ~7.3e-5 */
        dav2_tensor_t out = dav2_tensor_new(4, 32);
        dav2_add(&a, &b, &out);
        h = hash_tensor(h, &out);
    }

    /* --- convolution (im2col + GEMM), stride 1 and 2 ------------------- */
    {
        dav2_tensor_t in = dav2_tensor_new(6 * 6, 8);
        for (int i = 0; i < 6 * 6 * 8; i++) in.v[i] = rng_act();
        in.scale = xf_norm(944, 20);              /* ~9e-4 */
        dav2_qw_t w;
        fill_qw(&w, wbuf, sbuf, bbuf, 8, 9 * 8, 1);
        int oh, ow;
        size_t mark = dav2_arena_mark();
        dav2_tensor_t c1 = dav2_conv2d(&in, 6, 6, &w, 3, 1, 1, &oh, &ow);
        h = hash_tensor(h, &c1);
        dav2_arena_release(mark);
        dav2_tensor_t c2 = dav2_conv2d(&in, 6, 6, &w, 3, 2, 1, &oh, &ow);
        h = hash_tensor(h, &c2);
        dav2_arena_release(mark);

        /* transposed convolution (the DPT resize layers) */
        dav2_qw_t wt;
        fill_qw(&wt, wbuf, sbuf, bbuf, 4 * 8, 8, 1);   /* stride 2, cout 8 */
        dav2_tensor_t ct = dav2_conv_transpose(&in, 6, 6, &wt, 2, &oh, &ow);
        h = hash_tensor(h, &ct);
        dav2_arena_release(mark);

        /* bilinear resize, align_corners */
        dav2_tensor_t up = dav2_tensor_new(11 * 11, 8);
        dav2_interpolate(&in, 6, 6, 11, 11, &up);
        h = hash_tensor(h, &up);
        dav2_arena_release(mark);
    }

    /* --- the integer real-number routines (dav2_xf.h) ------------------ */
    {
        dav2_xf_t probes[6];
        probes[0] = xf_rsqrt(xf_from_int(2));
        probes[1] = xf_recip(xf_from_int(7));
        probes[2] = xf_add(xf_norm(3, 5), xf_norm(-7, 9));
        probes[3] = xf_mul(xf_norm(-12345, 20), xf_norm(6789, 3));
        probes[4] = xf_from_f32_bits(0x3fb8aa3bu);          /* 1.4426950f */
        probes[5] = xf_norm((int64_t)1 << 40, 0);
        h = hash_update(h, probes, (int)sizeof(probes));

        int64_t r[2] = { xf_round(xf_norm(5, 1), 0), xf_round(xf_norm(-5, 1), 0) };
        h = hash_update(h, r, (int)sizeof(r));             /* 2.5 -> 3, -2.5 -> -3 */
    }

    return h;
}

void dav2_selftest_report(void)
{
    uint32_t h = dav2_selftest();
    printf("DAV2_SELFTEST %08x\n", (unsigned)h);
}
