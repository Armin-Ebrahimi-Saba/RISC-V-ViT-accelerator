/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Deterministic kernel self-test.
 *
 * Purpose: prove that the RISC-V build of the engine computes bit-identically
 * to the host build. Every kernel in the engine is integer, and the float
 * bookkeeping is IEEE-754 single precision on both sides (libgcc soft-float on
 * the CV32E40P), so the checksum must match exactly. A mismatch points at the
 * toolchain, the soft-float path, or a 32-bit assumption -- not at the model.
 *
 * It is deliberately small (a few hundred thousand cycles, a 32 KB arena in
 * BRAM) so it can run inside an RTL simulation, where a full 126x126 inference
 * would take billions of cycles and is not feasible.
 */

#include "dav2.h"
#include "dav2_mathf.h"

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
    h = hash_update(h, &t->scale, (int)sizeof(float));
    return h;
}

static void fill_qw(dav2_qw_t *w, int8_t *wbuf, float *sbuf, float *bbuf,
                    int m, int k, int with_bias)
{
    for (int i = 0; i < m * k; i++)
        wbuf[i] = (int8_t)((int32_t)(rng_next() % 255u) - 127);
    for (int i = 0; i < m; i++) {
        sbuf[i] = (float)(1 + (rng_next() % 1000u)) * 1e-5f;
        if (bbuf)
            bbuf[i] = (float)((int32_t)(rng_next() % 2000u) - 1000) * 1e-3f;
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
    static float  sbuf[64], bbuf[64];

    /* --- GEMM: the kernel that dominates the whole network ------------- */
    {
        dav2_tensor_t a = dav2_tensor_new(8, 32);
        for (int i = 0; i < 8 * 32; i++) a.v[i] = rng_act();
        a.scale = 1.234e-3f;

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

    /* --- LayerNorm: exercises int64 accumulation and dav2_sqrtf -------- */
    {
        dav2_tensor_t a = dav2_tensor_new(6, 64);
        for (int i = 0; i < 6 * 64; i++) a.v[i] = rng_act();
        a.scale = 5.5e-4f;
        static float g[64], b[64];
        for (int i = 0; i < 64; i++) {
            g[i] = 0.5f + (float)(rng_next() % 100u) * 0.01f;
            b[i] = (float)((int32_t)(rng_next() % 200u) - 100) * 0.001f;
        }
        dav2_tensor_t out = dav2_tensor_new(6, 64);
        dav2_layernorm(&a, g, b, &out);
        h = hash_tensor(h, &out);
    }

    /* --- GELU: exercises the LUT build (erf, exp) and interpolation ---- */
    {
        dav2_tensor_t a = dav2_tensor_new(4, 64);
        for (int i = 0; i < 4 * 64; i++) a.v[i] = rng_act();
        a.scale = 2.0e-3f;
        dav2_gelu(&a);
        h = hash_tensor(h, &a);
    }

    /* --- residual add with mismatched scales --------------------------- */
    {
        dav2_tensor_t a = dav2_tensor_new(4, 32);
        dav2_tensor_t b = dav2_tensor_new(4, 32);
        for (int i = 0; i < 4 * 32; i++) { a.v[i] = rng_act(); b.v[i] = rng_act(); }
        a.scale = 1.0e-3f;
        b.scale = 7.3e-5f;
        dav2_tensor_t out = dav2_tensor_new(4, 32);
        dav2_add(&a, &b, &out);
        h = hash_tensor(h, &out);
    }

    /* --- convolution (im2col + GEMM), stride 1 and 2 ------------------- */
    {
        dav2_tensor_t in = dav2_tensor_new(6 * 6, 8);
        for (int i = 0; i < 6 * 6 * 8; i++) in.v[i] = rng_act();
        in.scale = 9.0e-4f;
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

    /* --- float helpers, checked by bit pattern ------------------------- */
    {
        float probes[6];
        probes[0] = dav2_sqrtf(2.0f);
        probes[1] = dav2_expf(-3.25f);
        probes[2] = dav2_erff(0.75f);
        probes[3] = dav2_gelu_f(-1.5f);
        int e;
        probes[4] = dav2_frexpf(1234.5f, &e);
        probes[5] = (float)e;
        h = hash_update(h, probes, (int)sizeof(probes));

        int32_t mult; int shift;
        dav2_make_multiplier(3.7e-5f, &mult, &shift);
        h = hash_update(h, &mult, (int)sizeof(mult));
        h = hash_update(h, &shift, (int)sizeof(shift));
    }

    return h;
}

void dav2_selftest_report(void)
{
    uint32_t h = dav2_selftest();
    printf("DAV2_SELFTEST %08x\n", (unsigned)h);
}
