/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Depth-Anything V2 Small: ViT-S/14 encoder + DPT depth head.
 *
 * Structure mirrors tools/dav2_numpy.py, which was validated against the
 * PyTorch reference to 2.9e-05 before any of this was written.
 *
 * The exporter has already folded away several things, so they do not appear
 * here: LayerScale (gamma folded into the preceding weight matrix and bias),
 * the attention 1/sqrt(head_dim) factor (folded into the Q rows of qkv), and
 * the positional-embedding interpolation (precomputed for the fixed input
 * resolution).
 */

#pragma GCC optimize ("O2")    /* the flow's -Os is wrong for these loops */

#include "dav2.h"
#include "dav2_accel.h"

#include <stdio.h>
#include <string.h>

/* Input image, supplied by the caller through dav2_set_image(). Kept as a
 * pointer rather than copied: on the board it is 95 kB in DDR3. */
static const int16_t *g_image;
static dav2_xf_t      g_image_scale = XF_ONE;

void dav2_set_image(const int16_t *hwc, uint32_t scale_f32_bits)
{
    g_image = hwc;
    g_image_scale = xf_from_f32_bits(scale_f32_bits);
}

extern const void *dav2_find_quiet(const char *name);
extern void dav2_dump(const char *path, const dav2_tensor_t *t);
extern int dav2_blob_check(void);

static const int INTERMEDIATE[4] = {2, 5, 8, 11};

/* ------------------------------------------------------------- attention */

/* Scores for one head, then softmax, then the context vector. q is shifted
 * to 11 bits and k keeps its 14 bits, so that the K=64 dot product cannot
 * overflow int32: 64 * 2047 * 8191 = 1.07e9 < 2^31. */

/* One attention head, on the GEMM accelerator.
 *
 * Both products of attention are matrix multiplications, but neither has an
 * int8 operand: q, k and v are all int16 activations. The accelerator only
 * multiplies int16 by int8, so the second operand is split into a high and a
 * low int8 part and the product is run twice:
 *
 *     x = hi * 2^s + lo,  a.x = 2^s * (a.hi) + (a.lo)         exactly.
 *
 * Scores:  S[t][m] = q_t . k_m      A = k (int16, read in place from qkv),
 *                                   W = q_hi (q >> 4), q_lo (q & 15)
 * Context: C[t][d] = sum_m P[t][m] v[m][d]
 *                                   A = P (int16 probabilities, Q15),
 *                                   W = v^T_hi (v >> 6), v^T_lo (v & 63)
 *
 * so the accelerator does the 2 x 82 x 82 x 64 multiply-accumulates per head
 * that the CPU spent ~16 cycles each on (47 % of the frame), and the CPU is
 * left with the gathers, the softmax and the final rounding. P holds each
 * row's probabilities already divided by the row's sum (Q15, clamped to
 * 32767 so it fits an int16), so the context needs no division.
 *
 * Operands are built in the DDR3 arena, packed four int8 per word, since a
 * byte store costs the CPU the same bus transaction as a word store. */
typedef struct {
    int8_t *q_hi, *q_lo;
    int32_t *s_hi, *s_lo;
    int16_t *p16;  int8_t *vt_hi, *vt_lo;
    int32_t *c_hi, *c_lo;
    int      q_sh;
    uint16_t *exp2;             /* exp2_tab, copied into the DDR3 arena */
} attn_bufs_t;

/* A GEMM that may be left running on the accelerator (see dav2_accel.h):
 * gemm_i32_start issues it -- or, without an accelerator, simply computes it
 * -- and gemm_i32_finish makes its result available, redoing it on the CPU
 * if the block failed. One at a time: starting a new one finishes the old.
 * acc[m*N + n] = sum_k a[n*a_stride + k] * w[m*w_stride + k], strides in
 * elements; integer either way, so host and board agree to the bit. */
static struct {
    int pending;
    const int16_t *a; int a_stride; const int8_t *w; int w_stride;
    int32_t *acc; int N, K, M;
} g_pend;

static void gemm_i32_cpu(const int16_t *a, int a_stride, const int8_t *w, int w_stride,
                         int32_t *acc, int N, int K, int M)
{
    for (int m = 0; m < M; m++) {
        const int8_t *wr = w + (size_t)m * w_stride;
        for (int n = 0; n < N; n++) {
            const int16_t *ar = a + (size_t)n * a_stride;
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)ar[k] * (int32_t)wr[k];
            acc[(size_t)m * N + n] = s;
        }
    }
}

static void gemm_i32_finish(void)
{
    if (!g_pend.pending)
        return;
    g_pend.pending = 0;
    if (!dav2_accel_finish())
        gemm_i32_cpu(g_pend.a, g_pend.a_stride, g_pend.w, g_pend.w_stride,
                     g_pend.acc, g_pend.N, g_pend.K, g_pend.M);
}

static void gemm_i32_start(const int16_t *a, int a_stride, const int8_t *w, int w_stride,
                           int32_t *acc, int N, int K, int M)
{
    gemm_i32_finish();
    int r = dav2_accel_gemm_raw_async(a, (uint32_t)a_stride * 2u, w, (uint32_t)w_stride,
                                      acc, N, K, M);
    if (r == 2) {
        g_pend.pending = 1;
        g_pend.a = a; g_pend.a_stride = a_stride; g_pend.w = w; g_pend.w_stride = w_stride;
        g_pend.acc = acc; g_pend.N = N; g_pend.K = K; g_pend.M = M;
    } else if (r == 0) {
        gemm_i32_cpu(a, a_stride, w, w_stride, acc, N, K, M);
    }
}

/* ---- the phases of one head (the code of the former attention_head) */

/* q of a head, shifted to 11 bits and split into two int8 halves. k needs
 * no preparation: the score GEMMs read it straight from qkv as their int16
 * operand, with qkv's row stride, at its full 14 bits. */
static void attn_prep_qk(const dav2_tensor_t *qkv, int head, int n, attn_bufs_t *b)
{
    const int HD = DAV2_HEAD_DIM;
    int8_t *q_hi = b->q_hi, *q_lo = b->q_lo;
    /* The range of q of this head, read straight from qkv in DDR3, two
     * values per load. The shift below depends only on the highest set bit
     * of the largest |value|, and the OR of all |values| has the same
     * highest bit: no compare and no branch per element. */
    uint32_t qmax = 0;
#define ABS16(x) ({ int32_t v_ = (x), s_ = v_ >> 31; (uint32_t)((v_ ^ s_) - s_); })
    for (int t = 0; t < n; t++) {
        const uint32_t *qr = (const uint32_t *)(qkv->v + (size_t)t * qkv->c + head * HD);
        #pragma GCC unroll 4
        for (int d = 0; d < HD / 2; d++) {
            uint32_t wq = qr[d];
            qmax |= ABS16((int16_t)(wq & 0xffffu)) | ABS16((int16_t)(wq >> 16));
        }
    }
#undef ABS16
    int q_sh = 0;
    while ((qmax >> q_sh) > (uint32_t)DAV2_QK_QMAX) q_sh++;

    /* shifted q split into two int8 halves; |q| <= 2047 after the shift,
     * so q >> 4 is within int8 */
    for (int t = 0; t < n; t++) {
        const uint32_t *qr = (const uint32_t *)(qkv->v + (size_t)t * qkv->c + head * HD);
        uint32_t *qh = (uint32_t *)(q_hi + (size_t)t * HD);
        uint32_t *ql = (uint32_t *)(q_lo + (size_t)t * HD);
        #pragma GCC unroll 2
        for (int d = 0; d < HD / 4; d++) {
            uint32_t wq0 = qr[2 * d], wq1 = qr[2 * d + 1];
            int32_t v0 = (int16_t)(wq0 & 0xffffu) >> q_sh, v1 = (int16_t)(wq0 >> 16) >> q_sh;
            int32_t v2 = (int16_t)(wq1 & 0xffffu) >> q_sh, v3 = (int16_t)(wq1 >> 16) >> q_sh;
            qh[d] = ((uint32_t)(v0 >> 4) & 0xffu) | (((uint32_t)(v1 >> 4) & 0xffu) << 8)
                  | (((uint32_t)(v2 >> 4) & 0xffu) << 16) | ((uint32_t)(v3 >> 4) << 24);
            ql[d] = ((uint32_t)v0 & 0xfu) | (((uint32_t)v1 & 0xfu) << 8)
                  | (((uint32_t)v2 & 0xfu) << 16) | (((uint32_t)v3 & 0xfu) << 24);
        }
    }
    b->q_sh = q_sh;
}

/* v^T of a head, split into int8 halves */
static void attn_prep_v(const dav2_tensor_t *qkv, int head, int n, attn_bufs_t *b)
{
    const int HD = DAV2_HEAD_DIM;
    const int ED = DAV2_EMBED_DIM;
    const int Kp = (n + 3) & ~3;
    int8_t *q_hi = b->q_hi, *q_lo = b->q_lo;
    int32_t *s_hi = b->s_hi, *s_lo = b->s_lo; int16_t *p16 = b->p16;
    int8_t *vt_hi = b->vt_hi, *vt_lo = b->vt_lo;
    int32_t *c_hi = b->c_hi, *c_lo = b->c_lo;
    (void)HD; (void)ED; (void)Kp; (void)q_hi; (void)q_lo; (void)s_hi; (void)s_lo;
    (void)p16; (void)vt_hi; (void)vt_lo; (void)c_hi; (void)c_lo;
    /* v^T, split, padded with zero columns up to Kp. Four tokens at a time so
     * that each store is a whole word. */
    for (int m0 = 0; m0 < Kp; m0 += 4) {
        const int16_t *vr[4];
        for (int j = 0; j < 4; j++)
            vr[j] = (m0 + j < n) ? qkv->v + (size_t)(m0 + j) * qkv->c + 2 * ED + head * HD
                                 : (const int16_t *)0;
        #pragma GCC unroll 4
        for (int d = 0; d < HD; d++) {
            uint32_t wh = 0, wl = 0;
            #pragma GCC unroll 4
            for (int j = 0; j < 4; j++) {
                int32_t v = vr[j] ? vr[j][d] : 0;
                wh |= ((uint32_t)(v >> 6) & 0xffu) << (8 * j);
                wl |= ((uint32_t)v & 0x3fu)        << (8 * j);
            }
            *(uint32_t *)(vt_hi + (size_t)d * Kp + m0) = wh;
            *(uint32_t *)(vt_lo + (size_t)d * Kp + m0) = wl;
        }
    }

}

/* 2^(-k/1024) in Q15 for k = 0..1023 (k = 0 gives 32768, clamped below).
 * Built once with an integer recurrence in Q30: T[k] = T[k-1] * r with
 * r = 2^(-1/1024) in Q31 = 2146030505; the error after 1023 steps is below
 * 3e-8, far below a Q15 unit. */
static uint16_t exp2_tab[1024];

static void exp2_tab_init(void)
{
    if (exp2_tab[0])
        return;
    uint32_t t = 1u << 30;
    for (int k = 0; k < 1024; k++) {
        exp2_tab[k] = (uint16_t)((t + (1u << 14)) >> 15);
        t = (uint32_t)(((uint64_t)t * 2146030505u + (1u << 30)) >> 31);
    }
}

/* scores -> Q15 probabilities, each row divided by its sum */
static void attn_softmax(const dav2_tensor_t *qkv, int n, attn_bufs_t *b)
{
    const int HD = DAV2_HEAD_DIM;
    const int ED = DAV2_EMBED_DIM;
    const int Kp = (n + 3) & ~3;
    int8_t *q_hi = b->q_hi, *q_lo = b->q_lo;
    int32_t *s_hi = b->s_hi, *s_lo = b->s_lo; int16_t *p16 = b->p16;
    int8_t *vt_hi = b->vt_hi, *vt_lo = b->vt_lo;
    int32_t *c_hi = b->c_hi, *c_lo = b->c_lo;
    (void)HD; (void)ED; (void)Kp; (void)q_hi; (void)q_lo; (void)s_hi; (void)s_lo;
    (void)p16; (void)vt_hi; (void)vt_lo; (void)c_hi; (void)c_lo;
    /* Softmax per query token, in fixed point: p = 2^(-(smax - s) * scale *
     * log2e) in Q15. kf is typically far below 1 (score_scale is ~1e-5), so
     * it must be carried as a (mult, shift) pair rather than a plain Q16
     * integer -- rounding it into a Q16 constant collapses it to 0 or 1 and
     * flattens the whole distribution. */
    dav2_xf_t score_scale = xf_mul(qkv->scale, qkv->scale);
    score_scale.sh -= b->q_sh;                          /* * 2^q_sh */
    /* -> exponent base 2: * log2(e), the float32 1.4426950f (0x3fb8aa3b) */
    dav2_xf_t kf = xf_mul(score_scale, xf_from_f32_bits(0x3fb8aa3bu));
    int32_t kmult; int kshift;
    xf_to_mult(kf, &kmult, &kshift);

    /* t_q16 = floor(d * kf * 2^16) = floor(d * kmult / 2^ks), ks constant
     * for the head. d < 2^30 and kmult < 2^31: the product is split into
     * mulhu and mul, with the shift hoisted. */
    int ks = kshift - 16;
    if (ks > 31) {
        /* keep one form in the loop: drop the multiplier's low bits instead
         * (it keeps at least 31 - (ks - 31) of them; in this model ks is
         * about 30) */
        kmult = (int32_t)((uint32_t)kmult >> (ks - 31));
        ks = 31;
    } else if (ks < 1) {
        /* kf >= 2^15: every nonzero gap underflows Q15. kmult = 2^31 - 1
         * with ks = 1 gives the same: d >= 1 maps to >= 2^30. */
        kmult = 2147483647;
        ks = 1;
    }
    const uint32_t over = 16u << 16;               /* underflows Q15 */
    const uint16_t *exp2t = b->exp2;
    for (int t = 0; t < n; t++) {
        const int32_t *sh = s_hi + (size_t)t * n, *sl = s_lo + (size_t)t * n;
        int32_t smax = -2147483647 - 1;
        #pragma GCC unroll 4
        for (int m = 0; m < n; m++) {
            int32_t sc = sh[m] * 16 + sl[m];
            if (sc > smax) smax = sc;
        }
        uint32_t sum = 0;
        int16_t *pr = p16 + (size_t)t * Kp;
        #pragma GCC unroll 2
        for (int m = 0; m < n; m++) {
            /* the score again from its two halves: two DDR3 cache hits cost
             * less than a store and a load of on-chip RAM */
            /* >= 0 and < 2^32: unsigned, since |s| < 1.07e9 each */
            uint32_t d = (uint32_t)smax - (uint32_t)(sh[m] * 16 + sl[m]);
            uint32_t hi = (uint32_t)(((uint64_t)d * (uint32_t)kmult) >> 32);
            uint32_t lo = d * (uint32_t)kmult;
            uint32_t t_q16 = (hi >> ks) ? over : (hi << (32 - ks)) | (lo >> ks);
            /* 2^-t = 2^-frac >> int: the fraction's 10 high bits index the
             * table (Q15, error below 11 units), the integer part shifts */
            int32_t p = t_q16 >= over ? 0
                      : (int32_t)exp2t[(t_q16 >> 6) & 1023u] >> (t_q16 >> 16);
            if (p > 32767) p = 32767;        /* the argmax: 2^15 -> int16 */
            pr[m] = (int16_t)p;
            sum += (uint32_t)p;
        }
        /* Normalise the row here, so that the context needs no division:
         * p' = round(p * 2^15 / sum). The row's maximum is 32767, so
         * sum >= 32767 and inv = round(2^31 / sum) <= 65538; p * inv < 2^32.
         * The p' of a row add up to 2^15 within rounding. */
        const uint32_t inv = (0x80000000u + sum / 2u) / sum;   /* 32-bit divu */
        #pragma GCC unroll 4
        for (int m = 0; m < n; m++) {
            uint32_t q = ((uint32_t)pr[m] * inv + 32768u) >> 16;
            pr[m] = (int16_t)(q > 32767u ? 32767u : q);
        }
        for (int m = n; m < Kp; m++) pr[m] = 0;
    }

}

/* context -> ctx columns of this head. Uses dav2_scratch. */
static void attn_normalise(int head, int n, attn_bufs_t *b, dav2_tensor_t *ctx)
{
    const int HD = DAV2_HEAD_DIM;
    const int ED = DAV2_EMBED_DIM;
    const int Kp = (n + 3) & ~3;
    int8_t *q_hi = b->q_hi, *q_lo = b->q_lo;
    int32_t *s_hi = b->s_hi, *s_lo = b->s_lo; int16_t *p16 = b->p16;
    int8_t *vt_hi = b->vt_hi, *vt_lo = b->vt_lo;
    int32_t *c_hi = b->c_hi, *c_lo = b->c_lo;
    (void)HD; (void)ED; (void)Kp; (void)q_hi; (void)q_lo; (void)s_hi; (void)s_lo;
    (void)p16; (void)vt_hi; (void)vt_lo; (void)c_hi; (void)c_lo;
    /* ctx[t][d] = (64 c_hi + c_lo) / 2^15, rounded: the probabilities are
     * already divided by their sum (attn_softmax), so this is 32-bit. The
     * p' of a row add up to 2^15 (within rounding) and |v| <= 8191, so
     * |64 c_hi + c_lo| < 2^28. Assembled in scratch as [t][d], then copied
     * out in rows. */
    int16_t *cbuf = dav2_scratch;
    for (int d = 0; d < HD; d += 2) {             /* two columns: word stores */
        const int32_t *ch0 = c_hi + (size_t)d * n, *cl0 = c_lo + (size_t)d * n;
        const int32_t *ch1 = ch0 + n, *cl1 = cl0 + n;
        for (int t = 0; t < n; t++) {
            int32_t r0 = (ch0[t] * 64 + cl0[t] + 16384) >> 15;
            int32_t r1 = (ch1[t] * 64 + cl1[t] + 16384) >> 15;
            if (r0 >  DAV2_ACT_QMAX) r0 =  DAV2_ACT_QMAX;
            if (r0 < -DAV2_ACT_QMAX) r0 = -DAV2_ACT_QMAX;
            if (r1 >  DAV2_ACT_QMAX) r1 =  DAV2_ACT_QMAX;
            if (r1 < -DAV2_ACT_QMAX) r1 = -DAV2_ACT_QMAX;
            *(uint32_t *)(cbuf + (size_t)t * HD + d) =
                ((uint32_t)r0 & 0xffffu) | ((uint32_t)r1 << 16);
        }
    }
    for (int t = 0; t < n; t++)
        dav2_copy16(ctx->v + (size_t)t * ED + head * HD, cbuf + (size_t)t * HD, (size_t)HD);

}

/* All heads, with the CPU's work overlapped with the accelerator's:
 *
 *   on the block     on the CPU meanwhile
 *   S_hi(h)          v^T of head h split
 *   S_lo(h)          --  (the softmax needs S_lo)
 *   C_hi(h), C_lo(h) q, k of head h+1 prepared (during C_lo)
 *   --               head h normalised
 *
 * The buffers are allocated once for all heads: head h+1's q goes into the
 * same q_hi/q_lo as head h's, which is safe because both S jobs of head
 * h have completed by then (one job at a time, and C_hi has run since).
 * Without an accelerator every GEMM is computed at its start, in the same
 * order, so the result is bit-identical. */
static void attention_all(const dav2_tensor_t *qkv, int n_tokens, dav2_tensor_t *ctx)
{
    const int HD = DAV2_HEAD_DIM;
    const int n  = n_tokens;
    const int Kp = (n + 3) & ~3;                 /* K must be a multiple of 4 */
    const size_t mark = dav2_arena_mark();
    attn_bufs_t b;

    b.q_hi  = (int8_t  *)dav2_arena_alloc((size_t)n * HD);
    b.q_lo  = (int8_t  *)dav2_arena_alloc((size_t)n * HD);
    b.s_hi  = (int32_t *)dav2_arena_alloc((size_t)n * n * sizeof(int32_t));
    b.s_lo  = (int32_t *)dav2_arena_alloc((size_t)n * n * sizeof(int32_t));
    b.p16   = (int16_t *)dav2_arena_alloc((size_t)n * Kp * sizeof(int16_t));
    b.vt_hi = (int8_t  *)dav2_arena_alloc((size_t)HD * Kp);
    b.vt_lo = (int8_t  *)dav2_arena_alloc((size_t)HD * Kp);
    b.c_hi  = (int32_t *)dav2_arena_alloc((size_t)HD * n * sizeof(int32_t));
    b.c_lo  = (int32_t *)dav2_arena_alloc((size_t)HD * n * sizeof(int32_t));
    b.exp2  = (uint16_t *)dav2_arena_alloc(1024 * sizeof(uint16_t));
    if (dav2_arena_failed) { dav2_arena_release(mark); return; }
    /* the table from the on-chip RAM into the DDR3 arena, where the
     * softmax's loads are about 4 cycles cheaper */
    exp2_tab_init();
    dav2_copy16((int16_t *)b.exp2, (const int16_t *)exp2_tab, 1024);

    uint64_t t0 = dav2_cycles(), t1;
#define LAP(d) do { t1 = dav2_cycles(); dav2_sub_add((d), t1 - t0); t0 = t1; } while (0)
    attn_prep_qk(qkv, 0, n, &b);
    LAP(DAV2_SUB_ATT_PREP_QK);
    for (int h = 0; h < DAV2_N_HEADS; h++) {
        /* S = k . q_hi, k . q_lo  ->  s[t][m] (W rows are t, A rows are m) */
        const int16_t *k = qkv->v + DAV2_EMBED_DIM + h * HD;   /* in place */
        gemm_i32_start(k, qkv->c, b.q_hi, HD, b.s_hi, n, HD, n);
        LAP(DAV2_SUB_ATT_WAIT);
        attn_prep_v(qkv, h, n, &b);
        LAP(DAV2_SUB_ATT_PREP_V);
        gemm_i32_start(k, qkv->c, b.q_lo, HD, b.s_lo, n, HD, n);
        gemm_i32_finish();
        LAP(DAV2_SUB_ATT_WAIT);
        attn_softmax(qkv, n, &b);
        LAP(DAV2_SUB_ATT_SOFTMAX);
        /* C = P . v^T_hi, P . v^T_lo  ->  c[d][t] */
        gemm_i32_start(b.p16, Kp, b.vt_hi, Kp, b.c_hi, n, Kp, HD);
        gemm_i32_start(b.p16, Kp, b.vt_lo, Kp, b.c_lo, n, Kp, HD);
        LAP(DAV2_SUB_ATT_WAIT);
        if (h + 1 < DAV2_N_HEADS)
            attn_prep_qk(qkv, h + 1, n, &b);
        LAP(DAV2_SUB_ATT_PREP_QK);
        gemm_i32_finish();
        LAP(DAV2_SUB_ATT_WAIT);
        attn_normalise(h, n, &b, ctx);
        LAP(DAV2_SUB_ATT_NORM);
    }
#undef LAP
    dav2_arena_release(mark);
}

/* ------------------------------------------------------------- encoder */

typedef struct {
    dav2_qw_t qkv, proj, fc1, fc2;
    const int32_t *n1w, *n1b, *n2w, *n2b;   /* LayerNorm gamma Q15, beta Q16 */
} block_w_t;

static void load_block(block_w_t *bw, int i)
{
    char base[64], nm[64];
    sprintf(base, "blk%d.", i);

    sprintf(nm, "%sqkv", base);  dav2_qw(&bw->qkv, nm, DAV2_EMBED_DIM);
    sprintf(nm, "%sproj", base); dav2_qw(&bw->proj, nm, DAV2_EMBED_DIM);
    sprintf(nm, "%sfc1", base);  dav2_qw(&bw->fc1, nm, DAV2_EMBED_DIM);
    sprintf(nm, "%sfc2", base);  dav2_qw(&bw->fc2, nm, 4 * DAV2_EMBED_DIM);

    sprintf(nm, "%snorm1.w", base); bw->n1w = (const int32_t *)dav2_find(nm, 0);
    sprintf(nm, "%snorm1.b", base); bw->n1b = (const int32_t *)dav2_find(nm, 0);
    sprintf(nm, "%snorm2.w", base); bw->n2w = (const int32_t *)dav2_find(nm, 0);
    sprintf(nm, "%snorm2.b", base); bw->n2b = (const int32_t *)dav2_find(nm, 0);
}

/* One transformer block, in place on x. */
static void run_block(dav2_tensor_t *x, int i, int n_tokens)
{
#ifdef DAV2_TRACE
#define BDUMP(nm, t) do { if (i == 0) dav2_dump("/tmp/dav2_b0_" nm ".bin", (t)); } while (0)
#else
#define BDUMP(nm, t) ((void)0)
#endif
    const int ED = DAV2_EMBED_DIM;
    block_w_t bw;
    load_block(&bw, i);

    size_t mark = dav2_arena_mark();

    dav2_tensor_t n = dav2_tensor_new(n_tokens, ED);
    dav2_layernorm(x, bw.n1w, bw.n1b, &n);

    BDUMP("ln1", &n);
    dav2_tensor_t qkv = dav2_tensor_new(n_tokens, 3 * ED);
    dav2_qgemm(&n, &bw.qkv, &qkv);
    BDUMP("qkv", &qkv);

    dav2_tensor_t ctx = dav2_tensor_new(n_tokens, ED);
    ctx.scale = qkv.scale;              /* context inherits the v scale */
    {
        uint64_t t0 = dav2_cycles();
        attention_all(&qkv, n_tokens, &ctx);
        dav2_prof_add(DAV2_PROF_ATTENTION, dav2_cycles() - t0);
    }

    BDUMP("ctx", &ctx);
    /* x = x + proj(ctx): the residual add is the requantisation job's
     * epilogue on the accelerator, and x is updated in place */
    dav2_qgemm_ex(&ctx, &bw.proj, x, 0, x);
    BDUMP("res1", x);
    dav2_arena_release(mark);

    mark = dav2_arena_mark();
    dav2_tensor_t n2 = dav2_tensor_new(n_tokens, ED);
    dav2_layernorm(x, bw.n2w, bw.n2b, &n2);

    BDUMP("ln2", &n2);
    dav2_tensor_t h1 = dav2_tensor_new(n_tokens, 4 * ED);
    dav2_qgemm(&n2, &bw.fc1, &h1);
    BDUMP("fc1", &h1);
    dav2_gelu(&h1);
    BDUMP("gelu", &h1);

    /* x = x + fc2(h1), likewise fused and in place */
    dav2_qgemm_ex(&h1, &bw.fc2, x, 0, x);
    BDUMP("res2", x);
    dav2_arena_release(mark);
}

/* --------------------------------------------------------------- DPT head */

/* ResidualConvUnit: relu -> conv3x3 -> relu -> conv3x3 -> +input */
static dav2_tensor_t res_conv_unit(const dav2_tensor_t *x, int h, int w,
                                   const char *prefix, int unit)
{
    char nm[64];
    dav2_qw_t c1, c2;
    sprintf(nm, "%su%dc1", prefix, unit); dav2_qw(&c1, nm, 9 * DAV2_FEATURES);
    sprintf(nm, "%su%dc2", prefix, unit); dav2_qw(&c2, nm, 9 * DAV2_FEATURES);

    dav2_tensor_t out = dav2_tensor_new(h * w, DAV2_FEATURES);
    size_t mark = dav2_arena_mark();

    dav2_tensor_t t = dav2_tensor_new(h * w, x->c);
    dav2_copy_relu(&t, x);

    /* relu(conv1) and x + conv2 each fused into the conv's requantisation */
    dav2_tensor_t a = dav2_conv2d_ex(&t, h, w, &c1, 3, 1, 1, 0, 1, 0, 0);
    dav2_tensor_t b = dav2_conv2d_ex(&a, h, w, &c2, 3, 1, 1, x, 0, 0, 0);
    dav2_copy16(out.v, b.v, (size_t)h * w * DAV2_FEATURES);
    out.scale = b.scale;
    out.amax_q = b.amax_q;

    dav2_arena_release(mark);
    return out;
}

/* FeatureFusionBlock */
static dav2_tensor_t fusion(int idx, const dav2_tensor_t *a,
                            const dav2_tensor_t *b, int h, int w,
                            int oh, int ow)
{
    char prefix[64], nm[64];
    sprintf(prefix, "rf%d.", idx);

    dav2_tensor_t out = dav2_tensor_new(oh * ow, DAV2_FEATURES);
    size_t mark = dav2_arena_mark();

    dav2_tensor_t cur = dav2_tensor_new(h * w, DAV2_FEATURES);
    dav2_copy16(cur.v, a->v, (size_t)h * w * DAV2_FEATURES);
    cur.scale = a->scale;

    if (b) {
        dav2_tensor_t res = res_conv_unit(b, h, w, prefix, 1);
        dav2_tensor_t s = dav2_tensor_new(h * w, DAV2_FEATURES);
        dav2_add(&cur, &res, &s);
        cur = s;
    }

    dav2_tensor_t u2 = res_conv_unit(&cur, h, w, prefix, 2);
    dav2_tensor_t up = dav2_tensor_new(oh * ow, DAV2_FEATURES);
    dav2_interpolate(&u2, h, w, oh, ow, &up);

    dav2_qw_t oc;
    sprintf(nm, "%sout", prefix); dav2_qw(&oc, nm, DAV2_FEATURES);
    dav2_tensor_t o = dav2_conv2d(&up, oh, ow, &oc, 1, 1, 0, 0, 0);
    dav2_copy16(out.v, o.v, (size_t)oh * ow * DAV2_FEATURES);
    out.scale = o.scale;

    dav2_arena_release(mark);
    return out;
}

/* -------------------------------------------------------------- profile */

/* One line per bucket, as a share of the whole frame. "other" is whatever
 * ran between the instrumented operators: patch extraction, memcpy, the
 * final float conversion, the progress prints themselves. */
void dav2_prof_report(uint64_t frame_cycles)
{
    uint64_t sum = 0;
    for (int b = 0; b < DAV2_PROF_OTHER; b++)
        sum += dav2_prof_get(b);
    dav2_prof_add(DAV2_PROF_OTHER, frame_cycles > sum ? frame_cycles - sum : 0);

    printf("profile: %u kcycles per frame\n", (unsigned)(frame_cycles / 1000u));
    for (int b = 0; b < DAV2_PROF_N; b++) {
        uint64_t c = dav2_prof_get(b);
        unsigned permille = frame_cycles ? (unsigned)(c * 1000u / frame_cycles) : 0;
        /* libsys's printf knows neither '-' nor '.*', so pad by hand */
        char name[24];
        int len = (int)strlen(dav2_prof_name[b]);
        memcpy(name, dav2_prof_name[b], (size_t)len);
        while (len < 20) name[len++] = ' ';
        name[len] = 0;
        printf("  %s %9u kcycles  %3u.%u %%\n", name,
               (unsigned)(c / 1000u), permille / 10u, permille % 10u);
    }
    printf("detail:\n");
    for (int d = 0; d < DAV2_SUB_N; d++) {
        char name[24];
        int len = (int)strlen(dav2_sub_name[d]);
        memcpy(name, dav2_sub_name[d], (size_t)len);
        while (len < 20) name[len++] = ' ';
        name[len] = 0;
        printf("  %s %9u kcycles\n", name, (unsigned)(dav2_sub_get(d) / 1000u));
    }
}

/* ------------------------------------------------------------------ main */

void dav2_infer(const dav2_cfg_t *cfg, int16_t *depth_q, dav2_xf_t *depth_scale)
{
    const int ED = DAV2_EMBED_DIM;
    const int grid = cfg->grid;
    const int n_tokens = cfg->n_tokens;
    const int n_patch = grid * grid;

    if (dav2_blob_check())
        return;

    dav2_prof_reset();
    const uint64_t t_frame = dav2_cycles();

    /* ---- patch embedding ---------------------------------------------- */
    dav2_progress("patch embedding");

    if (!g_image) {
        printf("dav2: no input image set (call dav2_set_image first)\n");
        return;
    }

    dav2_tensor_t image;
    image.v = (int16_t *)g_image;
    image.n = cfg->size * cfg->size;
    image.c = 3;
    image.scale = g_image_scale;
    image.amax_q = -1;

    dav2_qw_t pe_w;
    dav2_qw(&pe_w, "patch_embed", DAV2_PATCH * DAV2_PATCH * 3);

    dav2_tensor_t x = dav2_tensor_new(n_tokens, ED);
    {
        dav2_tensor_t patches = dav2_conv2d(&image, cfg->size, cfg->size,
                                            &pe_w, DAV2_PATCH, DAV2_PATCH, 0,
                                            0, 0);
        /* prepend the class token and add the (pre-interpolated) position
         * embedding, both Q24 integers in the blob, then requantise once */
        const int32_t *cls = (const int32_t *)dav2_find("cls_token", 0);
        const int32_t *pos = (const int32_t *)dav2_find("pos_embed", 0);
        dav2_embed_tokens(&patches, cls, pos, &x);
    }

#ifdef DAV2_TRACE
    dav2_dump("/tmp/dav2_tokens.bin", &x);
#endif

    /* ---- transformer blocks, capturing the four DPT taps --------------- */
    dav2_tensor_t feats[4];
    for (int i = 0; i < 4; i++)
        feats[i] = dav2_tensor_new(n_patch, ED);

    for (int i = 0; i < DAV2_N_BLOCKS; i++) {
        char msg[32];
        sprintf(msg, "block %d/12", i + 1);
        dav2_progress(msg);
        run_block(&x, i, n_tokens);
#ifdef DAV2_TRACE
        { char fn[64]; sprintf(fn, "/tmp/dav2_blk%d.bin", i); dav2_dump(fn, &x); }
#endif

        for (int j = 0; j < 4; j++) {
            if (INTERMEDIATE[j] != i) continue;
            /* final LayerNorm (gamma, beta folded into proj0..3), then
             * drop the class token */
            size_t mark = dav2_arena_mark();
            dav2_tensor_t nrm = dav2_tensor_new(n_tokens, ED);
            dav2_layernorm(&x, 0, 0, &nrm);
            dav2_copy16(feats[j].v, nrm.v + ED, (size_t)n_patch * ED);
            feats[j].scale = nrm.scale;
#ifdef DAV2_TRACE
            { char fn[64]; sprintf(fn, "/tmp/dav2_feat%d.bin", j);
              dav2_dump(fn, &feats[j]); }
#endif
            dav2_arena_release(mark);
        }
    }

    /* ---- DPT head ------------------------------------------------------ */
    dav2_progress("dpt head: projections");

    dav2_tensor_t rn[4];
    int rh[4], rw[4];
    for (int i = 0; i < 4; i++) {
        char nm[64];
        dav2_qw_t pw;
        sprintf(nm, "proj%d", i);
        dav2_qw(&pw, nm, ED);

        int h = grid, w = grid;
        dav2_tensor_t p = dav2_conv2d(&feats[i], grid, grid, &pw, 1, 1, 0, &h, &w);

        dav2_tensor_t resized;
        if (i == 0) {
            dav2_qw_t rz; dav2_qw(&rz, "resize0", pw.m);
            resized = dav2_conv_transpose(&p, h, w, &rz, 4, &h, &w);
        } else if (i == 1) {
            dav2_qw_t rz; dav2_qw(&rz, "resize1", pw.m);
            resized = dav2_conv_transpose(&p, h, w, &rz, 2, &h, &w);
        } else if (i == 3) {
            dav2_qw_t rz; dav2_qw(&rz, "resize3", 9 * pw.m);
            resized = dav2_conv2d(&p, h, w, &rz, 3, 2, 1, &h, &w);
        } else {
            resized = p;
        }

        char rnm[64];
        dav2_qw_t rw_w;
        sprintf(rnm, "rn%d", i + 1);
        dav2_qw(&rw_w, rnm, 9 * resized.c);
        rn[i] = dav2_conv2d(&resized, h, w, &rw_w, 3, 1, 1, &rh[i], &rw[i]);
    }

#ifdef DAV2_TRACE
    for (int i = 0; i < 4; i++) {
        char fn[64]; sprintf(fn, "/tmp/dav2_rn%d.bin", i + 1);
        dav2_dump(fn, &rn[i]);
    }
#endif
    dav2_progress("dpt head: fusion");
    dav2_tensor_t path4 = fusion(4, &rn[3], 0, rh[3], rw[3], rh[2], rw[2]);
    dav2_tensor_t path3 = fusion(3, &path4, &rn[2], rh[2], rw[2], rh[1], rw[1]);
    dav2_tensor_t path2 = fusion(2, &path3, &rn[1], rh[1], rw[1], rh[0], rw[0]);
    dav2_tensor_t path1 = fusion(1, &path2, &rn[0], rh[0], rw[0],
                                 rh[0] * 2, rw[0] * 2);

#ifdef DAV2_TRACE
    dav2_dump("/tmp/dav2_path4.bin", &path4);
    dav2_dump("/tmp/dav2_path3.bin", &path3);
    dav2_dump("/tmp/dav2_path2.bin", &path2);
    dav2_dump("/tmp/dav2_path1.bin", &path1);
#endif
    dav2_progress("dpt head: output convolutions");
    int h1 = rh[0] * 2, w1 = rw[0] * 2;
    dav2_qw_t o1, o2a, o2b;
    dav2_qw(&o1, "out1", 9 * DAV2_FEATURES);
    dav2_qw(&o2a, "out2a", 9 * 32);
    dav2_qw(&o2b, "out2b", 32);

    int oh, ow;
    dav2_tensor_t c1 = dav2_conv2d(&path1, h1, w1, &o1, 3, 1, 1, &oh, &ow);

    const int out_size = grid * DAV2_PATCH;
    dav2_tensor_t up = dav2_tensor_new(out_size * out_size, c1.c);
    dav2_interpolate(&c1, oh, ow, out_size, out_size, &up);

    dav2_tensor_t c2 = dav2_conv2d_ex(&up, out_size, out_size, &o2a, 3, 1, 1, 0, 1, &oh, &ow);
    dav2_tensor_t c3 = dav2_conv2d_ex(&c2, out_size, out_size, &o2b, 1, 1, 0, 0, 1, &oh, &ow);

    /* int16 depth and its scale; converting to float is the caller's
     * choice (the board leaves it to the PC) */
    dav2_copy16(depth_q, c3.v, (size_t)out_size * out_size);
    *depth_scale = c3.scale;

    dav2_progress("done");
    dav2_prof_report(dav2_cycles() - t_frame);
}
