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
    /* the int16-weight path (CTRL.w16): whole q and v^T, one score matrix
     * with each query's {max, min}, one context matrix */
    int16_t *q16, *vt16;
    int32_t *s, *sst, *c;
    int32_t *cpar;              /* {2^30, 45, 0} per d: (c + 2^14) >> 15 */
    /* the exponential on the accelerator: P^T from the requantisation job
     * ([key][query], rows Kp apart), its parameters, each query's sum */
    int16_t *pt;
    int32_t *epar;
    uint32_t *psum;
    int      q_direct;          /* the score GEMM reads q from qkv (CTRL.wsh) */
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

static int g16_pending;           /* an int16-weight GEMM left running */

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

/* 2^-t in Q15 for t = ((d >> j) * K) >> 11 in Q16, as below; u = t >> 6
 * carries the table index (low 10 bits) and the shift (the rest). 0 from
 * t >= 16 (u >= 16 << 10) and from d >= cutoff. At most 32767. */
static inline uint32_t softmax_p(uint32_t d, uint32_t cutoff, int j, uint32_t K,
                                 const uint16_t *e)
{
    if (d >= cutoff)
        return 0;
    uint32_t u = ((d >> j) * K) >> 17;
    if (u >= (16u << 10))
        return 0;
    uint32_t p = (uint32_t)e[u & 1023u] >> (u >> 10);
    return p > 32767u ? 32767u : p;
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
    /* t = d * kf in Q16 with one 32-bit multiply per element. At t >= 16
     * the probability underflows Q15, so only d < cutoff = ceil(16 / kf)
     * matters. Below the cutoff, d >> j has at most 16 bits (j from the
     * cutoff's length), and K = kf 2^(16 + j + 11) has 16 bits, so
     *
     *   t = ((d >> j) * K) >> 11      (d >> j) * K < 2^32.
     *
     * Dropping d's low j bits changes t by less than 2^-11, below the exp2
     * table's resolution (2^-10). Checked on 3000 random score rows with
     * this model's range of kf: the largest error of a probability against
     * float softmax is 24 Q15 units, the same as with the exact t (64-bit
     * product), and the mean error is 3.7e-5 against 3.3e-5. */
    uint32_t cutoff = 0xffffffffu;
    int j = 0;
    uint32_t K = 0;
    if (kf.m > 0) {
        int64_t c = xf_round(xf_div(xf_from_int(16), kf), 0) + 1;
        if (c < 0xffffffffLL)
            cutoff = (uint32_t)c;
        while ((cutoff >> j) > 0xffffu) j++;
        dav2_xf_t kk = kf;
        kk.sh -= 16 + j + 11;
        int64_t kr = xf_round(kk, 0);
        K = kr > 0xffff ? 0xffffu : (uint32_t)kr;
    }
    const uint32_t over = 16u << 16;               /* underflows Q15 */
    const uint16_t *exp2t = b->exp2;
    for (int t = 0; t < n; t++) {
        uint32_t sum = 0;
        int16_t *pr = p16 + (size_t)t * Kp;
        if (b->s) {
            /* one score matrix; the row maximum from the GEMM's statistics
             * (or found here without them) */
            const int32_t *sr = b->s + (size_t)t * n;
            int32_t smax;
            if (b->sst) {
                smax = b->sst[2 * t];
            } else {
                smax = -2147483647 - 1;
                for (int m = 0; m < n; m++)
                    if (sr[m] > smax) smax = sr[m];
            }
            /* two scores per step, one word store (pr is word aligned:
             * Kp is a multiple of 4) */
            uint32_t *pw = (uint32_t *)pr;
            int m = 0;
            for (; m + 1 < n; m += 2) {
                uint32_t p0 = softmax_p((uint32_t)smax - (uint32_t)sr[m], cutoff, j, K, exp2t);
                uint32_t p1 = softmax_p((uint32_t)smax - (uint32_t)sr[m + 1], cutoff, j, K, exp2t);
                pw[m >> 1] = p0 | (p1 << 16);
                sum += p0 + p1;
            }
            if (m < n) {
                uint32_t p0 = softmax_p((uint32_t)smax - (uint32_t)sr[m], cutoff, j, K, exp2t);
                pr[m] = (int16_t)p0;
                sum += p0;
            }
        } else {
        const int32_t *sh = s_hi + (size_t)t * n, *sl = s_lo + (size_t)t * n;
        int32_t smax = -2147483647 - 1;
        #pragma GCC unroll 4
        for (int m = 0; m < n; m++) {
            int32_t sc = sh[m] * 16 + sl[m];
            if (sc > smax) smax = sc;
        }
        #pragma GCC unroll 2
        for (int m = 0; m < n; m++) {
            /* the score again from its two halves: two DDR3 cache hits cost
             * less than a store and a load of on-chip RAM */
            /* >= 0 and < 2^32: unsigned, since |s| < 1.07e9 each */
            uint32_t d = (uint32_t)smax - (uint32_t)(sh[m] * 16 + sl[m]);
            uint32_t t_q16 = d < cutoff ? ((d >> j) * K) >> 11 : over;
            /* 2^-t = 2^-frac >> int: the fraction's 10 high bits index the
             * table (Q15, error below 11 units), the integer part shifts */
            int32_t p = t_q16 >= over ? 0
                      : (int32_t)exp2t[(t_q16 >> 6) & 1023u] >> (t_q16 >> 16);
            if (p > 32767) p = 32767;        /* the argmax: 2^15 -> int16 */
            pr[m] = (int16_t)p;
            sum += (uint32_t)p;
        }
        }
        /* Normalise the row here, so that the context needs no division:
         * p' = round(p * 2^15 / sum). The row's maximum is 32767, so
         * sum >= 32767 and inv = round(2^31 / sum) <= 65538; p * inv < 2^32.
         * The p' of a row add up to 2^15 within rounding. */
        const uint32_t inv = (0x80000000u + sum / 2u) / sum;   /* 32-bit divu */
        /* q <= 32768 (p <= 32767, inv <= 65538), so min(q, 32767) is
         * q - (q >> 15). Two per word; the padding up to Kp stays zero. */
        for (int m = n; m < Kp; m++) pr[m] = 0;
        uint32_t *pw = (uint32_t *)pr;
        #pragma GCC unroll 2
        for (int i = 0; i < Kp / 2; i++) {
            uint32_t w = pw[i];
            uint32_t q0 = ((w & 0xffffu) * inv + 32768u) >> 16;
            uint32_t q1 = ((w >> 16) * inv + 32768u) >> 16;
            q0 -= q0 >> 15;
            q1 -= q1 >> 15;
            pw[i] = q0 | (q1 << 16);
        }
    }

}

/* ---- the softmax's exponential on the accelerator
 *
 * The score job's requantisation computes v = r(s) - r(smax) per query,
 * r(x) = round(x kf 512 / 2^0) with the job's own rounding (multiplier and
 * shift of kf 512, the query's bias -r(smax)), so v = 0 at the query's
 * largest score and v <= 0 elsewhere; the block's lookup table maps v to
 * 2^(v/512) in Q15. The job reads S[query][key] and writes P^T[key][query].
 * The CPU then adds each query's column, and writes P[query][key] divided
 * by that sum, as attn_softmax does. The table is the CPU's formula,
 * exp2_tab[k & 1023] >> (k >> 10), at k = -2v: the exponent on a grid of
 * 2^-9 (rounded) instead of 2^-10 (truncated). */
static int16_t *g_exp_lut;        /* 16384 entries, E[v + 8192]; per frame */

static void exp_lut_build(int16_t *E)
{
    exp2_tab_init();
    for (int i = 0; i < 16384; i++) {
        const int v = i - 8192;
        int32_t p = 32767;
        if (v <= 0) {
            const uint32_t k = (uint32_t)(-2 * v);
            p = k >= (16u << 10) ? 0 : (int32_t)(exp2_tab[k & 1023u] >> (k >> 10));
            if (p > 32767) p = 32767;
        }
        E[i] = (int16_t)p;
    }
}

/* round(v mult / 2^sh), half up: the requantisation job's rounding */
static inline int32_t job_round(int32_t v, int32_t mult, int sh)
{
    return (int32_t)(((int64_t)v * mult + ((int64_t)1 << (sh - 1))) >> sh);
}

/* Start the exponential job for the head whose scores are b->s, b->sst:
 * 1 when started (then attn_softmax_accel_finish), 0 to use attn_softmax */
static int attn_softmax_accel_start(const dav2_tensor_t *qkv, int n, attn_bufs_t *b,
                                    int lut_load)
{
    const int Kp = (n + 3) & ~3;
    if (!g_exp_lut || (n & 1))
        return 0;
    dav2_xf_t kf = xf_mul(qkv->scale, qkv->scale);
    kf.sh -= b->q_sh;
    kf = xf_mul(kf, xf_from_f32_bits(0x3fb8aa3bu));          /* * log2(e) */
    kf.sh -= 9;                                               /* * 512 */
    if (kf.m <= 0 || kf.sh < 1 || kf.sh > 62)
        return 0;
    int32_t mult; int sh;
    xf_to_mult(kf, &mult, &sh);
    for (int q = 0; q < n; q++) {
        b->epar[3 * q]     = mult;
        b->epar[3 * q + 1] = sh;
        b->epar[3 * q + 2] = -job_round(b->sst[2 * q], mult, sh);
    }
    return dav2_accel_requant_lut_async(b->s, n, n, b->epar, b->pt, Kp, g_exp_lut, lut_load) != 0;
}

/* P (b->p16) from the exponential job's P^T (b->pt), which the caller has
 * collected (dav2_accel_finish), in two parts: each query's sum
 * (attn_softmax_accel_sums), then P (attn_softmax_accel_norm); the next
 * head's exponential job can run in between. */
static void attn_softmax_accel_sums(int n, attn_bufs_t *b)
{
    const int Kp = (n + 3) & ~3;

    /* Each query's sum: the columns of P^T, eight queries (four words) at
     * a time in registers. Per word w = p(q) + 2^16 p(q+1): T adds w and H
     * adds w >> 16, then sum(q+1) = H and sum(q) = T - 2^16 H, exact since
     * sum(q) < 2^32. */
    const int Kw = Kp / 2;                       /* P^T row in words */
    uint32_t *sum = b->psum;
    const uint32_t *ptw = (const uint32_t *)b->pt;
    for (int w0 = 0; w0 < n / 2; w0 += 4) {
        const int nw = n / 2 - w0 < 4 ? n / 2 - w0 : 4;
        uint32_t T0 = 0, T1 = 0, T2 = 0, T3 = 0, H0 = 0, H1 = 0, H2 = 0, H3 = 0;
        const uint32_t *r = ptw + w0;
        if (nw == 4) {
            for (int key = 0; key < n; key++, r += Kw) {
                const uint32_t a = r[0], c = r[1], d = r[2], e = r[3];
                T0 += a; H0 += a >> 16; T1 += c; H1 += c >> 16;
                T2 += d; H2 += d >> 16; T3 += e; H3 += e >> 16;
            }
        } else {
            for (int key = 0; key < n; key++, r += Kw) {
                T0 += r[0]; H0 += r[0] >> 16;
                if (nw > 1) { T1 += r[1]; H1 += r[1] >> 16; }
                if (nw > 2) { T2 += r[2]; H2 += r[2] >> 16; }
            }
        }
        const uint32_t T[4] = { T0, T1, T2, T3 }, H[4] = { H0, H1, H2, H3 };
        for (int i = 0; i < nw; i++) {
            sum[2 * (w0 + i)]     = T[i] - (H[i] << 16);
            sum[2 * (w0 + i) + 1] = H[i];
        }
    }
}

/* queries q_lo .. q_hi - 1 (both even) */
static void attn_softmax_accel_norm(int n, attn_bufs_t *b, int q_lo, int q_hi)
{
    const int Kp = (n + 3) & ~3;
    const int Kw = Kp / 2;                       /* P^T row in words */
    const uint32_t *sum = b->psum;
    const uint32_t *ptw = (const uint32_t *)b->pt;
    /* P[q][key] = round(p * 2^15 / sum), two queries and two keys per step
     * (two words of P^T in, one word to each of two rows of P out). With
     * inv = (2^31 - 2^16) / sum, p inv + 2^15 < 2^31 since p <= sum, so
     * q <= 32767 without a clamp; attn_softmax rounds inv to nearest, which
     * differs by at most one unit of q. */
    for (int q = q_lo; q < q_hi; q += 2) {
        const uint32_t s0 = sum[q] ? sum[q] : 1u, s1 = sum[q + 1] ? sum[q + 1] : 1u;
        const uint32_t i0 = (0x80000000u - 0x10000u) / s0, i1 = (0x80000000u - 0x10000u) / s1;
        const uint32_t *col = ptw + q / 2;
        uint32_t *r0 = (uint32_t *)(b->p16 + (size_t)q * Kp);
        uint32_t *r1 = (uint32_t *)(b->p16 + (size_t)(q + 1) * Kp);
        for (int key = 0; key < n; key += 2, col += 2 * Kw) {
            const uint32_t wa = col[0], wb = col[Kw];            /* keys key, key+1 */
            const uint32_t a0 = ((wa & 0xffffu) * i0 + 32768u) >> 16;
            const uint32_t b0 = ((wb & 0xffffu) * i0 + 32768u) >> 16;
            const uint32_t a1 = ((wa >> 16) * i1 + 32768u) >> 16;
            const uint32_t b1 = ((wb >> 16) * i1 + 32768u) >> 16;
            *r0++ = a0 | (b0 << 16);
            *r1++ = a1 | (b1 << 16);
        }
        for (int m = n; m < Kp; m++) {
            b->p16[(size_t)q * Kp + m] = 0;
            b->p16[(size_t)(q + 1) * Kp + m] = 0;
        }
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

/* ---- the int16-weight path (CTRL.w16) */

/* q of a head, shifted to 11 bits, as int16 rows [t][HD]: the score GEMM's
 * weights. No int8 split: the block multiplies by int16 weights. */
/* qkv's column extremes from its GEMM (dav2_qgemm_colext), or NULL */
static const int16_t *g_qkv_cmax, *g_qkv_cmin;

static void attn_prep_q16(const dav2_tensor_t *qkv, int head, int n, attn_bufs_t *b)
{
    const int HD = DAV2_HEAD_DIM;
    uint32_t qmax = 0;
#define ABS16(x) ({ int32_t v_ = (x), s_ = v_ >> 31; (uint32_t)((v_ ^ s_) - s_); })
    if (g_qkv_cmax) {
        /* the shift below depends only on the bit length of qmax, and the
         * largest |q| has the same bit length as the OR of all |q| */
        for (int d = head * HD; d < (head + 1) * HD; d++)
            qmax |= ABS16(g_qkv_cmax[d]) | ABS16(g_qkv_cmin[d]);
    } else
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
    b->q_sh = q_sh;
    if (b->q_direct)
        return;                  /* the block shifts q as it reads it */
    for (int t = 0; t < n; t++) {
        const uint32_t *qr = (const uint32_t *)(qkv->v + (size_t)t * qkv->c + head * HD);
        uint32_t *qd = (uint32_t *)(b->q16 + (size_t)t * HD);
        #pragma GCC unroll 4
        for (int d = 0; d < HD / 2; d++) {
            uint32_t w = qr[d];
            qd[d] = ((uint32_t)((int16_t)(w & 0xffffu) >> q_sh) & 0xffffu)
                  | ((uint32_t)((int16_t)(w >> 16) >> q_sh) << 16);
        }
    }
    b->q_sh = q_sh;
}

/* v^T of a head as int16 rows [d][Kp], zero-padded: the context GEMM's
 * weights. Two tokens at a time, so each store is a whole word. */
static void attn_prep_vt16(const dav2_tensor_t *qkv, int head, int n, attn_bufs_t *b)
{
    const int HD = DAV2_HEAD_DIM, ED = DAV2_EMBED_DIM;
    const int Kp = (n + 3) & ~3;
    for (int m0 = 0; m0 < Kp; m0 += 2) {
        const int16_t *v0 = m0 < n ? qkv->v + (size_t)m0 * qkv->c + 2 * ED + head * HD : 0;
        const int16_t *v1 = m0 + 1 < n ? qkv->v + (size_t)(m0 + 1) * qkv->c + 2 * ED + head * HD : 0;
        #pragma GCC unroll 4
        for (int d = 0; d < HD; d++) {
            int32_t a = v0 ? v0[d] : 0, c = v1 ? v1[d] : 0;
            *(uint32_t *)(b->vt16 + (size_t)d * Kp + m0) = ((uint32_t)a & 0xffffu) | ((uint32_t)c << 16);
        }
    }
}

/* acc[m][n] = sum_k a[n][k] w[m][k] with int16 weights, strides in
 * elements, and optionally each row's {max, min}: on the block (left
 * running, see gemm_i32_start) or computed here. */
/* wsh: the weights shifted right by wsh first (CTRL.wsh on the block) */
static void gemm16_start(const int16_t *a, int a_stride, const int16_t *w, int w_stride,
                         int32_t *acc, int N, int K, int M, int32_t *st, int wsh)
{
    gemm_i32_finish();
    if (wsh ? dav2_accel_gemm16_shift_async(a, (uint32_t)a_stride * 2u, w,
                                            (uint32_t)w_stride * 2u, wsh, acc, N, K, M, st)
            : dav2_accel_gemm16_async(a, (uint32_t)a_stride * 2u, w, (uint32_t)w_stride * 2u,
                                      acc, N, K, M, st)) {
        g16_pending = 1;                  /* collected by gemm16_finish */
        return;
    }
    for (int m = 0; m < M; m++) {
        const int16_t *wr = w + (size_t)m * w_stride;
        int32_t mx = -2147483647 - 1, mn = 2147483647;
        for (int n = 0; n < N; n++) {
            const int16_t *ar = a + (size_t)n * a_stride;
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)ar[k] * ((int32_t)wr[k] >> wsh);
            acc[(size_t)m * N + n] = s;
            if (s > mx) mx = s;
            if (s < mn) mn = s;
        }
        if (st) { st[2 * m] = mx; st[2 * m + 1] = mn; }
    }
}

static void gemm16_finish(const int16_t *a, int a_stride, const int16_t *w, int w_stride,
                          int32_t *acc, int N, int K, int M, int32_t *st, int wsh)
{
    if (!g16_pending)
        return;
    g16_pending = 0;
    if (!dav2_accel_finish()) {
        /* the block failed and has disabled itself: redo on the CPU */
        gemm16_start(a, a_stride, w, w_stride, acc, N, K, M, st, wsh);
    }
}

/* ctx[t][head*HD + d] = round(c[d][t] / 2^15), saturated: the requantisation
 * job writes the head's slice of ctx directly (rows ED apart), or the CPU. */
static void attn_context16_cpu(int head, int n, attn_bufs_t *b, dav2_tensor_t *ctx)
{
    const int HD = DAV2_HEAD_DIM, ED = DAV2_EMBED_DIM;
    for (int t = 0; t < n; t++) {
        int16_t *o = ctx->v + (size_t)t * ED + head * HD;
        for (int d = 0; d < HD; d++) {
            int32_t r = (b->c[(size_t)d * n + t] + 16384) >> 15;
            if (r >  DAV2_ACT_QMAX) r =  DAV2_ACT_QMAX;
            if (r < -DAV2_ACT_QMAX) r = -DAV2_ACT_QMAX;
            o[d] = (int16_t)r;
        }
    }
}

/* 1 when the requantisation job was started (collect it with
 * dav2_accel_finish; on failure attn_context16_cpu, b->c is intact), 0
 * when it was declined and the CPU has done it */
static int attn_context16_start(int head, int n, attn_bufs_t *b, dav2_tensor_t *ctx)
{
    const int HD = DAV2_HEAD_DIM, ED = DAV2_EMBED_DIM;
    if (dav2_accel_requant_stride_async(b->c, n, HD, b->cpar, ctx->v + head * HD, ED))
        return 1;
    attn_context16_cpu(head, n, b, ctx);
    return 0;
}

/* The heads with int16 weights: two GEMMs per head instead of four, no
 * int8 split of q and v, the query maxima from the score GEMM's statistics
 * and the context written by a requantisation job. The same numbers as the
 * split path: k . q = 16 (k . q_hi) + k . q_lo, and (c + 2^14) >> 15 is the
 * job's round(c * 2^30 / 2^45). */
static void attention_all16(const dav2_tensor_t *qkv, int n, dav2_tensor_t *ctx)
{
    const int HD = DAV2_HEAD_DIM, ED = DAV2_EMBED_DIM;
    const int Kp = (n + 3) & ~3;
    const size_t mark = dav2_arena_mark();
    attn_bufs_t b;
    memset(&b, 0, sizeof b);
    b.q16  = (int16_t *)dav2_arena_alloc((size_t)n * HD * sizeof(int16_t));
    b.vt16 = (int16_t *)dav2_arena_alloc((size_t)HD * Kp * sizeof(int16_t));
    b.s    = (int32_t *)dav2_arena_alloc((size_t)n * n * sizeof(int32_t));
    b.sst  = (int32_t *)dav2_arena_alloc((size_t)n * 2 * sizeof(int32_t));
    b.p16  = (int16_t *)dav2_arena_alloc((size_t)n * Kp * sizeof(int16_t));
    b.c    = (int32_t *)dav2_arena_alloc((size_t)HD * n * sizeof(int32_t));
    b.cpar = (int32_t *)dav2_arena_alloc((size_t)HD * 3 * sizeof(int32_t));
    b.exp2 = (uint16_t *)dav2_arena_alloc(1024 * sizeof(uint16_t));
    b.pt   = (int16_t *)dav2_arena_alloc((size_t)n * Kp * sizeof(int16_t));
    b.epar = (int32_t *)dav2_arena_alloc((size_t)n * 3 * sizeof(int32_t));
    b.psum = (uint32_t *)dav2_arena_alloc((size_t)n * sizeof(uint32_t));
    if (dav2_arena_failed) { dav2_arena_release(mark); return; }
    exp2_tab_init();
    dav2_copy16((int16_t *)b.exp2, (const int16_t *)exp2_tab, 1024);
    b.q_direct = dav2_accel_wsh_ok() && (qkv->c & 1) == 0;
    for (int d = 0; d < HD; d++) {
        b.cpar[3 * d] = 1 << 30;
        b.cpar[3 * d + 1] = 45;
        b.cpar[3 * d + 2] = 0;
    }

    /* two score matrices and two v^T, so that the next head's score GEMM
     * and v^T are made while this head's softmax and context are */
    int32_t *s2[2] = { b.s, (int32_t *)dav2_arena_alloc((size_t)n * n * sizeof(int32_t)) };
    int32_t *st2[2] = { b.sst, (int32_t *)dav2_arena_alloc((size_t)n * 2 * sizeof(int32_t)) };
    int16_t *vt2[2] = { b.vt16, (int16_t *)dav2_arena_alloc((size_t)HD * Kp * sizeof(int16_t)) };
    int16_t *pt2[2] = { b.pt, (int16_t *)dav2_arena_alloc((size_t)n * Kp * sizeof(int16_t)) };
    int qsh2[2] = { 0, 0 };
    if (dav2_arena_failed) { dav2_arena_release(mark); return; }

    uint64_t t0 = dav2_cycles(), t1;
    int lut_loaded = 0;
#define LAP(d) do { t1 = dav2_cycles(); dav2_sub_add((d), t1 - t0); t0 = t1; } while (0)
    /* S[t][m] = q_t . k_m: k in place (A, rows qkv->c apart), q as the
     * weights -- straight from qkv, shifted by the block as it reads it
     * (CTRL.wsh), or the shifted copy made by attn_prep_q16 -- so the
     * statistics are per query */
#define Q_W(h)   (b.q_direct ? qkv->v + (h) * HD : b.q16)
#define Q_WS     (b.q_direct ? qkv->c : HD)
#define K_A(h)   (qkv->v + ED + (h) * HD)
#define S_START(h) do { attn_prep_q16(qkv, (h), n, &b);                          \
        qsh2[(h) & 1] = b.q_direct ? b.q_sh : 0;                                  \
        gemm16_start(K_A(h), qkv->c, Q_W(h), Q_WS, s2[(h) & 1], n, HD, n,         \
                     st2[(h) & 1], qsh2[(h) & 1]); } while (0)
#define S_FINISH(h) gemm16_finish(K_A(h), qkv->c, Q_W(h), Q_WS, s2[(h) & 1], n, HD, n, \
                                  st2[(h) & 1], qsh2[(h) & 1])
    /* head 0: scores, v^T while they run, the exponential job */
    S_START(0);
    LAP(DAV2_SUB_ATT_PREP_QK);
    b.vt16 = vt2[0];
    attn_prep_vt16(qkv, 0, n, &b);
    LAP(DAV2_SUB_ATT_PREP_V);
    S_FINISH(0);
    b.s = s2[0]; b.sst = st2[0]; b.pt = pt2[0];
    int exp_ok = attn_softmax_accel_start(qkv, n, &b, !lut_loaded);
    if (exp_ok) lut_loaded = 1;
    exp_ok = exp_ok && dav2_accel_finish();
    if (exp_ok)
        attn_softmax_accel_sums(n, &b);
    LAP(DAV2_SUB_ATT_WAIT);
#define HEAD_BUFS(h) do { b.s = s2[(h) & 1]; b.sst = st2[(h) & 1]; b.pt = pt2[(h) & 1]; } while (0)
    /* Per head h, with head h's sums made and head h+1 prepared in turn:
     *   CPU                           block
     *   v^T of head h+1               context requantisation of head h-1
     *   first half of head h's P      scores of head h+1
     *   second half of head h's P     exponential job of head h+1
     *   head h+1's sums               context GEMM of head h
     * The context requantisation of head h-1 is collected before head h's
     * context GEMM overwrites its input (on a failure the CPU does it). */
    int ctx_pending = 0;
    const int half = (n / 2) & ~1;
    for (int h = 0; h < DAV2_N_HEADS; h++) {
        const int nx = h + 1 < DAV2_N_HEADS;
        if (nx) {
            b.vt16 = vt2[(h + 1) & 1];
            attn_prep_vt16(qkv, h + 1, n, &b);
        }
        LAP(DAV2_SUB_ATT_PREP_V);
        if (ctx_pending && !dav2_accel_finish())
            attn_context16_cpu(h - 1, n, &b, ctx);   /* b.c is still head h-1's */
        ctx_pending = 0;
        LAP(DAV2_SUB_ATT_NORM);
        if (nx) S_START(h + 1);
        HEAD_BUFS(h);
        if (exp_ok)
            attn_softmax_accel_norm(n, &b, 0, half);
        LAP(DAV2_SUB_ATT_SOFTMAX);
        int exp_next = 0;
        if (nx) {
            S_FINISH(h + 1);
            HEAD_BUFS(h + 1);
            exp_next = attn_softmax_accel_start(qkv, n, &b, !lut_loaded);
            if (exp_next) lut_loaded = 1;
        }
        LAP(DAV2_SUB_ATT_WAIT);
        HEAD_BUFS(h);
        if (exp_ok)
            attn_softmax_accel_norm(n, &b, half, n);
        else
            attn_softmax(qkv, n, &b);   /* no exponential job, or it failed */
        LAP(DAV2_SUB_ATT_SOFTMAX);
        /* the next head's exponential job, collected before the context
         * GEMM so that a failure is known (then that head's softmax runs
         * on the CPU; its S is intact in DDR3) */
        exp_next = exp_next && dav2_accel_finish();
        LAP(DAV2_SUB_ATT_WAIT);
        /* C[d][t] = sum_m P[t][m] v[m][d]; the next head's sums meanwhile */
        gemm16_start(b.p16, Kp, vt2[h & 1], Kp, b.c, n, Kp, HD, 0, 0);
        LAP(DAV2_SUB_ATT_WAIT);
        if (exp_next) {
            HEAD_BUFS(h + 1);
            attn_softmax_accel_sums(n, &b);
        }
        LAP(DAV2_SUB_ATT_SOFTMAX);
        gemm16_finish(b.p16, Kp, vt2[h & 1], Kp, b.c, n, Kp, HD, 0, 0);
        LAP(DAV2_SUB_ATT_WAIT);
        ctx_pending = attn_context16_start(h, n, &b, ctx);
        LAP(DAV2_SUB_ATT_NORM);
        exp_ok = exp_next;
    }
    if (ctx_pending && !dav2_accel_finish())
        attn_context16_cpu(DAV2_N_HEADS - 1, n, &b, ctx);
    LAP(DAV2_SUB_ATT_NORM);
#undef HEAD_BUFS
#undef S_START
#undef S_FINISH
#undef Q_W
#undef Q_WS
#undef K_A
#undef LAP
    dav2_arena_release(mark);
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
    /* int16 weights when the block has them, or on the CPU alone */
    if (dav2_accel_w16_ok() || !dav2_accel_present()) {
        attention_all16(qkv, n_tokens, ctx);
        return;
    }
    const int HD = DAV2_HEAD_DIM;
    const int n  = n_tokens;
    const int Kp = (n + 3) & ~3;                 /* K must be a multiple of 4 */
    const size_t mark = dav2_arena_mark();
    attn_bufs_t b;
    memset(&b, 0, sizeof b);

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
/* x = x + f(...) into the spare buffer, then swap the two: the GEMM's
 * requantisation job writes a buffer other than its residual input, so no
 * temporary and no copy back are needed, and a failed job still leaves x
 * intact for the CPU redo. */
static void residual_update(const dav2_tensor_t *a, const dav2_qw_t *w,
                            dav2_tensor_t *x, int16_t **spare, uint32_t *rst,
                            uint32_t *rsum)
{
    dav2_background_complete();   /* a tap's LayerNorm reads the old x */
    dav2_tensor_t y = *x;
    y.v = *spare;
    y.rst = rst;           /* ask for the new x's row ranges (for LayerNorm) */
    y.rsum = rsum;         /* ... and its row sums */
    dav2_qgemm_ex(a, w, x, 0, &y);
    *spare = x->v;
    *x = y;
}

static void run_block(dav2_tensor_t *x, int16_t **spare, uint32_t *rst, uint32_t *rsum,
                      int i, int n_tokens)
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
    /* with each column's extremes: the heads' q ranges for attention */
    int16_t *qext = (int16_t *)dav2_arena_alloc((size_t)qkv.c * 2 * sizeof(int16_t));
    if (qext) {
        dav2_qgemm_colext(&n, &bw.qkv, &qkv, qext, qext + qkv.c);
        g_qkv_cmax = qext;
        g_qkv_cmin = qext + qkv.c;
    } else {
        dav2_qgemm(&n, &bw.qkv, &qkv);
    }
    BDUMP("qkv", &qkv);

    dav2_tensor_t ctx = dav2_tensor_new(n_tokens, ED);
    ctx.scale = qkv.scale;              /* context inherits the v scale */
    {
        uint64_t t0 = dav2_cycles();
        attention_all(&qkv, n_tokens, &ctx);
        g_qkv_cmax = g_qkv_cmin = 0;
        dav2_prof_add(DAV2_PROF_ATTENTION, dav2_cycles() - t0);
    }

    BDUMP("ctx", &ctx);
    /* x = x + proj(ctx): the residual add is the requantisation job's
     * epilogue on the accelerator; the result goes to the spare buffer */
    residual_update(&ctx, &bw.proj, x, spare, rst, rsum);
    BDUMP("res1", x);
    dav2_arena_release(mark);

    mark = dav2_arena_mark();
    dav2_tensor_t n2 = dav2_tensor_new(n_tokens, ED);
    dav2_layernorm(x, bw.n2w, bw.n2b, &n2);

    BDUMP("ln2", &n2);
    dav2_tensor_t h1 = dav2_tensor_new(n_tokens, 4 * ED);
    dav2_qgemm_gelu(&n2, &bw.fc1, &h1);        /* GELU in the requantisation job */
    BDUMP("gelu", &h1);

    /* x = x + fc2(h1), likewise fused */
    residual_update(&h1, &bw.fc2, x, spare, rst, rsum);
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

    /* relu(x) read by the first conv's gather, relu(conv1) and x + conv2
     * fused into the convs' requantisation, conv2 written straight into out */
    dav2_tensor_t a = dav2_tensor_new(h * w, c1.m);
    dav2_conv2d_into(x, h, w, &c1, 3, 1, 1, 0, 1, 1, &a);
    dav2_conv2d_into(&a, h, w, &c2, 3, 1, 1, x, 0, 0, &out);

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

    /* a is read, never written: no copy of it. Its amax_q (known from the
     * producing conv) spares dav2_add a scan; a scan gives the same value. */
    dav2_tensor_t cur = *a, res, s;
    const dav2_tensor_t *u_in = &cur;
    dav2_add_t addp;
    int add_prod = 0;

    if (b) {
        /* the sum as a producer: the next unit's first convolution starts
         * on its first rows while the CPU adds the rest (same result) */
        res = res_conv_unit(b, h, w, prefix, 1);
        s = dav2_tensor_new(h * w, DAV2_FEATURES);
        dav2_add_begin(&addp, &cur, &res, &s, w);
        dav2_producer_set(&addp.base);
        add_prod = 1;
        u_in = &s;
    }

    dav2_tensor_t u2 = res_conv_unit(u_in, h, w, prefix, 2);
    if (add_prod) {
        dav2_producer_complete();
        dav2_producer_set(0);
    }
    dav2_tensor_t up = dav2_tensor_new(oh * ow, DAV2_FEATURES);
    dav2_qw_t oc;
    sprintf(nm, "%sout", prefix); dav2_qw(&oc, nm, DAV2_FEATURES);

    /* the upsampling as a producer: the output convolution's tiles start
     * as soon as their rows exist, and the CPU makes the next rows while
     * the block works (same result as interpolating first) */
    dav2_interp_t ip;
    dav2_interp_begin(&ip, &u2, h, w, oh, ow, &up);
    dav2_producer_set(&ip.base);
    dav2_conv2d_into(&up, oh, ow, &oc, 1, 1, 0, 0, 0, 0, &out);
    dav2_producer_complete();
    dav2_producer_set(0);

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
    image.rst = 0;
    image.rsum = 0;

    dav2_qw_t pe_w;
    dav2_qw(&pe_w, "patch_embed", DAV2_PATCH * DAV2_PATCH * 3);

    dav2_tensor_t x = dav2_tensor_new(n_tokens, ED);
    int16_t *x_spare = dav2_tensor_new(n_tokens, ED).v;   /* see residual_update */
    uint32_t *x_rst = (uint32_t *)dav2_arena_alloc((size_t)n_tokens * sizeof(uint32_t));
    uint32_t *x_rsum = (uint32_t *)dav2_arena_alloc((size_t)n_tokens * 3 * sizeof(uint32_t));
    /* the softmax's exponential table for the accelerator (fixed; built
     * once per frame, about 0.1 Mcycles) */
    g_exp_lut = 0;
    if (dav2_accel_lut_ok()) {
        g_exp_lut = (int16_t *)dav2_arena_alloc(16384 * sizeof(int16_t));
        if (g_exp_lut) exp_lut_build(g_exp_lut);
    }
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
    dav2_lnplain_t tap_ln[4];
    for (int i = 0; i < 4; i++)
        feats[i] = dav2_tensor_new(n_patch, ED);

    for (int i = 0; i < DAV2_N_BLOCKS; i++) {
        char msg[32];
        sprintf(msg, "block %d/12", i + 1);
        dav2_progress(msg);
        run_block(&x, &x_spare, x_rst, x_rsum, i, n_tokens);
#ifdef DAV2_TRACE
        { char fn[64]; sprintf(fn, "/tmp/dav2_blk%d.bin", i); dav2_dump(fn, &x); }
#endif

        for (int j = 0; j < 4; j++) {
            if (INTERMEDIATE[j] != i) continue;
            /* final LayerNorm (gamma, beta folded into proj0..3) without the
             * class token's row, straight into feats[j]: the statistics
             * now, the rows as background work in the next block's waits
             * (completed before x changes, see residual_update) */
            tap_ln[j].bg = 0;
            if (dav2_lnplain_begin(&tap_ln[j], &x, feats[j].v, 1)) {
                tap_ln[j].bg = 1;
                dav2_background_set(&tap_ln[j].base);
            }
            feats[j].scale = tap_ln[j].out_scale;
#ifdef DAV2_TRACE
            dav2_background_complete();
            { char fn[64]; sprintf(fn, "/tmp/dav2_feat%d.bin", j);
              dav2_dump(fn, &feats[j]); }
#endif
        }
    }

    dav2_background_complete();   /* the last tap's LayerNorm */

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
    /* the upsampling made while the next convolution's first tiles run */
    dav2_interp_t ip;
    dav2_interp_begin(&ip, &c1, oh, ow, out_size, out_size, &up);
    dav2_producer_set(&ip.base);
    dav2_tensor_t c2 = dav2_conv2d_ex(&up, out_size, out_size, &o2a, 3, 1, 1, 0, 1, &oh, &ow);
    dav2_producer_complete();
    dav2_producer_set(0);
    dav2_tensor_t c3 = dav2_conv2d_ex(&c2, out_size, out_size, &o2b, 1, 1, 0, 0, 1, &oh, &ow);

    /* int16 depth and its scale; converting to float is the caller's
     * choice (the board leaves it to the PC) */
    dav2_copy16(depth_q, c3.v, (size_t)out_size * out_size);
    *depth_scale = c3.scale;

    g_exp_lut = 0;                 /* its arena space is not kept after the frame */
    dav2_progress("done");
    dav2_prof_report(dav2_cycles() - t_frame);
}
