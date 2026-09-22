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
#include "dav2_mathf.h"

#include <stdio.h>
#include <string.h>

/* Input image, supplied by the caller through dav2_set_image(). Kept as a
 * pointer rather than copied: on the board it is 95 kB in DDR3. */
static const int16_t *g_image;
static float          g_image_scale = 1.0f;

void dav2_set_image(const int16_t *hwc, float scale)
{
    g_image = hwc;
    g_image_scale = scale;
}

extern const void *dav2_find_quiet(const char *name);
extern void dav2_dump(const char *path, const dav2_tensor_t *t);
extern int dav2_blob_check(void);

static const int INTERMEDIATE[4] = {2, 5, 8, 11};

/* ------------------------------------------------------------- attention */

/* Scores for one head, then softmax, then the context vector. Q/K are
 * requantised to 11 bits so that the K=64 dot product cannot overflow int32:
 * 64 * 2047^2 = 2.7e8. */

/* floor(a * b / 2^s) for a, b < 2^32 and 1 <= s <= 63, as two 32x32->64
 * halves and 32-bit shifts: mulhu + mul instead of a 64-bit library shift.
 * Saturates to 0xffffffff when the true quotient does not fit 32 bits. */
static inline uint32_t mul_shr_floor(uint32_t a, uint32_t b, int s)
{
    uint32_t hi = (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);   /* mulhu */
    uint32_t lo = a * b;                                            /* mul   */
    if (s >= 32)
        return hi >> (s - 32);
    if (hi >> s)
        return 0xffffffffu;
    return (hi << (32 - s)) | (lo >> s);
}

/* One attention head, on the GEMM accelerator.
 *
 * Both products of attention are matrix multiplications, but neither has an
 * int8 operand: q, k and v are all int16 activations. The accelerator only
 * multiplies int16 by int8, so the second operand is split into a high and a
 * low int8 part and the product is run twice:
 *
 *     x = hi * 2^s + lo,  a.x = 2^s * (a.hi) + (a.lo)         exactly.
 *
 * Scores:  S[t][m] = q_t . k_m      A = k (int16, shifted to 11 bits),
 *                                   W = q_hi (q >> 4), q_lo (q & 15)
 * Context: C[t][d] = sum_m P[t][m] v[m][d]
 *                                   A = P (int16 probabilities, Q15),
 *                                   W = v^T_hi (v >> 6), v^T_lo (v & 63)
 *
 * so the accelerator does the 2 x 82 x 82 x 64 multiply-accumulates per head
 * that the CPU spent ~16 cycles each on (47 % of the frame), and the CPU is
 * left with the gathers, the softmax and the final normalisation. The
 * numbers are identical to the all-CPU version except that the largest
 * probability, exactly 2^15, is clamped to 32767 so it fits an int16.
 *
 * Operands are built in the DDR3 arena, packed four int8 per word, since a
 * byte store costs the CPU the same bus transaction as a word store. */
typedef struct {
    int16_t *k16;  int8_t *q_hi, *q_lo;
    int32_t *s_hi, *s_lo;
    int16_t *p16;  int8_t *vt_hi, *vt_lo;
    int32_t *c_hi, *c_lo;
    int64_t *pinv; int32_t *psum;
    int      q_sh, k_sh;
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

/* q and k of a head, shifted to 11 bits: k as the int16 operand, q split
 * into two int8 halves. Uses dav2_scratch as a temporary. */
static void attn_prep_qk(const dav2_tensor_t *qkv, int head, int n, attn_bufs_t *b)
{
    const int HD = DAV2_HEAD_DIM;
    const int ED = DAV2_EMBED_DIM;
    const int Kp = (n + 3) & ~3;
    int16_t *k16 = b->k16; int8_t *q_hi = b->q_hi, *q_lo = b->q_lo;
    int32_t *s_hi = b->s_hi, *s_lo = b->s_lo; int16_t *p16 = b->p16;
    int8_t *vt_hi = b->vt_hi, *vt_lo = b->vt_lo;
    int32_t *c_hi = b->c_hi, *c_lo = b->c_lo; int64_t *pinv = b->pinv; int32_t *psum = b->psum;
    (void)HD; (void)ED; (void)Kp; (void)k16; (void)q_hi; (void)q_lo; (void)s_hi; (void)s_lo;
    (void)p16; (void)vt_hi; (void)vt_lo; (void)c_hi; (void)c_lo; (void)pinv; (void)psum;
    /* q and k of this head into on-chip scratch, with their ranges. */
    int16_t *q = dav2_scratch;
    int16_t *k = q + (size_t)n * HD;
    int32_t qmax = 0, kmax = 0;
    for (int t = 0; t < n; t++) {
        const int16_t *row = qkv->v + (size_t)t * qkv->c;
        const int16_t *qr = row + head * HD;
        const int16_t *kr = row + ED + head * HD;
        int16_t *qd = q + (size_t)t * HD, *kd = k + (size_t)t * HD;
        #pragma GCC unroll 8
        for (int d = 0; d < HD; d++) {
            int32_t vq = qr[d], vk = kr[d];
            qd[d] = (int16_t)vq;
            kd[d] = (int16_t)vk;
            if (vq < 0) vq = -vq;
            if (vk < 0) vk = -vk;
            if (vq > qmax) qmax = vq;
            if (vk > kmax) kmax = vk;
        }
    }
    int q_sh = 0, k_sh = 0;
    while ((qmax >> q_sh) > DAV2_QK_QMAX) q_sh++;
    while ((kmax >> k_sh) > DAV2_QK_QMAX) k_sh++;

    /* Shifted k as the int16 operand; shifted q split into two int8 halves.
     * |q| <= 2047 after the shift, so q >> 4 is within int8. */
    for (int t = 0; t < n; t++) {
        const int16_t *qr = q + (size_t)t * HD, *kr = k + (size_t)t * HD;
        int16_t  *kd = k16 + (size_t)t * HD;
        uint32_t *qh = (uint32_t *)(q_hi + (size_t)t * HD);
        uint32_t *ql = (uint32_t *)(q_lo + (size_t)t * HD);
        #pragma GCC unroll 4
        for (int d = 0; d < HD; d += 4) {
            uint32_t wh = 0, wl = 0;
            #pragma GCC unroll 4
            for (int j = 0; j < 4; j++) {
                int32_t vq = qr[d + j] >> q_sh;
                wh |= ((uint32_t)(vq >> 4) & 0xffu) << (8 * j);
                wl |= ((uint32_t)vq & 0xfu)         << (8 * j);
                kd[d + j] = (int16_t)(kr[d + j] >> k_sh);
            }
            qh[d >> 2] = wh;
            ql[d >> 2] = wl;
        }
    }
    b->q_sh = q_sh;
    b->k_sh = k_sh;
}

/* v^T of a head, split into int8 halves */
static void attn_prep_v(const dav2_tensor_t *qkv, int head, int n, attn_bufs_t *b)
{
    const int HD = DAV2_HEAD_DIM;
    const int ED = DAV2_EMBED_DIM;
    const int Kp = (n + 3) & ~3;
    int16_t *k16 = b->k16; int8_t *q_hi = b->q_hi, *q_lo = b->q_lo;
    int32_t *s_hi = b->s_hi, *s_lo = b->s_lo; int16_t *p16 = b->p16;
    int8_t *vt_hi = b->vt_hi, *vt_lo = b->vt_lo;
    int32_t *c_hi = b->c_hi, *c_lo = b->c_lo; int64_t *pinv = b->pinv; int32_t *psum = b->psum;
    (void)HD; (void)ED; (void)Kp; (void)k16; (void)q_hi; (void)q_lo; (void)s_hi; (void)s_lo;
    (void)p16; (void)vt_hi; (void)vt_lo; (void)c_hi; (void)c_lo; (void)pinv; (void)psum;
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

/* scores -> Q15 probabilities, with each token's sum and reciprocal.
 * Uses dav2_scratch for the score row. */
static void attn_softmax(const dav2_tensor_t *qkv, int n, attn_bufs_t *b)
{
    const int HD = DAV2_HEAD_DIM;
    const int ED = DAV2_EMBED_DIM;
    const int Kp = (n + 3) & ~3;
    int16_t *k16 = b->k16; int8_t *q_hi = b->q_hi, *q_lo = b->q_lo;
    int32_t *s_hi = b->s_hi, *s_lo = b->s_lo; int16_t *p16 = b->p16;
    int8_t *vt_hi = b->vt_hi, *vt_lo = b->vt_lo;
    int32_t *c_hi = b->c_hi, *c_lo = b->c_lo; int64_t *pinv = b->pinv; int32_t *psum = b->psum;
    (void)HD; (void)ED; (void)Kp; (void)k16; (void)q_hi; (void)q_lo; (void)s_hi; (void)s_lo;
    (void)p16; (void)vt_hi; (void)vt_lo; (void)c_hi; (void)c_lo; (void)pinv; (void)psum;
    /* Softmax per query token, in fixed point: p = 2^(-(smax - s) * scale *
     * log2e) in Q15. kf is typically far below 1 (score_scale is ~1e-5), so
     * it must be carried as a (mult, shift) pair rather than a plain Q16
     * integer -- rounding it into a Q16 constant collapses it to 0 or 1 and
     * flattens the whole distribution. */
    float score_scale = qkv->scale * qkv->scale
                      * (float)(1 << b->q_sh) * (float)(1 << b->k_sh);
    float kf = score_scale * 1.4426950408889634f;   /* -> exponent base 2 */
    int32_t kmult; int kshift;
    dav2_make_multiplier(kf, &kmult, &kshift);

    int32_t *scores = (int32_t *)dav2_scratch;   /* q/k are no longer needed */
    for (int t = 0; t < n; t++) {
        const int32_t *sh = s_hi + (size_t)t * n, *sl = s_lo + (size_t)t * n;
        int32_t smax = -2147483647 - 1;
        for (int m = 0; m < n; m++) {
            int32_t sc = sh[m] * 16 + sl[m];
            scores[m] = sc;
            if (sc > smax) smax = sc;
        }
        int32_t sum = 0;
        int16_t *pr = p16 + (size_t)t * Kp;
        for (int m = 0; m < n; m++) {
            uint32_t d = (uint32_t)(smax - scores[m]);      /* >= 0 */
            /* t_q16 = d * kf * 2^16, saturated: anything at or above 16.0
             * underflows Q15 anyway. All 32-bit: d < 2^30 and kmult < 2^31. */
            uint32_t t_q16;
            if (kshift > 16) {
                t_q16 = mul_shr_floor(d, (uint32_t)kmult, kshift - 16);
            } else if (kshift == 16) {
                uint32_t hi = (uint32_t)(((uint64_t)d * (uint64_t)kmult) >> 32);
                t_q16 = hi ? 0xffffffffu : d * (uint32_t)kmult;
            } else {
                /* kf >= 2^15: any nonzero gap underflows Q15 immediately */
                t_q16 = (d == 0) ? 0 : (16u << 16);
            }
            int32_t p;
            if (t_q16 >= (16u << 16)) {
                p = 0;                       /* underflows Q15 */
            } else {
                int32_t ip = (int32_t)(t_q16 >> 16);
                int32_t fp = (int32_t)(t_q16 & 0xffff);
                /* 2^-frac in Q15. Linear interpolation between 1.0 and 0.5
                 * overestimates because 2^-x is convex, so subtract a
                 * parabolic correction: the gap peaks at x=0.5, where
                 * 0.75 - 2^-0.5 = 0.0429 -> 1406 in Q15, and 4*1406 = 5623
                 * is the coefficient of the x(1-x) term. */
                int32_t lin = 32768 - ((fp * 16384) >> 16);
                int32_t u = (fp * (65536 - fp)) >> 16;   /* x(1-x) in Q16 */
                int32_t corr = (u * 5623) >> 16;
                p = (lin - corr) >> ip;
            }
            if (p > 32767) p = 32767;        /* the argmax: 2^15 -> int16 */
            pr[m] = (int16_t)p;
            sum += p;
        }
        for (int m = n; m < Kp; m++) pr[m] = 0;
        if (sum == 0) { sum = 1; pr[0] = 1; }
        psum[t] = sum;
        pinv[t] = ((int64_t)1 << 40) / sum;
    }

}

/* context -> ctx columns of this head. Uses dav2_scratch. */
static void attn_normalise(int head, int n, attn_bufs_t *b, dav2_tensor_t *ctx)
{
    const int HD = DAV2_HEAD_DIM;
    const int ED = DAV2_EMBED_DIM;
    const int Kp = (n + 3) & ~3;
    int16_t *k16 = b->k16; int8_t *q_hi = b->q_hi, *q_lo = b->q_lo;
    int32_t *s_hi = b->s_hi, *s_lo = b->s_lo; int16_t *p16 = b->p16;
    int8_t *vt_hi = b->vt_hi, *vt_lo = b->vt_lo;
    int32_t *c_hi = b->c_hi, *c_lo = b->c_lo; int64_t *pinv = b->pinv; int32_t *psum = b->psum;
    (void)HD; (void)ED; (void)Kp; (void)k16; (void)q_hi; (void)q_lo; (void)s_hi; (void)s_lo;
    (void)p16; (void)vt_hi; (void)vt_lo; (void)c_hi; (void)c_lo; (void)pinv; (void)psum;
    /* ctx[t][d] = (64 c_hi + c_lo) / sum_t, kept at the v scale, and exactly
     * the truncated quotient -- but without a 64-bit division per element
     * (5248 per head, hundreds of cycles each on this core). One reciprocal
     * per token, floor(2^40 / sum), multiplies in; it underestimates by less
     * than one unit of the quotient (|c| <= 8191 sum, so the error is under
     * 8191 sum / 2^40 < 1), so a single compare against the remainder
     * corrects it. Assembled in scratch as [t][d], then copied out in rows. */
    int16_t *cbuf = dav2_scratch + 2 * n;     /* past the score row */
    for (int d = 0; d < HD; d++) {
        const int32_t *ch = c_hi + (size_t)d * n, *cl = c_lo + (size_t)d * n;
        for (int t = 0; t < n; t++) {
            int64_t  a  = (int64_t)ch[t] * 64 + (int64_t)cl[t];
            uint64_t aa = (uint64_t)(a < 0 ? -a : a);              /* < 2^35 */
            /* aa * pinv >> 40 with 32-bit multiplies: aa = ah*2^32 + al,
             * H = ah*pinv + mulhu(al, pinv) < 2^29, and the low word of
             * al*pinv cannot carry into bit 40, so the quotient is H >> 8. */
            int64_t q;
            if (__builtin_expect(pinv[t] >> 32, 0)) {
                q = (int64_t)(aa / (uint64_t)psum[t]);   /* sum < 256: cannot happen */
            } else {
                uint32_t al = (uint32_t)aa, ah = (uint32_t)(aa >> 32);
                uint32_t pv = (uint32_t)pinv[t];
                uint32_t H  = ah * pv + (uint32_t)(((uint64_t)al * (uint64_t)pv) >> 32);
                q = (int64_t)(H >> 8);
                if (aa - (uint64_t)q * (uint64_t)psum[t] >= (uint64_t)psum[t]) q++;
            }
            int32_t r = (int32_t)(a < 0 ? -q : q);
            if (r >  DAV2_ACT_QMAX) r =  DAV2_ACT_QMAX;
            if (r < -DAV2_ACT_QMAX) r = -DAV2_ACT_QMAX;
            cbuf[(size_t)t * HD + d] = (int16_t)r;
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
 * The buffers are allocated once for all heads: head h+1's q/k go into the
 * same k16/q_hi/q_lo as head h's, which is safe because both S jobs of head
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

    b.k16   = (int16_t *)dav2_arena_alloc((size_t)n * HD * sizeof(int16_t));
    b.q_hi  = (int8_t  *)dav2_arena_alloc((size_t)n * HD);
    b.q_lo  = (int8_t  *)dav2_arena_alloc((size_t)n * HD);
    b.s_hi  = (int32_t *)dav2_arena_alloc((size_t)n * n * sizeof(int32_t));
    b.s_lo  = (int32_t *)dav2_arena_alloc((size_t)n * n * sizeof(int32_t));
    b.p16   = (int16_t *)dav2_arena_alloc((size_t)n * Kp * sizeof(int16_t));
    b.vt_hi = (int8_t  *)dav2_arena_alloc((size_t)HD * Kp);
    b.vt_lo = (int8_t  *)dav2_arena_alloc((size_t)HD * Kp);
    b.c_hi  = (int32_t *)dav2_arena_alloc((size_t)HD * n * sizeof(int32_t));
    b.c_lo  = (int32_t *)dav2_arena_alloc((size_t)HD * n * sizeof(int32_t));
    b.pinv  = (int64_t *)dav2_arena_alloc((size_t)n * sizeof(int64_t));
    b.psum  = (int32_t *)dav2_arena_alloc((size_t)n * sizeof(int32_t));
    if (dav2_arena_failed) { dav2_arena_release(mark); return; }

    attn_prep_qk(qkv, 0, n, &b);
    for (int h = 0; h < DAV2_N_HEADS; h++) {
        /* S = k . q_hi, k . q_lo  ->  s[t][m] (W rows are t, A rows are m) */
        gemm_i32_start(b.k16, HD, b.q_hi, HD, b.s_hi, n, HD, n);
        attn_prep_v(qkv, h, n, &b);
        gemm_i32_start(b.k16, HD, b.q_lo, HD, b.s_lo, n, HD, n);
        gemm_i32_finish();
        attn_softmax(qkv, n, &b);
        /* C = P . v^T_hi, P . v^T_lo  ->  c[d][t] */
        gemm_i32_start(b.p16, Kp, b.vt_hi, Kp, b.c_hi, n, Kp, HD);
        gemm_i32_start(b.p16, Kp, b.vt_lo, Kp, b.c_lo, n, Kp, HD);
        if (h + 1 < DAV2_N_HEADS)
            attn_prep_qk(qkv, h + 1, n, &b);
        gemm_i32_finish();
        attn_normalise(h, n, &b, ctx);
    }
    dav2_arena_release(mark);
}

/* ------------------------------------------------------------- encoder */

typedef struct {
    dav2_qw_t qkv, proj, fc1, fc2;
    const float *n1w, *n1b, *n2w, *n2b;
} block_w_t;

static void load_block(block_w_t *bw, int i)
{
    char base[64], nm[64];
    sprintf(base, "blk%d.", i);

    sprintf(nm, "%sqkv", base);  dav2_qw(&bw->qkv, nm, DAV2_EMBED_DIM);
    sprintf(nm, "%sproj", base); dav2_qw(&bw->proj, nm, DAV2_EMBED_DIM);
    sprintf(nm, "%sfc1", base);  dav2_qw(&bw->fc1, nm, DAV2_EMBED_DIM);
    sprintf(nm, "%sfc2", base);  dav2_qw(&bw->fc2, nm, 4 * DAV2_EMBED_DIM);

    sprintf(nm, "%snorm1.w", base); bw->n1w = (const float *)dav2_find(nm, 0);
    sprintf(nm, "%snorm1.b", base); bw->n1b = (const float *)dav2_find(nm, 0);
    sprintf(nm, "%snorm2.w", base); bw->n2w = (const float *)dav2_find(nm, 0);
    sprintf(nm, "%snorm2.b", base); bw->n2b = (const float *)dav2_find(nm, 0);
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
    dav2_tensor_t attn = dav2_tensor_new(n_tokens, ED);
    dav2_qgemm(&ctx, &bw.proj, &attn);
    BDUMP("attn", &attn);

    dav2_tensor_t sum1 = dav2_tensor_new(n_tokens, ED);
    dav2_add(x, &attn, &sum1);
    dav2_copy16(x->v, sum1.v, (size_t)n_tokens * ED);
    x->scale = sum1.scale;
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

    dav2_tensor_t h2 = dav2_tensor_new(n_tokens, ED);
    dav2_qgemm(&h1, &bw.fc2, &h2);

    BDUMP("fc2", &h2);
    dav2_tensor_t sum2 = dav2_tensor_new(n_tokens, ED);
    dav2_add(x, &h2, &sum2);
    dav2_copy16(x->v, sum2.v, (size_t)n_tokens * ED);
    x->scale = sum2.scale;
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
    dav2_copy16(t.v, x->v, (size_t)h * w * x->c);
    t.scale = x->scale;
    dav2_relu(&t);

    dav2_tensor_t a = dav2_conv2d(&t, h, w, &c1, 3, 1, 1, 0, 0);
    dav2_relu(&a);
    dav2_tensor_t b = dav2_conv2d(&a, h, w, &c2, 3, 1, 1, 0, 0);
    dav2_add(&b, x, &out);

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
}

/* ------------------------------------------------------------------ main */

void dav2_infer(const dav2_cfg_t *cfg, float *depth_out)
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
         * embedding; both are float in the blob, so this is done in float and
         * requantised once. */
        const float *cls = (const float *)dav2_find("cls_token", 0);
        const float *pos = (const float *)dav2_find("pos_embed", 0);
        size_t mark = dav2_arena_mark();
        float *tmp = (float *)dav2_arena_alloc((size_t)n_tokens * ED * sizeof(float));
        for (int c = 0; c < ED; c++)
            tmp[c] = cls[c] + pos[c];
        for (int p = 0; p < n_patch; p++)
            for (int c = 0; c < ED; c++)
                tmp[(size_t)(p + 1) * ED + c] =
                    (float)patches.v[(size_t)p * ED + c] * patches.scale
                    + pos[(size_t)(p + 1) * ED + c];
        dav2_quantize_f32(tmp, n_tokens, ED, &x);
        dav2_arena_release(mark);
    }

#ifdef DAV2_TRACE
    dav2_dump("/tmp/dav2_tokens.bin", &x);
#endif

    /* ---- transformer blocks, capturing the four DPT taps --------------- */
    const float *nw = (const float *)dav2_find("norm.w", 0);
    const float *nbf = (const float *)dav2_find("norm.b", 0);

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
            /* final LayerNorm, then drop the class token */
            size_t mark = dav2_arena_mark();
            dav2_tensor_t nrm = dav2_tensor_new(n_tokens, ED);
            dav2_layernorm(&x, nw, nbf, &nrm);
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

    dav2_tensor_t c2 = dav2_conv2d(&up, out_size, out_size, &o2a, 3, 1, 1, &oh, &ow);
    dav2_relu(&c2);
    dav2_tensor_t c3 = dav2_conv2d(&c2, out_size, out_size, &o2b, 1, 1, 0, &oh, &ow);
    dav2_relu(&c3);

    for (int i = 0; i < out_size * out_size; i++)
        depth_out[i] = (float)c3.v[i] * c3.scale;

    dav2_progress("done");
    dav2_prof_report(dav2_cycles() - t_frame);
}
