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

#include "dav2.h"
#include "dav2_mathf.h"

#include <stdio.h>
#include <string.h>

extern const void *dav2_find_quiet(const char *name);
extern void dav2_dump(const char *path, const dav2_tensor_t *t);
extern int dav2_blob_check(void);

static const int INTERMEDIATE[4] = {2, 5, 8, 11};

/* ------------------------------------------------------------- attention */

/* Scores for one head, then softmax, then the context vector. Q/K are
 * requantised to 11 bits so that the K=64 dot product cannot overflow int32:
 * 64 * 2047^2 = 2.7e8. */
static void attention_head(const dav2_tensor_t *qkv, int head, int n_tokens,
                           dav2_tensor_t *ctx)
{
    const int HD = DAV2_HEAD_DIM;
    const int ED = DAV2_EMBED_DIM;
    const size_t mark = dav2_arena_mark();

    int16_t *q = (int16_t *)dav2_arena_alloc((size_t)n_tokens * HD * sizeof(int16_t));
    int16_t *k = (int16_t *)dav2_arena_alloc((size_t)n_tokens * HD * sizeof(int16_t));
    int32_t *scores = (int32_t *)dav2_arena_alloc((size_t)n_tokens * sizeof(int32_t));
    int32_t *probs = (int32_t *)dav2_arena_alloc((size_t)n_tokens * sizeof(int32_t));

    /* extract and rescale q, k for this head */
    int32_t qmax = 0, kmax = 0;
    for (int t = 0; t < n_tokens; t++) {
        const int16_t *row = qkv->v + (size_t)t * qkv->c;
        for (int d = 0; d < HD; d++) {
            int32_t vq = row[head * HD + d];
            int32_t vk = row[ED + head * HD + d];
            q[t * HD + d] = (int16_t)vq;
            k[t * HD + d] = (int16_t)vk;
            if (vq < 0) vq = -vq;
            if (vk < 0) vk = -vk;
            if (vq > qmax) qmax = vq;
            if (vk > kmax) kmax = vk;
        }
    }
    int q_sh = 0, k_sh = 0;
    while ((qmax >> q_sh) > DAV2_QK_QMAX) q_sh++;
    while ((kmax >> k_sh) > DAV2_QK_QMAX) k_sh++;
    if (q_sh || k_sh) {
        for (int i = 0; i < n_tokens * HD; i++) {
            q[i] = (int16_t)(q[i] >> q_sh);
            k[i] = (int16_t)(k[i] >> k_sh);
        }
    }

    /* real score = acc * score_scale */
    float score_scale = qkv->scale * qkv->scale
                      * (float)(1 << q_sh) * (float)(1 << k_sh);

    for (int t = 0; t < n_tokens; t++) {
        const int16_t *qr = q + (size_t)t * HD;
        int32_t smax = -2147483647 - 1;
        for (int m = 0; m < n_tokens; m++) {
            const int16_t *kr = k + (size_t)m * HD;
            int32_t s = 0;
            for (int d = 0; d < HD; d++)
                s += (int32_t)qr[d] * (int32_t)kr[d];
            scores[m] = s;
            if (s > smax) smax = s;
        }

        /* softmax in fixed point: p = 2^(-(smax - s) * scale * log2e) in Q15.
         * kf is typically far below 1 (score_scale is ~1e-5), so it must be
         * carried as a (mult, shift) pair rather than a plain Q16 integer --
         * rounding it into a Q16 constant collapses it to 0 or 1 and flattens
         * the whole distribution. */
        float kf = score_scale * 1.4426950408889634f;   /* -> exponent base 2 */
        int32_t kmult; int kshift;
        dav2_make_multiplier(kf, &kmult, &kshift);
        int32_t sum = 0;
        for (int m = 0; m < n_tokens; m++) {
            int64_t d = (int64_t)(smax - scores[m]);
            /* t_q16 = d * kf * 2^16 */
            int64_t t_q16;
            if (kshift >= 16) {
                t_q16 = (d * (int64_t)kmult) >> (kshift - 16);
            } else {
                /* kf >= 2^15: any nonzero gap underflows Q15 immediately */
                t_q16 = (d == 0) ? 0 : ((int64_t)16 << 16);
            }
            int32_t p;
            if (t_q16 >= (int64_t)16 << 16) {
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
            probs[m] = p;
            sum += p;
        }
        if (sum == 0) { sum = 1; probs[0] = 1; }

        /* ctx = sum_m p_m * v_m; |sum p| == 32768 bounds this well inside
         * int32 even with 14-bit v. */
        int16_t *orow = ctx->v + (size_t)t * ED + head * HD;
        for (int d = 0; d < HD; d++) {
            int64_t a = 0;
            for (int m = 0; m < n_tokens; m++) {
                int32_t v = qkv->v[(size_t)m * qkv->c + 2 * ED + head * HD + d];
                a += (int64_t)probs[m] * (int32_t)v;
            }
            /* normalise by sum and keep the v scale */
            int32_t r = (int32_t)(a / sum);
            if (r >  DAV2_ACT_QMAX) r =  DAV2_ACT_QMAX;
            if (r < -DAV2_ACT_QMAX) r = -DAV2_ACT_QMAX;
            orow[d] = (int16_t)r;
        }
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
    for (int h = 0; h < DAV2_N_HEADS; h++)
        attention_head(&qkv, h, n_tokens, &ctx);

    BDUMP("ctx", &ctx);
    dav2_tensor_t attn = dav2_tensor_new(n_tokens, ED);
    dav2_qgemm(&ctx, &bw.proj, &attn);
    BDUMP("attn", &attn);

    dav2_tensor_t sum1 = dav2_tensor_new(n_tokens, ED);
    dav2_add(x, &attn, &sum1);
    memcpy(x->v, sum1.v, (size_t)n_tokens * ED * sizeof(int16_t));
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
    memcpy(x->v, sum2.v, (size_t)n_tokens * ED * sizeof(int16_t));
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
    memcpy(t.v, x->v, (size_t)h * w * x->c * sizeof(int16_t));
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
    memcpy(cur.v, a->v, (size_t)h * w * DAV2_FEATURES * sizeof(int16_t));
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
    memcpy(out.v, o.v, (size_t)oh * ow * DAV2_FEATURES * sizeof(int16_t));
    out.scale = o.scale;

    dav2_arena_release(mark);
    return out;
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

    /* ---- patch embedding ---------------------------------------------- */
    dav2_progress("patch embedding");

    uint32_t nb = 0;
    const int16_t *img = (const int16_t *)dav2_find("image", &nb);
    const float *img_scale = (const float *)dav2_find("image_scale", 0);
    if (!img || !img_scale) return;

    dav2_tensor_t image;
    image.v = (int16_t *)img;
    image.n = cfg->size * cfg->size;
    image.c = 3;
    image.scale = *img_scale;

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
            memcpy(feats[j].v, nrm.v + ED,
                   (size_t)n_patch * ED * sizeof(int16_t));
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
}
