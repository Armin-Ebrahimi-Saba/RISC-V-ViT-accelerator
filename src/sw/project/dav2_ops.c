/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Quantised kernels. See dav2.h for the numeric contract.
 */

#include "dav2.h"
#include "dav2_accel.h"
#include "dav2_mathf.h"

#include <string.h>

/* Build with -DDAV2_TRACE to print the dynamic range of every intermediate
 * tensor. Invaluable when a quantised network goes wrong, because the failure
 * always shows up first as a scale that runs away. */
#ifdef DAV2_TRACE
#include <stdio.h>
double dav2_mac_count = 0.0;      /* total multiply-accumulates, for costing */
double dav2_elem_count = 0.0;     /* total requantised output elements */
static const char *trace_tag = "?";
void dav2_trace_tag(const char *t) { trace_tag = t; }
static void trace_tensor(const char *op, const dav2_tensor_t *t)
{
    int32_t lo = 0, hi = 0;
    const int total = t->n * t->c;
    for (int i = 0; i < total; i++) {
        if (t->v[i] > hi) hi = t->v[i];
        if (t->v[i] < lo) lo = t->v[i];
    }
    printf("    %-12s %-8s n=%-6d c=%-5d q=[%6d,%6d] scale=%.6g real_max=%.6g\n",
           trace_tag, op, t->n, t->c, lo, hi, (double)t->scale,
           (double)((hi > -lo ? hi : -lo) * t->scale));
}
/* Dump a dequantised tensor for offline comparison against the numpy
 * blueprint. Host-only; the target build compiles this away. */
void dav2_dump(const char *path, const dav2_tensor_t *t)
{
    FILE *f = fopen(path, "wb");
    if (!f) return;
    const int total = t->n * t->c;
    for (int i = 0; i < total; i++) {
        float v = (float)t->v[i] * t->scale;
        fwrite(&v, sizeof(float), 1, f);
    }
    fclose(f);
    printf("    dumped %s (%d x %d)\n", path, t->n, t->c);
}
#else
#define trace_tensor(op, t) ((void)0)
void dav2_trace_tag(const char *t) { (void)t; }
void dav2_dump(const char *path, const dav2_tensor_t *t) { (void)path; (void)t; }
#endif

/* ------------------------------------------------------------------ arena */

static uint8_t *arena_base;
static size_t   arena_size, arena_used, arena_high;

/* Set when an allocation fails. On the target a NULL dereference just hangs
 * the core, which is expensive to diagnose over JTAG, so the failure is
 * recorded and checked by the caller instead. */
int dav2_arena_failed;

void dav2_arena_init(void *base, size_t bytes)
{
    arena_base = (uint8_t *)base;
    arena_size = bytes;
    arena_used = 0;
    arena_high = 0;
    dav2_arena_failed = 0;
}

void *dav2_arena_alloc(size_t bytes)
{
    bytes = (bytes + 15u) & ~(size_t)15u;
    if (arena_used + bytes > arena_size) {
        dav2_arena_failed = 1;
        return 0;
    }
    void *p = arena_base + arena_used;
    arena_used += bytes;
    if (arena_used > arena_high)
        arena_high = arena_used;
    return p;
}

size_t dav2_arena_mark(void)        { return arena_used; }
void   dav2_arena_release(size_t m) { arena_used = m; }
size_t dav2_arena_peak(void)        { return arena_high; }

dav2_tensor_t dav2_tensor_new(int n, int c)
{
    dav2_tensor_t t;
    t.n = n;
    t.c = c;
    t.scale = 1.0f;
    t.v = (int16_t *)dav2_arena_alloc((size_t)n * (size_t)c * sizeof(int16_t));
    return t;
}

/* ---------------------------------------------------- fixed-point helpers */

/* Split a positive float multiplier into (mult, shift) with mult in
 * [2^30, 2^31), so that round(x*m) == (x*mult + rounding) >> shift. */
void dav2_make_multiplier(float m, int32_t *mult, int *shift)
{
    if (m <= 0.0f) { *mult = 0; *shift = 0; return; }
    int e;
    float f = dav2_frexpf(m, &e);                 /* m = f * 2^e, f in [0.5,1) */
    int64_t q = (int64_t)(f * 2147483648.0f + 0.5f);
    if (q > 2147483647LL) { q >>= 1; e += 1; }
    int sh = 31 - e;
    if (sh < 0) {                                  /* |m| >= 1 */
        if (-sh >= 31) { q = 2147483647LL; sh = 0; }
        else { q >>= (-sh); sh = 0; }
    } else if (sh > 62) {                          /* underflows to zero */
        q = 0; sh = 0;
    }
    *mult = (int32_t)q;
    *shift = sh;
}

static inline int32_t apply_multiplier(int32_t acc, int32_t mult, int shift)
{
    int64_t p = (int64_t)acc * (int64_t)mult;
    if (shift > 0)
        p += ((int64_t)1 << (shift - 1));
    return (int32_t)(p >> shift);
}

static inline int16_t sat_act(int32_t v)
{
    if (v >  DAV2_ACT_QMAX) return (int16_t) DAV2_ACT_QMAX;
    if (v < -DAV2_ACT_QMAX) return (int16_t)-DAV2_ACT_QMAX;
    return (int16_t)v;
}

static inline int32_t iround(float f)
{
    return (int32_t)(f >= 0.0f ? f + 0.5f : f - 0.5f);
}

void dav2_quantize_f32(const float *src, int n, int c, dav2_tensor_t *out)
{
    const int total = n * c;
    float amax = 0.0f;
    for (int i = 0; i < total; i++) {
        float a = dav2_fabsf(src[i]);
        if (a > amax) amax = a;
    }
    float s = (amax > 0.0f) ? amax * (1.0f / (float)DAV2_ACT_QMAX) : 1.0f;
    float inv = 1.0f / s;
    for (int i = 0; i < total; i++)
        out->v[i] = sat_act(iround(src[i] * inv));
    out->scale = s;
    out->n = n;
    out->c = c;
}

/* ------------------------------------------------------------------- GEMM */

/* Activation rows are staged in this buffer so the inner loop reads them from
 * fast BRAM while the weight row streams sequentially from DDR3. Sequential
 * weight access matters: the DDR3 last-level cache is direct-mapped. */
#define TILE_A_ELEMS 8192                       /* 16 KB of BRAM */
static int16_t tile_a[TILE_A_ELEMS];

/* Software reference kernel. acc is in [m][n] order -- see dav2_qgemm(). */
void dav2_qgemm_cpu(const int16_t *av, const int8_t *w, int32_t *acc,
                    int N, int K, int M)
{
    int rows_per_tile = TILE_A_ELEMS / K;
    if (rows_per_tile < 1) rows_per_tile = 1;    /* K > tile: read from DDR3 */

    for (int n0 = 0; n0 < N; n0 += rows_per_tile) {
        int nt = N - n0;
        if (nt > rows_per_tile) nt = rows_per_tile;

        const int16_t *arows;
        if ((size_t)nt * K <= TILE_A_ELEMS) {
            memcpy(tile_a, av + (size_t)n0 * K, (size_t)nt * K * sizeof(int16_t));
            arows = tile_a;
        } else {
            arows = av + (size_t)n0 * K;
        }

        for (int m = 0; m < M; m++) {
            const int8_t *wr = w + (size_t)m * K;
            int32_t *orow = acc + (size_t)m * N + n0;
            for (int t = 0; t < nt; t++) {
                const int16_t *ar = arows + (size_t)t * K;
                int32_t s = 0;
                int k = 0;
                /* unrolled by 4: fewer loop branches on a scalar in-order core */
                for (; k + 3 < K; k += 4) {
                    s += (int32_t)ar[k]     * (int32_t)wr[k];
                    s += (int32_t)ar[k + 1] * (int32_t)wr[k + 1];
                    s += (int32_t)ar[k + 2] * (int32_t)wr[k + 2];
                    s += (int32_t)ar[k + 3] * (int32_t)wr[k + 3];
                }
                for (; k < K; k++)
                    s += (int32_t)ar[k] * (int32_t)wr[k];
                orow[t] = s;
            }
        }
    }
}

/* out[n][m] = requant(sum_k a[n][k] * w[m][k] + bias[m])
 *
 * The int32 accumulator block is held in [m][n] order. That is the layout the
 * accelerator produces -- it emits one contiguous run of rows per output
 * channel, which keeps its write bursts sequential -- and it also suits the
 * CPU, because the requantisation pass below can then hoist mult/shift/bias
 * out of the inner loop. The software kernel uses the same layout so that
 * everything downstream is bit-identical whichever path ran. */
void dav2_qgemm(const dav2_tensor_t *a, const dav2_qw_t *wt, dav2_tensor_t *out)
{
    const int N = a->n, K = a->c, M = wt->m;
#ifdef DAV2_TRACE
    dav2_mac_count += (double)N * (double)K * (double)M;
    dav2_elem_count += (double)N * (double)M;
#endif

    const size_t mark = dav2_arena_mark();
    int32_t *acc    = (int32_t *)dav2_arena_alloc((size_t)N * M * sizeof(int32_t));
    int32_t *mult   = (int32_t *)dav2_arena_alloc((size_t)M * sizeof(int32_t));
    int32_t *biasq  = (int32_t *)dav2_arena_alloc((size_t)M * sizeof(int32_t));
    int     *shift  = (int     *)dav2_arena_alloc((size_t)M * sizeof(int));
    if (!acc || !mult || !biasq || !shift) {
        /* dav2_arena_failed is set; unwinding here beats faulting on NULL,
         * which on the target just hangs the core. */
        dav2_arena_release(mark);
        out->n = N;
        out->c = M;
        out->scale = 1.0f;
        return;
    }

    if (!dav2_accel_qgemm(a, wt, acc))
        dav2_qgemm_cpu(a->v, wt->w, acc, N, K, M);

    /* Exact output range, including bias, so nothing clips. The min/max scan
     * is O(N*M) against the O(N*M*K) product, so it stays in software even
     * when the accelerator ran. */
    float amax = 0.0f;
    for (int m = 0; m < M; m++) {
        const int32_t *ar = acc + (size_t)m * N;
        int32_t cmax = ar[0], cmin = ar[0];
        for (int n = 1; n < N; n++) {
            if (ar[n] > cmax) cmax = ar[n];
            if (ar[n] < cmin) cmin = ar[n];
        }
        float k_c = a->scale * wt->s[m];
        float bias = wt->b ? wt->b[m] : 0.0f;
        float hi = (float)cmax * k_c + bias;
        float lo = (float)cmin * k_c + bias;
        float ah = dav2_fabsf(hi), al = dav2_fabsf(lo);
        if (ah > amax) amax = ah;
        if (al > amax) amax = al;
    }
    float out_scale = (amax > 0.0f) ? amax * (1.0f / (float)DAV2_ACT_QMAX) : 1.0f;
    float inv_out = 1.0f / out_scale;

    for (int m = 0; m < M; m++) {
        dav2_make_multiplier(a->scale * wt->s[m] * inv_out, &mult[m], &shift[m]);
        biasq[m] = wt->b ? iround(wt->b[m] * inv_out) : 0;
    }

    for (int m = 0; m < M; m++) {
        const int32_t *ar = acc + (size_t)m * N;
        int16_t *ocol = out->v + m;
        const int32_t mu = mult[m], bq = biasq[m];
        const int sh = shift[m];
        for (int n = 0; n < N; n++)
            ocol[(size_t)n * M] = sat_act(apply_multiplier(ar[n], mu, sh) + bq);
    }

    out->n = N;
    out->c = M;
    out->scale = out_scale;
    trace_tensor("qgemm", out);
    dav2_arena_release(mark);
}

/* -------------------------------------------------------------- LayerNorm */

void dav2_layernorm(const dav2_tensor_t *in, const float *g, const float *b,
                    dav2_tensor_t *out)
{
    const int N = in->n, C = in->c;
    const size_t mark = dav2_arena_mark();
    float *tmp = (float *)dav2_arena_alloc((size_t)N * C * sizeof(float));

    const float s = in->scale;
    for (int n = 0; n < N; n++) {
        const int16_t *row = in->v + (size_t)n * C;
        int32_t sum = 0;
        for (int c = 0; c < C; c++)
            sum += row[c];
        /* mean/variance in the integer domain, then converted once per row */
        float mean_q = (float)sum / (float)C;
        int64_t sq = 0;
        for (int c = 0; c < C; c++) {
            int32_t d = (int32_t)row[c];
            sq += (int64_t)d * (int64_t)d;
        }
        /* sq <= 384 * 8191^2 = 2.6e10, so sq/C fits comfortably in int32 */
        float mean_sq = (float)(int32_t)(sq / C)
                      + (float)(int32_t)(sq % C) / (float)C;
        float var_q = mean_sq - mean_q * mean_q;
        if (var_q < 0.0f) var_q = 0.0f;
        /* variance in real units = var_q * s^2; eps matches PyTorch's 1e-6 */
        float inv_std = 1.0f / dav2_sqrtf(var_q * s * s + 1e-6f);
        float *orow = tmp + (size_t)n * C;
        for (int c = 0; c < C; c++)
            orow[c] = ((float)row[c] - mean_q) * s * inv_std * g[c] + b[c];
    }

    dav2_quantize_f32(tmp, N, C, out);
    trace_tensor("layernorm", out);
    dav2_arena_release(mark);
}

/* ----------------------------------------------------------- elementwise */

void dav2_add(const dav2_tensor_t *a, const dav2_tensor_t *b, dav2_tensor_t *out)
{
    const int total = a->n * a->c;
    /* Upper bound on the sum's magnitude; at most one bit of range is lost. */
    int32_t amax_a = 0, amax_b = 0;
    for (int i = 0; i < total; i++) {
        int32_t va = a->v[i] < 0 ? -a->v[i] : a->v[i];
        int32_t vb = b->v[i] < 0 ? -b->v[i] : b->v[i];
        if (va > amax_a) amax_a = va;
        if (vb > amax_b) amax_b = vb;
    }
    float bound = (float)amax_a * a->scale + (float)amax_b * b->scale;
    float out_scale = (bound > 0.0f) ? bound * (1.0f / (float)DAV2_ACT_QMAX) : 1.0f;
    float inv = 1.0f / out_scale;

    int32_t ma, mb; int sa, sb;
    dav2_make_multiplier(a->scale * inv, &ma, &sa);
    dav2_make_multiplier(b->scale * inv, &mb, &sb);

    for (int i = 0; i < total; i++) {
        int32_t v = apply_multiplier(a->v[i], ma, sa)
                  + apply_multiplier(b->v[i], mb, sb);
        out->v[i] = sat_act(v);
    }
    out->n = a->n;
    out->c = a->c;
    out->scale = out_scale;
    trace_tensor("add", out);
}

void dav2_relu(dav2_tensor_t *t)
{
    const int total = t->n * t->c;
    for (int i = 0; i < total; i++)
        if (t->v[i] < 0) t->v[i] = 0;
}

/* GELU is an elementwise map on a 14-bit input, so a 257-entry table with
 * linear interpolation reproduces it to well below quantisation noise. The
 * table costs 257 float evaluations per call, versus 126k for direct
 * evaluation. */
void dav2_gelu(dav2_tensor_t *t)
{
    float lut_f[257];
    const float s = t->scale;
    float amax = 0.0f;
    for (int i = 0; i < 257; i++) {
        float x = (float)(i * 64 - 8192) * s;
        lut_f[i] = dav2_gelu_f(x);
        float a = dav2_fabsf(lut_f[i]);
        if (a > amax) amax = a;
    }
    float out_scale = (amax > 0.0f) ? amax * (1.0f / (float)DAV2_ACT_QMAX) : 1.0f;
    float inv = 1.0f / out_scale;

    int16_t lut[257];
    for (int i = 0; i < 257; i++)
        lut[i] = sat_act(iround(lut_f[i] * inv));

    const int total = t->n * t->c;
    for (int i = 0; i < total; i++) {
        int32_t u = (int32_t)t->v[i] + 8192;     /* [1, 16383] */
        int32_t idx = u >> 6;                    /* [0, 255]   */
        int32_t frac = u & 63;
        int32_t lo = lut[idx], hi = lut[idx + 1];
        t->v[i] = (int16_t)(lo + (((hi - lo) * frac) >> 6));
    }
    t->scale = out_scale;
    trace_tensor("gelu", t);
}

/* ------------------------------------------------------------ resampling */

void dav2_interpolate(const dav2_tensor_t *in, int h, int w,
                      int oh, int ow, dav2_tensor_t *out)
{
    const int C = in->c;
    const size_t mark = dav2_arena_mark();
    int *y0a = (int *)dav2_arena_alloc((size_t)oh * sizeof(int));
    int *y1a = (int *)dav2_arena_alloc((size_t)oh * sizeof(int));
    int *wya = (int *)dav2_arena_alloc((size_t)oh * sizeof(int));
    int *x0a = (int *)dav2_arena_alloc((size_t)ow * sizeof(int));
    int *x1a = (int *)dav2_arena_alloc((size_t)ow * sizeof(int));
    int *wxa = (int *)dav2_arena_alloc((size_t)ow * sizeof(int));

    /* align_corners=true, matching F.interpolate. Weights in Q8. */
    float sy = (oh > 1) ? (float)(h - 1) / (float)(oh - 1) : 0.0f;
    float sx = (ow > 1) ? (float)(w - 1) / (float)(ow - 1) : 0.0f;
    for (int i = 0; i < oh; i++) {
        float f = (float)i * sy;
        int i0 = (int)f;
        if (i0 > h - 1) i0 = h - 1;
        int i1 = (i0 + 1 < h) ? i0 + 1 : h - 1;
        y0a[i] = i0; y1a[i] = i1;
        wya[i] = iround((f - (float)i0) * 256.0f);
    }
    for (int j = 0; j < ow; j++) {
        float f = (float)j * sx;
        int j0 = (int)f;
        if (j0 > w - 1) j0 = w - 1;
        int j1 = (j0 + 1 < w) ? j0 + 1 : w - 1;
        x0a[j] = j0; x1a[j] = j1;
        wxa[j] = iround((f - (float)j0) * 256.0f);
    }

    for (int i = 0; i < oh; i++) {
        const int16_t *r0 = in->v + (size_t)y0a[i] * w * C;
        const int16_t *r1 = in->v + (size_t)y1a[i] * w * C;
        int wy = wya[i];
        int16_t *orow = out->v + (size_t)i * ow * C;
        for (int j = 0; j < ow; j++) {
            const int16_t *a = r0 + (size_t)x0a[j] * C;
            const int16_t *b = r0 + (size_t)x1a[j] * C;
            const int16_t *c = r1 + (size_t)x0a[j] * C;
            const int16_t *d = r1 + (size_t)x1a[j] * C;
            int wx = wxa[j];
            int16_t *o = orow + (size_t)j * C;
            for (int ch = 0; ch < C; ch++) {
                int32_t top = ((int32_t)a[ch] * (256 - wx) + (int32_t)b[ch] * wx) >> 8;
                int32_t bot = ((int32_t)c[ch] * (256 - wx) + (int32_t)d[ch] * wx) >> 8;
                o[ch] = (int16_t)((top * (256 - wy) + bot * wy) >> 8);
            }
        }
    }
    out->n = oh * ow;
    out->c = C;
    out->scale = in->scale;
    dav2_arena_release(mark);
}

/* --------------------------------------------------------- convolutions */

void dav2_im2col(const dav2_tensor_t *in, int h, int w,
                 int kh, int kw, int stride, int pad, dav2_tensor_t *cols)
{
    const int C = in->c;
    const int oh = (h + 2 * pad - kh) / stride + 1;
    const int ow = (w + 2 * pad - kw) / stride + 1;
    const int K = kh * kw * C;
    int16_t *dst = cols->v;

    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            for (int ky = 0; ky < kh; ky++) {
                int iy = oy * stride + ky - pad;
                for (int kx = 0; kx < kw; kx++) {
                    int ix = ox * stride + kx - pad;
                    if (iy < 0 || iy >= h || ix < 0 || ix >= w) {
                        memset(dst, 0, (size_t)C * sizeof(int16_t));
                    } else {
                        memcpy(dst, in->v + ((size_t)iy * w + ix) * C,
                               (size_t)C * sizeof(int16_t));
                    }
                    dst += C;
                }
            }
        }
    }
    cols->n = oh * ow;
    cols->c = K;
    cols->scale = in->scale;
}

dav2_tensor_t dav2_conv2d(const dav2_tensor_t *in, int h, int w,
                          const dav2_qw_t *wt, int k, int stride, int pad,
                          int *oh_out, int *ow_out)
{
    const int oh = (h + 2 * pad - k) / stride + 1;
    const int ow = (w + 2 * pad - k) / stride + 1;

    dav2_tensor_t out = dav2_tensor_new(oh * ow, wt->m);
    const size_t mark = dav2_arena_mark();
    dav2_tensor_t cols = dav2_tensor_new(oh * ow, k * k * in->c);
    dav2_im2col(in, h, w, k, k, stride, pad, &cols);
    dav2_qgemm(&cols, wt, &out);
    dav2_arena_release(mark);

    if (oh_out) *oh_out = oh;
    if (ow_out) *ow_out = ow;
    return out;
}

dav2_tensor_t dav2_conv_transpose(const dav2_tensor_t *in, int h, int w,
                                  const dav2_qw_t *wt, int stride,
                                  int *oh_out, int *ow_out)
{
    /* wt is (stride*stride*Cout) x Cin: each input pixel expands into a
     * contiguous (ky, kx, cout) run, which is then scattered into the NHWC
     * output image. */
    const int cout = wt->m / (stride * stride);
    const int oh = h * stride, ow = w * stride;

    dav2_tensor_t out = dav2_tensor_new(oh * ow, cout);
    const size_t mark = dav2_arena_mark();
    dav2_tensor_t flat = dav2_tensor_new(h * w, wt->m);
    dav2_qgemm(in, wt, &flat);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const int16_t *src = flat.v + ((size_t)y * w + x) * wt->m;
            for (int ky = 0; ky < stride; ky++) {
                int16_t *dst = out.v + ((size_t)(y * stride + ky) * ow
                                        + (size_t)x * stride) * cout;
                memcpy(dst, src + (size_t)ky * stride * cout,
                       (size_t)stride * cout * sizeof(int16_t));
            }
        }
    }
    out.scale = flat.scale;
    dav2_arena_release(mark);

    if (oh_out) *oh_out = oh;
    if (ow_out) *ow_out = ow;
    return out;
}
