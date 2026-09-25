/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Quantised kernels. See dav2.h for the numeric contract.
 *
 * Everything the network is built from lives here: the arena allocator, the
 * fixed-point helpers, GEMM, LayerNorm, GELU, residual add and the bilinear
 * resampler. dav2_engine.c composes these into layers; it never touches a
 * number itself.
 *
 * Terms:
 *   quantised   stored as a small integer plus one float scale: real = q*scale
 *   requantise  turn an int32 accumulator (the sum of many int8*int16
 *               products) back into a 14-bit int16 activation, using a
 *               precomputed (multiplier, shift) pair instead of a float divide
 *   GEMM        general matrix multiply, C = A * W^T; every linear layer,
 *               attention projection and MLP is one
 *   arena       bump allocator over a DDR3 region: alloc moves a pointer up,
 *               release moves it back to a saved mark
 *
 * Every GEMM goes through dav2_qgemm(), which asks the accelerator first and
 * falls back to dav2_qgemm_cpu() if it declines. Both produce the same int32
 * accumulators, so the requantisation after them is common code and the
 * result is bit-identical whichever ran.
 */

/* The build flow compiles every file with -Os. That is right for the boot
 * code and wrong for these loops, which are where the CPU spends every cycle
 * it does not spend waiting for the accelerator. -O2 here only; program
 * memory (224 kB) has room. */
#pragma GCC optimize ("O2")

#include "dav2.h"
#include "dav2_accel.h"
#include "dav2_gelu_phi.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------- profiler */

const char *const dav2_prof_name[DAV2_PROF_N] = {
    "gemm (accelerator)", "gemm (cpu)", "requantise", "attention",
    "layernorm", "gelu", "add/relu", "im2col", "interpolate", "other",
};
static uint64_t prof_acc[DAV2_PROF_N];
static uint64_t sub_acc[DAV2_SUB_N];
void     dav2_prof_reset(void)
{
    memset(prof_acc, 0, sizeof prof_acc);
    memset(sub_acc, 0, sizeof sub_acc);
}
void     dav2_prof_add(int b, uint64_t c)      { prof_acc[b] += c; }
uint64_t dav2_prof_get(int b)                  { return prof_acc[b]; }
const char *const dav2_sub_name[DAV2_SUB_N] = {
    "ln stats", "ln gamma/beta", "ln requant", "rq range", "rq params",
    "att prep q,k", "att prep v", "att softmax", "att normalise", "att wait",
    "gelu table",
};
void     dav2_sub_add(int d, uint64_t c)       { sub_acc[d] += c; }
uint64_t dav2_sub_get(int d)                   { return sub_acc[d]; }
#define SUB_START()       uint64_t sub_t0 = dav2_cycles()
#define SUB_LAP(d)        do { uint64_t sub_t1 = dav2_cycles(); \
                               dav2_sub_add((d), sub_t1 - sub_t0); sub_t0 = sub_t1; } while (0)
#define PROF_START()      uint64_t prof_t0 = dav2_cycles()
#define PROF_STOP(b)      dav2_prof_add((b), dav2_cycles() - prof_t0)
/* ------------------------------------------------------ copies, scratch */

/* libsys's memcpy copies one byte per iteration, and on this SoC the CPU has
 * no data cache of its own: every byte is a full bus transaction to the DDR3
 * block cache. The engine copies tens of megabytes per frame (im2col alone
 * gathers 36 MB), so a word-wise copy is a 4x win on all of it. Both ends
 * are int16 tensors from a 16-byte aligned arena, so the word path is the
 * common case; the byte tail handles odd element counts. */
void dav2_copy16(int16_t *dst, const int16_t *src, size_t n)
{
    if ((((uintptr_t)dst | (uintptr_t)src) & 3u) == 0) {
        uint32_t *d = (uint32_t *)dst;
        const uint32_t *q = (const uint32_t *)src;
        size_t w = n >> 1;
        for (; w >= 4; w -= 4) {
            uint32_t a = q[0], b = q[1], c = q[2], e = q[3];
            d[0] = a; d[1] = b; d[2] = c; d[3] = e;
            d += 4; q += 4;
        }
        for (; w; w--) *d++ = *q++;
        if (n & 1) *(int16_t *)d = *(const int16_t *)q;
        return;
    }
    for (; n; n--) *dst++ = *src++;
}

void dav2_zero16(int16_t *dst, size_t n)
{
    if (((uintptr_t)dst & 3u) == 0) {
        uint32_t *d = (uint32_t *)dst;
        for (size_t w = n >> 1; w; w--) *d++ = 0;
        if (n & 1) *(int16_t *)d = 0;
        return;
    }
    for (; n; n--) *dst++ = 0;
}

/* On-chip scratch memory. The program's own RAM is block RAM with single-
 * cycle access; the activation arena is DDR3 behind a 16 kB direct-mapped
 * cache. Anything reused many times inside an operator -- a tile of
 * activations, the q/k/v of one attention head -- is copied here first.
 * Users never overlap in time, so one buffer serves them all. */
int16_t dav2_scratch[DAV2_SCRATCH_ELEMS] __attribute__((aligned(16)));

#define PROF_LAP(b)       do { uint64_t prof_t1 = dav2_cycles(); \
                               dav2_prof_add((b), prof_t1 - prof_t0); \
                               prof_t0 = prof_t1; } while (0)

/* Build with -DDAV2_TRACE to print the dynamic range of every intermediate
 * tensor. Invaluable when a quantised network goes wrong, because the failure
 * always shows up first as a scale that runs away. */
#ifdef DAV2_TRACE
#include <stdio.h>
#include <math.h>
/* Host-only: a scale as a double, for printing. */
static double xf_d(dav2_xf_t a) { return ldexp((double)a.m, -a.sh); }
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
           trace_tag, op, t->n, t->c, lo, hi, xf_d(t->scale),
           (double)(hi > -lo ? hi : -lo) * xf_d(t->scale));
}
/* Dump a dequantised tensor for offline comparison against the numpy
 * blueprint. Host-only; the target build compiles this away. */
void dav2_dump(const char *path, const dav2_tensor_t *t)
{
    FILE *f = fopen(path, "wb");
    if (!f) return;
    const int total = t->n * t->c;
    for (int i = 0; i < total; i++) {
        float v = (float)((double)t->v[i] * xf_d(t->scale));
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
    t.scale = XF_ONE;
    t.amax_q = -1;
    t.v = (int16_t *)dav2_arena_alloc((size_t)n * (size_t)c * sizeof(int16_t));
    return t;
}

/* ---------------------------------------------------- fixed-point helpers */


static inline int32_t apply_multiplier(int32_t acc, int32_t mult, int shift)
{
    /* round(acc * mult / 2^shift), exactly, without a 64-bit shift.
     *
     * The multiplier is a normalised dav2_xf_t mantissa, in [2^30, 2^31),
     * so the shift is 31..62 for factors below 1 and smaller for larger
     * factors. Two straight-line cases cover everything this engine does;
     * on the CV32E40P a taken branch costs ~4 cycles, as much as the
     * multiply itself, so both are written without one.
     *
     * shift >= 33: the rounding constant 2^(shift-1) lies entirely in the
     * high word of the product, and the low word only adds a fraction below
     * one to a value that is then floored:
     *     (acc*mult + 2^(s-1)) >> s  ==  (hi + 2^(s-33)) >> (s-32).
     *
     * 1 <= shift <= 32 (the residual adds, factors near 1): add the rounding
     * constant to the low word, carry into the high word, and assemble the
     * shifted result from both halves. ">> (s-1) >> 1" is ">> s" that is
     * also defined for s == 32. */
    int32_t  hi = (int32_t)(((int64_t)acc * (int64_t)mult) >> 32);   /* mulh */
    if (shift >= 33)
        return (hi + (1 << (shift - 33))) >> (shift - 32);
    uint32_t lo  = (uint32_t)acc * (uint32_t)mult;                   /* mul  */
    if (shift == 0)                       /* factor >= 1 with no fraction bits */
        return (int32_t)lo;
    uint32_t lo2 = lo + (1u << (shift - 1));
    hi += (lo2 < lo);
    return (int32_t)(((uint32_t)hi << (32 - shift)) | ((lo2 >> (shift - 1)) >> 1));
}

/* The elementwise operators below read and write two int16 per 32-bit word
 * wherever the buffers allow it (they always do: the arena is 16-byte aligned
 * and every tensor here has an even element count). On this core each load
 * or store is a bus transaction of ~8 cycles, so halving them is the single
 * biggest lever these loops have. Results are the same as the scalar code. */
static inline int32_t lo16(uint32_t w) { return (int32_t)(int16_t)(w & 0xffffu); }
static inline int32_t hi16(uint32_t w) { return (int32_t)(int16_t)(w >> 16); }
static inline uint32_t pack16(int32_t lo, int32_t hi)
{
    return ((uint32_t)(uint16_t)lo) | ((uint32_t)(uint16_t)hi << 16);
}
static inline int words_ok(const void *p, const void *q, const void *r, int total)
{
    return ((((uintptr_t)p | (uintptr_t)q | (uintptr_t)r) & 3u) == 0) && (total & 1) == 0;
}

static inline int16_t sat_act(int32_t v)
{
    if (v >  DAV2_ACT_QMAX) return (int16_t) DAV2_ACT_QMAX;
    if (v < -DAV2_ACT_QMAX) return (int16_t)-DAV2_ACT_QMAX;
    return (int16_t)v;
}



/* ------------------------------------------------------------ embedding */

/* v * s * 2^k rounded half away from zero, v an int16-range value. */
static inline int64_t mul_xf_round(int32_t v, dav2_xf_t s, int k)
{
    int64_t p = (int64_t)v * s.m;
    int32_t sh = s.sh - k;
    if (sh <= 0)
        return p * ((int64_t)1 << (-sh > 20 ? 20 : -sh));
    if (sh > 62)
        return 0;
    int64_t a = p < 0 ? -p : p;
    a = (a + ((int64_t)1 << (sh - 1))) >> sh;
    return p < 0 ? -a : a;
}

void dav2_embed_tokens(const dav2_tensor_t *patches, const int32_t *cls,
                       const int32_t *pos, dav2_tensor_t *out)
{
    const int C = patches->c, NP = patches->n, N = NP + 1;
    const size_t mark = dav2_arena_mark();

    /* Every token value in Q24 (DAV2_POS_Q), as int64: row 0 is
     * cls + pos[0], row p+1 is patch[p] * scale + pos[p+1]. */
    int64_t *v = (int64_t *)dav2_arena_alloc((size_t)N * C * sizeof(int64_t));
    if (!v) { dav2_arena_release(mark); return; }
    int64_t amax = 0;
    for (int i = 0; i < N * C; i++) {
        int64_t x = (i < C) ? (int64_t)cls[i]
                            : mul_xf_round(patches->v[i - C], patches->scale, DAV2_POS_Q);
        x += pos[i];
        v[i] = x;
        int64_t a = x < 0 ? -x : x;
        if (a > amax) amax = a;
    }

    /* Common output scale amax / 8191. The values are brought into int32
     * range first (a right shift by r, normally 0), so apply_multiplier can
     * scale them. */
    int r = 0;
    while ((amax >> r) >= ((int64_t)1 << 30))
        r++;
    dav2_xf_t out_scale = amax ? xf_div(xf_norm(amax, DAV2_POS_Q),
                                        xf_from_int(DAV2_ACT_QMAX)) : XF_ONE;
    int32_t om; int osh;
    xf_to_mult(xf_div(xf_norm(1, DAV2_POS_Q - r), out_scale), &om, &osh);
    int32_t omax = 0;
    for (int i = 0; i < N * C; i++) {
        int64_t x = v[i];
        if (r) {
            int64_t a = x < 0 ? -x : x;
            a = (a + ((int64_t)1 << (r - 1))) >> r;
            x = x < 0 ? -a : a;
        }
        int32_t q = sat_act(apply_multiplier((int32_t)x, om, osh));
        out->v[i] = (int16_t)q;
        if (q < 0) q = -q;
        if (q > omax) omax = q;
    }
    out->n = N;
    out->c = C;
    out->scale = out_scale;
    out->amax_q = omax;
    dav2_arena_release(mark);
}

/* ------------------------------------------------------------------- GEMM */

/* Activation rows are staged in the on-chip scratch so the inner loop reads
 * them from fast BRAM while the weight row streams sequentially from DDR3.
 * Sequential weight access matters: the DDR3 last-level cache is
 * direct-mapped. */
#define TILE_A_ELEMS DAV2_SCRATCH_ELEMS
#define tile_a       dav2_scratch

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
            dav2_copy16(tile_a, av + (size_t)n0 * K, (size_t)nt * K);
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
/* Largest |value| of a tensor, the scan dav2_add does when its producer did
 * not record amax_q. */
static int32_t amax_scan(const dav2_tensor_t *t)
{
    const int total = t->n * t->c;
    int32_t m = 0;
    for (int i = 0; i < total; i++) {
        int32_t v = t->v[i] < 0 ? -t->v[i] : t->v[i];
        if (v > m) m = v;
    }
    return m;
}

/* The parameters of out = a + b, from the operands' largest |values| and
 * scales. The sum is bounded by amax_a*sa + amax_b*sb, which becomes the
 * output's full range; ma/sha and mb/shb rescale a and b to the output
 * scale. dav2_add and the requantisation job's add epilogue both use this,
 * so the fused and the separate add agree to the bit. */
static void add_params(int32_t amax_a, dav2_xf_t sa, int32_t amax_b, dav2_xf_t sb,
                       dav2_xf_t *out_scale, int32_t *ma, int *sha,
                       int32_t *mb, int *shb)
{
    dav2_xf_t bound = xf_add(xf_mul(xf_from_int(amax_a), sa),
                             xf_mul(xf_from_int(amax_b), sb));
    *out_scale = bound.m ? xf_div(bound, xf_from_int(DAV2_ACT_QMAX)) : XF_ONE;
    dav2_xf_t inv = xf_recip(*out_scale);
    xf_to_mult(xf_mul(sa, inv), ma, sha);
    xf_to_mult(xf_mul(sb, inv), mb, shb);
}

/* The mantissa of a (m, sh) pair at the common exponent e <= sh:
 * m * 2^-(sh - e), truncated. */
static inline int32_t align_mant(dav2_xf_t v, int e)
{
    int d = v.sh - e;
    return d > 31 ? (v.m < 0 ? -1 : 0) : (v.m >> d);
}

/* round(v * 2^-s), half up, for |v| < 2^62. s <= 0 shifts left, saturated
 * at +-2^62. */
static inline int64_t shr_round64(int64_t v, int s)
{
    if (s > 62)
        return 0;
    if (s > 0)
        return (v + ((int64_t)1 << (s - 1))) >> s;
    const int64_t lim = (int64_t)1 << 62;
    for (; s < 0; s++) {
        if (v >= lim / 2) return lim;
        if (v <= -lim / 2) return -lim;
        v *= 2;
    }
    return v;
}

/* the high word of a * b */
static inline int32_t mulh32(int32_t a, int32_t b)
{
    return (int32_t)(((int64_t)a * (int64_t)b) >> 32);
}

static inline int32_t sat_i32(int64_t v)
{
    return v > 2147483647 ? 2147483647 : v < -2147483647 ? -2147483647 : (int32_t)v;
}

/* A convolution's geometry, for a GEMM whose A matrix is the im2col of an
 * image rather than a tensor in memory. */
typedef struct { int h, w, k, stride, pad; } conv_desc_t;

static void qgemm_impl(const dav2_tensor_t *a, const conv_desc_t *cv,
                       const dav2_qw_t *wt, const dav2_tensor_t *res, int relu,
                       int gelu, dav2_tensor_t *out);
static int16_t *gelu_table(dav2_xf_t in_scale, dav2_xf_t *out_scale_ret);

void dav2_qgemm(const dav2_tensor_t *a, const dav2_qw_t *wt, dav2_tensor_t *out)
{
    qgemm_impl(a, 0, wt, 0, 0, 0, out);
}

/* dav2_qgemm then dav2_gelu, bit-identical; on the accelerator the GELU
 * table is applied inside the requantisation job (CTRL.lut). */
void dav2_qgemm_gelu(const dav2_tensor_t *a, const dav2_qw_t *wt, dav2_tensor_t *out)
{
    qgemm_impl(a, 0, wt, 0, 0, 1, out);
}

/* dav2_qgemm followed by dav2_add(res, result) and/or dav2_relu, with the
 * same result bit for bit. On the accelerator the add and the ReLU happen
 * inside the requantisation job (its epilogue), so neither the
 * intermediate tensor nor the CPU pass over it exists. out may be res. */
void dav2_qgemm_ex(const dav2_tensor_t *a, const dav2_qw_t *wt,
                   const dav2_tensor_t *res, int relu, dav2_tensor_t *out)
{
    qgemm_impl(a, 0, wt, res, relu, 0, out);
}

/* a is the input image when cv is set: N = output pixels, K = k*k*C.
 * res, relu: the epilogue, see dav2_qgemm_ex. */
static void qgemm_impl(const dav2_tensor_t *a, const conv_desc_t *cv,
                       const dav2_qw_t *wt, const dav2_tensor_t *res, int relu,
                       int gelu, dav2_tensor_t *out)
{
    const int M = wt->m;
    const int N = cv ? ((cv->h + 2 * cv->pad - cv->k) / cv->stride + 1)
                     * ((cv->w + 2 * cv->pad - cv->k) / cv->stride + 1)
                     : a->n;
    const int K = cv ? cv->k * cv->k * a->c : a->c;
#ifdef DAV2_TRACE
    dav2_mac_count += (double)N * (double)K * (double)M;
    dav2_elem_count += (double)N * (double)M;
#endif

    const size_t mark = dav2_arena_mark();
    int32_t *acc    = (int32_t *)dav2_arena_alloc((size_t)N * M * sizeof(int32_t));
    /* per-row requantisation parameters, {mult, shift, bias} interleaved:
     * the layout the accelerator's requantisation job reads */
    int32_t *par    = (int32_t *)dav2_arena_alloc((size_t)M * 3 * sizeof(int32_t));
    /* with a residual: each row's extreme accumulators, kept for the add */
    int32_t *rmax   = res ? (int32_t *)dav2_arena_alloc((size_t)M * 2 * sizeof(int32_t)) : 0;
    int32_t *rmin   = rmax ? rmax + M : 0;
    if (!acc || !par || (res && !rmax)) {
        /* dav2_arena_failed is set; unwinding here beats faulting on NULL,
         * which on the target just hangs the core. */
        dav2_arena_release(mark);
        out->n = N;
        out->c = M;
        out->scale = XF_ONE;
        return;
    }

    /* ---- the product ------------------------------------------------------
     *
     * On the accelerator the operation's last job is left running (run == 2)
     * and the per-row range below is computed while it drains: the block
     * writes each weight row's {max, min} as it finishes the row, so the CPU
     * takes row m's range -- a dozen soft-float operations -- while rows
     * m+1.. are still being computed. Profile: "gemm (accelerator)" counts
     * only the time the CPU actually waits for the block. */
    uint64_t t_start = dav2_cycles(), waited = 0;
    dav2_accel_stats_t st;
    int run;
    if (cv && (run = dav2_accel_conv_async(a->v, cv->h, cv->w, a->c, cv->k, cv->stride,
                                           cv->pad, wt->w, M, acc, &st)) != 0) {
        /* the block gathers the patches itself: no im2col matrix */
    } else {
        dav2_tensor_t cols = *a;
        if (cv) {
            cols = dav2_tensor_new(N, K);
            if (!cols.v) { dav2_arena_release(mark); return; }
            dav2_im2col(a, cv->h, cv->w, cv->k, cv->k, cv->stride, cv->pad, &cols);
            t_start = dav2_cycles();          /* im2col counted in its own bucket */
        }
        run = dav2_accel_qgemm_async(&cols, wt, acc, &st);
        if (!run) {
            dav2_qgemm_cpu(cols.v, wt->w, acc, N, K, M);
            dav2_prof_add(DAV2_PROF_GEMM_CPU, dav2_cycles() - t_start);
            t_start = dav2_cycles();
        }
    }
    if (run == 2 && !st.tiles) {
        /* running, but without statistics to stream: acc is scanned below */
        uint64_t w0 = dav2_cycles();
        if (!dav2_accel_finish()) goto redo_on_cpu;
        waited += dav2_cycles() - w0;
        run = 1;
    }

    /* Row scales and biases at one exponent per matrix:
     *
     *   s[m] = S_m * 2^-es,  b[m] = B_m * 2^-eb,
     *
     * with es, eb the smallest exponents (the largest values) of the matrix.
     * Each row then costs a shift instead of normalised (m, sh) arithmetic,
     * and the range and parameter passes below are a few 64-bit integer
     * products per row. The largest S_m and |B_m| keep all 31 bits; a row
     * 2^k smaller keeps 31 - k. The largest spread in this model is 2^16
     * (blk0.proj); such a row contributes at most 2^-16 of the output range,
     * so its lost bits do not matter. The biases' lost bits are absolute
     * errors below 2^-30 of the largest bias. */
    uint64_t range_t0 = dav2_cycles();
    const uint64_t range_w0 = waited;
    int es = 1 << 30, eb = 1 << 30;
    for (int m = 0; m < M; m++) {
        if (wt->s[m].m && wt->s[m].sh < es) es = wt->s[m].sh;
        if (wt->b && wt->b[m].m && wt->b[m].sh < eb) eb = wt->b[m].sh;
    }
    /* The range is computed in units of V = 2^(32-es) * a->scale: the high
     * word of acc * S_m (mulh), and the bias as mulh(B_m, beta.m) >> beta.sh.
     * 32-bit operations with shifts that are constant for the GEMM; the
     * floor of the high word underestimates by less than one V, about 2^-27
     * of a typical range, and the output saturates at DAV2_ACT_QMAX anyway.
     * A bias exponent outside 0..31 (never in this model) takes the 64-bit
     * path. */
    const dav2_xf_t a_scale = a->scale.m ? a->scale : XF_ONE;
    dav2_xf_t beta = xf_recip(a_scale);
    beta.sh += eb - es;
    const int b_fast = beta.sh >= 0 && beta.sh <= 31;

    /* Exact output range, including bias, so nothing clips. The per-row
     * extremes come from the accelerator's drain when it produced them (two
     * words per row per tile); otherwise from a scan of acc. */
    int64_t vmax = 0, vmin = 0;                 /* in units of V */
    for (int m = 0; m < M; m++) {
        int32_t cmax, cmin;
        if (st.tiles) {
            const volatile int32_t *sv = st.v + (size_t)m * 2;
            if (run == 2) {
                /* the last tile's slot, filled while the job runs */
                const volatile int32_t *lv = sv + (size_t)(st.tiles - 1) * M * 2;
                if (lv[0] == DAV2_STATS_EMPTY_MAX && lv[1] == DAV2_STATS_EMPTY_MIN) {
                    uint64_t w0 = dav2_cycles();
                    while (lv[0] == DAV2_STATS_EMPTY_MAX && lv[1] == DAV2_STATS_EMPTY_MIN
                           && dav2_accel_busy())
                        ;
                    waited += dav2_cycles() - w0;
                }
            }
            cmax = sv[0]; cmin = sv[1];
            for (int t = 1; t < st.tiles; t++) {
                const volatile int32_t *tv = sv + (size_t)t * M * 2;
                if (tv[0] > cmax) cmax = tv[0];
                if (tv[1] < cmin) cmin = tv[1];
            }
        } else {
            const int32_t *ar = acc + (size_t)m * N;
            cmax = ar[0]; cmin = ar[0];
            for (int n = 1; n < N; n++) {
                if (ar[n] > cmax) cmax = ar[n];
                if (ar[n] < cmin) cmin = ar[n];
            }
        }
        if (rmax) { rmax[m] = cmax; rmin[m] = cmin; }
        /* S_m, kept in the row's first parameter word for the pass below */
        const int32_t S = align_mant(wt->s[m], es);
        par[3 * m] = S;
        int32_t bv = 0;
        if (wt->b) {
            int32_t B = align_mant(wt->b[m], eb);
            bv = b_fast ? mulh32(B, beta.m) >> beta.sh
                        : (int32_t)(shr_round64((int64_t)B * beta.m, beta.sh + 32));
        }
        /* S >= 0, so the row's largest value comes from cmax, the smallest
         * from cmin */
        int64_t hv = (int64_t)mulh32(cmax, S) + bv;
        int64_t lv = (int64_t)mulh32(cmin, S) + bv;
        if (hv > vmax) vmax = hv;
        if (lv < vmin) vmin = lv;
    }
    dav2_xf_t amax = xf_mul(xf_norm(vmax > -vmin ? vmax : -vmin, es - 32), a_scale);
    dav2_sub_add(DAV2_SUB_RQ_RANGE, (dav2_cycles() - range_t0) - (waited - range_w0));
    if (run == 2) {
        /* every row's statistics have been read, so the job is done or
         * failed; collect it (and learn which) */
        uint64_t w0 = dav2_cycles();
        if (!dav2_accel_finish()) goto redo_on_cpu;
        waited += dav2_cycles() - w0;
    }
    const dav2_xf_t qmax = xf_from_int(DAV2_ACT_QMAX);
    dav2_xf_t out_scale = amax.m ? xf_div(amax, qmax) : XF_ONE;
    dav2_xf_t inv_out = xf_recip(out_scale);

    /* ---- requantisation ---------------------------------------------------
     *
     * Per-row parameters, then the int32 -> int16 conversion. On the
     * accelerator this runs in chunks of RQ_CHUNK rows, and the parameters
     * of chunk c+1 are computed while the block converts chunk c. The
     * parameters are the same numbers in any order, so the result is too.
     *
     * The row multiplier is a->scale * s[m] / out_scale = S_m * F with
     * F = a->scale * 2^-es / out_scale, one value per GEMM:
     *
     *   mult = mulh(S_m, F.m) >> r,  shift = F.sh - 32 - r,
     *
     * the same shift for every row. S_m, F.m < 2^31, so mult < 2^30. r
     * keeps the shift within 62 (the block's shift field has 6 bits). The
     * multiplier need not be normalised: apply_multiplier and the block
     * compute round(acc * mult / 2^shift) for any mult. The bias is
     * round(B_m * 2^-eb / out_scale) = round(mulh(B_m, G.m) / 2^(G.sh-32)),
     * with a 64-bit fallback when that shift is outside 1..31. */
    dav2_xf_t fk = xf_mul(a_scale, inv_out);
    fk.sh += es;
    int par_r = 0, par_shift = fk.sh - 32;
    if (par_shift > 62) { par_r = par_shift - 62; par_shift = 62; }
    if (par_r > 31) par_r = 31;                   /* mult is then 0 */
    dav2_xf_t gb = inv_out;
    gb.sh += eb;
    const int gb_k = gb.sh - 32;
    const int gb_fast = gb_k >= 1 && gb_k <= 31;
    int par_done = 0;
#define PAR_UPTO(end_m) do {                                                  \
        uint64_t par_t0_ = dav2_cycles();                                     \
        for (; par_done < (end_m); par_done++) {                              \
            int m_ = par_done;                                                \
            if (par_shift >= 1) {                                             \
                par[3 * m_] = mulh32(par[3 * m_], fk.m) >> par_r;             \
                par[3 * m_ + 1] = par_shift;                                  \
            } else {                  /* factor >= 1: never in this model */  \
                int sh_;                                                      \
                xf_to_mult(xf_mul(xf_mul(a_scale, wt->s[m_]), inv_out),       \
                           &par[3 * m_], &sh_);                               \
                par[3 * m_ + 1] = sh_;                                        \
            }                                                                 \
            if (!wt->b) {                                                     \
                par[3 * m_ + 2] = 0;                                          \
            } else if (gb_fast) {                                             \
                par[3 * m_ + 2] = (mulh32(align_mant(wt->b[m_], eb), gb.m)    \
                                   + (1 << (gb_k - 1))) >> gb_k;              \
            } else {                                                          \
                par[3 * m_ + 2] = sat_i32(shr_round64(                        \
                    (int64_t)align_mant(wt->b[m_], eb) * gb.m, gb.sh));       \
            }                                                                 \
        }                                                                     \
        dav2_sub_add(DAV2_SUB_RQ_PAR, dav2_cycles() - par_t0_);               \
    } while (0)

    const int res_ok = !res || (res->n == N && res->c == M
                                && (((uintptr_t)res->v) & 3u) == 0);
    if ((M & 1) == 0 && run && res_ok) {
        enum { RQ_CHUNK = 256 };          /* <= CAPS.KMAX/4, the add-mode limit */
        int32_t rq_amax = 0;
        int ok = 1;
        dav2_xf_t fin_scale = out_scale;
        dav2_rq_epi_t epi;
        memset(&epi, 0, sizeof epi);
        epi.relu = relu;
        uint64_t gelu_cycles = 0;
        if (gelu && dav2_accel_lut_ok()) {
            /* GELU inside the job: the table for this result's scale goes
             * into the block's lookup-table RAM with the first chunk */
            uint64_t g0 = dav2_cycles();
            dav2_xf_t gscale;
            int16_t *tab = gelu_table(out_scale, &gscale);
            if (tab) {
                epi.lut = tab;
                epi.lut_load = 1;
                fin_scale = gscale;
            }
            gelu_cycles = dav2_cycles() - g0;
        }
        if (res) {
            /* The add's scale needs the largest |h| of this result before
             * any h exists. Per row, h is a non-decreasing function of the
             * accumulator (the multiplier is >= 0), so that maximum is
             * reached at the row's largest or smallest accumulator -- which
             * the range pass kept. Exactly the value a scan of h would give,
             * so the parameters below are dav2_add's, bit for bit. This
             * needs every row's parameters first, so in add mode they are
             * not overlapped with the chunks. */
            PAR_UPTO(M);
            int32_t amax_h = 0;
            for (int m = 0; m < M; m++) {
                int32_t e0 = sat_act(apply_multiplier(rmax[m], par[3 * m], par[3 * m + 1])
                                     + par[3 * m + 2]);
                int32_t e1 = sat_act(apply_multiplier(rmin[m], par[3 * m], par[3 * m + 1])
                                     + par[3 * m + 2]);
                if (e0 < 0) e0 = -e0;
                if (e1 < 0) e1 = -e1;
                if (e0 > amax_h) amax_h = e0;
                if (e1 > amax_h) amax_h = e1;
            }
            int32_t amax_x = res->amax_q >= 0 ? res->amax_q : amax_scan(res);
            /* exactly as dav2_add(res, h) computes them */
            add_params(amax_x, res->scale, amax_h, out_scale,
                       &fin_scale, &epi.mx, &epi.sx, &epi.mh, &epi.sh);
            epi.x = res->v;
            epi.add = 1;
        }
        /* In place (out is the residual): write elsewhere and copy back
         * only once every job has succeeded, so a failure leaves the
         * residual intact for the CPU redo. */
        int16_t *dst = out->v;
        if (res && out->v == res->v) {
            dst = (int16_t *)dav2_arena_alloc((size_t)N * M * sizeof(int16_t));
            if (!dst) ok = 0;
        }
        for (int m0 = 0; m0 < M && ok; m0 += RQ_CHUNK) {
            int mc = M - m0 < RQ_CHUNK ? M - m0 : RQ_CHUNK;
            PAR_UPTO(m0 + mc);                    /* overlaps the previous chunk */
            uint64_t w0 = dav2_cycles();
            ok = dav2_accel_requant_rows_async(acc, N, M, m0, mc, par, dst, &rq_amax,
                                               (epi.add || epi.relu || epi.lut) ? &epi : 0) != 0;
            epi.lut_load = 0;                     /* the block keeps the table */
            waited += dav2_cycles() - w0;         /* includes settling the previous */
        }
        if (ok) {
            uint64_t w0 = dav2_cycles();
            ok = dav2_accel_finish();
            waited += dav2_cycles() - w0;
        }
        if (ok && dst != out->v)
            dav2_copy16(out->v, dst, (size_t)N * M);
        if (ok) {
            uint64_t now = dav2_cycles();
            dav2_prof_add(DAV2_PROF_GEMM_ACCEL, waited);
            dav2_prof_add(DAV2_PROF_GELU, gelu_cycles);
            dav2_prof_add(DAV2_PROF_REQUANT, (now - t_start) - waited - gelu_cycles);
            out->n = N;
            out->c = M;
            out->scale = fin_scale;
            out->amax_q = epi.lut ? -1 : rq_amax;
            trace_tensor("qgemm", out);
            if (gelu && !epi.lut)
                dav2_gelu(out);                   /* no lookup table: on the CPU */
            dav2_arena_release(mark);
            return;
        }
        /* Declined or failed: the CPU path below does the whole conversion
         * (the residual is intact, see dst above). */
    }
    PAR_UPTO(M);
#undef PAR_UPTO
    dav2_prof_add(DAV2_PROF_GEMM_ACCEL, waited);
    uint64_t prof_t0 = t_start + waited;

    /* with an epilogue, the plain result goes to a temporary first */
    dav2_tensor_t htmp, *hdst = out;
    if (res) {
        htmp = dav2_tensor_new(N, M);
        if (!htmp.v) { dav2_arena_release(mark); return; }
        hdst = &htmp;
    }

    /* acc is [m][n]; out is [n][m]. Either loop order strides one side by
     * hundreds of bytes and misses the 16 kB direct-mapped DDR3 cache on
     * every access (a "miss" fetches a 32-byte line from DRAM; a "direct-
     * mapped" cache has exactly one slot per address, so strided accesses
     * evict each other). So: read acc in chunks of RQ_MB rows -- sequential,
     * one pass -- into on-chip scratch, and write the output RQ_MB elements
     * (32 bytes, one cache line) at a time per row. Both sides now touch
     * each line exactly once. Bit-identical to the plain loop. */
    enum { RQ_MB = 16 };
    int32_t *tile = (int32_t *)dav2_scratch;          /* RQ_MB x N int32 */
    const int max_n = (int)(DAV2_SCRATCH_ELEMS / 2 / RQ_MB);
    int32_t omax = 0;
    for (int m0 = 0; m0 < M; m0 += RQ_MB) {
        int mb = M - m0;
        if (mb > RQ_MB) mb = RQ_MB;
        for (int n0 = 0; n0 < N; n0 += max_n) {
            int nb = N - n0;
            if (nb > max_n) nb = max_n;
            for (int j = 0; j < mb; j++)
                dav2_copy16((int16_t *)(tile + (size_t)j * nb),
                            (const int16_t *)(acc + (size_t)(m0 + j) * N + n0),
                            (size_t)nb * 2);
            for (int n = 0; n < nb; n++) {
                int16_t *orow = hdst->v + (size_t)(n0 + n) * M + m0;
                int j = 0;
                /* Two outputs per word store: every store is a bus
                 * transaction on this core, so pairs halve them. M is a
                 * multiple of 16 in this model, but odd tails are handled. */
                if ((((uintptr_t)orow) & 3u) == 0) {
                    for (; j + 1 < mb; j += 2) {
                        int m = m0 + j;
                        int32_t r0 = sat_act(apply_multiplier(tile[(size_t)j * nb + n],
                                                              par[3 * m], par[3 * m + 1]) + par[3 * m + 2]);
                        int32_t r1 = sat_act(apply_multiplier(tile[(size_t)(j + 1) * nb + n],
                                                              par[3 * m + 3], par[3 * m + 4]) + par[3 * m + 5]);
                        *(uint32_t *)(orow + j) = pack16(r0, r1);
                        if (r0 < 0) r0 = -r0;
                        if (r1 < 0) r1 = -r1;
                        if (r0 > omax) omax = r0;
                        if (r1 > omax) omax = r1;
                    }
                }
                for (; j < mb; j++) {
                    int m = m0 + j;
                    int32_t r = sat_act(apply_multiplier(tile[(size_t)j * nb + n],
                                                         par[3 * m], par[3 * m + 1]) + par[3 * m + 2]);
                    orow[j] = (int16_t)r;
                    if (r < 0) r = -r;
                    if (r > omax) omax = r;
                }
            }
        }
    }

    PROF_STOP(DAV2_PROF_REQUANT);
    hdst->n = N;
    hdst->c = M;
    hdst->scale = out_scale;
    hdst->amax_q = omax;
    trace_tensor("qgemm", hdst);
    if (res)
        dav2_add(res, hdst, out);
    if (relu)
        dav2_relu(out);
    if (gelu)
        dav2_gelu(out);
    dav2_arena_release(mark);
    return;

redo_on_cpu:
    /* The accelerator failed mid-operation (it has disabled itself and
     * printed why); acc is incomplete. Start over -- every path is now the
     * CPU's. */
    dav2_arena_release(mark);
    qgemm_impl(a, cv, wt, res, relu, gelu, out);
}

/* -------------------------------------------------------------- LayerNorm */

/* round(d * mult / 2^shift) for shift >= 33: apply_multiplier's first
 * case, with rnd = 2^(shift-33) and s = shift - 32 computed by the caller
 * once per row or tensor */
static inline int32_t mul_hi_round(int32_t d, int32_t mult, int32_t rnd, int s)
{
    return (mulh32(d, mult) + rnd) >> s;
}

static void layernorm_affine_cpu(const dav2_tensor_t *in, const int32_t *gq,
                             const int32_t *bq, dav2_tensor_t *out)
{
    const int N = in->n, C = in->c;
    const size_t mark = dav2_arena_mark();

    /* All integer. Per element (N*C = 31k):
     *
     *   out = (z * g[c] + b[c]) / scale,   z = (x - mean) * r,
     *
     * with r = s * inv_std per row, g in Q15 and b in Q16 (converted
     * offline, dav2_blob_int.py). Per row (N = 82), the statistics are
     * exact integers:
     *
     *   var_q = (C * sum(x^2) - sum(x)^2) / C^2        (in units of s^2)
     *   r = s / sqrt(var_q * s^2 + eps) = 1 / sqrt(var_q + eps/s^2)
     *
     * and the reciprocal square root is a Newton iteration (xf_rsqrt).
     *
     * The second half, out = z * G[c] + B[c], is a per-channel multiply,
     * add and saturation: exactly the accelerator's requantisation job
     * with the channels as its rows. So the CPU computes only z (Q16, one
     * mulh per element), channel by channel into zc[c][n], and keeps each
     * channel's smallest and largest z. Those give the exact range of
     * z * g[c] + b[c] per channel, hence the output scale, before any
     * output exists. The job then applies g and b and the scale, saturates,
     * writes out[n][c] and reports the largest |out|. Without an
     * accelerator the CPU runs the same arithmetic (apply_multiplier), so
     * both give the same result. */
    int32_t *zc = (int32_t *)dav2_arena_alloc((size_t)N * C * sizeof(int32_t));
    int32_t *rowp = (int32_t *)dav2_arena_alloc((size_t)N * 5 * sizeof(int32_t));
    int32_t *par = (int32_t *)dav2_arena_alloc((size_t)C * 3 * sizeof(int32_t));
    if (!zc || !rowp || !par) { dav2_arena_release(mark); return; }

    /* per call: 1/C^2 and eps/s^2, eps = 1e-6 as in PyTorch (the exact
     * float32 value 1e-6f, 0x358637bd) */
    const dav2_xf_t inv_c2 = xf_recip(xf_from_int((int64_t)C * C));
    const dav2_xf_t inv_s  = xf_recip(in->scale);
    const dav2_xf_t eps_s2 = xf_mul(xf_from_f32_bits(0x358637bdu), xf_mul(inv_s, inv_s));
    const int wide = (((uintptr_t)in->v | (uintptr_t)out->v) & 3u) == 0 && (C & 3) == 0;
    const int fast = wide;
    SUB_START();
    for (int n = 0; n < N; n++) {
        const int16_t *row = in->v + (size_t)n * C;
        int32_t sum = 0;
        uint64_t sq = 0;
        if (wide) {
            /* x^2 in 32 bits, 64 at a time: activations are within
             * +-DAV2_ACT_QMAX, and 64 * 8191^2 < 2^32 */
            const uint32_t *rw = (const uint32_t *)row;
            for (int c0 = 0; c0 < C / 2; c0 += 32) {
                const int c1 = c0 + 32 < C / 2 ? c0 + 32 : C / 2;
                uint32_t sq32 = 0;
                #pragma GCC unroll 4
                for (int c = c0; c < c1; c++) {
                    uint32_t x = rw[c];
                    int32_t d0 = lo16(x), d1 = hi16(x);
                    sum += d0 + d1;
                    sq32 += (uint32_t)(d0 * d0) + (uint32_t)(d1 * d1);
                }
                sq += sq32;
            }
        } else {
            for (int c = 0; c < C; c++) {
                int32_t d = (int32_t)row[c];
                sum += d;
                sq += (uint64_t)((int64_t)d * (int64_t)d);
            }
        }
        int64_t num = (int64_t)C * (int64_t)sq - (int64_t)sum * sum;  /* C^2 * var_q, exact */
        if (num < 0) num = 0;
        dav2_xf_t v = xf_add(xf_mul(xf_from_int(num), inv_c2), eps_s2);
        int32_t am; int ash;
        xf_to_mult(xf_rsqrt(v), &am, &ash);                   /* r */
        /* mean in Q16, rounded half away from zero */
        int64_t s16 = (int64_t)sum * 65536;
        /* z = round(d16 * am / 2^ash) as one mulh: (mulh(d16 << up, am) +
         * rnd) >> (sh - 32) with sh = max(ash, 33), up = sh - ash. A row with
         * ash < 33 has a small spread: r ~ 1/std = am 2^-ash gives std <
         * 2^(ash - 30), and |d16| <= sqrt(C - 1) std 2^16 < 2^(ash - 10), so
         * d16 << up < 2^23 fits. The rounding constant is a multiple of 2^32,
         * so this is exactly apply_multiplier's result. */
        const int shz = ash > 33 ? ash : 33;
        int32_t *p = rowp + (size_t)n * 5;
        p[0] = (int32_t)(s16 >= 0 ? (s16 + C / 2) / C : -((-s16 + C / 2) / C));
        p[1] = am;
        p[2] = shz - 32;
        p[3] = 1 << (shz - 33);
        p[4] = shz - ash;
    }
    SUB_LAP(DAV2_SUB_LN_STATS);

    /* z = (x - mean) * r in Q16, two channels at a time, into zc[c][n],
     * with each channel's extremes. |z| <= sqrt(C - 1) < 2^5, so z16 fits. */
    int64_t ymax_all = 0;                          /* largest |y| in Q16 */
    for (int c = 0; c < C; c += (fast ? 2 : 1)) {
        int32_t lo0 = 0x7fffffff, hi0 = -0x7fffffff - 1, lo1 = lo0, hi1 = hi0;
        int32_t *z0 = zc + (size_t)c * N;
        if (fast) {
            int32_t *z1 = z0 + N;
            const int16_t *col = in->v + c;
            #pragma GCC unroll 2
            for (int n = 0; n < N; n++) {
                const int32_t *p = rowp + (size_t)n * 5;
                uint32_t x = *(const uint32_t *)(col + (size_t)n * C);
                const int sh = p[2], up = p[4];
                int32_t a = (mulh32(((lo16(x) << 16) - p[0]) << up, p[1]) + p[3]) >> sh;
                int32_t b = (mulh32(((hi16(x) << 16) - p[0]) << up, p[1]) + p[3]) >> sh;
                z0[n] = a;
                z1[n] = b;
                if (a < lo0) lo0 = a;
                if (a > hi0) hi0 = a;
                if (b < lo1) lo1 = b;
                if (b > hi1) hi1 = b;
            }
        } else {
            for (int n = 0; n < N; n++) {
                const int32_t *p = rowp + (size_t)n * 5;
                int32_t a = (mulh32((((int32_t)in->v[(size_t)n * C + c] << 16) - p[0]) << p[4],
                                    p[1]) + p[3]) >> p[2];
                z0[n] = a;
                if (a < lo0) lo0 = a;
                if (a > hi0) hi0 = a;
            }
        }
        /* y = z g / 2^15 + b, Q16, at the channel's extremes */
        for (int k = 0; k < (fast ? 2 : 1); k++) {
            const int64_t g = gq[c + k], b = bq[c + k];
            const int64_t zl = k ? lo1 : lo0, zh = k ? hi1 : hi0;
            int64_t y0 = ((zl * g) >> 15) + b, y1 = ((zh * g) >> 15) + b;
            if (y0 < 0) y0 = -y0;
            if (y1 < 0) y1 = -y1;
            if (y0 > ymax_all) ymax_all = y0;
            if (y1 > ymax_all) ymax_all = y1;
        }
    }
    SUB_LAP(DAV2_SUB_LN_Y);

    /* the scale, then per channel: mult = g / 2^31 / scale, bias = b / 2^16
     * / scale. mult = mulh(g << 12, F.m) with F = 2^-31 / scale: the same
     * shift F.sh - 20 for every channel; |g| < 2^19, so g << 12 fits. */
    dav2_xf_t out_scale = ymax_all ? xf_div(xf_norm(ymax_all, 16), xf_from_int(DAV2_ACT_QMAX))
                                   : XF_ONE;
    dav2_xf_t inv = xf_recip(out_scale);
    dav2_xf_t F = inv;
    F.sh += 31;
    int shift = F.sh - 20, r = 0;
    if (shift > 62) { r = shift - 62; shift = 62; }
    if (r > 31) r = 31;
    const int shift_ok = shift >= 1;
    for (int c = 0; c < C; c++) {
        if (shift_ok) {
            par[3 * c] = mulh32(gq[c] << 12, F.m) >> r;
            par[3 * c + 1] = shift;
        } else {                           /* factor >= 1: never in this model */
            int sh;
            xf_to_mult(xf_mul(xf_norm(gq[c], 31), inv), &par[3 * c], &sh);
            par[3 * c + 1] = sh;
        }
        par[3 * c + 2] = sat_i32(shr_round64((int64_t)bq[c] * inv.m, inv.sh + 16));
    }

    int32_t omax = -1;
    if (!(C & 1) && !dav2_accel_requant(zc, N, C, par, out->v, &omax)) {
        omax = -1;
    }
    if (omax < 0) {
        /* no accelerator: the same arithmetic on the CPU */
        omax = 0;
        for (int n = 0; n < N; n++) {
            int16_t *orow = out->v + (size_t)n * C;
            for (int c = 0; c < C; c++) {
                int32_t o = sat_act(apply_multiplier(zc[(size_t)c * N + n], par[3 * c],
                                                     par[3 * c + 1]) + par[3 * c + 2]);
                orow[c] = (int16_t)o;
                if (o < 0) o = -o;
                if (o > omax) omax = o;
            }
        }
    }
    SUB_LAP(DAV2_SUB_LN_OUT);
    out->scale = out_scale;
    out->n = N;
    out->c = C;
    out->amax_q = omax;
    dav2_arena_release(mark);
}

/* A requantisation job over int16 input (the block's A16 mode):
 * out[n][m] = sat14(round(in[m][n] * mult_m / 2^shift_m) + bias_m), in
 * rows of N int16, out rows of M int16. On the accelerator when it has the
 * mode, otherwise the same arithmetic on the CPU. Returns the largest
 * |out|. */
static int32_t requant16(const int16_t *in, int N, int M, const int32_t *par, int16_t *out)
{
    int32_t amax = -1;
    if (dav2_accel_requant16(in, N, M, par, out, &amax))
        return amax;
    amax = 0;
    for (int n = 0; n < N; n++) {
        int16_t *orow = out + (size_t)n * M;
        for (int m = 0; m < M; m++) {
            int32_t o = sat_act(apply_multiplier(in[(size_t)m * N + n], par[3 * m],
                                                 par[3 * m + 1]) + par[3 * m + 2]);
            orow[m] = (int16_t)o;
            if (o < 0) o = -o;
            if (o > amax) amax = o;
        }
    }
    return amax;
}

static void layernorm_affine(const dav2_tensor_t *in, const int32_t *gq,
                             const int32_t *bq, dav2_tensor_t *out)
{
    const int N = in->n, C = in->c;
    const size_t mark = dav2_arena_mark();

    /* All integer, in three steps, two of them requantisation jobs:
     *
     *   1. CPU, per token n: mean, r = s * inv_std (exact integer
     *      statistics, xf_rsqrt) and the extremes of x. They give the
     *      largest |z| of the tensor, z = (x - mean) * r, hence a common
     *      14-bit scale zs for z.
     *   2. Job, tokens as rows: zq[c][n] = sat14(round(x[n][c] * mult_n)
     *      + bias_n), mult_n = r_n / zs, bias_n = -round(mean_n r_n / zs).
     *      The output is channel by channel, the layout step 3 reads.
     *   3. CPU: each channel's smallest and largest zq (contiguous rows,
     *      one pass), hence the exact range of y = zq zs g[c] + b[c] and
     *      the output scale. Job, channels as rows: out[n][c] =
     *      sat14(round(zq[c][n] * G_c) + B_c) with G_c = g_c zs / scale,
     *      B_c = b_c / scale.
     *
     * Both jobs read int16 (the block's A16 mode); without it the CPU runs
     * the same arithmetic. zq keeps 14 bits of z, the same resolution as
     * the output. */
    int16_t *zq   = (int16_t *)dav2_arena_alloc((size_t)N * C * sizeof(int16_t));
    int32_t *rowp = (int32_t *)dav2_arena_alloc((size_t)N * 3 * sizeof(int32_t));
    int32_t *parn = (int32_t *)dav2_arena_alloc((size_t)(N > C ? N : C) * 3 * sizeof(int32_t));
    if (!zq || !rowp || !parn || (N & 1)) {
        dav2_arena_release(mark);
        layernorm_affine_cpu(in, gq, bq, out);
        return;
    }

    const dav2_xf_t inv_c2 = xf_recip(xf_from_int((int64_t)C * C));
    const dav2_xf_t inv_s  = xf_recip(in->scale);
    const dav2_xf_t eps_s2 = xf_mul(xf_from_f32_bits(0x358637bdu), xf_mul(inv_s, inv_s));
    const int wide = (((uintptr_t)in->v) & 3u) == 0 && (C & 1) == 0;
    SUB_START();
    dav2_xf_t zmax_all = XF_ZERO;
    for (int n = 0; n < N; n++) {
        const int16_t *row = in->v + (size_t)n * C;
        int32_t sum = 0, xmax = -32768, xmin = 32767;
        uint64_t sq = 0;
        if (wide) {
            /* x^2 in 32 bits, 64 at a time: activations are within
             * +-DAV2_ACT_QMAX, and 64 * 8191^2 < 2^32 */
            const uint32_t *rw = (const uint32_t *)row;
            for (int c0 = 0; c0 < C / 2; c0 += 32) {
                const int c1 = c0 + 32 < C / 2 ? c0 + 32 : C / 2;
                uint32_t sq32 = 0;
                #pragma GCC unroll 4
                for (int c = c0; c < c1; c++) {
                    uint32_t x = rw[c];
                    int32_t d0 = lo16(x), d1 = hi16(x);
                    sum += d0 + d1;
                    sq32 += (uint32_t)(d0 * d0) + (uint32_t)(d1 * d1);
                    if (d0 > xmax) xmax = d0;
                    if (d0 < xmin) xmin = d0;
                    if (d1 > xmax) xmax = d1;
                    if (d1 < xmin) xmin = d1;
                }
                sq += sq32;
            }
        } else {
            for (int c = 0; c < C; c++) {
                int32_t d = (int32_t)row[c];
                sum += d;
                sq += (uint64_t)((int64_t)d * (int64_t)d);
                if (d > xmax) xmax = d;
                if (d < xmin) xmin = d;
            }
        }
        int64_t num = (int64_t)C * (int64_t)sq - (int64_t)sum * sum;  /* C^2 * var_q, exact */
        if (num < 0) num = 0;
        dav2_xf_t r = xf_rsqrt(xf_add(xf_mul(xf_from_int(num), inv_c2), eps_s2));
        int64_t s16 = (int64_t)sum * 65536;
        int32_t mean16 = (int32_t)(s16 >= 0 ? (s16 + C / 2) / C : -((-s16 + C / 2) / C));
        int32_t dmax = (xmax << 16) - mean16, dmin = (xmin << 16) - mean16;
        int32_t dev = dmax > -dmin ? dmax : -dmin;
        dav2_xf_t zmax = xf_mul(xf_norm(dev, 16), r);
        if (xf_cmp_abs(zmax, zmax_all) > 0) zmax_all = zmax;
        int32_t *p = rowp + (size_t)n * 3;
        p[0] = mean16; p[1] = r.m; p[2] = r.sh;
    }

    /* step 2: per token, mult = r / zs, bias = -mean r / zs */
    dav2_xf_t zs = zmax_all.m ? xf_div(zmax_all, xf_from_int(DAV2_ACT_QMAX)) : XF_ONE;
    dav2_xf_t izs = xf_recip(zs);
    for (int n = 0; n < N; n++) {
        const int32_t *p = rowp + (size_t)n * 3;
        dav2_xf_t k = xf_mul((dav2_xf_t){ p[1], p[2] }, izs);
        int sh;
        xf_to_mult(k, &parn[3 * n], &sh);
        parn[3 * n + 1] = sh;
        parn[3 * n + 2] = (int32_t)-xf_round(xf_mul(xf_norm(p[0], 16), k), 0);
    }
    SUB_LAP(DAV2_SUB_LN_STATS);
    requant16(in->v, C, N, parn, zq);            /* rows n, C columns -> zq[c][n] */

    /* step 3: channel extremes of zq, the exact range of y in Q16, with
     * ZS = zs 2^32 (|zq g| < 2^33, ZS < 2^24: the product fits int64) */
    const int64_t ZS = xf_round(zs, 32);
    int64_t ymax_all = 0;
    for (int c = 0; c < C; c++) {
        const int16_t *zr = zq + (size_t)c * N;
        int32_t lo = 32767, hi = -32768;
        if ((N & 1) == 0 && (((uintptr_t)zr) & 3u) == 0) {
            const uint32_t *zw = (const uint32_t *)zr;
            #pragma GCC unroll 4
            for (int n = 0; n < N / 2; n++) {
                uint32_t w = zw[n];
                int32_t a = lo16(w), b = hi16(w);
                if (a < lo) lo = a;
                if (a > hi) hi = a;
                if (b < lo) lo = b;
                if (b > hi) hi = b;
            }
        } else {
            for (int n = 0; n < N; n++) {
                if (zr[n] < lo) lo = zr[n];
                if (zr[n] > hi) hi = zr[n];
            }
        }
        const int64_t g = gq[c], b = bq[c];
        int64_t y0 = (((int64_t)lo * g * ZS) >> 31) + b;   /* Q15 * 2^32 >> 31 = Q16 */
        int64_t y1 = (((int64_t)hi * g * ZS) >> 31) + b;
        if (y0 < 0) y0 = -y0;
        if (y1 < 0) y1 = -y1;
        if (y0 > ymax_all) ymax_all = y0;
        if (y1 > ymax_all) ymax_all = y1;
    }
    SUB_LAP(DAV2_SUB_LN_Y);

    /* per channel: G = g zs / scale, B = b / scale; G = mulh(g << 12, F.m)
     * with F = zs 2^-15 / scale, one shift F.sh - 20 for every channel */
    dav2_xf_t out_scale = ymax_all ? xf_div(xf_norm(ymax_all, 16), xf_from_int(DAV2_ACT_QMAX))
                                   : XF_ONE;
    dav2_xf_t inv = xf_recip(out_scale);
    dav2_xf_t F = xf_mul(zs, inv);
    F.sh += 15;
    int shift = F.sh - 20, rr = 0;
    if (shift > 62) { rr = shift - 62; shift = 62; }
    if (rr > 31) rr = 31;
    for (int c = 0; c < C; c++) {
        if (shift >= 1) {
            parn[3 * c] = mulh32(gq[c] << 12, F.m) >> rr;
            parn[3 * c + 1] = shift;
        } else {                           /* factor >= 1: never in this model */
            int sh;
            xf_to_mult(xf_mul(xf_norm(gq[c], 15), F), &parn[3 * c], &sh);
            parn[3 * c + 1] = sh;
        }
        parn[3 * c + 2] = sat_i32(shr_round64((int64_t)bq[c] * inv.m, inv.sh + 16));
    }
    int32_t omax = requant16(zq, N, C, parn, out->v);   /* rows c, N columns -> out[n][c] */
    SUB_LAP(DAV2_SUB_LN_OUT);
    out->scale = out_scale;
    out->n = N;
    out->c = C;
    out->amax_q = omax;
    dav2_arena_release(mark);
}

static void layernorm_plain(const dav2_tensor_t *in, dav2_tensor_t *out)
{
    const int N = in->n, C = in->c;
    const size_t mark = dav2_arena_mark();

    /* out = (x - mean) / std, without gamma and beta. The final norm's are
     * folded into the weights of the layers that read its output
     * (dav2_blob_int.py). All integer.
     *
     * Pass 1, per row: sum(x), sum(x^2), max(x) and min(x). The statistics
     * are exact integers:
     *
     *   var_q = (C * sum(x^2) - sum(x)^2) / C^2        (in units of s^2)
     *   r = s * inv_std = 1 / sqrt(var_q + eps/s^2)    (xf_rsqrt)
     *
     * and z = (x - mean) * r. z is a non-decreasing function of x, so the
     * row's largest |z| is at its max or min. The largest |z| of all rows
     * gives the output scale before any output exists.
     *
     * Pass 2, per element: one multiply, out = round((x - mean) * r / scale),
     * with x - mean in Q16 and r / scale as one (mult, shift) pair per row.
     * With gamma and beta (layernorm_affine), the output range is not known
     * before the outputs exist, so that version stores an int32
     * intermediate and requantises it in a third pass. */
    int32_t *rp = (int32_t *)dav2_arena_alloc((size_t)N * 5 * sizeof(int32_t));
    if (!rp) { dav2_arena_release(mark); return; }

    /* per call: 1/C^2 and eps/s^2, eps = 1e-6 as in PyTorch (the exact
     * float32 value 1e-6f, 0x358637bd) */
    const dav2_xf_t inv_c2 = xf_recip(xf_from_int((int64_t)C * C));
    const dav2_xf_t inv_s  = xf_recip(in->scale);
    const dav2_xf_t eps_s2 = xf_mul(xf_from_f32_bits(0x358637bdu), xf_mul(inv_s, inv_s));
    const int wide = (((uintptr_t)in->v | (uintptr_t)out->v) & 3u) == 0 && (C & 3) == 0;
    dav2_xf_t zmax_all = XF_ZERO;
    for (int n = 0; n < N; n++) {
        const int16_t *row = in->v + (size_t)n * C;
        int32_t sum = 0, xmax = -32768, xmin = 32767;
        uint64_t sq = 0;
        if (wide) {
            /* x^2 in 32 bits, 64 at a time: activations are within
             * +-DAV2_ACT_QMAX, and 64 * 8191^2 < 2^32. No 64-bit multiply. */
            const uint32_t *rw = (const uint32_t *)row;
            for (int c0 = 0; c0 < C / 2; c0 += 32) {
                const int c1 = c0 + 32 < C / 2 ? c0 + 32 : C / 2;
                uint32_t sq32 = 0;
                #pragma GCC unroll 4
                for (int c = c0; c < c1; c++) {
                    uint32_t x = rw[c];
                    int32_t d0 = lo16(x), d1 = hi16(x);
                    sum += d0 + d1;
                    sq32 += (uint32_t)(d0 * d0) + (uint32_t)(d1 * d1);
                    if (d0 > xmax) xmax = d0;
                    if (d0 < xmin) xmin = d0;
                    if (d1 > xmax) xmax = d1;
                    if (d1 < xmin) xmin = d1;
                }
                sq += sq32;
            }
        } else {
            for (int c = 0; c < C; c++) {
                int32_t d = (int32_t)row[c];
                sum += d;
                sq += (uint64_t)((int64_t)d * (int64_t)d);
                if (d > xmax) xmax = d;
                if (d < xmin) xmin = d;
            }
        }
        int64_t num = (int64_t)C * (int64_t)sq - (int64_t)sum * sum;  /* C^2 * var_q, exact */
        if (num < 0) num = 0;
        dav2_xf_t r = xf_rsqrt(xf_add(xf_mul(xf_from_int(num), inv_c2), eps_s2));

        /* mean in Q16, rounded half away from zero */
        int64_t s16 = (int64_t)sum * 65536;
        int32_t mean16 = (int32_t)(s16 >= 0 ? (s16 + C / 2) / C : -((-s16 + C / 2) / C));
        int32_t dmax = (xmax << 16) - mean16, dmin = (xmin << 16) - mean16;
        int32_t dev = dmax > -dmin ? dmax : -dmin;
        dav2_xf_t zmax = xf_mul(xf_norm(dev, 16), r);
        if (xf_cmp_abs(zmax, zmax_all) > 0) zmax_all = zmax;

        int32_t *p = rp + (size_t)n * 5;
        p[0] = mean16; p[1] = r.m; p[2] = r.sh; p[3] = dmax; p[4] = dmin;
    }

    dav2_xf_t out_scale = zmax_all.m ? xf_div(zmax_all, xf_from_int(DAV2_ACT_QMAX)) : XF_ONE;
    dav2_xf_t inv16 = xf_recip(out_scale);
    inv16.sh += 16;                                  /* x - mean is in Q16 */
    int32_t omax = 0;
    for (int n = 0; n < N; n++) {
        const int32_t *p = rp + (size_t)n * 5;
        const int32_t mean16 = p[0];
        dav2_xf_t r = { p[1], p[2] };
        int32_t cm; int csh;
        xf_to_mult(xf_mul(r, inv16), &cm, &csh);

        /* the row's largest |out|, from its extremes (see above) */
        int32_t e0 = sat_act(apply_multiplier(p[3], cm, csh));
        int32_t e1 = sat_act(apply_multiplier(p[4], cm, csh));
        if (e0 < 0) e0 = -e0;
        if (e1 < 0) e1 = -e1;
        if (e0 > omax) omax = e0;
        if (e1 > omax) omax = e1;

        const int16_t *row = in->v + (size_t)n * C;
        int16_t *orow = out->v + (size_t)n * C;
        if (wide && csh >= 33) {
            /* apply_multiplier's shift >= 33 case with the row's constants
             * hoisted: one mulh, an add and a shift per element */
            const int32_t rnd = 1 << (csh - 33);
            const int sh = csh - 32;
            const uint32_t *rw = (const uint32_t *)row;
            uint32_t *ow = (uint32_t *)orow;
            #pragma GCC unroll 4
            for (int c = 0; c < C / 2; c++) {
                uint32_t x = rw[c];
                int32_t d0 = (lo16(x) << 16) - mean16, d1 = (hi16(x) << 16) - mean16;
                int32_t h0 = (int32_t)(((int64_t)d0 * (int64_t)cm) >> 32);
                int32_t h1 = (int32_t)(((int64_t)d1 * (int64_t)cm) >> 32);
                ow[c] = pack16(sat_act((h0 + rnd) >> sh), sat_act((h1 + rnd) >> sh));
            }
        } else {
            for (int c = 0; c < C; c++)
                orow[c] = sat_act(apply_multiplier(((int32_t)row[c] << 16) - mean16, cm, csh));
        }
    }
    out->scale = out_scale;
    out->n = N;
    out->c = C;
    out->amax_q = omax;
    dav2_arena_release(mark);
}

void dav2_layernorm(const dav2_tensor_t *in, const int32_t *gq, const int32_t *bq,
                    dav2_tensor_t *out)
{
    PROF_START();
    if (gq)
        layernorm_affine(in, gq, bq, out);
    else
        layernorm_plain(in, out);
    PROF_STOP(DAV2_PROF_LAYERNORM);
    trace_tensor("layernorm", out);
}

void dav2_add(const dav2_tensor_t *a, const dav2_tensor_t *b, dav2_tensor_t *out)
{
    const int total = a->n * a->c;
    PROF_START();
    /* Upper bound on the sum's magnitude; at most one bit of range is lost.
     * The operands' ranges come from their producers when known (every
     * streaming operator records the largest |value| it wrote), otherwise
     * from a scan; either way the same number. */
    int32_t amax_a = a->amax_q, amax_b = b->amax_q;
    const int wide = words_ok(a->v, b->v, out->v, total);
    if (amax_a >= 0 && amax_b >= 0) {
        /* nothing to scan */
    } else if (wide) {
        amax_a = 0; amax_b = 0;
        const uint32_t *aw = (const uint32_t *)a->v, *bw = (const uint32_t *)b->v;
        for (int i = 0; i < total / 2; i++) {
            uint32_t x = aw[i], y = bw[i];
            int32_t v0 = lo16(x), v1 = hi16(x), u0 = lo16(y), u1 = hi16(y);
            if (v0 < 0) v0 = -v0;
            if (v1 < 0) v1 = -v1;
            if (u0 < 0) u0 = -u0;
            if (u1 < 0) u1 = -u1;
            if (v0 > amax_a) amax_a = v0;
            if (v1 > amax_a) amax_a = v1;
            if (u0 > amax_b) amax_b = u0;
            if (u1 > amax_b) amax_b = u1;
        }
    } else {
        amax_a = 0; amax_b = 0;
        for (int i = 0; i < total; i++) {
            int32_t va = a->v[i] < 0 ? -a->v[i] : a->v[i];
            int32_t vb = b->v[i] < 0 ? -b->v[i] : b->v[i];
            if (va > amax_a) amax_a = va;
            if (vb > amax_b) amax_b = vb;
        }
    }
    dav2_xf_t out_scale;
    int32_t ma, mb; int sa, sb;
    add_params(amax_a, a->scale, amax_b, b->scale, &out_scale, &ma, &sa, &mb, &sb);

    int32_t omax = 0;
    if (wide && sa >= 17 && sb >= 17) {
        /* The factors are near 1, so the shifts are about 31 and
         * apply_multiplier takes its slower branch (shift <= 32). With the
         * int16 input moved up by 16 bits the shift becomes >= 33 and one
         * mulh does it: round(v m / 2^s) = round((v 2^16) m / 2^(s+16)),
         * the same value. |v| < 2^15, so v << 16 fits. */
        const int32_t ra = 1 << (sa + 16 - 33), rb = 1 << (sb + 16 - 33);
        const int ka = sa + 16 - 32, kb = sb + 16 - 32;
        const uint32_t *aw = (const uint32_t *)a->v, *bw = (const uint32_t *)b->v;
        uint32_t *ow = (uint32_t *)out->v;
        #pragma GCC unroll 2
        for (int i = 0; i < total / 2; i++) {
            uint32_t x = aw[i], y = bw[i];
            int32_t r0 = sat_act(((mulh32(lo16(x) << 16, ma) + ra) >> ka)
                               + ((mulh32(lo16(y) << 16, mb) + rb) >> kb));
            int32_t r1 = sat_act(((mulh32(hi16(x) << 16, ma) + ra) >> ka)
                               + ((mulh32(hi16(y) << 16, mb) + rb) >> kb));
            ow[i] = pack16(r0, r1);
            if (r0 < 0) r0 = -r0;
            if (r1 < 0) r1 = -r1;
            if (r0 > omax) omax = r0;
            if (r1 > omax) omax = r1;
        }
    } else if (wide) {
        const uint32_t *aw = (const uint32_t *)a->v, *bw = (const uint32_t *)b->v;
        uint32_t *ow = (uint32_t *)out->v;
        #pragma GCC unroll 4
        for (int i = 0; i < total / 2; i++) {
            uint32_t x = aw[i], y = bw[i];
            int32_t r0 = sat_act(apply_multiplier(lo16(x), ma, sa) + apply_multiplier(lo16(y), mb, sb));
            int32_t r1 = sat_act(apply_multiplier(hi16(x), ma, sa) + apply_multiplier(hi16(y), mb, sb));
            ow[i] = pack16(r0, r1);
            if (r0 < 0) r0 = -r0;
            if (r1 < 0) r1 = -r1;
            if (r0 > omax) omax = r0;
            if (r1 > omax) omax = r1;
        }
    } else {
        for (int i = 0; i < total; i++) {
            int32_t v = sat_act(apply_multiplier(a->v[i], ma, sa)
                              + apply_multiplier(b->v[i], mb, sb));
            out->v[i] = (int16_t)v;
            if (v < 0) v = -v;
            if (v > omax) omax = v;
        }
    }
    out->n = a->n;
    out->c = a->c;
    out->scale = out_scale;
    out->amax_q = omax;
    PROF_STOP(DAV2_PROF_ADD_RELU);
    trace_tensor("add", out);
}

void dav2_relu(dav2_tensor_t *t)
{
    const int total = t->n * t->c;
    PROF_START();
    int32_t omax = 0;
    if (words_ok(t->v, t->v, t->v, total)) {
        uint32_t *w = (uint32_t *)t->v;
        #pragma GCC unroll 4
        for (int i = 0; i < total / 2; i++) {
            uint32_t x = w[i];
            /* clear each half whose sign bit is set; skip the store if nothing changed */
            uint32_t neg = ((x >> 15) & 1u) * 0xffffu | ((x >> 31) & 1u) * 0xffff0000u;
            if (neg) { x &= ~neg; w[i] = x; }
            int32_t v0 = (int32_t)(x & 0xffffu), v1 = (int32_t)(x >> 16);
            if (v0 > omax) omax = v0;
            if (v1 > omax) omax = v1;
        }
    } else {
        for (int i = 0; i < total; i++) {
            if (t->v[i] < 0) t->v[i] = 0;
            if (t->v[i] > omax) omax = t->v[i];
        }
    }
    t->amax_q = omax;
    PROF_STOP(DAV2_PROF_ADD_RELU);
}

void dav2_copy_relu(dav2_tensor_t *dst, const dav2_tensor_t *src)
{
    const int total = src->n * src->c;
    PROF_START();
    int32_t omax = 0;
    if (words_ok(dst->v, src->v, src->v, total)) {
        const uint32_t *sw = (const uint32_t *)src->v;
        uint32_t *dw = (uint32_t *)dst->v;
        #pragma GCC unroll 4
        for (int i = 0; i < total / 2; i++) {
            uint32_t x = sw[i];
            /* clear each half whose sign bit is set */
            x &= ~(((x >> 15) & 1u) * 0xffffu | ((x >> 31) & 1u) * 0xffff0000u);
            dw[i] = x;
            int32_t v0 = (int32_t)(x & 0xffffu), v1 = (int32_t)(x >> 16);
            if (v0 > omax) omax = v0;
            if (v1 > omax) omax = v1;
        }
    } else {
        for (int i = 0; i < total; i++) {
            int16_t v = src->v[i] < 0 ? 0 : src->v[i];
            dst->v[i] = v;
            if (v > omax) omax = v;
        }
    }
    dst->n = src->n;
    dst->c = src->c;
    dst->scale = src->scale;
    dst->amax_q = omax;
    PROF_STOP(DAV2_PROF_ADD_RELU);
}

/* GELU is an elementwise map on a 14-bit input, so a 257-entry table with
 * linear interpolation reproduces it to well below quantisation noise. The
 * table costs 257 float evaluations per call, versus 126k for direct
 * evaluation. */
/* gelu(x) = x * Phi(x), with Phi the normal distribution function, read
 * from the table dav2_gelu_phi (Q30, z = -8 + i/128, generated offline by
 * dav2_blob_int.py --phi-header) with linear interpolation. The
 * interpolation error is below 2e-6; torch.nn.GELU's default is this exact
 * erf form. */
#define gelu_phi dav2_gelu_phi

static dav2_xf_t gelu_xf(dav2_xf_t x)
{
    const dav2_xf_t eight = { 1 << 30, 27 };             /* 8.0 */
    if (xf_cmp_abs(x, eight) >= 0)
        return x.m > 0 ? x : XF_ZERO;                   /* Phi = 1 or 0 */
    int64_t u = xf_round(x, 20) + ((int64_t)-DAV2_PHI_Z0 << 20);  /* (z + 8) in Q20 */
    int32_t idx = (int32_t)(u >> (20 - DAV2_PHI_STEP_LOG2));
    int32_t frac = (int32_t)(u & ((1 << (20 - DAV2_PHI_STEP_LOG2)) - 1));
    int64_t lo = gelu_phi[idx], hi = gelu_phi[idx + 1];
    int64_t phi = lo + (((hi - lo) * frac) >> (20 - DAV2_PHI_STEP_LOG2));
    return xf_mul(x, xf_norm(phi, DAV2_PHI_Q));
}

/* The GELU table for inputs of scale in_scale: T[x + 8192] for x in
 * -8192..8191 (16384 int16 in the DDR3 arena, 4-byte aligned), and the
 * scale of its outputs. Used by the CPU pass (dav2_gelu) and, loaded into
 * the accelerator's lookup-table RAM, by fc1's requantisation job
 * (dav2_qgemm_gelu); the same table, so the same result. */
static int16_t *gelu_table(dav2_xf_t in_scale, dav2_xf_t *out_scale_ret)
{
    dav2_xf_t g[257];
    dav2_xf_t amax = XF_ZERO;
    for (int i = 0; i < 257; i++) {
        g[i] = gelu_xf(xf_mul(xf_from_int(i * 64 - 8192), in_scale));
        if (xf_cmp_abs(g[i], amax) > 0) amax = xf_abs(g[i]);
    }
    dav2_xf_t out_scale = amax.m ? xf_div(amax, xf_from_int(DAV2_ACT_QMAX)) : XF_ONE;
    dav2_xf_t inv = xf_recip(out_scale);

    int16_t lut[257];
    for (int i = 0; i < 257; i++)
        lut[i] = sat_act((int32_t)xf_round(xf_mul(g[i], inv), 0));

    /* A direct table: one int16 output per input value, T[x + 8192] for
     * x in -8192..8191, each entry the linear interpolation between the
     * 257 points above,
     *
     *   lut[i] + ((lut[i+1] - lut[i]) * f >> 6),  i = u >> 6, f = u & 63,
     *
     * so the element loop is one load per element and no arithmetic. The
     * table is built interval by interval with a running product (one add
     * per entry) and costs about a tenth of the element loop. It lives in
     * the DDR3 arena (32 kB): the stack is in the on-chip RAM, where a load
     * takes about 4 cycles longer than a DDR3 cache hit. The activations
     * cluster near zero, so the part of the table in use stays mostly in
     * the cache. */
    int16_t *tab = (int16_t *)dav2_arena_alloc(16384 * sizeof(int16_t));
    if (!tab)
        return 0;
    for (int i = 0; i < 256; i++) {
        const int32_t lo = lut[i], step = lut[i + 1] - lut[i];
        uint32_t *tw = (uint32_t *)(tab + 64 * i);
        int32_t acc = 0;
        #pragma GCC unroll 4
        for (int f = 0; f < 64; f += 2) {
            int32_t v0 = lo + (acc >> 6);
            acc += step;
            int32_t v1 = lo + (acc >> 6);
            acc += step;
            tw[f >> 1] = pack16(v0, v1);
        }
    }
    *out_scale_ret = out_scale;
    return tab;
}

void dav2_gelu(dav2_tensor_t *t)
{
    PROF_START();
    SUB_START();
    const size_t mark = dav2_arena_mark();
    dav2_xf_t out_scale;
    int16_t *tab = gelu_table(t->scale, &out_scale);
    if (!tab) { dav2_arena_release(mark); PROF_STOP(DAV2_PROF_GELU); return; }
    const int16_t *T = tab + 8192;

    SUB_LAP(DAV2_SUB_GELU_TABLE);
    const int total = t->n * t->c;
    /* The output's largest |value| is not tracked (amax_q = -1): GELU's
     * output only feeds fc2's GEMM, which needs the scale alone. */
    if (words_ok(t->v, t->v, t->v, total)) {
        uint32_t *w = (uint32_t *)t->v;
        #pragma GCC unroll 4
        for (int i = 0; i < total / 2; i++) {
            uint32_t x = w[i];
            w[i] = pack16(T[lo16(x)], T[hi16(x)]);
        }
    } else {
        for (int i = 0; i < total; i++)
            t->v[i] = T[t->v[i]];
    }
    t->amax_q = -1;
    dav2_arena_release(mark);
    t->scale = out_scale;
    PROF_STOP(DAV2_PROF_GELU);
    trace_tensor("gelu", t);
}

/* ------------------------------------------------------------ resampling */

/* One source row, resampled horizontally: dst[j][ch] = (a (256 - wx) +
 * b wx) >> 8 with a, b the row's pixels x0[j], x1[j]. */
static void interp_row_h(const int16_t *src, int C, int ow, const int *x0a,
                         const int *x1a, const int *wxa, int16_t *dst, int wide)
{
    for (int j = 0; j < ow; j++) {
        const int16_t *a = src + (size_t)x0a[j] * C;
        const int16_t *b = src + (size_t)x1a[j] * C;
        const int wx = wxa[j];
        int16_t *o = dst + (size_t)j * C;
        if (wx == 0) {                          /* (a * 256) >> 8 == a */
            dav2_copy16(o, a, (size_t)C);
        } else if (wide) {
            const uint32_t *aw = (const uint32_t *)a, *bw = (const uint32_t *)b;
            uint32_t *ow32 = (uint32_t *)o;
            const int32_t wa = 256 - wx;
            #pragma GCC unroll 4
            for (int c = 0; c < C / 2; c++) {
                uint32_t x = aw[c], y = bw[c];
                ow32[c] = pack16((lo16(x) * wa + lo16(y) * wx) >> 8,
                                 (hi16(x) * wa + hi16(y) * wx) >> 8);
            }
        } else {
            for (int c = 0; c < C; c++)
                o[c] = (int16_t)((a[c] * (256 - wx) + b[c] * wx) >> 8);
        }
    }
}

void dav2_interpolate(const dav2_tensor_t *in, int h, int w,
                      int oh, int ow, dav2_tensor_t *out)
{
    const int C = in->c;
    const size_t mark = dav2_arena_mark();
    PROF_START();
    int *y0a = (int *)dav2_arena_alloc((size_t)oh * sizeof(int));
    int *y1a = (int *)dav2_arena_alloc((size_t)oh * sizeof(int));
    int *wya = (int *)dav2_arena_alloc((size_t)oh * sizeof(int));
    int *x0a = (int *)dav2_arena_alloc((size_t)ow * sizeof(int));
    int *x1a = (int *)dav2_arena_alloc((size_t)ow * sizeof(int));
    int *wxa = (int *)dav2_arena_alloc((size_t)ow * sizeof(int));

    /* align_corners=true, matching F.interpolate. Weights in Q8. The
     * source position of output row i is i*(h-1)/(oh-1): its integer part
     * and its fraction, as an exact ratio of integers, rounded to Q8. */
    for (int i = 0; i < oh; i++) {
        int i0 = 0, wq = 0;
        if (oh > 1) {
            int num = i * (h - 1), den = oh - 1;
            i0 = num / den;
            wq = ((num % den) * 512 + den) / (2 * den);
        }
        if (i0 > h - 1) i0 = h - 1;
        y0a[i] = i0;
        y1a[i] = (i0 + 1 < h) ? i0 + 1 : h - 1;
        wya[i] = wq;
    }
    for (int j = 0; j < ow; j++) {
        int j0 = 0, wq = 0;
        if (ow > 1) {
            int num = j * (w - 1), den = ow - 1;
            j0 = num / den;
            wq = ((num % den) * 512 + den) / (2 * den);
        }
        if (j0 > w - 1) j0 = w - 1;
        x0a[j] = j0;
        x1a[j] = (j0 + 1 < w) ? j0 + 1 : w - 1;
        wxa[j] = wq;
    }

    /* Separable, horizontal first:
     *   top = (a (256 - wx) + b wx) >> 8     on source row y0
     *   bot = (c (256 - wx) + d wx) >> 8     on source row y1
     *   out = (top (256 - wy) + bot wy) >> 8
     * A source row's horizontal pass is the same for every output row that
     * reads it (about oh / h of them), so it is computed once into one of
     * two row buffers and kept while it is needed. Same arithmetic, same
     * result as computing all four terms per output element; about half
     * the work for the 2x upsamplings of the DPT head. Two channels per
     * 32-bit word where C and the buffers allow it. */
    const size_t rowlen = (size_t)ow * C;
    int16_t *hbuf[2];
    hbuf[0] = (int16_t *)dav2_arena_alloc(rowlen * sizeof(int16_t));
    hbuf[1] = (int16_t *)dav2_arena_alloc(rowlen * sizeof(int16_t));
    int hrow[2] = { -1, -1 };
    if (!hbuf[0] || !hbuf[1]) { dav2_arena_release(mark); return; }
    const int wide = (C & 1) == 0 && ((((uintptr_t)in->v | (uintptr_t)out->v) & 3u) == 0);

    for (int i = 0; i < oh; i++) {
        const int need[2] = { y0a[i], y1a[i] };
        int16_t *hr[2];
        for (int k = 0; k < 2; k++) {
            int slot = hrow[0] == need[k] ? 0 : hrow[1] == need[k] ? 1 : -1;
            if (slot < 0) {
                /* compute into the slot the other needed row does not use */
                slot = (k == 1 && hrow[0] == need[0]) ? 1
                     : (k == 1 && hrow[1] == need[0]) ? 0
                     : (hrow[0] == need[1 - k]) ? 1 : 0;
                interp_row_h(in->v + (size_t)need[k] * w * C, C, ow, x0a, x1a, wxa,
                             hbuf[slot], wide);
                hrow[slot] = need[k];
            }
            hr[k] = hbuf[slot];
        }
        const int wy = wya[i];
        int16_t *orow = out->v + (size_t)i * rowlen;
        if (wy == 0) {                          /* (top * 256) >> 8 == top */
            dav2_copy16(orow, hr[0], rowlen);
        } else if (wide) {
            const uint32_t *tw = (const uint32_t *)hr[0], *bw = (const uint32_t *)hr[1];
            uint32_t *ow32 = (uint32_t *)orow;
            const int32_t wt = 256 - wy;
            #pragma GCC unroll 4
            for (size_t e = 0; e < rowlen / 2; e++) {
                uint32_t t = tw[e], b = bw[e];
                ow32[e] = pack16((lo16(t) * wt + lo16(b) * wy) >> 8,
                                 (hi16(t) * wt + hi16(b) * wy) >> 8);
            }
        } else {
            for (size_t e = 0; e < rowlen; e++)
                orow[e] = (int16_t)((hr[0][e] * (256 - wy) + hr[1][e] * wy) >> 8);
        }
    }
    out->n = oh * ow;
    out->c = C;
    out->scale = in->scale;
    PROF_STOP(DAV2_PROF_INTERP);
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
    PROF_START();

    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            for (int ky = 0; ky < kh; ky++) {
                int iy = oy * stride + ky - pad;
                for (int kx = 0; kx < kw; kx++) {
                    int ix = ox * stride + kx - pad;
                    if (iy < 0 || iy >= h || ix < 0 || ix >= w) {
                        dav2_zero16(dst, (size_t)C);
                    } else {
                        dav2_copy16(dst, in->v + ((size_t)iy * w + ix) * C, (size_t)C);
                    }
                    dst += C;
                }
            }
        }
    }
    cols->n = oh * ow;
    cols->c = K;
    cols->scale = in->scale;
    PROF_STOP(DAV2_PROF_IM2COL);
}

dav2_tensor_t dav2_conv2d(const dav2_tensor_t *in, int h, int w,
                          const dav2_qw_t *wt, int k, int stride, int pad,
                          int *oh_out, int *ow_out)
{
    return dav2_conv2d_ex(in, h, w, wt, k, stride, pad, 0, 0, oh_out, ow_out);
}

/* dav2_conv2d followed by dav2_add(res, result) and/or dav2_relu -- see
 * dav2_qgemm_ex. */
dav2_tensor_t dav2_conv2d_ex(const dav2_tensor_t *in, int h, int w,
                             const dav2_qw_t *wt, int k, int stride, int pad,
                             const dav2_tensor_t *res, int relu,
                             int *oh_out, int *ow_out)
{
    const int oh = (h + 2 * pad - k) / stride + 1;
    const int ow = (w + 2 * pad - k) / stride + 1;

    dav2_tensor_t out = dav2_tensor_new(oh * ow, wt->m);
    const size_t mark = dav2_arena_mark();
    if (k == 1 && stride == 1 && pad == 0) {
        /* A 1x1 convolution is a GEMM over the pixels as they are; im2col
         * would only copy the tensor. */
        qgemm_impl(in, 0, wt, res, relu, 0, &out);
    } else {
        /* Gathered in hardware when the accelerator can (no patch matrix is
         * ever built); otherwise im2col in software, inside qgemm_impl. */
        conv_desc_t cv = { h, w, k, stride, pad };
        qgemm_impl(in, &cv, wt, res, relu, 0, &out);
    }
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

    PROF_START();
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const int16_t *src = flat.v + ((size_t)y * w + x) * wt->m;
            for (int ky = 0; ky < stride; ky++) {
                int16_t *dst = out.v + ((size_t)(y * stride + ky) * ow
                                        + (size_t)x * stride) * cout;
                dav2_copy16(dst, src + (size_t)ky * stride * cout,
                            (size_t)stride * cout);
            }
        }
    }
    PROF_STOP(DAV2_PROF_INTERP);
    out.scale = flat.scale;
    out.amax_q = flat.amax_q;
    dav2_arena_release(mark);

    if (oh_out) *oh_out = oh;
    if (ow_out) *ow_out = ow;
    return out;
}
