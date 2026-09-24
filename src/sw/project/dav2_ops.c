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
#include "dav2_mathf.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------- profiler */

const char *const dav2_prof_name[DAV2_PROF_N] = {
    "gemm (accelerator)", "gemm (cpu)", "requantise", "attention",
    "layernorm", "gelu", "add/relu", "im2col", "interpolate", "other",
};
static uint64_t prof_acc[DAV2_PROF_N];
void     dav2_prof_reset(void)                 { memset(prof_acc, 0, sizeof prof_acc); }
void     dav2_prof_add(int b, uint64_t c)      { prof_acc[b] += c; }
uint64_t dav2_prof_get(int b)                  { return prof_acc[b]; }
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
    t.amax_q = -1;
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
    int64_t q;
    uint32_t bits;
    /* __builtin_memcpy, not memcpy: the flow compiles with -fno-builtin, so
     * a plain memcpy here is a call to libsys's byte-by-byte copy. The
     * builtin is a register move. (The same holds for every 4-byte bit
     * copy in this file.) */
    __builtin_memcpy(&bits, &m, 4);
    if (((bits >> 23) & 0xffu) != 0u) {
        /* Normal float: the mantissa with its hidden bit is m's frexp
         * fraction times 2^24, so f * 2^31 is exactly mantissa << 7 and the
         * "+ 0.5" of the general path truncates away. Same (mult, shift) as
         * below, without frexpf and two soft-float operations -- this runs
         * once per weight row, ~3500 times per transformer block. */
        q = (int64_t)(((bits & 0x7fffffu) | 0x800000u) << 7);
        e = (int)((bits >> 23) & 0xffu) - 126;
    } else {
        float f = dav2_frexpf(m, &e);             /* m = f * 2^e, f in [0.5,1) */
        q = (int64_t)(f * 2147483648.0f + 0.5f);
        if (q > 2147483647LL) { q >>= 1; e += 1; }
    }
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
    /* round(acc * mult / 2^shift), exactly, without a 64-bit shift.
     *
     * The multiplier is scaled into [2^30, 2^31) by dav2_make_multiplier,
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

static inline int32_t iround(float f)
{
    /* Round half away from zero. The sign is taken from the float's sign
     * bit, not from a comparison: the core has no FPU, so "f >= 0.0f" is a
     * library call. The results are the same, including for -0.0f (both
     * forms give 0). */
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    return (int32_t)((bits >> 31) ? f - 0.5f : f + 0.5f);
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

/* A convolution's geometry, for a GEMM whose A matrix is the im2col of an
 * image rather than a tensor in memory. */
typedef struct { int h, w, k, stride, pad; } conv_desc_t;

static void qgemm_impl(const dav2_tensor_t *a, const conv_desc_t *cv,
                       const dav2_qw_t *wt, const dav2_tensor_t *res, int relu,
                       dav2_tensor_t *out);

void dav2_qgemm(const dav2_tensor_t *a, const dav2_qw_t *wt, dav2_tensor_t *out)
{
    qgemm_impl(a, 0, wt, 0, 0, out);
}

/* dav2_qgemm followed by dav2_add(res, result) and/or dav2_relu, with the
 * same result bit for bit. On the accelerator the add and the ReLU happen
 * inside the requantisation job (its epilogue), so neither the
 * intermediate tensor nor the CPU pass over it exists. out may be res. */
void dav2_qgemm_ex(const dav2_tensor_t *a, const dav2_qw_t *wt,
                   const dav2_tensor_t *res, int relu, dav2_tensor_t *out)
{
    qgemm_impl(a, 0, wt, res, relu, out);
}

/* a is the input image when cv is set: N = output pixels, K = k*k*C.
 * res, relu: the epilogue, see dav2_qgemm_ex. */
static void qgemm_impl(const dav2_tensor_t *a, const conv_desc_t *cv,
                       const dav2_qw_t *wt, const dav2_tensor_t *res, int relu,
                       dav2_tensor_t *out)
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
        out->scale = 1.0f;
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

    /* Exact output range, including bias, so nothing clips. The per-row
     * extremes come from the accelerator's drain when it produced them (two
     * words per row per tile); otherwise from a scan of acc. */
    float amax;
    uint32_t amax_bits = 0;               /* bit pattern of +0.0f */
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
        float k_c = a->scale * wt->s[m];
        /* Keep k_c for the parameter pass below, which needs the same
         * product. It is stored in the row's first parameter slot, which
         * that pass overwrites after reading it. */
        __builtin_memcpy(&par[3 * m], &k_c, 4);
        float bias = wt->b ? wt->b[m] : 0.0f;
        float hi = (float)cmax * k_c + bias;
        float lo = (float)cmin * k_c + bias;
        /* Compare the magnitudes as integers. For floats that are not
         * negative, the order of the bit patterns is the order of the
         * values, so this finds the same maximum without two library
         * calls per row. */
        uint32_t ah, al;
        float fah = dav2_fabsf(hi), fal = dav2_fabsf(lo);
        __builtin_memcpy(&ah, &fah, 4);
        __builtin_memcpy(&al, &fal, 4);
        if (ah > amax_bits) amax_bits = ah;
        if (al > amax_bits) amax_bits = al;
    }
    __builtin_memcpy(&amax, &amax_bits, 4);
    if (run == 2) {
        /* every row's statistics have been read, so the job is done or
         * failed; collect it (and learn which) */
        uint64_t w0 = dav2_cycles();
        if (!dav2_accel_finish()) goto redo_on_cpu;
        waited += dav2_cycles() - w0;
    }
    float out_scale = (amax > 0.0f) ? amax * (1.0f / (float)DAV2_ACT_QMAX) : 1.0f;
    float inv_out = 1.0f / out_scale;

    /* ---- requantisation ---------------------------------------------------
     *
     * Per-row parameters, then the int32 -> int16 conversion. On the
     * accelerator this runs in chunks of RQ_CHUNK rows, and the parameters
     * of chunk c+1 are computed while the block converts chunk c. The
     * parameters are the same numbers in any order, so the result is too. */
    int par_done = 0;
#define PAR_UPTO(end_m) do {                                                  \
        for (; par_done < (end_m); par_done++) {                              \
            int m_ = par_done, sh_;                                           \
            float k_;                /* a->scale * wt->s[m_], see above */ \
            __builtin_memcpy(&k_, &par[3 * m_], 4);                                     \
            dav2_make_multiplier(k_ * inv_out, &par[3 * m_], &sh_);           \
            par[3 * m_ + 1] = sh_;                                            \
            par[3 * m_ + 2] = wt->b ? iround(wt->b[m_] * inv_out) : 0;        \
        }                                                                     \
    } while (0)

    const int res_ok = !res || (res->n == N && res->c == M
                                && (((uintptr_t)res->v) & 3u) == 0);
    if ((M & 1) == 0 && run && res_ok) {
        enum { RQ_CHUNK = 256 };          /* <= CAPS.KMAX/4, the add-mode limit */
        int32_t rq_amax = 0;
        int ok = 1;
        float fin_scale = out_scale;
        dav2_rq_epi_t epi;
        memset(&epi, 0, sizeof epi);
        epi.relu = relu;
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
            /* as in dav2_add (the sum is commutative, so operand order
             * does not matter) */
            float bound = (float)amax_x * res->scale + (float)amax_h * out_scale;
            fin_scale = (bound > 0.0f) ? bound * (1.0f / (float)DAV2_ACT_QMAX) : 1.0f;
            float inv = 1.0f / fin_scale;
            dav2_make_multiplier(res->scale * inv, &epi.mx, &epi.sx);
            dav2_make_multiplier(out_scale * inv, &epi.mh, &epi.sh);
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
                                               (epi.add || epi.relu) ? &epi : 0) != 0;
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
            dav2_prof_add(DAV2_PROF_REQUANT, (now - t_start) - waited);
            out->n = N;
            out->c = M;
            out->scale = fin_scale;
            out->amax_q = rq_amax;
            trace_tensor("qgemm", out);
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
    dav2_arena_release(mark);
    return;

redo_on_cpu:
    /* The accelerator failed mid-operation (it has disabled itself and
     * printed why); acc is incomplete. Start over -- every path is now the
     * CPU's. */
    dav2_arena_release(mark);
    qgemm_impl(a, cv, wt, res, relu, out);
}

/* -------------------------------------------------------------- LayerNorm */

/* One LayerNorm output in Q16: (x - mean) / std * g + b, see dav2_layernorm. */
static inline int32_t ln_y(int32_t x, int32_t mean16, int32_t am, int ash,
                           int32_t gq, int32_t bq)
{
    int32_t d16 = (x << 16) - mean16;
    int32_t z16 = apply_multiplier(d16, am, ash);        /* (x-mean)/std, Q16 */
    int64_t pg  = (int64_t)z16 * (int64_t)gq;
    return (int32_t)((pg + (1 << 14)) >> 15) + bq;
}


void dav2_layernorm(const dav2_tensor_t *in, const float *g, const float *b,
                    dav2_tensor_t *out)
{
    const int N = in->n, C = in->c;
    const size_t mark = dav2_arena_mark();
    PROF_START();

    /* The core has no FPU: every float operation is a library call of
     * 50-150 cycles. The statistics are per row (N = 82 of them) and stay
     * in float; the per-element work (N*C = 31k) is fixed point:
     *
     *   y = (x - mean) * (s * inv_std) * g[c] + b[c]
     *
     * with x - mean in Q16, s*inv_std as a (mult, shift) pair per row, g in
     * Q15 and b in Q16. y comes out in Q16 -- resolution 1.5e-5 against an
     * output step of ~1e-3 after quantisation to 14 bits -- and is then
     * requantised to the tensor's common scale.
     *
     * One pass computes y and stores it (int32, arena) while tracking the
     * range; a second requantises. A version that recomputed y instead of
     * storing it was slower: on this core a load and a store cost less
     * than the twelve instructions of arithmetic they replace. */
    int32_t *y16 = (int32_t *)dav2_arena_alloc((size_t)N * C * sizeof(int32_t));
    int32_t *gq  = (int32_t *)dav2_scratch;               /* g[c] in Q15 */
    int32_t *bq  = gq + C;                                /* b[c] in Q16 */
    if (!y16) { dav2_arena_release(mark); return; }
    for (int c = 0; c < C; c++) {
        gq[c] = iround(g[c] * 32768.0f);
        bq[c] = iround(b[c] * 65536.0f);
    }

    const float s = in->scale;
    const int wide = (((uintptr_t)in->v | (uintptr_t)out->v) & 3u) == 0 && (C & 3) == 0;
    int32_t amax = 0;
    for (int n = 0; n < N; n++) {
        const int16_t *row = in->v + (size_t)n * C;
        int32_t sum = 0;
        int64_t sq = 0;
        if (wide) {
            const uint32_t *rw = (const uint32_t *)row;
            #pragma GCC unroll 4
            for (int c = 0; c < C / 2; c++) {
                uint32_t x = rw[c];
                int32_t d0 = lo16(x), d1 = hi16(x);
                sum += d0 + d1;
                sq += (int64_t)d0 * d0 + (int64_t)d1 * d1;
            }
        } else {
            for (int c = 0; c < C; c++) {
                int32_t d = (int32_t)row[c];
                sum += d;
                sq += (int64_t)d * (int64_t)d;
            }
        }
        /* mean/variance in the integer domain, then converted once per row.
         * sq <= 384 * 8191^2 = 2.6e10, so sq/C fits comfortably in int32 */
        float mean_q = (float)sum / (float)C;
        float mean_sq = (float)(int32_t)(sq / C)
                      + (float)(int32_t)(sq % C) / (float)C;
        float var_q = mean_sq - mean_q * mean_q;
        if (var_q < 0.0f) var_q = 0.0f;
        /* variance in real units = var_q * s^2; eps matches PyTorch's 1e-6 */
        float inv_std = 1.0f / dav2_sqrtf(var_q * s * s + 1e-6f);

        int32_t mean16 = iround(mean_q * 65536.0f);        /* |x| <= 8191: fits */
        int32_t am; int ash;
        dav2_make_multiplier(s * inv_std, &am, &ash);

        int32_t *orow = y16 + (size_t)n * C;
        if (wide) {
            const uint32_t *rw = (const uint32_t *)row;
            #pragma GCC unroll 4
            for (int c = 0; c < C / 2; c++) {
                uint32_t x = rw[c];
                int32_t y0 = ln_y(lo16(x), mean16, am, ash, gq[2 * c],     bq[2 * c]);
                int32_t y1 = ln_y(hi16(x), mean16, am, ash, gq[2 * c + 1], bq[2 * c + 1]);
                orow[2 * c] = y0;
                orow[2 * c + 1] = y1;
                if (y0 < 0) y0 = -y0;
                if (y1 < 0) y1 = -y1;
                if (y0 > amax) amax = y0;
                if (y1 > amax) amax = y1;
            }
        } else {
            for (int c = 0; c < C; c++) {
                int32_t y = ln_y(row[c], mean16, am, ash, gq[c], bq[c]);
                orow[c] = y;
                if (y < 0) y = -y;
                if (y > amax) amax = y;
            }
        }
    }

    /* common output scale, then the second pass writes the result */
    float out_scale = (amax > 0) ? (float)amax * (1.0f / 65536.0f)
                                  * (1.0f / (float)DAV2_ACT_QMAX) : 1.0f;
    int32_t om; int osh;
    dav2_make_multiplier((1.0f / 65536.0f) / out_scale, &om, &osh);
    const int total = N * C;
    int32_t omax = 0;
    if (wide) {
        uint32_t *ow = (uint32_t *)out->v;
        #pragma GCC unroll 4
        for (int i = 0; i < total / 2; i++) {
            int32_t r0 = sat_act(apply_multiplier(y16[2 * i],     om, osh));
            int32_t r1 = sat_act(apply_multiplier(y16[2 * i + 1], om, osh));
            ow[i] = pack16(r0, r1);
            if (r0 < 0) r0 = -r0;
            if (r1 < 0) r1 = -r1;
            if (r0 > omax) omax = r0;
            if (r1 > omax) omax = r1;
        }
    } else {
        for (int i = 0; i < total; i++) {
            int32_t r = sat_act(apply_multiplier(y16[i], om, osh));
            out->v[i] = (int16_t)r;
            if (r < 0) r = -r;
            if (r > omax) omax = r;
        }
    }
    out->scale = out_scale;
    out->n = N;
    out->c = C;
    out->amax_q = omax;

    PROF_STOP(DAV2_PROF_LAYERNORM);
    trace_tensor("layernorm", out);
    dav2_arena_release(mark);
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
    float bound = (float)amax_a * a->scale + (float)amax_b * b->scale;
    float out_scale = (bound > 0.0f) ? bound * (1.0f / (float)DAV2_ACT_QMAX) : 1.0f;
    float inv = 1.0f / out_scale;

    int32_t ma, mb; int sa, sb;
    dav2_make_multiplier(a->scale * inv, &ma, &sa);
    dav2_make_multiplier(b->scale * inv, &mb, &sb);

    int32_t omax = 0;
    if (wide) {
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

/* GELU is an elementwise map on a 14-bit input, so a 257-entry table with
 * linear interpolation reproduces it to well below quantisation noise. The
 * table costs 257 float evaluations per call, versus 126k for direct
 * evaluation. */
void dav2_gelu(dav2_tensor_t *t)
{
    float lut_f[257];
    const float s = t->scale;
    PROF_START();
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

    /* One 32-bit word per interval: the value at the left end in the low
     * half and the step to the right end in the high half. The step is at
     * most 2 * 8191 in magnitude, so it fits an int16. The interpolation
     * below then needs one load per element instead of two; a load from
     * on-chip RAM costs about 10 cycles on this core. Same arithmetic as
     * lut[i] + ((lut[i+1] - lut[i]) * f >> 6). */
    uint32_t lutp[256];
    for (int i = 0; i < 256; i++)
        lutp[i] = pack16(lut[i], lut[i + 1] - lut[i]);

    const int total = t->n * t->c;
#define GELU_LUT(x) ({ int32_t u_ = (int32_t)(x) + 8192;   /* [1, 16383] */ \
                       uint32_t e_ = lutp[u_ >> 6];       /* [0, 255]   */ \
                       lo16(e_) + ((hi16(e_) * (u_ & 63)) >> 6); })
    int32_t omax = 0;
    if (words_ok(t->v, t->v, t->v, total)) {
        uint32_t *w = (uint32_t *)t->v;
        #pragma GCC unroll 4
        for (int i = 0; i < total / 2; i++) {
            uint32_t x = w[i];
            int32_t r0 = GELU_LUT(lo16(x)), r1 = GELU_LUT(hi16(x));
            w[i] = pack16(r0, r1);
            if (r0 < 0) r0 = -r0;
            if (r1 < 0) r1 = -r1;
            if (r0 > omax) omax = r0;
            if (r1 > omax) omax = r1;
        }
    } else {
        for (int i = 0; i < total; i++) {
            int32_t r = GELU_LUT(t->v[i]);
            t->v[i] = (int16_t)r;
            if (r < 0) r = -r;
            if (r > omax) omax = r;
        }
    }
    t->amax_q = omax;
#undef GELU_LUT
    t->scale = out_scale;
    PROF_STOP(DAV2_PROF_GELU);
    trace_tensor("gelu", t);
}

/* ------------------------------------------------------------ resampling */

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
        qgemm_impl(in, 0, wt, res, relu, &out);
    } else {
        /* Gathered in hardware when the accelerator can (no patch matrix is
         * ever built); otherwise im2col in software, inside qgemm_impl. */
        conv_desc_t cv = { h, w, k, stride, pad };
        qgemm_impl(in, &cv, wt, res, relu, &out);
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
