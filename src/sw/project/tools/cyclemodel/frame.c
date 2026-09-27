/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 */
/* Cycle-estimate harness: engine kernels on synthetic data, accelerator
 * stubbed. Runs in tools/emu.py (Unicorn + a CV32E40P timing model). */
#include <stdint.h>
#include <stddef.h>
#include "dav2.h"
#include "dav2_accel.h"


#define MMIO        ((volatile uint32_t *)0x10000000u)
/* MMIO[0] write: phase marker. MMIO[1],[2] read: cycle counter lo, hi. */
uint64_t dav2_cycles(void) { uint32_t lo = MMIO[1], hi = MMIO[2]; return ((uint64_t)hi << 32) | lo; }
void dav2_progress(const char *s) { (void)s; }
static void mark(uint32_t id) { MMIO[0] = id; }

/* minimal libc */
void *memset(void *d, int c, size_t n) { uint8_t *p = d; while (n--) *p++ = (uint8_t)c; return d; }
void *memcpy(void *d, const void *s, size_t n) { uint8_t *p = d; const uint8_t *q = s; while (n--) *p++ = *q++; return d; }
size_t strlen(const char *s) { size_t n = 0; while (s[n]) n++; return n; }
int strcmp(const char *a, const char *b) { while (*a && *a == *b) { a++; b++; } return (uint8_t)*a - (uint8_t)*b; }
int printf(const char *f, ...) { (void)f; return 0; }
#include <stdarg.h>
int sprintf(char *o, const char *f, ...)
{
    va_list ap; va_start(ap, f);
    char *p = o;
    for (; *f; f++) {
        if (*f != '%') { *p++ = *f; continue; }
        f++;
        if (*f == 's') { const char *s = va_arg(ap, const char *); while (*s) *p++ = *s++; }
        else if (*f == 'd') {
            int v = va_arg(ap, int); char t[12]; int n = 0;
            if (v < 0) { *p++ = '-'; v = -v; }
            do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v);
            while (n) *p++ = t[--n];
        } else *p++ = *f;
    }
    *p = 0; va_end(ap);
    return (int)(p - o);
}
char *strcpy(char *d, const char *s) { char *r = d; while ((*d++ = *s++)) ; return r; }
char *strcat(char *d, const char *s) { strcpy(d + strlen(d), s); return d; }
int puts(const char *s) { (void)s; return 0; }
int putchar(int c) { return c; }

/* Accelerator model. A job starts when the previous one has finished (the
 * driver settles a pending job before it starts the next, so the CPU waits
 * for it) and then runs for an estimated time while the CPU continues; a
 * finish call waits until that time. MMIO[6] adds cycles to the CPU clock.
 * Job times from the block's design, with 2.1 cycles per bus word as
 * measured on the board (boot report, "cycles/beat"):
 *   GEMM, per tile of nt <= 128 rows: A tile nt*K/2 words, then per weight
 *        row the MAC time, the pipeline tail and nt + 2 drain writes. MAC
 *        time: K / MACW cycles (MACW weights per cycle), but not less than
 *        the row's weight words (K/4, or K/2 for int16 weights) at the bus
 *        rate. Pipeline tail: PIPE cycles from the last weight to the drain.
 *        With the result RAM (ONCHIP_C) the drain takes nt/4 + 2 cycles and
 *        the next requantisation reads no input words from the bus.
 *   requantisation: parameters 3*M words, input N*M words (int16: half),
 *        output one element per cycle, N*M/2 words written. */
#define BEAT 21                         /* tenths of a cycle per bus word */
static uint64_t busy_until;
enum { K_GEMM_ENC, K_GEMM_CONV, K_ATT, K_RQ, K_RQ_ADD, K_RQ16, K_RQ_CTX, K_RQ_EXP, K_LERP, K_TP, K_LN, K_SMX, K_N };
static uint64_t acc_time[K_N];          /* modelled job time by kind, reported at the end */
static void wait_done(void)
{
    /* as the driver's wait loops: background and producer steps while the
     * job runs (dav2_producer_idle), then the rest of the job's time */
    uint64_t now = dav2_cycles();
    while (busy_until > now && dav2_producer_idle())
        now = dav2_cycles();
    if (busy_until > now) MMIO[6] = (uint32_t)(busy_until - now);
}
static int job_kind;
static uint64_t idle_before[K_N];       /* block idle before a job, by the job's kind */
static void start_job_at(uint64_t cyc, uintptr_t site)
{
    const uint64_t now0 = dav2_cycles();
    if (busy_until && now0 > busy_until) {
        idle_before[job_kind] += now0 - busy_until;
        /* by call site (IDLE_SITES=n in emu.py prints the largest) */
        MMIO[7] = 1;
        MMIO[3] = (uint32_t)site; MMIO[4] = (uint32_t)(now0 - busy_until);
        MMIO[7] = 0;
    }
    acc_time[job_kind] += cyc;
    wait_done();
    busy_until = dav2_cycles() + cyc;
}
#define start_job(c) start_job_at((c), (uintptr_t)__builtin_return_address(0))
#ifndef MACW
#define MACW 2                          /* weights per MAC cycle (1 before round 15) */
#endif
#ifndef PIPE
#define PIPE 5                          /* MAC pipeline stages (2 before round 15) */
#endif
#ifndef CR_WORDS
#define CR_WORDS 131072                 /* result RAM words (round 16) */
#endif
#ifndef ONCHIP_C
#define ONCHIP_C 1                      /* results kept on chip (round 16; 0 before) */
#endif
static int onchip_last;                 /* the last GEMM kept its result on chip */
#ifndef ONCHIP_CONV
#define ONCHIP_CONV 1                   /* convolutions use the result RAM too (round 17) */
#endif
#ifndef REUSE_C
#define REUSE_C 1                       /* gather tap reuse, CTRL.greuse (round 17) */
#endif
/* A GEMM job's time. rk > 1: a gather with tap reuse over an output row of
 * ow pixels: a tile row whose left neighbour is the previous tile row
 * copies (rk-1)/rk of its K/2 words at one per cycle and reads the rest,
 * the reads running ahead through the FIFO. */
static uint64_t gemm_core(int N, int K, int M, int w16, int onchip, int rk, int ow)
{
    uint64_t c = 0;
    uint64_t mac = (uint64_t)K / MACW;
    uint64_t bus = (uint64_t)K / (w16 ? 2 : 4) * BEAT / 10;
    uint64_t row = (mac > bus ? mac : bus) + PIPE;
    const uint64_t full = (uint64_t)K / 2 * BEAT / 10;
    uint64_t reused = full;
    if (rk > 1) {
        uint64_t rd = (uint64_t)K / 2 / rk * BEAT / 10;
        reused = (uint64_t)K / 2 > rd ? (uint64_t)K / 2 : rd;
    }
    for (int n0 = 0; n0 < N; n0 += 128) {
        int nt = N - n0 < 128 ? N - n0 : 128;
        /* tile rows without a left neighbour in the tile: the first, and
         * each one that starts an output row (a multiple of ow). Counted
         * without a loop: the model's own instructions count as CPU time. */
        uint64_t fresh = (uint64_t)nt;
        if (rk > 1)
            fresh = 1u + (uint64_t)((n0 + nt - 1) / ow - n0 / ow);
        c += fresh * full + ((uint64_t)nt - fresh) * reused;
        c += (uint64_t)M * (row + (onchip ? (uint64_t)nt / 4 + 2 : (uint64_t)(nt + 2) * BEAT / 10));
    }
    return c;
}
#ifndef RNG_C
#define RNG_C 1                         /* the range unit and CTRL.dpar (round 31) */
#endif
static int32_t stats_buf[2 * 4096];      /* the statistics the stubs return (below) */
static int rng_armed, rng_valid, rng_m;
static dav2_accel_rng_t rng_cfg;
int dav2_accel_rng_ok(void) { return RNG_C; }
void dav2_accel_rng_next(const dav2_accel_rng_t *r) { rng_cfg = *r; rng_armed = RNG_C; }
static int32_t st_mulh(int32_t a, int32_t b) { return (int32_t)(((int64_t)a * b) >> 32); }
static int32_t st_align(dav2_xf_t v, int e)
{ int d = v.sh - e; return d > 31 ? (v.m < 0 ? -1 : 0) : d < 0 ? 0 : (v.m >> d); }
/* the range the block would report for the statistics the stubs return
 * (stats_buf), computed in a section the model does not count */
int dav2_accel_rng_get(int64_t *vmax, int64_t *vmin)
{
    if (!rng_valid) return 0;
    rng_valid = 0;
    MMIO[7] = 1;
    int64_t hx = 0, hn = 0;
    for (int m = 0; m < rng_m; m++) {
        const int32_t S = st_align(rng_cfg.s[m], rng_cfg.es);
        const int32_t B = rng_cfg.b ? st_align(rng_cfg.b[m], rng_cfg.eb) : 0;
        const int32_t bv = st_mulh(B, rng_cfg.beta_m) >> rng_cfg.beta_sh;
        const int64_t hv = (int64_t)st_mulh(stats_buf[2 * m], S) + bv;
        const int64_t lv = (int64_t)st_mulh(stats_buf[2 * m + 1], S) + bv;
        if (hv > hx) hx = hv;
        if (lv < hn) hn = lv;
    }
    MMIO[7] = 0;
    *vmax = hx; *vmin = hn; return 1;
}
/* the table load of an operation's first job (scales and biases, 4M words) */
static uint64_t rng_take(int M)
{ uint64_t c = rng_armed ? (uint64_t)4 * M * BEAT / 10 : 0; rng_valid = rng_armed; rng_armed = 0;
  rng_m = M; return c; }
static uint64_t gemm_cycles(int N, int K, int M, int w16)
{
    const int onchip = ONCHIP_C && N <= 128 && (long)N * M <= CR_WORDS && !w16;
    onchip_last = onchip;
    return gemm_core(N, K, M, w16, onchip, 0, 1);
}
static uint64_t requant_cycles(int N, int M, int in16, int add)
{
    uint64_t e = (uint64_t)N * M;
    uint64_t tiles = (uint64_t)(N + 127) / 128;
    uint64_t words = 3u * M * tiles + (onchip_last && !in16 ? 0 : (in16 ? e / 2 : e)) + (add ? e / 2 : 0);
    uint64_t out = e > e / 2 * BEAT / 10 ? e : e / 2 * BEAT / 10;
    return words * BEAT / 10 + out;
}
static int32_t stats_buf[2 * 4096];
int dav2_accel_qgemm_async(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc,
                           dav2_accel_stats_t *st)
{ (void)acc; job_kind = K_GEMM_ENC; start_job(gemm_cycles(a->n, a->c, wt->m, 0) + rng_take(wt->m));
  st->v = stats_buf; st->tiles = 1; return 2; }
#ifndef GRELU_C
#define GRELU_C 1                       /* ReLU while gathering, CTRL.grelu (round 19) */
#endif
int dav2_accel_grelu_ok(void) { return GRELU_C; }
#ifndef WSH_C
#define WSH_C 1                         /* CTRL.wsh and CTRL.lutint (round 22) */
#endif
int dav2_accel_wsh_ok(void) { return WSH_C; }
#ifndef LERP_C
#define LERP_C 1                        /* the LERP job, CTRL.lerp (round 27) */
#endif
int dav2_accel_lerp_ok(void) { return LERP_C; }
/* two rows of n int16 read (n words), n/2 words written, at the bus rate */
int dav2_accel_lerp_async(const int16_t *t, uint32_t row_stride, int16_t *out, int n, int w)
{ (void)t;(void)row_stride;(void)out;(void)w;
  if (!LERP_C) return 0;
  job_kind = K_LERP; start_job((uint64_t)(n + n / 2) * BEAT / 10 + 20); return 2; }
int dav2_accel_lutint_ok(void) { return WSH_C; }
int dav2_accel_gemm16_shift_async(const int16_t *a, uint32_t as, const int16_t *w, uint32_t ws,
                                  int wsh, int32_t *acc, int N, int K, int M, int32_t *st)
{ (void)a;(void)as;(void)w;(void)ws;(void)wsh;(void)acc;(void)st;
  if (!WSH_C) return 0;
  job_kind = K_ATT; start_job(gemm_cycles(N, K, M, 1)); return 2; }
int dav2_accel_conv_async(const int16_t *img, int h, int w, int C, int k, int stride,
                          int pad, const int8_t *wt, int M, int32_t *acc, dav2_accel_stats_t *st,
                          int in_relu)
{ (void)img;(void)wt;(void)acc;(void)in_relu;
  int oh = (h + 2 * pad - k) / stride + 1, ow = (w + 2 * pad - k) / stride + 1;
  int single = k * k * C <= 2048;
  onchip_last = 0;
  job_kind = K_GEMM_CONV;
  start_job(gemm_core(oh * ow, k * k * C, M, 0, 0, (REUSE_C && stride == 1 && single) ? k : 0, ow)
            + (single ? rng_take(M) : (rng_armed = 0)));
  st->v = stats_buf; st->tiles = 1; return 2; }
int dav2_accel_conv_onchip_async(const int16_t *img, int h, int w, int C, int k, int stride,
                                 int pad, const int8_t *wt, int M, dav2_accel_stats_t *st,
                                 int in_relu)
{ (void)img;(void)wt;(void)in_relu;
  int oh = (h + 2 * pad - k) / stride + 1, ow = (w + 2 * pad - k) / stride + 1;
  int single = k * k * C <= 2048;
  if (!ONCHIP_C || !ONCHIP_CONV || !single || (long)oh * ow * M > CR_WORDS) return 0;
  onchip_last = 1;
  job_kind = K_GEMM_CONV;
  start_job(gemm_core(oh * ow, k * k * C, M, 0, 1, (REUSE_C && stride == 1) ? k : 0, ow) + rng_take(M));
  st->v = stats_buf; st->tiles = 1; return 2; }
int dav2_accel_gemm_raw_async(const int16_t *a, uint32_t as, const int8_t *w, uint32_t ws,
                              int32_t *acc, int N, int K, int M)
{ (void)a;(void)as;(void)w;(void)ws;(void)acc; job_kind = K_ATT; start_job(gemm_cycles(N, K, M, 0)); return 2; }
int dav2_accel_requant_rows_async(const int32_t *acc, int N, int M, int m0, int mc,
                                  const int32_t *par, int16_t *dst, int32_t *amax,
                                  const dav2_rq_epi_t *epi)
{ (void)acc;(void)M;(void)m0;(void)par;(void)dst;
  job_kind = (epi && epi->add) ? K_RQ_ADD : K_RQ;
  const int wpr = (epi && epi->ostats) ? (epi->osums ? 4 : 1) : 0;
  for (int n = 0; wpr && n < N; n++) {
      /* plausible statistics: a range of +-8000, a spread of 3000 */
      uint32_t *r = epi->ostats + (size_t)n * wpr;
      r[0] = (8000u << 16) | (uint16_t)-8000;
      if (wpr == 4) {
          uint64_t sq = (uint64_t)mc * 3000u * 3000u;
          r[1] = (uint32_t)(mc * 100); r[2] = (uint32_t)sq; r[3] = (uint32_t)(sq >> 32);
      }
  }
  /* a table load first: 8192 words, or 256 for the interpolating table */
  const uint64_t lutw = (epi && epi->lut && epi->lut_load) ? (epi->lut_int ? 256u : 8192u) : 0u;
  uint64_t c = requant_cycles(N, mc, 0, epi && epi->add);
  /* CTRL.dpar: no parameter words; one row per cycle through the unit */
  if (epi && epi->dpar)
      c = c - (uint64_t)3 * mc * ((N + 127) / 128) * BEAT / 10 + (uint64_t)mc + 10;
  start_job(c + (uint64_t)N * wpr * BEAT / 10 + lutw * BEAT / 10);
  *amax = 8000; return 2; }
#ifndef OSUMS_C
#define OSUMS_C 1                       /* output row sums, CTRL.osums (round 18) */
#endif
int dav2_accel_osums_ok(void) { return OSUMS_C; }
int dav2_accel_requant(const int32_t *acc, int N, int M, const int32_t *params,
                       int16_t *out, int32_t *amax_out)
{ (void)acc;(void)params;(void)out; job_kind = K_RQ; start_job(requant_cycles(N, M, 0, 0)); wait_done();
  *amax_out = 8000; return 1; }
int dav2_accel_finish(void) { wait_done(); return 1; }
int dav2_accel_busy(void) { return dav2_cycles() < busy_until; }
int dav2_accel_lut_ok(void) { return 1; }
int dav2_accel_w16_ok(void) { return 1; }
int dav2_accel_present(void) { return 1; }
int dav2_accel_gemm16_async(const int16_t *a, uint32_t as, const int16_t *w, uint32_t ws,
                            int32_t *acc, int N, int K, int M, int32_t *st)
{ (void)a;(void)as;(void)w;(void)ws;(void)acc;(void)st; job_kind = K_ATT; start_job(gemm_cycles(N, K, M, 1)); return 2; }
#ifndef EXP_C
#define EXP_C 1                         /* the softmax's exponential on the block (round 20) */
#endif
int dav2_accel_requant_lut_async(const int32_t *acc, int N, int M, const int32_t *params,
                                 int16_t *out, int out_stride, const int16_t *lut, int lut_load)
{ (void)acc;(void)params;(void)out;(void)out_stride;(void)lut;
  if (!EXP_C) return 0;
  onchip_last = 0;
  job_kind = K_RQ_EXP;
  start_job(requant_cycles(N, M, 0, 0) + (lut_load ? 8192u * BEAT / 10 : 0));
  return 2; }
int dav2_accel_requant_stride_async(const int32_t *acc, int N, int M, const int32_t *params,
                                    int16_t *out, int out_stride)
{ (void)acc;(void)params;(void)out;(void)out_stride;
  onchip_last = 0;
  job_kind = K_RQ_CTX; start_job(requant_cycles(N, M, 0, 0)); return 2; }
int dav2_accel_requant_stride(const int32_t *acc, int N, int M, const int32_t *params,
                              int16_t *out, int out_stride, int32_t *amax)
{ (void)acc;(void)params;(void)out;(void)out_stride;
  job_kind = K_RQ_CTX; start_job(requant_cycles(N, M, 0, 0)); wait_done(); *amax = 8000; return 1; }
int dav2_accel_requant16(const int16_t *in, int N, int M, const int32_t *params,
                         int16_t *out, int32_t *amax, uint32_t *ostats)
{ (void)in;(void)params;(void)out;
  if (ostats) for (int n = 0; n < N; n++) ostats[n] = (8000u << 16) | (uint16_t)-8000;
  job_kind = K_RQ16; start_job(requant_cycles(N, M, 1, 0) + (ostats ? (uint64_t)N * BEAT / 10 : 0)); wait_done();
  *amax = 8000; return 1; }
int dav2_accel_ostats_ok(void) { return 1; }
#ifndef SMX_C
#define SMX_C 1                         /* the softmax's normalisation on the block (round 32) */
#endif
int dav2_accel_smx_ok(void) { return SMX_C; }
/* the exponential job, and M column sums written at the end */
int dav2_accel_exp_async(const int32_t *acc, uint32_t cr_base, int N, int M,
                         const int32_t *params, int16_t *out, int out_stride,
                         const int16_t *lut, int lut_load, int32_t *msums)
{ (void)params;(void)out;(void)out_stride;(void)lut;
  if (!SMX_C || !EXP_C) return 0;
  MMIO[7] = 1;
  for (int m = 0; m < M; m++) msums[m] = 100000;     /* plausible sums */
  MMIO[7] = 0;
  onchip_last = acc == 0 && cr_base + (long)N * M <= CR_WORDS;
  job_kind = K_RQ_EXP;
  start_job(requant_cycles(N, M, 0, 0) + (lut_load ? 8192u * BEAT / 10 : 0) + (uint64_t)M * BEAT / 10);
  return 2; }
/* int16 in (M rows of N), parameters per row n (3N words), N x M out */
int dav2_accel_norm16_async(const int16_t *pt, uint32_t pt_pitch, int N, int M,
                            const int32_t *params, int16_t *out, int out_stride)
{ (void)pt;(void)pt_pitch;(void)params;(void)out;(void)out_stride;
  if (!SMX_C) return 0;
  job_kind = K_SMX;
  start_job(requant_cycles(N, M, 1, 0) - (uint64_t)3 * M * BEAT / 10 + (uint64_t)3 * N * BEAT / 10);
  return 2; }
#ifndef TP_C
#define TP_C 1                          /* v^T by a transposition job (round 29) */
#endif
/* an int16-input requantisation: M rows of N int16 in, the same out */
int dav2_accel_transpose16_async(const int16_t *in, uint32_t in_pitch, int N, int M,
                                 int16_t *out, int out_stride)
{ (void)in;(void)in_pitch;(void)out;(void)out_stride;
  if (!TP_C) return 0;
  job_kind = K_TP; start_job(requant_cycles(N, M, 1, 0)); return 2; }
#ifndef LN_C
#define LN_C 1                          /* the LayerNorm job, CTRL.ln (round 30) */
#endif
int dav2_accel_ln_ok(void) { return LN_C; }
/* phase A: 3N + 2C words of parameters, the N x C/2-word tile, then one
 * element pair per cycle and about 15 cycles per channel pair */
int dav2_accel_ln_a(const int16_t *x, uint32_t x_pitch, int N, int C, const int32_t *tokpar,
                    const int32_t *g, const int32_t *b, uint32_t zs, uint64_t *ymax)
{ (void)x;(void)x_pitch;(void)tokpar;(void)g;(void)b;(void)zs;
  if (!LN_C) return 0;
  job_kind = K_LN;
  start_job((uint64_t)(3 * N + 2 * C + N * C / 2) * BEAT / 10 + (uint64_t)N * C / 2
            + 15u * (uint64_t)C / 2);
  wait_done();
  *ymax = (uint64_t)1 << 30; return 1; }
/* phase B: four products per channel pair, then N x C/2 words out */
int dav2_accel_ln_b(int N, int C, int16_t *out, uint32_t out_pitch, int32_t fm, int32_t invm,
                    int shift, int rr, int bsr, int32_t *amax)
{ (void)out;(void)out_pitch;(void)fm;(void)invm;(void)shift;(void)rr;(void)bsr;
  if (!LN_C) return 0;
  job_kind = K_LN;
  start_job(2u * (uint64_t)C + (uint64_t)N * C / 2 * BEAT / 10 + 20);
  wait_done();
  *amax = 8000; return 1; }
#ifndef ATTCR_C
#define ATTCR_C 1                       /* the attention's S and C in the result RAM (round 28) */
#endif
int dav2_accel_gemm16_cr_async(const int16_t *a, uint32_t as, const int16_t *w, uint32_t ws,
                               int wsh, uint32_t cr_base, int N, int K, int M, int32_t *st)
{ (void)a;(void)as;(void)w;(void)ws;(void)wsh;(void)st;
  if (!ATTCR_C || !ONCHIP_C || cr_base + (long)N * M > CR_WORDS) return 0;
  job_kind = K_ATT; start_job(gemm_core(N, K, M, 1, 1, 0, 1)); return 2; }
int dav2_accel_requant_lut_cr_async(uint32_t cr_base, int N, int M, const int32_t *params,
                                    int16_t *out, int out_stride, const int16_t *lut, int lut_load)
{ (void)params;(void)out;(void)out_stride;(void)lut;
  if (!EXP_C || !ATTCR_C || cr_base + (long)N * M > CR_WORDS) return 0;
  onchip_last = 1;
  job_kind = K_RQ_EXP;
  start_job(requant_cycles(N, M, 0, 0) + (lut_load ? 8192u * BEAT / 10 : 0));
  return 2; }
int dav2_accel_requant_stride_cr_async(uint32_t cr_base, int N, int M, const int32_t *params,
                                       int16_t *out, int out_stride)
{ (void)params;(void)out;(void)out_stride;
  if (!ATTCR_C || cr_base + (long)N * M > CR_WORDS) return 0;
  onchip_last = 1;
  job_kind = K_RQ_CTX; start_job(requant_cycles(N, M, 0, 0)); return 2; }
int dav2_accel_onchip_ok(void) { return ONCHIP_C; }
int dav2_accel_qgemm_onchip_async(const dav2_tensor_t *a, const dav2_qw_t *wt,
                                  dav2_accel_stats_t *st)
{ if ((long)a->n * wt->m > CR_WORDS) return 0;
  job_kind = K_GEMM_ENC; start_job(gemm_cycles(a->n, a->c, wt->m, 0) + rng_take(wt->m));
  st->v = stats_buf; st->tiles = 1; return 2; }


#define DDR ((uint8_t *)0x80000000u)
#define IMG ((const int16_t *)0x81E00000u)
static int16_t depth[126 * 126];

int main(void)
{
    uint32_t scale_bits = *(const uint32_t *)(0x81E00000u + 126u * 126u * 3u * 2u);
    dav2_blob_init(DDR);
    dav2_arena_init(DDR + 0x02000000u, 32u << 20);
    dav2_set_image(IMG, scale_bits);
    dav2_cfg_t cfg = { 126, 9, 82 };
    dav2_xf_t sc;
    for (int i = 0; i < 4096; i++) { stats_buf[2 * i] = 300000; stats_buf[2 * i + 1] = -290000; }
    mark(1);
    dav2_infer(&cfg, depth, &sc);
    mark(0);
    wait_done();
    for (int b = 0; b < DAV2_PROF_N; b++) { MMIO[3] = (uint32_t)b; MMIO[4] = (uint32_t)dav2_prof_get(b); }
    for (int k = 0; k < K_N; k++) { MMIO[3] = (uint32_t)(100 + k); MMIO[4] = (uint32_t)acc_time[k]; }
    for (int k = 0; k < K_N; k++) { MMIO[3] = (uint32_t)(300 + k); MMIO[4] = (uint32_t)idle_before[k]; }
    for (int d = 0; d < DAV2_SUB_N; d++) { MMIO[3] = (uint32_t)(200 + d); MMIO[4] = (uint32_t)dav2_sub_get(d); }
    mark(0xdead);
    for (;;) ;
}
