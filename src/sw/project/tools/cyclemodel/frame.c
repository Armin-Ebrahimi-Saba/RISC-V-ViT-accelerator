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
static uint64_t acc_time[8];            /* modelled job time by kind, reported at the end */
enum { K_GEMM_ENC, K_GEMM_CONV, K_ATT, K_RQ, K_RQ_ADD, K_RQ16, K_RQ_CTX, K_N };
static void wait_done(void)
{
    uint64_t now = dav2_cycles();
    if (busy_until > now) MMIO[6] = (uint32_t)(busy_until - now);
}
static int job_kind;
static void start_job(uint64_t cyc)
{
    acc_time[job_kind] += cyc;
    wait_done();
    busy_until = dav2_cycles() + cyc;
}
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
static uint64_t gemm_cycles(int N, int K, int M, int w16)
{
    const int onchip = ONCHIP_C && N <= 128 && (long)N * M <= CR_WORDS && !w16;
    onchip_last = onchip;
    uint64_t c = 0;
    uint64_t mac = (uint64_t)K / MACW;
    uint64_t bus = (uint64_t)K / (w16 ? 2 : 4) * BEAT / 10;
    uint64_t row = (mac > bus ? mac : bus) + PIPE;
    for (int n0 = 0; n0 < N; n0 += 128) {
        int nt = N - n0 < 128 ? N - n0 : 128;
        c += (uint64_t)nt * K / 2 * BEAT / 10;
        c += (uint64_t)M * (row + (onchip ? (uint64_t)nt / 4 + 2 : (uint64_t)(nt + 2) * BEAT / 10));
    }
    return c;
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
{ (void)acc; job_kind = K_GEMM_ENC; start_job(gemm_cycles(a->n, a->c, wt->m, 0));
  st->v = stats_buf; st->tiles = 1; return 2; }
int dav2_accel_conv_async(const int16_t *img, int h, int w, int C, int k, int stride,
                          int pad, const int8_t *wt, int M, int32_t *acc, dav2_accel_stats_t *st)
{ (void)img;(void)wt;(void)acc;
  int oh = (h + 2 * pad - k) / stride + 1, ow = (w + 2 * pad - k) / stride + 1;
  job_kind = K_GEMM_CONV; start_job(gemm_cycles(oh * ow, k * k * C, M, 0));
  st->v = stats_buf; st->tiles = 1; return 2; }
int dav2_accel_gemm_raw_async(const int16_t *a, uint32_t as, const int8_t *w, uint32_t ws,
                              int32_t *acc, int N, int K, int M)
{ (void)a;(void)as;(void)w;(void)ws;(void)acc; job_kind = K_ATT; start_job(gemm_cycles(N, K, M, 0)); return 2; }
int dav2_accel_requant_rows_async(const int32_t *acc, int N, int M, int m0, int mc,
                                  const int32_t *par, int16_t *dst, int32_t *amax,
                                  const dav2_rq_epi_t *epi)
{ (void)acc;(void)M;(void)m0;(void)par;(void)dst;
  job_kind = (epi && epi->add) ? K_RQ_ADD : K_RQ;
  start_job(requant_cycles(N, mc, 0, epi && epi->add)); *amax = 8000; return 2; }
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
int dav2_accel_onchip_ok(void) { return ONCHIP_C; }
int dav2_accel_qgemm_onchip_async(const dav2_tensor_t *a, const dav2_qw_t *wt,
                                  dav2_accel_stats_t *st)
{ if ((long)a->n * wt->m > CR_WORDS) return 0;
  job_kind = K_GEMM_ENC; start_job(gemm_cycles(a->n, a->c, wt->m, 0));
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
    mark(0xdead);
    for (;;) ;
}
