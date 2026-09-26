/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 */
/* Cycle-estimate harness: engine kernels on synthetic data, accelerator
 * stubbed. Runs in tools/emu.py (Unicorn + a CV32E40P timing model). */
#include <stdint.h>
#include <stddef.h>
#include "dav2.h"
#include "dav2_accel.h"
void bench_attention(const dav2_tensor_t *qkv, int n, dav2_tensor_t *ctx);

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

/* accelerator stubs: every job "completes" at once without computing */
static int32_t stats_buf[2 * 4096];
int dav2_accel_qgemm_async(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc,
                           dav2_accel_stats_t *st)
{ (void)a; (void)acc; st->v = stats_buf; st->tiles = 1; (void)wt; return 1; }
int dav2_accel_conv_async(const int16_t *img, int h, int w, int C, int k, int stride,
                          int pad, const int8_t *wt, int M, int32_t *acc, dav2_accel_stats_t *st,
                          int in_relu)
{ (void)img;(void)h;(void)w;(void)C;(void)k;(void)stride;(void)pad;(void)wt;(void)M;(void)acc;(void)in_relu;
  st->v = stats_buf; st->tiles = 1; return 1; }
int dav2_accel_gemm_raw_async(const int16_t *a, uint32_t as, const int8_t *w, uint32_t ws,
                              int32_t *acc, int N, int K, int M)
{ (void)a;(void)as;(void)w;(void)ws;(void)acc;(void)N;(void)K;(void)M; return 1; }
int dav2_accel_requant_rows_async(const int32_t *acc, int N, int M, int m0, int mc,
                                  const int32_t *par, int16_t *dst, int32_t *amax,
                                  const dav2_rq_epi_t *epi)
{ (void)acc;(void)N;(void)M;(void)m0;(void)mc;(void)par;(void)dst;(void)epi; *amax = 8000; return 1; }
/* synchronous requantisation: the CPU waits for the job. Charged as
 * 1.5 bus beats per element at 2 cycles per beat (MMIO[6] adds cycles). */
int dav2_accel_requant(const int32_t *acc, int N, int M, const int32_t *params,
                       int16_t *out, int32_t *amax_out)
{ (void)acc;(void)params;(void)out; MMIO[6] = (uint32_t)(3 * N * M); *amax_out = 8000; return 1; }
int dav2_accel_finish(void) { return 1; }
int dav2_accel_lut_ok(void) { return 1; }   /* GELU in the requantisation job */
/* int16-weight GEMM (attention): instant, rows' statistics left as they are;
 * strided requantisation (the context): 1.5 beats per element, 2 cycles each */
int dav2_accel_w16_ok(void) { return 1; }
int dav2_accel_present(void) { return 1; }
int dav2_accel_gemm16_async(const int16_t *a, uint32_t as, const int16_t *w, uint32_t ws,
                            int32_t *acc, int N, int K, int M, int32_t *st)
{ (void)a;(void)as;(void)w;(void)ws;(void)acc;(void)N;(void)K;(void)M;(void)st; return 1; }
int dav2_accel_requant_lut_async(const int32_t *acc, int N, int M, const int32_t *params,
                                 int16_t *out, int out_stride, const int16_t *lut, int lut_load)
{ (void)acc;(void)N;(void)M;(void)params;(void)out;(void)out_stride;(void)lut;(void)lut_load;
  return 0; }
int dav2_accel_requant_stride_async(const int32_t *acc, int N, int M, const int32_t *params,
                                    int16_t *out, int out_stride)
{ (void)acc;(void)N;(void)M;(void)params;(void)out;(void)out_stride; return 0; }
int dav2_accel_requant_stride(const int32_t *acc, int N, int M, const int32_t *params,
                              int16_t *out, int out_stride, int32_t *amax)
{ (void)acc;(void)params;(void)out;(void)out_stride; MMIO[6] = (uint32_t)(3 * N * M); *amax = 8000; return 1; }
/* int16-input requantisation (LayerNorm): N*M/2 words in and out, charged
 * at 2 cycles per beat */
int dav2_accel_requant16(const int16_t *in, int N, int M, const int32_t *params,
                         int16_t *out, int32_t *amax, uint32_t *ostats)
{ (void)in;(void)params;(void)out;
  if (ostats) for (int n = 0; n < N; n++) ostats[n] = (8000u << 16) | (uint16_t)-8000;
  MMIO[6] = (uint32_t)(2 * N * M + N); *amax = 8000; return 1; }
int dav2_accel_ostats_ok(void) { return 1; }
int dav2_accel_osums_ok(void) { return 0; }
int dav2_accel_onchip_ok(void) { return 0; }
int dav2_accel_qgemm_onchip_async(const dav2_tensor_t *a, const dav2_qw_t *wt,
                                  dav2_accel_stats_t *st)
{ (void)a; (void)wt; st->v = 0; st->tiles = 0; return 0; }
int dav2_accel_conv_onchip_async(const int16_t *img, int h, int w, int C, int k, int stride,
                                 int pad, const int8_t *wt, int M, dav2_accel_stats_t *st,
                                 int in_relu)
{ (void)img;(void)h;(void)w;(void)C;(void)k;(void)stride;(void)pad;(void)wt;(void)M;(void)in_relu;
  st->v = 0; st->tiles = 0; return 0; }
int dav2_accel_grelu_ok(void) { return 0; }
int dav2_accel_wsh_ok(void) { return 0; }
int dav2_accel_lerp_ok(void) { return 0; }
int dav2_accel_lerp_async(const int16_t *t, uint32_t row_stride, int16_t *out, int n, int w)
{ (void)t;(void)row_stride;(void)out;(void)n;(void)w; return 0; }
int dav2_accel_lutint_ok(void) { return 0; }
int dav2_accel_gemm16_shift_async(const int16_t *a, uint32_t as, const int16_t *w, uint32_t ws,
                                  int wsh, int32_t *acc, int N, int K, int M, int32_t *st)
{ (void)a;(void)as;(void)w;(void)ws;(void)wsh;(void)acc;(void)N;(void)K;(void)M;(void)st; return 0; }
int dav2_accel_gemm16_cr_async(const int16_t *a, uint32_t as, const int16_t *w, uint32_t ws,
                               int wsh, uint32_t cr_base, int N, int K, int M, int32_t *st)
{ (void)a;(void)as;(void)w;(void)ws;(void)wsh;(void)cr_base;(void)N;(void)K;(void)M;(void)st;
  return 0; }
int dav2_accel_requant_lut_cr_async(uint32_t cr_base, int N, int M, const int32_t *params,
                                    int16_t *out, int out_stride, const int16_t *lut, int lut_load)
{ (void)cr_base;(void)N;(void)M;(void)params;(void)out;(void)out_stride;(void)lut;(void)lut_load;
  return 0; }
int dav2_accel_requant_stride_cr_async(uint32_t cr_base, int N, int M, const int32_t *params,
                                       int16_t *out, int out_stride)
{ (void)cr_base;(void)N;(void)M;(void)params;(void)out;(void)out_stride; return 0; }
int dav2_accel_busy(void) { return 0; }

static uint32_t rs = 12345;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static int16_t ract(int amp) { return (int16_t)((int)(rnd() % (2u * amp + 1u)) - amp); }

#define DDR ((uint8_t *)0x80000000u)

int main(void)
{
    dav2_arena_init(DDR + (4u << 20), 32u << 20);
    const int N = 82, C = 384;
    dav2_prof_reset();

    /* fake weight descriptors in "DDR" (the blob region) */
    dav2_xf_t *s = (dav2_xf_t *)(DDR + (1u << 20));
    dav2_xf_t *b = s + 4096;
    int32_t *g = (int32_t *)(b + 4096), *be = g + 1024;
    for (int i = 0; i < 4096; i++) {
        s[i] = xf_norm(1000 + rnd() % 3000, 36 + (int)(rnd() % 4));
        b[i] = xf_norm((int32_t)(rnd() % 2000) - 1000, 30 + (int)(rnd() % 6));
    }
    for (int i = 0; i < 1024; i++) { g[i] = 20000 + (int32_t)(rnd() % 30000); be[i] = (int32_t)(rnd() % 60000) - 30000; }
    for (int i = 0; i < 4096; i++) { int32_t v = (int32_t)(rnd() % 400000); stats_buf[2*i] = v; stats_buf[2*i+1] = -v + 5000; }

    dav2_tensor_t x = dav2_tensor_new(N, C);
    for (int i = 0; i < N * C; i++) x.v[i] = ract(8000);
    x.scale = xf_norm(577, 20); x.amax_q = -1;
    dav2_tensor_t o = dav2_tensor_new(N, C);

    mark(1); dav2_layernorm(&x, g, be, &o); mark(0);  /* affine */
    mark(2); dav2_layernorm(&x, 0, 0, &o);  mark(0);  /* plain  */
    mark(3);

    dav2_qw_t wt = { (const int8_t *)DDR, s, b, 1152, 384 };
    dav2_tensor_t q = dav2_tensor_new(N, 1152);
    dav2_qgemm(&x, &wt, &q);                          /* qkv-shaped requant */
    mark(0);
    dav2_qw_t w2 = { (const int8_t *)DDR, s, b, 384, 384 };
    mark(4);
    dav2_qgemm_ex(&x, &w2, &x, 0, &x);                /* proj with residual */
    mark(0);

    dav2_tensor_t h = dav2_tensor_new(N, 1536);
    for (int i = 0; i < N * 1536; i++) h.v[i] = ract(8000);
    h.scale = xf_norm(1049, 20);
    mark(5);
    dav2_gelu(&h);
    mark(0);

    for (int i = 0; i < N * 1152; i++) q.v[i] = ract(8000);
    q.scale = xf_norm(1049, 21);
    dav2_tensor_t ctx = dav2_tensor_new(N, C);
    mark(6);
    bench_attention(&q, N, &ctx);
    mark(0);

    dav2_tensor_t a2 = dav2_tensor_new(N, C);
    for (int i = 0; i < N * C; i++) a2.v[i] = ract(8000);
    a2.scale = xf_norm(999, 20); a2.amax_q = -1; x.amax_q = -1;
    mark(7);
    dav2_add(&x, &a2, &o);
    mark(0);
    {
        dav2_tensor_t src = dav2_tensor_new(72 * 72, 32);
        for (int i = 0; i < 72 * 72 * 32; i++) src.v[i] = ract(8000);
        src.scale = xf_norm(999, 20);
        dav2_tensor_t dst = dav2_tensor_new(126 * 126, 32);
        mark(8);
        dav2_interpolate(&src, 72, 72, 126, 126, &dst);
        mark(0);
        uint32_t hsh = 0;
        for (int i = 0; i < 126 * 126 * 32; i++) hsh = hsh * 31u + (uint16_t)dst.v[i];
        MMIO[5] = hsh;
    }
    for (int d = 0; d < DAV2_SUB_N; d++) { MMIO[3] = (uint32_t)d; MMIO[4] = (uint32_t)dav2_sub_get(d); }
    mark(0xdead);
    for (;;) ;
}
