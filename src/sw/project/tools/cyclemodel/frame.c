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

/* accelerator stubs: every job "completes" at once without computing */
static int32_t stats_buf[2 * 4096];
int dav2_accel_qgemm_async(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc,
                           dav2_accel_stats_t *st)
{ (void)a; (void)acc; st->v = stats_buf; st->tiles = 1; (void)wt; return 1; }
int dav2_accel_conv_async(const int16_t *img, int h, int w, int C, int k, int stride,
                          int pad, const int8_t *wt, int M, int32_t *acc, dav2_accel_stats_t *st)
{ (void)img;(void)h;(void)w;(void)C;(void)k;(void)stride;(void)pad;(void)wt;(void)M;(void)acc;
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
int dav2_accel_busy(void) { return 0; }


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
    for (int b = 0; b < DAV2_PROF_N; b++) { MMIO[3] = (uint32_t)b; MMIO[4] = (uint32_t)dav2_prof_get(b); }
    mark(0xdead);
    for (;;) ;
}
