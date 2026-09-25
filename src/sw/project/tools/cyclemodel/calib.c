/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 */
#include <stdint.h>
#define MMIO ((volatile uint32_t *)0x10000000u)
static void mark(uint32_t id) { MMIO[0] = id; }
static volatile uint32_t bench_sink;
static uint32_t scratch[10240] __attribute__((aligned(16)));
int main(void)
{
    enum { N = 4096 };
    uint32_t x = 1, s = 0;
    const uint32_t *bram = scratch;
    volatile uint32_t *ddr = (volatile uint32_t *)0x82000000u;
    mark(1);
    for (uint32_t i = 0; i < N; i++) { x = (x << 1) ^ (x + i); }
    bench_sink = x; mark(2);
    for (uint32_t i = 0; i < N / 8; i++) {
#define A4 x = (x << 1) ^ i; x = x + 7u; x = x ^ (x >> 3); x = x + i;
        A4 A4 A4 A4 A4 A4 A4 A4
    }
    bench_sink = x; mark(3);
    for (uint32_t i = 0; i < N; i++) { x += (uint32_t)(((int64_t)(int32_t)x * (int32_t)(i | 1)) >> 32); }
    bench_sink = x; mark(4);
    for (uint32_t i = 0; i < N; i++) { s += bram[i & 8191u]; }
    bench_sink = s; mark(5);
    for (uint32_t i = 0; i < N; i++) { s += ddr[i]; }
    bench_sink = s; mark(6);
    for (uint32_t i = 0; i < N; i++) { s += ddr[i * 8u]; }
    bench_sink = s; mark(7);
    for (uint32_t i = 0; i < N; i++) { ddr[i] = i; }
    mark(8);
    { volatile int16_t *h = (volatile int16_t *)scratch;
      for (uint32_t i = 0; i < N; i++) { h[i & 16383u] = (int16_t)(h[(i + 1) & 16383u] + 1); } }
    mark(9);
    mark(0xdead);
    for (;;) ;
}
