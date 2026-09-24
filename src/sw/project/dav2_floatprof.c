/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Measure the software floating point.
 *
 * The CV32E40P has no FPU. Every float operation in C is a call into
 * libgcc: __addsf3, __mulsf3, __floatsisf and so on. This file counts
 * those calls and the cycles they take, per routine and per call site.
 *
 * How it is switched on: build_flags.txt in this directory passes
 * -DDAV2_FLOAT_PROFILE=1 and one -Wl,--wrap=<routine> per routine. The
 * linker then sends every call of <routine> to __wrap_<routine> below,
 * which times the real routine (__real_<routine>) and returns its result
 * unchanged. Without those flags (the default: they are commented out)
 * this file compiles to two empty functions, and the program is exactly
 * as before.
 *
 * Cost: two mcycle reads and a table update per call. Measured on the
 * board: about 93 cycles per call, 10.26 s per frame instead of 8.19 s.
 * The cycles reported are those of the real routine only (plus a few
 * cycles for the mcycle read), but the frame time printed in the same run
 * includes the overhead.
 *
 * Call sites are reported as return addresses. Map them to functions and
 * source lines on the PC with addr2line (tools/dav2_floatprof_map.py).
 */
#include <stdint.h>
#include <stdio.h>

#include "dav2.h"

#if DAV2_FLOAT_PROFILE

enum {
    F_ADD, F_SUB, F_MUL, F_DIV,
    F_I2F, F_U2F, F_F2I, F_F2L,
    F_GE, F_GT, F_LE, F_LT,
    F_N
};
static const char *const f_name[F_N] = {
    "__addsf3", "__subsf3", "__mulsf3", "__divsf3",
    "__floatsisf", "__floatunsisf", "__fixsfsi", "__fixsfdi",
    "__gesf2", "__gtsf2", "__lesf2", "__ltsf2",
};

static uint32_t f_cnt[F_N];
static uint64_t f_cyc[F_N];

/* Call sites: open addressing on the return address. The program has a
 * few hundred float call sites; 512 slots keep the table sparse. */
#define SITES 512u
static uint32_t site_pc[SITES];
static uint32_t site_cnt[SITES];
static uint64_t site_cyc[SITES];
static uint32_t site_overflow;

static inline uint32_t cyc(void)
{
    uint32_t v;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(v));
    return v;
}

static void note(int f, uint32_t pc, uint32_t d)
{
    f_cnt[f]++;
    f_cyc[f] += d;
    uint32_t h = (pc >> 1) & (SITES - 1u);
    for (uint32_t n = 0; n < SITES; n++, h = (h + 1u) & (SITES - 1u)) {
        if (site_pc[h] == pc || site_pc[h] == 0) {
            site_pc[h] = pc;
            site_cnt[h]++;
            site_cyc[h] += d;
            return;
        }
    }
    site_overflow++;
}

#define RA() ((uint32_t)(uintptr_t)__builtin_return_address(0))

#define WRAP2(ret, name, idx, ta, tb)                                        \
    extern ret __real_##name(ta, tb);                                        \
    ret __wrap_##name(ta a, tb b)                                            \
    {                                                                        \
        uint32_t t0 = cyc();                                                 \
        ret r = __real_##name(a, b);                                         \
        note(idx, RA(), cyc() - t0);                                         \
        return r;                                                            \
    }
#define WRAP1(ret, name, idx, ta)                                            \
    extern ret __real_##name(ta);                                            \
    ret __wrap_##name(ta a)                                                  \
    {                                                                        \
        uint32_t t0 = cyc();                                                 \
        ret r = __real_##name(a);                                            \
        note(idx, RA(), cyc() - t0);                                         \
        return r;                                                            \
    }

WRAP2(float, __addsf3, F_ADD, float, float)
WRAP2(float, __subsf3, F_SUB, float, float)
WRAP2(float, __mulsf3, F_MUL, float, float)
WRAP2(float, __divsf3, F_DIV, float, float)
WRAP1(float, __floatsisf, F_I2F, int)
WRAP1(float, __floatunsisf, F_U2F, unsigned)
WRAP1(int, __fixsfsi, F_F2I, float)
WRAP1(long long, __fixsfdi, F_F2L, float)
WRAP2(int, __gesf2, F_GE, float, float)
WRAP2(int, __gtsf2, F_GT, float, float)
WRAP2(int, __lesf2, F_LE, float, float)
WRAP2(int, __ltsf2, F_LT, float, float)

void dav2_floatprof_report(uint64_t frame_cycles)
{
    uint64_t tot_cyc = 0;
    uint32_t tot_cnt = 0;
    for (int f = 0; f < F_N; f++) {
        tot_cyc += f_cyc[f];
        tot_cnt += f_cnt[f];
    }
    unsigned permille = frame_cycles ? (unsigned)(tot_cyc * 1000u / frame_cycles) : 0;
    printf("float profile: %u calls, %u kcycles, %u.%u %% of the frame\n",
           (unsigned)tot_cnt, (unsigned)(tot_cyc / 1000u), permille / 10u, permille % 10u);
    for (int f = 0; f < F_N; f++)
        if (f_cnt[f])
            printf("  %s %u calls %u kcycles (%u cycles/call)\n", f_name[f],
                   (unsigned)f_cnt[f], (unsigned)(f_cyc[f] / 1000u),
                   (unsigned)(f_cyc[f] / f_cnt[f]));

    /* all sites, one line each: "FPSITE <pc> <calls> <cycles>" for the
     * host-side mapping script */
    unsigned used = 0;
    for (uint32_t i = 0; i < SITES; i++) {
        if (!site_pc[i])
            continue;
        used++;
        printf("FPSITE %08x %u %u\n", (unsigned)site_pc[i], (unsigned)site_cnt[i],
               (unsigned)site_cyc[i]);
    }
    printf("FPSITE_END %u sites, %u not recorded\n", used, (unsigned)site_overflow);

    for (int f = 0; f < F_N; f++) { f_cnt[f] = 0; f_cyc[f] = 0; }
    for (uint32_t i = 0; i < SITES; i++) { site_pc[i] = 0; site_cnt[i] = 0; site_cyc[i] = 0; }
    site_overflow = 0;
}

void dav2_floatprof_reset(void)
{
    for (int f = 0; f < F_N; f++) { f_cnt[f] = 0; f_cyc[f] = 0; }
    for (uint32_t i = 0; i < SITES; i++) { site_pc[i] = 0; site_cnt[i] = 0; site_cyc[i] = 0; }
    site_overflow = 0;
}

#else

void dav2_floatprof_report(uint64_t frame_cycles) { (void)frame_cycles; }
void dav2_floatprof_reset(void) { }

#endif
