/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Register-level C model of student_gemm -- see accel_emu.h.
 *
 * Timing is not modelled, but *progress* is: a GEMM job completes a few
 * weight rows each time STATUS is read, writing each row's accumulators and
 * then its {max, min} statistics exactly as the hardware's drain does. The
 * driver's streaming consumers therefore really do see a job half done.
 * A job is latched from the registers at the CTRL write, as in the RTL
 * (detected on the next register access of any kind, before that access
 * takes effect).
 */
#include "accel_emu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STUDENT_GEMM0_BASE_ADDR 0x20010000u
#include <reggen/student_gemm.h>

enum { NROWS = 128, KMAX = 2048, ROWS_PER_POLL = 5, NREGS = 64 };

static uint32_t regs[NREGS];
#define R(name) regs[(STUDENT_GEMM_##name##_OFFSET) >> 2]

/* the latched job */
static struct {
    int      busy, requant, gather, add, relu, lut, lut_load, a16, w16, ostats, onchip, osums;
    uint32_t lut_addr;
    uint32_t x_addr, add_mx, add_mh, add_shift;
    uint32_t a_addr, a_stride, w_addr, w_stride, c_addr, c_stride, s_addr, p_addr;
    uint32_t k, m, n;
    uint32_t g_addr, g_geom, g_chan, g_conv, g_start;
    uint32_t m_done;                 /* GEMM: weight rows finished */
    uint32_t writes;                 /* for DBG2 */
} job;

static void *ptr(uint32_t a) { return (void *)(uintptr_t)a; }

/* the block's lookup-table RAM (CTRL.lut), kept between jobs */
static int16_t lut_tab[16384];
static int32_t cr_ram[131072];          /* the result RAM (CTRL.onchip) */
static int     lut_valid;

static void fail(const char *what)
{
    fprintf(stderr, "accel_emu: %s\n", what);
    exit(3);
}

/* A-tile element (row t, column k) of the current GEMM job. */
static int32_t a_elem(uint32_t t, uint32_t kk)
{
    if (!job.gather) {
        uint32_t stride = job.a_stride ? job.a_stride : job.k * 2u;
        return ((const int16_t *)ptr(job.a_addr + t * stride))[kk];
    }
    /* gather: row t is output pixel g_start + t, column kk = pos*C + c */
    uint32_t h = job.g_geom >> 16, w = job.g_geom & 0xffffu;
    uint32_t ow = job.g_chan >> 16, C = job.g_chan & 0xffffu;
    uint32_t ks = job.g_conv & 15u, st = (job.g_conv >> 4) & 15u, pad = (job.g_conv >> 8) & 15u;
    uint32_t ky0 = (job.g_conv >> 12) & 15u, kx0 = (job.g_conv >> 16) & 15u;
    uint32_t pix = (job.g_start >> 16) * ow + (job.g_start & 0xffffu) + t;
    uint32_t oy = pix / ow, ox = pix % ow;
    uint32_t pos = kk / C, c = kk % C;
    uint32_t lin = ky0 * ks + kx0 + pos;          /* kernel positions row-major */
    int32_t ky = (int32_t)(lin / ks), kx = (int32_t)(lin % ks);
    int32_t iy = (int32_t)(oy * st) + ky - (int32_t)pad;
    int32_t ix = (int32_t)(ox * st) + kx - (int32_t)pad;
    if (iy < 0 || iy >= (int32_t)h || ix < 0 || ix >= (int32_t)w)
        return 0;
    return ((const int16_t *)ptr(job.g_addr))[((uint32_t)iy * w + (uint32_t)ix) * C + c];
}

static void gemm_row(uint32_t m)
{
    uint32_t wstride = job.w_stride ? job.w_stride : (job.w16 ? 2u * job.k : job.k);
    const int8_t *wr = (const int8_t *)ptr(job.w_addr + m * wstride);
    const int16_t *wr16 = (const int16_t *)ptr(job.w_addr + m * wstride);
    int32_t *cr = job.onchip ? cr_ram + job.c_addr + m * job.c_stride
                             : (int32_t *)ptr(job.c_addr + m * job.c_stride);
    int32_t mx = 0, mn = 0;
    for (uint32_t t = 0; t < job.n; t++) {
        int32_t s = 0;
        for (uint32_t kk = 0; kk < job.k; kk++)
            s += a_elem(t, kk) * (job.w16 ? (int32_t)wr16[kk] : (int32_t)wr[kk]);
        cr[t] = s;
        if (t == 0 || s > mx) mx = s;
        if (t == 0 || s < mn) mn = s;
    }
    job.writes += job.n;
    if (job.s_addr) {
        int32_t *sv = (int32_t *)ptr(job.s_addr + m * 8u);
        sv[0] = mx;
        sv[1] = mn;
        job.writes += 2;
    }
}

/* apply_multiplier, as the hardware and the C engine compute it */
static int64_t apply_mult(int64_t v, int64_t mult, int sh)
{
    int64_t r = v * mult;
    if (sh > 0) r += (int64_t)1 << (sh - 1);
    return r >> sh;
}

static void requant_all(void)
{
    const int32_t *par = (const int32_t *)ptr(job.p_addr);
    int32_t amax = 0;
    if (job.lut_load) {
        memcpy(lut_tab, ptr(job.lut_addr), sizeof lut_tab);
        lut_valid = 1;
    }
    if (job.lut && !lut_valid) fail("CTRL.lut before any table was loaded");
    for (uint32_t n = 0; n < job.n; n++) {
        int16_t *orow = (int16_t *)ptr(job.c_addr + n * job.c_stride);
        for (uint32_t m = 0; m < job.m; m++) {
            int64_t acc = job.onchip ? cr_ram[job.a_addr + m * job.a_stride + n]
                        : job.a16 ? ((const int16_t *)ptr(job.a_addr + m * job.a_stride))[n]
                                  : ((const int32_t *)ptr(job.a_addr + m * job.a_stride))[n];
            int64_t mult = par[3 * m];
            int     sh   = par[3 * m + 1];
            int64_t r = acc * mult;
            if (sh > 0) r += (int64_t)1 << (sh - 1);
            r = (r >> sh) + par[3 * m + 2];
            if (r >  8191) r =  8191;
            if (r < -8191) r = -8191;
            if (job.add) {
                /* the epilogue: residual and value rescaled, added, saturated */
                int64_t x = ((const int16_t *)ptr(job.x_addr + n * job.c_stride))[m];
                int64_t v = (int64_t)(int32_t)apply_mult(x, job.add_mx, (int)(job.add_shift & 63u))
                          + (int64_t)(int32_t)apply_mult(r, job.add_mh, (int)((job.add_shift >> 8) & 63u));
                r = v > 8191 ? 8191 : v < -8191 ? -8191 : v;
            }
            if (job.relu && r < 0) r = 0;
            if (job.lut) r = lut_tab[r + 8192];
            orow[m] = (int16_t)r;
            int32_t a = r < 0 ? (int32_t)-r : (int32_t)r;
            if (a > amax) amax = a;
        }
        if (job.ostats) {
            int16_t mx = orow[0], mn = orow[0];
            for (uint32_t m = 1; m < job.m; m++) {
                if (orow[m] > mx) mx = orow[m];
                if (orow[m] < mn) mn = orow[m];
            }
            uint32_t *sv = (uint32_t *)ptr(job.s_addr) + (job.osums ? 4u * n : n);
            sv[0] = ((uint32_t)(uint16_t)mx << 16) | (uint16_t)mn;
            if (job.osums) {
                /* CTRL.osums: the row's sum and sum of squares (40 bits) */
                int32_t sm = 0;
                uint64_t sq = 0;
                for (uint32_t m = 0; m < job.m; m++) {
                    sm += orow[m];
                    sq += (uint64_t)((int64_t)orow[m] * orow[m]);
                }
                sq &= 0xffffffffffull;
                sv[1] = (uint32_t)sm;
                sv[2] = (uint32_t)sq;
                sv[3] = (uint32_t)(sq >> 32);
            }
        }
    }
    R(RQ_AMAX) = (uint32_t)amax;
}

static void latch(void)
{
    uint32_t ctrl = R(CTRL);
    R(CTRL) = 0;
    if (!(ctrl & 1u))
        return;
    if (job.busy)
        fail("CTRL.start written while a job is running");
    memset(&job, 0, sizeof job);
    job.busy     = 1;
    job.requant  = (ctrl >> 1) & 1u;
    job.gather   = ((ctrl >> 2) & 1u) && !job.requant;
    job.add      = ((ctrl >> 3) & 1u) && job.requant;
    job.relu     = ((ctrl >> 4) & 1u) && job.requant;
    job.lut      = ((ctrl >> 5) & 1u) && job.requant;
    job.lut_load = ((ctrl >> 6) & 1u) && job.requant;
    job.lut_addr = R(LUT_ADDR);
    job.a16      = ((ctrl >> 7) & 1u) && job.requant;
    job.w16      = ((ctrl >> 8) & 1u) && !job.requant;
    job.ostats   = ((ctrl >> 9) & 1u) && job.requant;
    job.onchip   = (ctrl >> 10) & 1u;
    job.osums    = ((ctrl >> 12) & 1u) && job.ostats;
    if (job.a16 && (job.n & 1u)) fail("requant A16: N_ROWS odd");
    if (job.lut_load && (job.lut_addr & 3u)) fail("LUT_ADDR unaligned");
    job.x_addr   = R(X_ADDR);   job.add_mx = R(ADD_MULT_X);
    job.add_mh   = R(ADD_MULT_H); job.add_shift = R(ADD_SHIFT);
    job.a_addr   = R(A_ADDR);   job.a_stride = R(A_STRIDE);
    job.w_addr   = R(W_ADDR);   job.w_stride = R(W_STRIDE);
    job.c_addr   = R(C_ADDR);   job.c_stride = R(C_STRIDE);
    job.s_addr   = R(S_ADDR);   job.p_addr   = R(P_ADDR);
    job.k        = R(K_LEN);    job.m        = R(M_LEN);   job.n = R(N_ROWS);
    job.g_addr   = R(G_ADDR);   job.g_geom   = R(G_GEOM);  job.g_chan = R(G_CHAN);
    job.g_conv   = R(G_CONV);   job.g_start  = R(G_START);

    /* the contract, as student_gemm.hjson states it */
    if (job.n < 1 || job.n > NROWS) fail("N_ROWS out of range");
    if (job.requant) {
        if (job.m < 2 || (job.m & 1u) || job.m > KMAX / 2) fail("requant M_LEN out of range");
        if (!job.onchip && (job.a_stride & 3u)) fail("requant A_STRIDE not word aligned");
        if (job.add && job.m > KMAX / 4) fail("requant+add: M_LEN over KMAX/4");
        if (job.add && (job.x_addr & 3u)) fail("requant+add: X_ADDR unaligned");
    } else {
        if (job.k < 4 || (job.k & 3u) || job.k > KMAX) fail("K_LEN out of range");
        if (job.gather && job.k != ((job.g_conv >> 20) & 255u) * (job.g_chan & 0xffffu))
            fail("gather: K_LEN != kernel positions * C");
        if (job.gather && (job.g_chan & 1u)) fail("gather: C odd");
    }
    /* with CTRL.onchip A_ADDR (requant) or C_ADDR (GEMM) count words */
    if (((job.onchip && job.requant ? 0u : job.a_addr) | job.w_addr
         | (job.onchip && !job.requant ? 0u : job.c_addr)
         | job.s_addr | job.p_addr | job.g_addr) & 3u)
        fail("unaligned address");
    if (job.onchip && job.requant && job.a16) fail("requant: CTRL.onchip with CTRL.a16");
    /* the result RAM's words used by the job must exist */
    if (job.onchip && !job.requant
        && (uint64_t)job.c_addr + (uint64_t)(job.m - 1) * job.c_stride + job.n > 131072u)
        fail("GEMM: result RAM overrun");
    if (job.onchip && job.requant
        && (uint64_t)job.a_addr + (uint64_t)(job.m - 1) * job.a_stride + job.n > 131072u)
        fail("requant: result RAM overrun");
    /* CTRL.greuse (bit 11) changes the block's timing only, never the result */
    if (((ctrl >> 11) & 1u) && !job.gather) fail("CTRL.greuse without CTRL.gather");
}

/* Advance the running job; called on every STATUS read. */
static void step(void)
{
    if (!job.busy)
        return;
    if (job.requant) {
        requant_all();
        job.writes = job.n * job.m / 2u;
        job.busy = 0;
    } else {
        for (int i = 0; i < ROWS_PER_POLL && job.m_done < job.m; i++)
            gemm_row(job.m_done++);
        if (job.m_done == job.m)
            job.busy = 0;
    }
    if (!job.busy) {
        /* DAV2_EMU_FAIL_JOB=n: report a bus error on the n-th job, to
         * exercise the driver's and the engine's recovery paths */
        static long jobs_done, fail_at = -1;
        if (fail_at < 0) {
            const char *e = getenv("DAV2_EMU_FAIL_JOB");
            fail_at = e ? atol(e) : 0;
        }
        jobs_done++;
        R(STATUS) = (fail_at && jobs_done == fail_at) ? 6u : 2u;   /* done (+ error) */
        R(DBG2)   = ((job.writes & 0xffffu) << 16) | (job.writes & 0xffffu);
        R(CYCLES) = job.m * (job.k / 4u + 1u) * 3u;       /* plausible, unmodelled */
    }
}

volatile uint32_t *dav2_emu_reg(uint32_t addr)
{
    static int init;
    if (!init) {
        init = 1;
        R(CAPS) = (127u << 24) | ((uint32_t)KMAX << 8) | NROWS;   /* bits 24-30: table, int16 input and weights, row statistics, result RAM, tap reuse, row sums */
    }
    if (addr < STUDENT_GEMM0_BASE_ADDR || addr >= STUDENT_GEMM0_BASE_ADDR + NREGS * 4u)
        fail("register access outside the block");
    latch();                               /* a CTRL write since the last access */
    uint32_t off = addr - STUDENT_GEMM0_BASE_ADDR;
    if (off == STUDENT_GEMM_STATUS_OFFSET) {
        step();
        R(STATUS) = job.busy ? 1u : (R(STATUS) & ~1u);   /* keeps done/error */
    }
    return &regs[off >> 2];
}

uint32_t dav2_emu_mcycle(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 50000000ull + (uint64_t)ts.tv_nsec / 20u);
}
