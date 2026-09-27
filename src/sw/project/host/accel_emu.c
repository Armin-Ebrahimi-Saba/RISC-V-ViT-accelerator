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
    int      busy, requant, gather, add, relu, lut, lut_load, a16, w16, ostats, onchip, osums, grelu;
    int      lutint, wsh, lerp, ln, lnb, rng, rngl, dpar, msums, nparam, sat16;
    uint32_t dp_fk, dp_gb, dp_sh, dp_m0, rng_beta_m, rng_beta_sh;
    uint32_t ln_zs, ln_fm, ln_invm, ln_sh;
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
/* CTRL.ln: what phase A leaves in the block for phase B */
static int16_t ln_zq[NROWS * KMAX];
static int32_t ln_g[KMAX], ln_b[KMAX];
static int     ln_valid;
/* CTRL.rng: the table (aligned scales and biases) and the running range */
static int32_t rng_S[2048], rng_B[2048];
static int     rng_valid;
static uint32_t rng_rows;
static int64_t rng_vmax, rng_vmin;
static int32_t emu_align(int32_t m, int32_t sh, int32_t e)
{
    const int32_t d = sh - e;
    return d > 31 ? (m < 0 ? -1 : 0) : d < 0 ? 0 : (m >> d);
}
static inline int32_t emu_mulh(int32_t a, int32_t b) { return (int32_t)(((int64_t)a * b) >> 32); }
static uint32_t ln_n, ln_k;

static void fail(const char *what)
{
    fprintf(stderr, "accel_emu: %s\n", what);
    exit(3);
}

/* A-tile element (row t, column k) of the current GEMM job. */
static int32_t a_elem_mem(uint32_t t, uint32_t kk)
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
    int32_t v = ((const int16_t *)ptr(job.g_addr))[((uint32_t)iy * w + (uint32_t)ix) * C + c];
    return (job.grelu && v < 0) ? 0 : v;          /* CTRL.grelu */
}

/* The A tile, read when the job starts, as the block loads it before its
 * first MAC: a caller that writes A later (a producer that is late) gives
 * a wrong result here too. */
static int32_t a_snap[NROWS * KMAX];
static void a_snapshot(void)
{
    for (uint32_t t = 0; t < job.n; t++)
        for (uint32_t kk = 0; kk < job.k; kk++)
            a_snap[t * job.k + kk] = a_elem_mem(t, kk);
}
static inline int32_t a_elem(uint32_t t, uint32_t kk) { return a_snap[t * job.k + kk]; }

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
            s += a_elem(t, kk) * (job.w16 ? ((int32_t)wr16[kk] >> job.wsh) : (int32_t)wr[kk]);
        cr[t] = s;
        if (t == 0 || s > mx) mx = s;
        if (t == 0 || s < mn) mn = s;
    }
    job.writes += job.n;
    if (job.rng) {
        /* the range unit: this row's extremes through the table */
        const int32_t bv = emu_mulh(rng_B[m], (int32_t)job.rng_beta_m) >> (job.rng_beta_sh & 31u);
        const int64_t hv = (int64_t)emu_mulh(mx, rng_S[m]) + bv;
        const int64_t lv = (int64_t)emu_mulh(mn, rng_S[m]) + bv;
        if (hv > rng_vmax) rng_vmax = hv;
        if (lv < rng_vmin) rng_vmin = lv;
    }
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
        /* CTRL.lutint loads 256 words (into the RAM's first words) */
        memcpy(lut_tab, ptr(job.lut_addr), job.lutint ? 256u * 4u : sizeof lut_tab);
        lut_valid = 1;
    }
    if (job.lut && !lut_valid) fail("CTRL.lut before any table was loaded");
    static int32_t dpar_tab[3 * KMAX];
    if (job.dpar) {
        /* CTRL.dpar: the parameters from the table */
        const int k = (int)((job.dp_sh >> 16) & 31u), r = (int)((job.dp_sh >> 8) & 31u);
        for (uint32_t m = 0; m < job.m; m++) {
            const int32_t S = rng_S[job.dp_m0 + m], B = rng_B[job.dp_m0 + m];
            dpar_tab[3 * m]     = emu_mulh(S, (int32_t)job.dp_fk) >> r;
            dpar_tab[3 * m + 1] = (int32_t)(job.dp_sh & 63u);
            dpar_tab[3 * m + 2] = ((job.dp_sh >> 24) & 1u) ? 0
                                : (int32_t)(((int64_t)emu_mulh(B, (int32_t)job.dp_gb) + ((int64_t)1 << (k - 1))) >> k);
        }
        par = dpar_tab;
    }
    for (uint32_t n = 0; n < job.n; n++) {
        int16_t *orow = (int16_t *)ptr(job.c_addr + n * job.c_stride);
        for (uint32_t m = 0; m < job.m; m++) {
            int64_t acc = job.onchip ? cr_ram[job.a_addr + m * job.a_stride + n]
                        : job.a16 ? ((const int16_t *)ptr(job.a_addr + m * job.a_stride))[n]
                                  : ((const int32_t *)ptr(job.a_addr + m * job.a_stride))[n];
            const uint32_t pr = job.nparam ? n : m;          /* CTRL.nparam: by output row */
            int64_t mult = par[3 * pr];
            int     sh   = par[3 * pr + 1];
            int64_t r = acc * mult;
            if (sh > 0) r += (int64_t)1 << (sh - 1);
            r = (int64_t)(int32_t)(r >> sh) + par[3 * pr + 2];
            const int64_t lim = job.sat16 ? 32767 : 8191;    /* CTRL.sat16 */
            if (r >  lim) r =  lim;
            if (r < -lim) r = -lim;
            if (job.add) {
                /* the epilogue: residual and value rescaled, added, saturated */
                int64_t x = ((const int16_t *)ptr(job.x_addr + n * job.c_stride))[m];
                int64_t v = (int64_t)(int32_t)apply_mult(x, job.add_mx, (int)(job.add_shift & 63u))
                          + (int64_t)(int32_t)apply_mult(r, job.add_mh, (int)((job.add_shift >> 8) & 63u));
                r = v > 8191 ? 8191 : v < -8191 ? -8191 : v;
            }
            if (job.relu && r < 0) r = 0;
            if (job.lut && job.lutint) {
                /* CTRL.lutint: word i = {L[i+1], L[i]} of the first 256 */
                const uint32_t u = (uint32_t)(r + 8192);
                const int32_t lo = lut_tab[2 * (u >> 6)], hi = lut_tab[2 * (u >> 6) + 1];
                r = lo + (((hi - lo) * (int32_t)(u & 63u)) >> 6);
            } else if (job.lut) {
                r = lut_tab[r + 8192];
            }
            orow[m] = (int16_t)r;
            int32_t a = r < 0 ? (int32_t)-r : (int32_t)r;
            if (a > amax) amax = a;
        }
        if (job.msums) {
            /* CTRL.msums: each m's sum over the job's rows */
            int32_t *sv = (int32_t *)ptr(job.s_addr);
            for (uint32_t m = 0; m < job.m; m++)
                sv[m] = (n ? sv[m] : 0) + orow[m];
        } else if (job.ostats) {
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
    job.ln       = ((ctrl >> 20) & 1u) && !job.requant;
    job.lnb      = ((ctrl >> 21) & 1u) && job.ln;
    job.gather   = ((ctrl >> 2) & 1u) && !job.requant && !job.ln;
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
    job.grelu    = ((ctrl >> 13) & 1u) && job.gather;
    job.lerp     = ((ctrl >> 19) & 1u) && !job.requant && !job.gather && !job.ln;
    job.wsh      = job.w16 ? (int)((ctrl >> 14) & 15u) : 0;
    job.lutint   = ((ctrl >> 18) & 1u) && job.lut;
    if (((ctrl >> 13) & 1u) && !job.gather) fail("CTRL.grelu without CTRL.gather");
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
    job.rng      = ((ctrl >> 22) & 1u) && !job.requant && !job.ln && !job.lerp;
    job.rngl     = ((ctrl >> 23) & 1u) && job.rng;
    job.dpar     = ((ctrl >> 24) & 1u) && job.requant;
    job.msums    = ((ctrl >> 25) & 1u) && job.requant;
    job.nparam   = ((ctrl >> 26) & 1u) && job.requant && !job.dpar;
    job.sat16    = ((ctrl >> 27) & 1u) && job.requant;
    if (job.sat16 && (job.lut || job.add)) fail("CTRL.sat16 with the table or the add");
    if (job.msums && (R(S_ADDR) & 3u)) fail("CTRL.msums: S_ADDR unaligned");
    job.rng_beta_m = R(RNG_BETA_M); job.rng_beta_sh = R(RNG_BETA_SH);
    job.dp_fk    = R(DP_FK);    job.dp_gb    = R(DP_GB);
    job.dp_sh    = R(DP_SH);    job.dp_m0    = R(DP_M0) & 0x7ffu;
    if (job.rngl) {
        /* the table, read at the start as the block loads it first */
        const int32_t es = (int16_t)(R(RNG_E) & 0xffffu), eb = (int16_t)(R(RNG_E) >> 16);
        const int32_t *sv = (const int32_t *)ptr(R(RNG_S_ADDR));
        const int32_t *bv = R(RNG_B_ADDR) ? (const int32_t *)ptr(R(RNG_B_ADDR)) : 0;
        if (job.m > 2048) fail("CTRL.rng_load: M_LEN over 2048");
        if ((R(RNG_S_ADDR) | R(RNG_B_ADDR)) & 3u) fail("CTRL.rng_load: unaligned table");
        for (uint32_t m = 0; m < job.m; m++) {
            rng_S[m] = emu_align(sv[2 * m], sv[2 * m + 1], es);
            rng_B[m] = bv ? emu_align(bv[2 * m], bv[2 * m + 1], eb) : 0;
        }
        rng_valid = 1;
        rng_rows = job.m;
        rng_vmax = rng_vmin = 0;
    }
    if (job.rng && (!rng_valid || job.m > rng_rows)) fail("CTRL.rng without its table");
    if (job.dpar) {
        const uint32_t k = (job.dp_sh >> 16) & 31u;
        if (!rng_valid || job.dp_m0 + job.m > rng_rows) fail("CTRL.dpar outside the table");
        if (!((job.dp_sh >> 24) & 1u) && (k < 1 || k > 31)) fail("CTRL.dpar: DP_SH.k out of range");
    }
    job.ln_zs    = R(LN_ZS);    job.ln_fm    = R(LN_FM);
    job.ln_invm  = R(LN_INVM);  job.ln_sh    = R(LN_SH);
    if (job.ln) {
        /* CTRL.ln: its own contract (see student_gemm.hjson) */
        if (job.n < 1 || job.n > NROWS) fail("CTRL.ln: N_ROWS out of range");
        if (job.k < 2 || (job.k & 1u) || job.k > KMAX) fail("CTRL.ln: K_LEN out of range");
        if (!job.lnb) {
            if (job.m != job.n) fail("CTRL.ln: M_LEN != N_ROWS");
            if ((job.a_addr | job.a_stride | job.p_addr | job.w_addr | job.x_addr) & 3u)
                fail("CTRL.ln: unaligned address");
            if (job.ln_zs >> 25) fail("CTRL.ln: LN_ZS over 25 bits");
            ln_valid = 0;
            a_snapshot();
        } else {
            const uint32_t bsr = (job.ln_sh >> 16) & 31u;
            if (!ln_valid || job.n != ln_n || job.k != ln_k)
                fail("CTRL.lnb without a matching phase A just before");
            if ((job.c_addr | job.c_stride) & 3u) fail("CTRL.lnb: unaligned output");
            if (bsr < 1 || bsr > 30) fail("CTRL.lnb: LN_SH.bsr out of range");
        }
        return;
    }
    ln_valid = 0;                          /* any other job overwrites the tile */

    if (job.lerp && job.n != 2) fail("CTRL.lerp: N_ROWS must be 2");
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
    if (!job.requant)
        a_snapshot();
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

/* CTRL.ln, phase A: zq in place, the channel extremes, the largest |y| */
static void ln_phase_a(void)
{
    const int32_t *par = (const int32_t *)ptr(job.p_addr);
    const int32_t *g = (const int32_t *)ptr(job.w_addr), *b = (const int32_t *)ptr(job.x_addr);
    uint64_t ymax = 0;
    for (uint32_t c = 0; c < job.k; c++) { ln_g[c] = g[c]; ln_b[c] = b[c]; }
    for (uint32_t t = 0; t < job.n; t++)
        for (uint32_t c = 0; c < job.k; c++) {
            int64_t r = (int64_t)(int32_t)apply_mult(a_elem(t, c), par[3 * t], par[3 * t + 1])
                      + par[3 * t + 2];
            ln_zq[t * job.k + c] = (int16_t)(r > 8191 ? 8191 : r < -8191 ? -8191 : r);
        }
    for (uint32_t c = 0; c < job.k; c++) {
        int32_t lo = ln_zq[c], hi = ln_zq[c];
        for (uint32_t t = 1; t < job.n; t++) {
            const int32_t z = ln_zq[t * job.k + c];
            if (z < lo) lo = z;
            if (z > hi) hi = z;
        }
        const int64_t G = (int64_t)ln_g[c] * (int64_t)(job.ln_zs & 0x1ffffffu);
        for (int e = 0; e < 2; e++) {
            int64_t y = (((int64_t)(e ? hi : lo) * G) >> 31) + ln_b[c];
            uint64_t a = y < 0 ? (uint64_t)-y : (uint64_t)y;
            if (a > ymax) ymax = a;
        }
    }
    R(LN_YMAX_LO) = (uint32_t)ymax;
    R(LN_YMAX_HI) = (uint32_t)(ymax >> 32);
    ln_n = job.n; ln_k = job.k;
    ln_valid = 1;
}

/* CTRL.ln | CTRL.lnb, phase B: channel multipliers and biases, the output */
static void ln_phase_b(void)
{
    const int shift = (int)(job.ln_sh & 63u), rr = (int)((job.ln_sh >> 8) & 31u);
    const int bsr = (int)((job.ln_sh >> 16) & 31u);
    int32_t amax = 0;
    for (uint32_t c = 0; c < job.k; c++) {
        const int32_t G = (int32_t)(((int64_t)(int32_t)((uint32_t)ln_g[c] << 12)
                                     * (int64_t)(int32_t)job.ln_fm) >> 32) >> rr;
        const int32_t B = (int32_t)(((int64_t)(int32_t)(((int64_t)ln_b[c] * (int64_t)(int32_t)job.ln_invm) >> 32)
                                     + ((int64_t)1 << (bsr - 1))) >> bsr);
        for (uint32_t t = 0; t < job.n; t++) {
            int64_t r = (int64_t)(int32_t)apply_mult(ln_zq[t * job.k + c], G, shift) + B;
            r = r > 8191 ? 8191 : r < -8191 ? -8191 : r;
            ((int16_t *)ptr(job.c_addr + t * job.c_stride))[c] = (int16_t)r;
            const int32_t a = r < 0 ? (int32_t)-r : (int32_t)r;
            if (a > amax) amax = a;
        }
    }
    R(RQ_AMAX) = (uint32_t)amax;
    ln_valid = 0;
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
    } else if (job.ln) {
        if (job.lnb) ln_phase_b(); else ln_phase_a();
        job.writes = job.lnb ? job.n * job.k / 2u : 0u;
        job.busy = 0;
    } else if (job.lerp) {
        /* CTRL.lerp: the two A rows (read at the start), interpolated */
        int16_t *o = (int16_t *)ptr(job.c_addr);
        const int32_t w = (int32_t)(job.add_mx & 511u);
        for (uint32_t e = 0; e < job.k; e++) {
            const int32_t t = a_elem(0, e), b = a_elem(1, e);
            o[e] = (int16_t)(t + (((b - t) * w) >> 8));
        }
        job.writes = job.k / 2u;
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
        R(RNG_VMAX_LO) = (uint32_t)rng_vmax;  R(RNG_VMAX_HI) = (uint32_t)((uint64_t)rng_vmax >> 32);
        R(RNG_VMIN_LO) = (uint32_t)rng_vmin;  R(RNG_VMIN_HI) = (uint32_t)((uint64_t)rng_vmin >> 32);
        R(CYCLES) = job.m * (job.k / 4u + 1u) * 3u;       /* plausible, unmodelled */
    }
}

volatile uint32_t *dav2_emu_reg(uint32_t addr)
{
    static int init;
    if (!init) {
        init = 1;
        R(CAPS) = (255u << 24) | (7u << 21) | ((uint32_t)KMAX << 8) | NROWS;   /* bits 24-31: table, int16 input and weights, row statistics, result RAM, tap reuse, row sums, gather ReLU; bit 23: LayerNorm */
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
