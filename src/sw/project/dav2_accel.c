/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Driver for the student_gemm int8 GEMM accelerator.
 *
 * On the host (and in any build without the block) every entry point reports
 * "not available" and the engine falls back to the software kernel, so the
 * numerical result is identical either way.
 *
 * How a GEMM runs on the block: the accelerator holds NROWS (16) activation
 * rows at a time, so dav2_accel_qgemm() cuts an N-row problem into ceil(N/16)
 * "jobs", programs each one's addresses into the registers, sets CTRL.start,
 * and polls STATUS until busy drops. All M output columns are produced per
 * job. The int32 results land directly in the caller's accumulator buffer in
 * DDR3; requantisation happens afterwards in dav2_ops.c.
 *
 * The debug registers (DBG..DBG4) are read only when something goes wrong
 * -- a timeout -- or by the self-checks at the bottom of this file. Their
 * packing is defined in src/design/reggen/student_gemm.hjson.
 */

#include "dav2_accel.h"

#if defined(__riscv) && !defined(DAV2_NO_ACCEL)
#define DAV2_ACCEL 1
#elif defined(DAV2_ACCEL_EMU)
/* Host build against the register-level model in host/accel_emu.c: the
 * whole driver runs, bit-exactness with the plain host build checks it. */
#define DAV2_ACCEL 1
#else
#define DAV2_ACCEL 0
#endif

#if DAV2_ACCEL

#include <stdio.h>
#include <string.h>
#ifdef DAV2_ACCEL_EMU
#include "host/accel_emu.h"
#else
#include <rvlab.h>
#endif

/* The fast device window is split by address inside student.sv:
 *   0x2000_0000 student_dma, 0x2001_0000 student_gemm. */
#define STUDENT_GEMM0_BASE_ADDR 0x20010000
#include <reggen/student_gemm.h>

#define GEMM_STATUS   STUDENT_GEMM_STATUS(0)
#define GEMM_CTRL     STUDENT_GEMM_CTRL(0)
#define GEMM_A_ADDR   STUDENT_GEMM_A_ADDR(0)
#define GEMM_W_ADDR   STUDENT_GEMM_W_ADDR(0)
#define GEMM_C_ADDR   STUDENT_GEMM_C_ADDR(0)
#define GEMM_C_STRIDE STUDENT_GEMM_C_STRIDE(0)
#define GEMM_K_LEN    STUDENT_GEMM_K_LEN(0)
#define GEMM_M_LEN    STUDENT_GEMM_M_LEN(0)
#define GEMM_N_ROWS   STUDENT_GEMM_N_ROWS(0)
#define GEMM_CAPS     STUDENT_GEMM_CAPS(0)
#define GEMM_CYCLES   STUDENT_GEMM_CYCLES(0)
#define GEMM_DBG      STUDENT_GEMM_DBG(0)
#define GEMM_DBG2     STUDENT_GEMM_DBG2(0)
#define GEMM_DBG3     STUDENT_GEMM_DBG3(0)
#define GEMM_DBG4     STUDENT_GEMM_DBG4(0)
#define GEMM_A_STRIDE STUDENT_GEMM_A_STRIDE(0)
#define GEMM_W_STRIDE STUDENT_GEMM_W_STRIDE(0)
#define GEMM_S_ADDR   STUDENT_GEMM_S_ADDR(0)
#define GEMM_P_ADDR   STUDENT_GEMM_P_ADDR(0)
#define CTRL_START    0x1u
#define CTRL_REQUANT  0x2u
#define CTRL_GATHER   0x4u
#define GEMM_RQ_AMAX  STUDENT_GEMM_RQ_AMAX(0)
#define GEMM_G_ADDR   STUDENT_GEMM_G_ADDR(0)
#define GEMM_G_GEOM   STUDENT_GEMM_G_GEOM(0)
#define GEMM_G_CHAN   STUDENT_GEMM_G_CHAN(0)
#define GEMM_G_CONV   STUDENT_GEMM_G_CONV(0)
#define GEMM_G_START  STUDENT_GEMM_G_START(0)
#define GEMM_X_ADDR   STUDENT_GEMM_X_ADDR(0)
#define GEMM_ADD_MX   STUDENT_GEMM_ADD_MULT_X(0)
#define GEMM_ADD_MH   STUDENT_GEMM_ADD_MULT_H(0)
#define GEMM_ADD_SH   STUDENT_GEMM_ADD_SHIFT(0)
#define GEMM_LUT_ADDR STUDENT_GEMM_LUT_ADDR(0)
#define CTRL_ADD      0x8u
#define CTRL_RELU     0x10u
#define CTRL_LUT      0x20u
#define CTRL_LUT_LOAD 0x40u
#define CTRL_A16      0x80u
#define CTRL_W16      0x100u
#define CTRL_OSTATS   0x200u
#define CTRL_ONCHIP   0x400u
#define CTRL_GREUSE   0x800u
#define CTRL_OSUMS    0x1000u
#define CTRL_GRELU    0x2000u

#define STATUS_BUSY 0x1u
#define STATUS_DONE 0x2u
#define STATUS_ERR  0x4u

static uint32_t accel_mcycle(void)
{
#ifdef DAV2_ACCEL_EMU
    return dav2_emu_mcycle();
#else
    uint32_t v;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(v));
    return v;
#endif
}

/* Deliberately huge. An earlier 200M-cycle bound disabled a perfectly correct
 * accelerator: simulation against a slow, shallow memory shows the block is
 * latency-bound rather than stalled (0.116 MAC/cycle vs 12.4 against a fast
 * model), and real DDR3 here is slower still. This exists only to stop a true
 * hardware hang wedging the program forever, not to police throughput. */
#define ACCEL_TIMEOUT_CYCLES 2000000000u

static int      accel_probed;
static int      accel_gather_ok = 1;   /* cleared if the gather self-test fails */
static int      accel_epi_ok = 1;      /* cleared if the add/ReLU self-test fails */
static int      accel_lut_ok;          /* CAPS bit 24, cleared if its self-test fails */
static int      accel_a16_ok;          /* CAPS bit 25, cleared if its self-test fails */
static int      accel_w16_ok;          /* CAPS bit 26, cleared if its self-test fails */
static int      accel_ostats_ok;       /* CAPS bit 27, cleared if its self-test fails */
static int      accel_onchip_ok;       /* CAPS bit 28, cleared if its self-test fails */
static int      accel_greuse_ok;       /* CAPS bit 29, cleared if its self-test fails */
static int      accel_osums_ok;        /* CAPS bit 30, cleared if its self-test fails */
static int      accel_grelu_ok;        /* CAPS bit 31, cleared if its self-test fails */
static int      accel_conv_relu;       /* dav2_accel_conv: ReLU on the image (CTRL.grelu) */
static int      accel_conv_onchip;     /* dav2_accel_conv: result into the result RAM */
static uint32_t accel_gemm_ctrl;       /* extra CTRL bits for accel_run (CTRL.w16) */
static int      accel_out_stride;      /* requant output row length in int16, 0 = M */
static int      accel_ok;
static unsigned accel_nrows;
static unsigned accel_kmax;

/* Cumulative hardware cycles, for the boot log. */
static unsigned long accel_cycles;
static unsigned long accel_jobs;
static unsigned long accel_beats;     /* words read + written by the block  */
static unsigned long accel_retries;
static unsigned long accel_overlapped;  /* jobs left running while the CPU worked */   /* lost responses re-issued (dbg4 hi) */

int dav2_accel_init(void)
{
    if (accel_probed)
        return accel_ok;
    accel_probed = 1;

    uint32_t caps = REG32(GEMM_CAPS);
    accel_nrows = caps & 0xffu;
    accel_kmax  = (caps >> 8) & 0xffffu;
    accel_lut_ok = (int)((caps >> 24) & 1u);
    accel_a16_ok = (int)((caps >> 25) & 1u);
    accel_w16_ok = (int)((caps >> 26) & 1u);
    accel_ostats_ok = (int)((caps >> 27) & 1u);
    accel_onchip_ok = (int)((caps >> 28) & 1u);
    accel_greuse_ok = (int)((caps >> 29) & 1u);
    accel_osums_ok = (int)((caps >> 30) & 1u);
    accel_grelu_ok = (int)((caps >> 31) & 1u);

    /* A missing block reads back as zero (or the bus errors out, which the
     * core reports separately); either way we simply stay on the CPU path. */
    accel_ok = (accel_nrows > 0u) && (accel_kmax >= 4u);
    return accel_ok;
}

int dav2_accel_present(void)
{
    return accel_ok;
}

void dav2_accel_report(void)
{
    dav2_accel_init();
    if (!accel_ok) {
        printf("GEMM accelerator: absent, using the CPU kernel\n");
        return;
    }
    printf("GEMM accelerator: %u rows/pass, K<=%u\n", accel_nrows, accel_kmax);
    if (accel_jobs)
        printf("  %lu jobs (%lu overlapped with CPU work), %lu kcycles, %lu kbeats "
               "(%lu.%lu cycles/beat), %lu lost-response retries\n",
               accel_jobs, accel_overlapped, accel_cycles / 1000u, accel_beats / 1000u,
               accel_cycles / (accel_beats ? accel_beats : 1),
               (accel_cycles * 10u / (accel_beats ? accel_beats : 1)) % 10u,
               accel_retries);
    accel_jobs = 0; accel_cycles = 0; accel_beats = 0; accel_retries = 0;
    accel_overlapped = 0;
}

/* ------------------------------------------------------------ async jobs
 *
 * An operation may leave its LAST job running and return, so the CPU can do
 * work that does not depend on that job's result meanwhile (see
 * dav2_qgemm's streaming range scan and chunked requantisation, and
 * attention in dav2_engine.c). At most one job is ever outstanding: every
 * entry point first settles a pending one, and dav2_accel_finish() collects
 * it. accel_defer is set only for the duration of an *_async call. */
static int accel_defer;
static struct {
    int            active;
    unsigned long  beats;
    int32_t       *amax;         /* requant job: fold RQ_AMAX in here */
} pend;

static int accel_wait_simple(const char *what);

int dav2_accel_busy(void)
{
    return pend.active && (REG32(GEMM_STATUS) & STATUS_BUSY);
}

int dav2_accel_finish(void)
{
    if (!pend.active)
        return 1;
    pend.active = 0;
    if (!accel_wait_simple("async job"))
        return 0;
    accel_cycles  += REG32(GEMM_CYCLES);
    accel_jobs++;
    accel_beats   += pend.beats;
    accel_retries += REG32(GEMM_DBG4) >> 16;
    if (pend.amax) {
        int32_t a = (int32_t)REG32(GEMM_RQ_AMAX);
        if (a > *pend.amax) *pend.amax = a;
        pend.amax = 0;
    }
    return 1;
}

/* Before a job whose statistics the caller will stream: mark its {max, min}
 * slots with a pair no real row can produce (max < min). A slot that still
 * holds it has not been drained yet. */
static void accel_prefill_stats(int32_t *sv, int M)
{
    for (int m = 0; m < M; m++) {
        sv[2 * m]     = DAV2_STATS_EMPTY_MAX;
        sv[2 * m + 1] = DAV2_STATS_EMPTY_MIN;
    }
}

/* Leave the job just started running if the caller asked for that and it
 * is the operation's last one. Returns 1 when deferred. */
static int accel_maybe_defer(int last, unsigned long beats, int32_t *amax)
{
    if (!accel_defer || !last)
        return 0;
    accel_overlapped++;
    pend.active = 1;
    pend.beats  = beats;
    pend.amax   = amax;
    return 1;
}

/* One accelerator pass over all N rows: acc[m][n] = sum_k A[n][k] W[m][k]
 * for K columns, with A rows a_stride bytes apart and W rows w_stride bytes
 * apart (both in elements of their own type when 0). K must already satisfy
 * the block's contract. Returns 0 if the block failed and disabled itself. */
static int accel_run(const int16_t *av, uint32_t a_stride,
                     const int8_t *w, uint32_t w_stride,
                     int32_t *acc, int N, int K, int M, int32_t *stats)
{
    const int nrows = (int)accel_nrows;
    const size_t a_row = a_stride ? a_stride / 2 : (size_t)K;   /* int16 elements */
    int tile = 0;
    if (!dav2_accel_finish())
        return 0;

    REG32(GEMM_W_ADDR)   = (uint32_t)(uintptr_t)w;
    REG32(GEMM_A_STRIDE) = a_stride;
    REG32(GEMM_W_STRIDE) = w_stride;
    /* with CTRL.onchip C_ADDR/C_STRIDE count words of the result RAM */
    const int onchip = (accel_gemm_ctrl & CTRL_ONCHIP) != 0;
    REG32(GEMM_C_STRIDE) = onchip ? (uint32_t)N : (uint32_t)N * 4u;
    REG32(GEMM_K_LEN)    = (uint32_t)K;
    REG32(GEMM_M_LEN)    = (uint32_t)M;

    for (int n0 = 0; n0 < N; n0 += nrows) {
        int nt = N - n0;
        if (nt > nrows)
            nt = nrows;

        /* a producer of A (dav2_producer_t) makes the tile's rows first */
        dav2_producer_need(n0 + nt);
        REG32(GEMM_A_ADDR) = (uint32_t)(uintptr_t)(av + (size_t)n0 * a_row);
        REG32(GEMM_C_ADDR) = onchip ? (uint32_t)n0 : (uint32_t)(uintptr_t)(acc + n0);
        REG32(GEMM_N_ROWS) = (uint32_t)nt;
        /* per-row {max, min} of this tile, 2 words per weight row */
        REG32(GEMM_S_ADDR) = stats ? (uint32_t)(uintptr_t)(stats + (size_t)tile * M * 2) : 0u;
        if (stats && accel_defer && n0 + nt >= N)
            accel_prefill_stats(stats + (size_t)tile * M * 2, M);
        REG32(GEMM_CTRL)   = CTRL_START | accel_gemm_ctrl;
        tile++;
        if (accel_maybe_defer(n0 + nt >= N,
                              (unsigned long)M * (K / ((accel_gemm_ctrl & CTRL_W16) ? 2 : 4))
                              + (unsigned long)nt * (K / 2) + (unsigned long)nt * M, 0))
            return 2;

        /* Bounded wait. The block has only ever been exercised against BRAM
         * (dav2_accel_check runs from .bss); the model's tensors live in the
         * DDR3 arena, so a stall that never appears in simulation or at boot
         * is possible here. An unbounded poll would wedge the program with no
         * diagnostic at all. */
        {
            uint32_t t0 = accel_mcycle();
            while (REG32(GEMM_STATUS) & STATUS_BUSY) {
                if (dav2_producer_idle())
                    continue;              /* make the next tiles' input */
                if ((accel_mcycle() - t0) > ACCEL_TIMEOUT_CYCLES) {
                    uint32_t d0 = REG32(GEMM_DBG);
                    uint32_t c0 = REG32(GEMM_CYCLES);
                    /* Sample twice: if rd_left or cycles move, the block is
                     * advancing and this is throughput, not a deadlock. */
                    for (volatile int w = 0; w < 200000; w++) ;
                    uint32_t d1 = REG32(GEMM_DBG);
                    uint32_t c1 = REG32(GEMM_CYCLES);
                    printf("GEMM accelerator: TIMEOUT status=0x%08lx "
                           "N=%d K=%d M=%d nt=%d a=%08lx w=%08lx c=%08lx\n",
                           (unsigned long)REG32(GEMM_STATUS), N, K, M, nt,
                           (unsigned long)(uintptr_t)(av + (size_t)n0 * a_row),
                           (unsigned long)(uintptr_t)w,
                           (unsigned long)(uintptr_t)(acc + n0));
                    printf("  dbg0=%08lx state=%lu rb_cnt=%lu wr_out=%lu "
                           "rd_left=%lu cycles=%lu\n",
                           (unsigned long)d0, (unsigned long)(d0 & 7u),
                           (unsigned long)((d0 >> 4) & 15u),
                           (unsigned long)((d0 >> 8) & 15u),
                           (unsigned long)(d0 >> 16), (unsigned long)c0);
                    printf("  dbg1=%08lx state=%lu rb_cnt=%lu wr_out=%lu "
                           "rd_left=%lu cycles=%lu  (delta cycles=%lu)\n",
                           (unsigned long)d1, (unsigned long)(d1 & 7u),
                           (unsigned long)((d1 >> 4) & 15u),
                           (unsigned long)((d1 >> 8) & 15u),
                           (unsigned long)(d1 >> 16), (unsigned long)c1,
                           (unsigned long)(c1 - c0));
                    /* The decisive pair: was the outstanding request ever
                     * accepted, and did a response ever come back? */
                    {
                        /* dbg4 packs {retry_n_q[15:0], rsp_cnt_q[15:0]} --
                         * printing it raw as "responses" reports a number
                         * larger than the request count as soon as any retry
                         * has happened, which is exactly the situation this
                         * diagnostic exists to describe. */
                        uint32_t d4 = REG32(GEMM_DBG4);
                        printf("  bus: areq_valid=%lu a_ready=%lu d_valid=%lu "
                               "err=%lu addr=%08lx accepted=%lu responses=%lu "
                               "retries=%lu\n",
                               (unsigned long)((d1 >> 12) & 1u),
                               (unsigned long)((d1 >> 13) & 1u),
                               (unsigned long)((d1 >> 14) & 1u),
                               (unsigned long)((d1 >> 15) & 1u),
                               (unsigned long)REG32(GEMM_DBG2),
                               (unsigned long)REG32(GEMM_DBG3),
                               (unsigned long)(d4 & 0xffffu),
                               (unsigned long)(d4 >> 16));
                    }
                    printf("GEMM accelerator: disabled, using the CPU kernel\n");
                    accel_ok = 0;
                    return 0;
                }
            }
        }

        if (REG32(GEMM_STATUS) & STATUS_ERR) {
            /* A bus error means acc is partly garbage and the block cannot be
             * trusted for the rest of the run either. */
            printf("GEMM accelerator: bus error, falling back to the CPU\n");
            accel_ok = 0;
            return 0;
        }

        {
            uint32_t jc = REG32(GEMM_CYCLES);
            accel_cycles += jc;
            accel_jobs++;
            accel_beats += (unsigned long)M * (K / 4) + (unsigned long)nt * (K / 2)
                         + (unsigned long)nt * M;
            accel_retries += REG32(GEMM_DBG4) >> 16;
            /* First few jobs only: enough to measure cycles-per-beat on real
             * DDR3 without flooding the hostio link. */
            /* Print the first few tiles of each large job. dbg4 packs the
             * retry count in its high half, which is the number that says
             * whether lost-response recovery is rare or constant. */
            /* Off by default. The hostio console ring is 1 kB and the
             * host drains it over JTAG at tens of bytes per second; a device
             * that outruns that spins in obuf_putc (hostio.c:23) and looks
             * exactly like a hung program. Keep target output minimal. */
            if (0) {
                uint32_t d4 = REG32(GEMM_DBG4);
                printf("  tile: %lu cycles, %lu beats, retries=%lu, rsps=%lu\n",
                       (unsigned long)jc,
                       (unsigned long)((size_t)M * (K / 4) + (size_t)nt * (K / 2)),
                       (unsigned long)(d4 >> 16), (unsigned long)(d4 & 0xffffu));
            }

            /* Did this job issue every write it owed? dbg2 packs
             * {writes issued, writes acknowledged} for the job just
             * finished. Expected is nt * M: one word per output row per
             * tile row. Any shortfall is a write that was never put on the
             * bus, which is the lost-word signature; only such jobs print,
             * so this costs nothing on a healthy run. */
            {
                uint32_t d2 = REG32(GEMM_DBG2);
                uint32_t issued = d2 >> 16, acked = d2 & 0xffffu;
                /* The hardware counters are 16 bits wide and a 64-row tile
                 * with M = 1536 owes 98304 writes, so compare modulo 2^16. */
                uint32_t owed = ((uint32_t)nt * (uint32_t)M
                                 + (stats ? 2u * (uint32_t)M : 0u)) & 0xffffu;
                /* (This used to print the last tile of every job as well,
                 * as a check that the counters were wired. They are; the
                 * console link is slow enough that the lines cost more than
                 * a second per frame, so now only a shortfall speaks.) */
                if (issued != owed || acked != issued)
                    printf("  WRITE SHORTFALL job %lu: nt=%d M=%d owed=%lu "
                           "issued=%lu acked=%lu\n",
                           (unsigned long)accel_jobs, nt, M,
                           (unsigned long)owed, (unsigned long)issued,
                           (unsigned long)acked);
            }
        }
    }

    return 1;
}

/* Wait for the current job. Returns 0 on timeout or bus error (and disables
 * the block), 1 when done. The GEMM path has its own, more talkative wait. */
static int accel_wait_simple(const char *what)
{
    uint32_t t0 = accel_mcycle();
    while (REG32(GEMM_STATUS) & STATUS_BUSY) {
        if (dav2_producer_idle())
            continue;                      /* make the next tiles' input */
        if ((accel_mcycle() - t0) > ACCEL_TIMEOUT_CYCLES) {
            printf("GEMM accelerator: TIMEOUT in %s, status=0x%08lx dbg=%08lx; disabled\n",
                   what, (unsigned long)REG32(GEMM_STATUS), (unsigned long)REG32(GEMM_DBG));
            accel_ok = 0;
            return 0;
        }
    }
    if (REG32(GEMM_STATUS) & STATUS_ERR) {
        printf("GEMM accelerator: bus error in %s; disabled\n", what);
        accel_ok = 0;
        return 0;
    }
    return 1;
}

/* Requantise rows m0..m0+mc of acc (all N columns) into out; *amax is
 * raised to the largest |out| written (directly, or when a deferred last job
 * is collected by dav2_accel_finish). mc even, at most KMAX/2. */
/* acc: int32 elements, or int16 with a16 (CTRL.a16) */
static int accel_requant_rows(const void *acc, int N, int M, int m0, int mc,
                              const int32_t *params, int16_t *out, int32_t *amax,
                              const dav2_rq_epi_t *epi, int a16)
{
    const unsigned esz = a16 ? 2u : 4u;
    const int nc_max = (int)accel_nrows;
    uint32_t ctrl = CTRL_START | CTRL_REQUANT | (a16 ? CTRL_A16 : 0u);
    if (!dav2_accel_finish())
        return 0;
    if (epi && epi->add) {
        REG32(GEMM_ADD_MX) = (uint32_t)epi->mx;
        REG32(GEMM_ADD_MH) = (uint32_t)epi->mh;
        REG32(GEMM_ADD_SH) = ((uint32_t)epi->sh << 8) | (uint32_t)epi->sx;
        ctrl |= CTRL_ADD;
    }
    if (epi && epi->relu)
        ctrl |= CTRL_RELU;
    if (epi && epi->lut) {
        REG32(GEMM_LUT_ADDR) = (uint32_t)(uintptr_t)epi->lut;
        ctrl |= CTRL_LUT;
    }
    int lut_load = epi && epi->lut && epi->lut_load;
    if (epi && epi->ostats)
        ctrl |= CTRL_OSTATS;
    const unsigned os_words = (epi && epi->osums) ? 4u : 1u;   /* per row */
    if (epi && epi->osums)
        ctrl |= CTRL_OSUMS;
    const int onchip = epi && epi->onchip;
    if (onchip)
        ctrl |= CTRL_ONCHIP;
    const size_t orow = accel_out_stride ? (size_t)accel_out_stride : (size_t)M;
    REG32(GEMM_A_STRIDE) = onchip ? (uint32_t)N : (uint32_t)N * esz;
    REG32(GEMM_C_STRIDE) = (uint32_t)orow * 2u;
    REG32(GEMM_S_ADDR)   = 0u;
    REG32(GEMM_P_ADDR)   = (uint32_t)(uintptr_t)(params + (size_t)m0 * 3);
    REG32(GEMM_M_LEN)    = (uint32_t)mc;
    for (int n0 = 0; n0 < N; n0 += nc_max) {
        int nc = N - n0;
        if (nc > nc_max) nc = nc_max;
        unsigned long beats = (unsigned long)mc * 3u + (unsigned long)mc * nc * esz / 4u
                            + (unsigned long)mc * nc / 2u + (lut_load ? 8192u : 0u);
        REG32(GEMM_A_ADDR) = onchip ? (uint32_t)((size_t)m0 * N + n0)
                                    : (uint32_t)(uintptr_t)((const uint8_t *)acc
                                                            + ((size_t)m0 * N + n0) * esz);
        REG32(GEMM_C_ADDR) = (uint32_t)(uintptr_t)(out + (size_t)n0 * orow + m0);
        if (epi && epi->add)
            REG32(GEMM_X_ADDR) = (uint32_t)(uintptr_t)(epi->x + (size_t)n0 * M + m0);
        REG32(GEMM_N_ROWS) = (uint32_t)nc;
        if (epi && epi->ostats)
            REG32(GEMM_S_ADDR) = (uint32_t)(uintptr_t)(epi->ostats + (size_t)n0 * os_words);
        REG32(GEMM_CTRL)   = ctrl | (lut_load ? CTRL_LUT_LOAD : 0u);
        lut_load = 0;                       /* the table stays in the block */
        if (accel_maybe_defer(n0 + nc >= N, beats, amax))
            return 2;
        if (!accel_wait_simple("requant"))
            return 0;
        accel_cycles += REG32(GEMM_CYCLES);
        accel_jobs++;
        accel_beats += beats;
        {
            int32_t a = (int32_t)REG32(GEMM_RQ_AMAX);
            if (a > *amax) *amax = a;
        }
    }
    return 1;
}

static int requant_ok(const int32_t *acc, int N, int M, const int32_t *params,
                      const int16_t *out)
{
    /* Contract: M even (pairs of int16 per word), everything word aligned. */
    if (N < 1 || M < 2 || (M & 1))
        return 0;
    return ((((uintptr_t)acc) | ((uintptr_t)params) | ((uintptr_t)out)) & 3u) == 0;
}

int dav2_accel_requant(const int32_t *acc, int N, int M, const int32_t *params,
                       int16_t *out, int32_t *amax_out)
{
    if (amax_out) *amax_out = -1;
    if (!dav2_accel_init() || !requant_ok(acc, N, M, params, out))
        return 0;
    const int mc_max = (int)(accel_kmax / 2u) & ~1;
    int32_t amax = 0;
    for (int m0 = 0; m0 < M; m0 += mc_max) {
        int mc = M - m0;
        if (mc > mc_max) mc = mc_max;
        if (!accel_requant_rows(acc, N, M, m0, mc, params, out, &amax, 0, 0))
            return 0;
    }
    if (amax_out) *amax_out = amax;
    return 1;
}

int dav2_accel_requant_rows_async(const int32_t *acc, int N, int M, int m0, int mc,
                                  const int32_t *params, int16_t *out, int32_t *amax,
                                  const dav2_rq_epi_t *epi)
{
    if (!dav2_accel_init() || !requant_ok(acc, N, M, params, out))
        return 0;
    const int add = epi && epi->add;
    if (epi && (epi->add || epi->relu) && !accel_epi_ok)
        return 0;
    if (epi && epi->lut && (!accel_lut_ok || (((uintptr_t)epi->lut) & 3u)))
        return 0;
    if (epi && epi->ostats && (!accel_ostats_ok || (((uintptr_t)epi->ostats) & 3u)))
        return 0;
    if (epi && epi->osums && (!epi->ostats || !accel_osums_ok))
        return 0;
    if (epi && epi->onchip && (!accel_onchip_ok || (long)N * M > DAV2_ACCEL_CR_WORDS))
        return 0;
    if (m0 < 0 || mc < 2 || (mc & 1) || m0 + mc > M
        || mc > (int)(accel_kmax / (add ? 4u : 2u)))
        return 0;
    if (add && ((((uintptr_t)epi->x) & 3u) || epi->sx < 0 || epi->sx > 63
                || epi->sh < 0 || epi->sh > 63 || epi->mx < 0 || epi->mh < 0))
        return 0;
    accel_defer = 1;
    int r = accel_requant_rows(acc, N, M, m0, mc, params, out, amax, epi, 0);
    accel_defer = 0;
    return r;
}

/* Convolution GEMM with the A tile gathered by the block from the NHWC image
 * (CTRL.gather): acc[m][n] for the oh*ow output pixels n, no im2col matrix
 * in memory. Kernel positions are split into jobs of at most KMAX/C
 * positions; partial sums over position chunks are added here, as for any
 * long reduction. Statistics come out only when one chunk covers the whole
 * kernel. Returns 0 (caller falls back to im2col) outside the contract. */
int dav2_accel_conv(const int16_t *img, int h, int w, int C, int k, int stride,
                    int pad, const int8_t *wt, int M, int32_t *acc,
                    dav2_accel_stats_t *st)
{
    if (st) { st->v = 0; st->tiles = 0; }
    if (!dav2_accel_init() || !accel_gather_ok)
        return 0;
    const int oh = (h + 2 * pad - k) / stride + 1;
    const int ow = (w + 2 * pad - k) / stride + 1;
    const int N = oh * ow, K = k * k * C;
    /* register field widths and the tile contract */
    if ((C & 3) || C < 4 || k < 1 || k > 15 || stride < 1 || stride > 15 || pad > 15)
        return 0;
    if (h > 65535 || w > 65535 || ow > 65535 || M < 1)
        return 0;
    if ((((uintptr_t)img) | ((uintptr_t)wt) | ((uintptr_t)acc)) & 3u)
        return 0;
    int pc_max = (int)(accel_kmax / (unsigned)C);
    if (pc_max < 1) return 0;
    if (pc_max > k * k) pc_max = k * k;
    if (pc_max > 255) pc_max = 255;
    if (!dav2_accel_finish())
        return 0;

    const int nrows = (int)accel_nrows;
    const int tiles = (N + nrows - 1) / nrows;
    const int single = (pc_max == k * k);
    /* the result RAM: one chunk, the statistics as the only way back */
    const int onchip = accel_conv_onchip;
    if (onchip && (!single || !st || !accel_onchip_ok || (long)N * M > DAV2_ACCEL_CR_WORDS))
        return 0;
    if (accel_conv_relu && !accel_grelu_ok)
        return 0;
    /* tap reuse (CTRL.greuse): stride 1, all k*k positions in one job */
    const uint32_t ctrl = CTRL_START | CTRL_GATHER | (onchip ? CTRL_ONCHIP : 0u)
                        | (accel_conv_relu ? CTRL_GRELU : 0u)
                        | ((accel_greuse_ok && stride == 1 && single) ? CTRL_GREUSE : 0u);
    int32_t *stats = 0, *part = 0;
    const size_t mark = dav2_arena_mark();
    if (single && st) {
        stats = (int32_t *)dav2_arena_alloc((size_t)tiles * M * 2 * sizeof(int32_t));
        if (stats) { st->v = stats; st->tiles = tiles; }
        else if (onchip) return 0;
    }
    if (!single) {
        part = (int32_t *)dav2_arena_alloc((size_t)N * M * sizeof(int32_t));
        if (!part) { dav2_arena_release(mark); return 0; }
    }

    REG32(GEMM_G_ADDR)   = (uint32_t)(uintptr_t)img;
    REG32(GEMM_G_GEOM)   = ((uint32_t)h << 16) | (uint32_t)w;
    REG32(GEMM_G_CHAN)   = ((uint32_t)ow << 16) | (uint32_t)C;
    REG32(GEMM_W_STRIDE) = (uint32_t)K;
    REG32(GEMM_A_STRIDE) = 0u;
    REG32(GEMM_C_STRIDE) = onchip ? (uint32_t)N : (uint32_t)N * 4u;
    REG32(GEMM_M_LEN)    = (uint32_t)M;

    for (int p0 = 0; p0 < k * k; p0 += pc_max) {
        int pc = k * k - p0;
        if (pc > pc_max) pc = pc_max;
        int32_t *dst = p0 ? part : acc;
        REG32(GEMM_G_CONV)  = (uint32_t)k | ((uint32_t)stride << 4) | ((uint32_t)pad << 8)
                            | ((uint32_t)(p0 / k) << 12) | ((uint32_t)(p0 % k) << 16)
                            | ((uint32_t)pc << 20);
        REG32(GEMM_W_ADDR)  = (uint32_t)(uintptr_t)(wt + (size_t)p0 * C);
        REG32(GEMM_K_LEN)   = (uint32_t)(pc * C);
        for (int n0 = 0, t = 0; n0 < N; n0 += nrows, t++) {
            int nt = N - n0;
            if (nt > nrows) nt = nrows;
            {
                /* a producer of the image makes the input rows this tile
                 * reads: up to the last output row's lowest kernel row */
                int iy = ((n0 + nt - 1) / ow) * stride - pad + k - 1;
                if (iy > h - 1) iy = h - 1;
                dav2_producer_need((iy + 1) * w);
            }
            REG32(GEMM_G_START) = ((uint32_t)(n0 / ow) << 16) | (uint32_t)(n0 % ow);
            REG32(GEMM_C_ADDR)  = onchip ? (uint32_t)n0 : (uint32_t)(uintptr_t)(dst + n0);
            REG32(GEMM_N_ROWS)  = (uint32_t)nt;
            REG32(GEMM_S_ADDR)  = stats ? (uint32_t)(uintptr_t)(stats + (size_t)t * M * 2) : 0u;
            if (stats && accel_defer && single && n0 + nt >= N)
                accel_prefill_stats(stats + (size_t)t * M * 2, M);
            REG32(GEMM_CTRL)    = ctrl;
            {
                unsigned long beats = (unsigned long)M * (pc * C / 4)
                                    + (unsigned long)nt * (pc * C / 2)
                                    + (onchip ? 0ul : (unsigned long)nt * M);
                /* only a single-chunk conv may leave its last job running:
                 * a split one adds partial sums right after each chunk */
                if (accel_maybe_defer(single && n0 + nt >= N, beats, 0))
                    return 2;       /* single chunk: nothing to release but stats */
            }
            if (!accel_wait_simple("gather")) {
                dav2_arena_release(mark);
                if (st) { st->v = 0; st->tiles = 0; }
                return 0;
            }
            accel_cycles += REG32(GEMM_CYCLES);
            accel_jobs++;
            accel_beats += (unsigned long)M * (pc * C / 4) + (unsigned long)nt * (pc * C / 2)
                         + (onchip ? 0ul : (unsigned long)nt * M);
            accel_retries += REG32(GEMM_DBG4) >> 16;
        }
        if (p0) {
            const size_t total = (size_t)N * M;
            for (size_t i = 0; i < total; i++)
                acc[i] += part[i];
        }
    }
    /* stats (if any) must outlive this call: keep them, drop only part */
    if (!stats) dav2_arena_release(mark);
    return 1;
}

int dav2_accel_w16_ok(void) { return dav2_accel_init() && accel_w16_ok; }

int dav2_accel_gemm16_async(const int16_t *a, uint32_t a_stride,
                            const int16_t *w, uint32_t w_stride,
                            int32_t *acc, int N, int K, int M, int32_t *stats)
{
    if (!dav2_accel_init() || !accel_w16_ok)
        return 0;
    if (K < 2 || (K & 1) || (unsigned)K > accel_kmax || N < 1 || M < 1)
        return 0;
    if ((((uintptr_t)a) | ((uintptr_t)w) | ((uintptr_t)acc) | ((uintptr_t)stats)
         | a_stride | w_stride) & 3u)
        return 0;
    if (stats && N > (int)accel_nrows)
        return 0;                          /* one tile: one set of statistics */
    accel_defer = 1;
    accel_gemm_ctrl = CTRL_W16;
    int r = accel_run(a, a_stride, (const int8_t *)w, w_stride ? w_stride : (uint32_t)K * 2u,
                      acc, N, K, M, stats);
    accel_gemm_ctrl = 0;
    accel_defer = 0;
    return r;
}

int dav2_accel_requant_lut_async(const int32_t *acc, int N, int M, const int32_t *params,
                                 int16_t *out, int out_stride, const int16_t *lut,
                                 int lut_load)
{
    if (!dav2_accel_init() || !accel_lut_ok || !requant_ok(acc, N, M, params, out))
        return 0;
    if (M > (int)(accel_kmax / 2u) || (out_stride & 1) || out_stride < M
        || (((uintptr_t)lut) & 3u))
        return 0;
    dav2_rq_epi_t epi;
    memset(&epi, 0, sizeof epi);
    epi.lut = lut;
    epi.lut_load = lut_load;
    int32_t amax = 0;
    accel_out_stride = out_stride;
    accel_defer = 1;
    int r = accel_requant_rows(acc, N, M, 0, M, params, out, &amax, &epi, 0);
    accel_defer = 0;
    accel_out_stride = 0;
    return r;
}

int dav2_accel_requant_stride(const int32_t *acc, int N, int M, const int32_t *params,
                              int16_t *out, int out_stride, int32_t *amax)
{
    accel_out_stride = out_stride;
    int r = dav2_accel_requant(acc, N, M, params, out, amax);
    accel_out_stride = 0;
    return r;
}

int dav2_accel_gemm_raw_async(const int16_t *a, uint32_t a_stride,
                              const int8_t *w, uint32_t w_stride,
                              int32_t *acc, int N, int K, int M)
{
    accel_defer = 1;
    int r = dav2_accel_gemm_raw(a, a_stride, w, w_stride, acc, N, K, M);
    accel_defer = 0;
    return r;
}

int dav2_accel_qgemm_async(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc,
                           dav2_accel_stats_t *st)
{
    accel_defer = 1;
    int r = dav2_accel_qgemm(a, wt, acc, st);
    accel_defer = 0;
    return r;
}

int dav2_accel_conv_async(const int16_t *img, int h, int w, int C, int k, int stride,
                          int pad, const int8_t *wt, int M, int32_t *acc,
                          dav2_accel_stats_t *st, int in_relu)
{
    accel_defer = 1;
    accel_conv_relu = in_relu;
    int r = dav2_accel_conv(img, h, w, C, k, stride, pad, wt, M, acc, st);
    accel_conv_relu = 0;
    accel_defer = 0;
    return r;
}

int dav2_accel_grelu_ok(void) { return dav2_accel_init() && accel_gather_ok && accel_grelu_ok; }

int dav2_accel_conv_onchip_async(const int16_t *img, int h, int w, int C, int k, int stride,
                                 int pad, const int8_t *wt, int M, dav2_accel_stats_t *st,
                                 int in_relu)
{
    st->v = 0; st->tiles = 0;
    if (!dav2_accel_init() || !accel_onchip_ok)
        return 0;
    accel_defer = 1;
    accel_conv_onchip = 1;
    accel_conv_relu = in_relu;
    int r = dav2_accel_conv(img, h, w, C, k, stride, pad, wt, M, 0, st);
    accel_conv_relu = 0;
    accel_conv_onchip = 0;
    accel_defer = 0;
    if (!r) { st->v = 0; st->tiles = 0; }
    return r;
}

int dav2_accel_gemm_raw(const int16_t *a, uint32_t a_stride,
                        const int8_t *w, uint32_t w_stride,
                        int32_t *acc, int N, int K, int M)
{
    if (!dav2_accel_init())
        return 0;
    if (K < 4 || (K & 3) != 0 || (unsigned)K > accel_kmax || N < 1 || M < 1)
        return 0;
    if ((((uintptr_t)a) | ((uintptr_t)w) | ((uintptr_t)acc) | a_stride | w_stride) & 3u)
        return 0;
    return accel_run(a, a_stride, w, w_stride, acc, N, K, M, 0);
}

int dav2_accel_qgemm(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc,
                     dav2_accel_stats_t *st)
{
    if (st) { st->v = 0; st->tiles = 0; }
    if (!dav2_accel_init())
        return 0;

    const int N = a->n, K = a->c, M = wt->m;

    /* Shape and alignment contract of student_gemm.hjson. Anything outside it
     * is handed back to the software kernel rather than approximated. */
    if (K < 4 || (K & 3) != 0)
        return 0;
    if (N < 1 || M < 1)
        return 0;
    if ((((uintptr_t)a->v) | ((uintptr_t)wt->w) | ((uintptr_t)acc)) & 3u)
        return 0;

    if ((unsigned)K <= accel_kmax) {
        int32_t *stats = 0;
        int tiles = (N + (int)accel_nrows - 1) / (int)accel_nrows;
        if (st) {
            stats = (int32_t *)dav2_arena_alloc((size_t)tiles * M * 2 * sizeof(int32_t));
            if (stats) { st->v = stats; st->tiles = tiles; }
        }
        int r = accel_run(a->v, 0, wt->w, 0, acc, N, K, M, stats);
        /* Failed part-way: the caller recomputes acc on the CPU and must not
         * take its row ranges from half-written statistics. (Found by
         * injecting a bus error on a multi-tile job in the emulator.) */
        if (!r && st) { st->v = 0; st->tiles = 0; }
        return r;
    }

    /* K longer than the A-tile RAM: the 3x3 convolutions on the 384-channel
     * level have K = 9*384 = 3456 against KMAX = 2048. The row strides let a
     * job read a column slice of A and W, so the reduction is done in chunks
     * of at most KMAX columns and the partial sums are added here. The
     * shapes that need this are small (N <= 81), so the adds are cheap; the
     * alternative was the whole GEMM on the CPU, 5 s per frame. */
    const size_t mark = dav2_arena_mark();
    int32_t *part = (int32_t *)dav2_arena_alloc((size_t)N * M * sizeof(int32_t));
    if (!part) {
        dav2_arena_release(mark);
        return 0;
    }
    const int kc_max = (int)(accel_kmax & ~3u);
    int ok = 1;
    const int defer_saved = accel_defer;
    accel_defer = 0;         /* partial sums are added right after each chunk */
    for (int k0 = 0; k0 < K && ok; k0 += kc_max) {
        int kc = K - k0;
        if (kc > kc_max) kc = kc_max;
        int32_t *dst = k0 ? part : acc;
        /* no statistics here: the extremes of partial sums say nothing
         * about the extremes of the total, so the caller scans acc */
        ok = accel_run(a->v + k0, (uint32_t)K * 2u,
                       wt->w + k0, (uint32_t)K,
                       dst, N, kc, M, 0);
        if (ok && k0) {
            const size_t total = (size_t)N * M;
            for (size_t i = 0; i < total; i++)
                acc[i] += part[i];
        }
    }
    accel_defer = defer_saved;
    dav2_arena_release(mark);
    return ok;
}

/* Small fixed test case run once at start-up. Kept in .bss so it works before
 * the DDR3 arena exists, and deliberately sized so that N straddles a tile
 * boundary (130 = 128 + 2 for the board's 128-row array). */
#define CHK_N 130
#define CHK_K 64
#define CHK_M 6

static int16_t chk_a[CHK_N * CHK_K]   __attribute__((aligned(4)));
static int8_t  chk_w[CHK_M * CHK_K]   __attribute__((aligned(4)));
static int32_t chk_hw[CHK_M * CHK_N];
static int32_t chk_sw[CHK_M * CHK_N];

static uint32_t chk_rand(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

/* Same comparison as dav2_accel_check, but at model dimensions and with every
 * buffer in DDR3 rather than BRAM. Returns the number of mismatching
 * accumulator words, or -1 if it could not run.
 *
 * Why it exists: the small check passed on hardware while the full inference
 * was wrong (r = 0.19 against the host), so the fault was one the small case
 * could not reach. This found it: 81x588x384 -- the patch-embedding shape --
 * lost exactly one word, deterministically, while every other shape passed.
 * That word, and the "reread after eviction" probe below, showed the write
 * was acknowledged but never reached DDR3, which led to the cache write-back
 * bug in rvlab_ddr_block_cache.sv. Since that fix every shape is bit-exact.
 *
 * Run it by setting DAV2_STARTUP_CHECKS in main.c; it costs about a second
 * per shape on the board, so it is off by default. */
int dav2_accel_bigcheck(int N, int K, int M)
{
    if (!dav2_accel_init())
        return -1;

    const size_t mark = dav2_arena_mark();
    int16_t *a  = (int16_t *)dav2_arena_alloc((size_t)N * K * sizeof(int16_t));
    int8_t  *w  = (int8_t  *)dav2_arena_alloc((size_t)M * K * sizeof(int8_t));
    int32_t *hw = (int32_t *)dav2_arena_alloc((size_t)N * M * sizeof(int32_t));
    int32_t *sw = (int32_t *)dav2_arena_alloc((size_t)N * M * sizeof(int32_t));
    if (!a || !w || !hw || !sw) {
        printf("GEMM bigcheck: arena too small\n");
        return -1;
    }

    uint32_t seed = 0x2468aceu;
    for (int i = 0; i < N * K; i++)
        a[i] = (int16_t)((int32_t)(chk_rand(&seed) % 16383u) - 8191);
    for (int i = 0; i < M * K; i++)
        w[i] = (int8_t)((int32_t)(chk_rand(&seed) % 255u) - 127);
    for (int i = 0; i < N * M; i++) { hw[i] = 0; sw[i] = 0; }

    dav2_tensor_t at = { a, XF_ONE, N, K, -1, 0, 0 };
    dav2_qw_t     wt = { w, 0, 0, M, K };

    dav2_qgemm_cpu(a, w, sw, N, K, M);
    if (!dav2_accel_qgemm(&at, &wt, hw, 0)) {
        printf("GEMM bigcheck: accelerator declined\n");
        dav2_arena_release(mark);
        return -1;
    }

    int bad = 0, first = -1;
    for (int i = 0; i < N * M; i++) {
        if (hw[i] != sw[i]) {
            if (first < 0) first = i;
            bad++;
        }
    }
    printf("GEMM bigcheck %dx%dx%d: %d/%d words differ", N, K, M, bad, N * M);
    if (bad)
        printf(", first at %d (hw %ld sw %ld)", first,
               (long)hw[first], (long)sw[first]);
    printf("\n");

    if (bad) {
        /* Was the word never written, or written and read stale?
         *
         * Those need opposite fixes and nothing so far separates them: the
         * value read is 0 either way, because the buffer is pre-zeroed. Evict
         * the line by walking well past the 16 kB direct-mapped cache, then
         * read it again. If the second read finds the accelerator's data, the
         * write landed and the first read was stale. If it is still 0, the
         * write never reached DDR3. */
        volatile int32_t sink = 0;
        for (int i = 0; i < 16384; i++)
            sink += ((volatile int32_t *)a)[i % (N * K / 2)];
        (void)sink;

        printf("  reread after eviction: hw %ld (sw %ld) -> %s\n",
               (long)((volatile int32_t *)hw)[first], (long)sw[first],
               (((volatile int32_t *)hw)[first] == sw[first])
                   ? "STALE READ, the write did land"
                   : "still wrong, the write was lost");
    }

    dav2_arena_release(mark);
    return bad;
}

/* Gather mode against im2col + the software kernel, on a 5x4x8 image with a
 * 3x3 kernel, padding 1: every border case (all four edges and corners) and
 * a non-square image. All in BRAM, so it runs before DDR3 exists -- and in
 * the whole-SoC simulation, which has no DDR3. A failure disables gather
 * mode only; convolutions then build im2col in software as before. */
static int accel_check_gather(void)
{
    enum { H = 5, W = 4, C = 8, KS = 3, M = 4, N = H * W, K = KS * KS * C };
    int16_t *img  = chk_a;                  /* H*W*C = 160  */
    int16_t *cols = chk_a + 256;            /* N*K  = 1440 <= 4480 - 256 */
    uint32_t seed = 0x2468aceu;
    for (int i = 0; i < H * W * C; i++)
        img[i] = (int16_t)((int32_t)(chk_rand(&seed) % 16383u) - 8191);
    for (int i = 0; i < M * K; i++)
        chk_w[i] = (int8_t)((int32_t)(chk_rand(&seed) % 255u) - 127);

    int16_t *dst = cols;                    /* im2col, pad 1, stride 1 */
    for (int oy = 0; oy < H; oy++)
        for (int ox = 0; ox < W; ox++)
            for (int ky = 0; ky < KS; ky++)
                for (int kx = 0; kx < KS; kx++) {
                    int iy = oy + ky - 1, ix = ox + kx - 1;
                    for (int c = 0; c < C; c++)
                        *dst++ = (iy < 0 || iy >= H || ix < 0 || ix >= W)
                               ? 0 : img[(iy * W + ix) * C + c];
                }
    dav2_qgemm_cpu(cols, chk_w, chk_sw, N, K, M);

    /* plain gather first, then with tap reuse (CTRL.greuse) */
    const int greuse = accel_greuse_ok;
    accel_greuse_ok = 0;
    if (!dav2_accel_conv(img, H, W, C, KS, 1, 1, chk_w, M, chk_hw, 0)) {
        printf("GEMM accelerator: gather self-test could not run\n");
        accel_gather_ok = 0;
        return 0;
    }
    int bad = 0;
    for (int i = 0; i < M * N; i++)
        if (chk_hw[i] != chk_sw[i]) {
            if (bad < 4)
                printf("  gather mismatch at %d: hw %d != sw %d\n",
                       i, (int)chk_hw[i], (int)chk_sw[i]);
            bad++;
        }
    if (bad) {
        printf("GEMM accelerator: GATHER SELF-TEST FAILED (%d/%d), gather disabled\n",
               bad, M * N);
        accel_gather_ok = 0;
    } else {
        printf("GEMM accelerator: gather self-test ok (%dx%dx%d, 3x3 pad 1)\n", H, W, C);
        if (greuse) {
            accel_greuse_ok = 1;
            for (int i = 0; i < M * N; i++) chk_hw[i] = 0x55555555;
            int rbad = dav2_accel_conv(img, H, W, C, KS, 1, 1, chk_w, M, chk_hw, 0) ? 0 : -1;
            for (int i = 0; rbad >= 0 && i < M * N; i++)
                if (chk_hw[i] != chk_sw[i]) rbad++;
            if (rbad) {
                printf("GEMM accelerator: TAP-REUSE SELF-TEST FAILED (%d), reuse disabled\n", rbad);
                accel_greuse_ok = 0;
            } else {
                printf("GEMM accelerator: tap-reuse self-test ok\n");
            }
        }
        if (accel_grelu_ok) {
            /* ReLU on the image while gathering (CTRL.grelu), against
             * the CPU on the rectified patch matrix */
            dst = cols;
            for (int oy = 0; oy < H; oy++)
                for (int ox = 0; ox < W; ox++)
                    for (int ky = 0; ky < KS; ky++)
                        for (int kx = 0; kx < KS; kx++) {
                            int iy = oy + ky - 1, ix = ox + kx - 1;
                            for (int c = 0; c < C; c++) {
                                int16_t v = (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                          ? 0 : img[(iy * W + ix) * C + c];
                                *dst++ = v < 0 ? 0 : v;
                            }
                        }
            dav2_qgemm_cpu(cols, chk_w, chk_sw, N, K, M);
            for (int i = 0; i < M * N; i++) chk_hw[i] = 0x55555555;
            accel_conv_relu = 1;
            int gbad = dav2_accel_conv(img, H, W, C, KS, 1, 1, chk_w, M, chk_hw, 0) ? 0 : -1;
            accel_conv_relu = 0;
            for (int i = 0; gbad >= 0 && i < M * N; i++)
                if (chk_hw[i] != chk_sw[i]) gbad++;
            if (gbad) {
                printf("GEMM accelerator: GATHER-RELU SELF-TEST FAILED (%d), ReLU on the CPU\n", gbad);
                accel_grelu_ok = 0;
            } else {
                printf("GEMM accelerator: gather-ReLU self-test ok\n");
            }
        }
    }
    return 0;      /* the plain GEMM path is still good */
}

/* The requantisation epilogue (CTRL.add, then CTRL.add + CTRL.relu)
 * against the same arithmetic in C, on a 20 x 8 chunk in BRAM, before DDR3
 * exists. A mismatch disables the epilogue alone; the engine then adds and
 * applies ReLU on the CPU as before. */
static int64_t chk_apply(int64_t v, int64_t mult, int sh)
{
    int64_t r = v * mult;
    if (sh > 0) r += (int64_t)1 << (sh - 1);
    return r >> sh;
}
static int32_t chk_sat(int64_t v) { return v > 8191 ? 8191 : v < -8191 ? -8191 : (int32_t)v; }

static int accel_check_epilogue(void)
{
    enum { N = 20, M = 8 };
    static int32_t par[3 * M] __attribute__((aligned(4)));
    int32_t *acc = chk_sw;                      /* M x N int32 */
    int16_t *x   = chk_a;                       /* N x M int16 */
    int16_t *out = chk_a + 256;                 /* N x M int16 */
    uint32_t seed = 0x13579bdu;
    for (int i = 0; i < M * N; i++)
        acc[i] = (int32_t)(chk_rand(&seed) % 2000001u) - 1000000;
    for (int i = 0; i < N * M; i++)
        x[i] = (int16_t)((int32_t)(chk_rand(&seed) % 16383u) - 8191);
    for (int m = 0; m < M; m++) {
        par[3 * m]     = (int32_t)(0x40000000u + chk_rand(&seed) % 0x3fffffffu);
        par[3 * m + 1] = 38 + (int32_t)(chk_rand(&seed) % 4u);
        par[3 * m + 2] = (int32_t)(chk_rand(&seed) % 2001u) - 1000;
    }
    dav2_rq_epi_t epi = { x, 0x5a000000, 0x61000000, 31, 32, 1, 0, 0, 0, 0, 0, 0 };
    int bad = 0;
    for (int pass = 0; pass < 2 && !bad; pass++) {
        int32_t amax = 0;
        epi.relu = pass;
        int r = dav2_accel_requant_rows_async(acc, N, M, 0, M, par, out, &amax, &epi);
        if (r == 0 || !dav2_accel_finish()) {
            printf("GEMM accelerator: add/ReLU self-test could not run\n");
            accel_epi_ok = 0;
            return 0;
        }
        for (int n = 0; n < N; n++)
            for (int m = 0; m < M; m++) {
                int32_t h = chk_sat(chk_apply(acc[m * N + n], par[3 * m], par[3 * m + 1])
                                    + par[3 * m + 2]);
                int32_t e = chk_sat((int64_t)(int32_t)chk_apply(x[n * M + m], epi.mx, epi.sx)
                                  + (int64_t)(int32_t)chk_apply(h, epi.mh, epi.sh));
                if (epi.relu && e < 0) e = 0;
                if (out[n * M + m] != e) {
                    if (bad < 4)
                        printf("  add/ReLU mismatch n=%d m=%d: hw %d != %d\n",
                               n, m, (int)out[n * M + m], (int)e);
                    bad++;
                }
            }
    }
    if (bad) {
        printf("GEMM accelerator: ADD/RELU SELF-TEST FAILED (%d), epilogue disabled\n", bad);
        accel_epi_ok = 0;
    } else {
        printf("GEMM accelerator: add/ReLU self-test ok\n");
    }
    return 0;      /* the plain paths are still good */
}

/* The lookup table (CTRL.lut): a random table in dav2_scratch (40 kB of
 * on-chip RAM, free at boot), loaded with the first of two jobs and reused
 * by the second, with and without ReLU. */
static int accel_check_lut(void)
{
    if (!accel_lut_ok)
        return 0;                      /* an older bitstream: GELU on the CPU */
    enum { N = 20, M = 8 };
    static int32_t par[3 * M] __attribute__((aligned(4)));
    int32_t *acc = chk_sw;
    int16_t *out = chk_a;
    int16_t *tab = dav2_scratch;       /* 16384 int16 */
    uint32_t seed = 0x2468aceu;
    for (int i = 0; i < 16384; i++)
        tab[i] = (int16_t)chk_rand(&seed);
    for (int i = 0; i < M * N; i++)
        acc[i] = (int32_t)(chk_rand(&seed) % 2000001u) - 1000000;
    for (int m = 0; m < M; m++) {
        par[3 * m]     = (int32_t)(0x40000000u + chk_rand(&seed) % 0x3fffffffu);
        par[3 * m + 1] = 38 + (int32_t)(chk_rand(&seed) % 4u);
        par[3 * m + 2] = (int32_t)(chk_rand(&seed) % 2001u) - 1000;
    }
    int bad = 0;
    for (int pass = 0; pass < 2 && !bad; pass++) {
        dav2_rq_epi_t epi = { 0, 0, 0, 0, 0, 0, pass, tab, pass == 0, 0, 0, 0 };
        int32_t amax = 0;
        int r = dav2_accel_requant_rows_async(acc, N, M, 0, M, par, out, &amax, &epi);
        if (r == 0 || !dav2_accel_finish()) {
            printf("GEMM accelerator: lookup-table self-test could not run\n");
            accel_lut_ok = 0;
            return 0;
        }
        for (int n = 0; n < N; n++)
            for (int m = 0; m < M; m++) {
                int32_t h = chk_sat(chk_apply(acc[m * N + n], par[3 * m], par[3 * m + 1])
                                    + par[3 * m + 2]);
                if (pass && h < 0) h = 0;
                int32_t e = tab[h + 8192];
                if (out[n * M + m] != e) {
                    if (bad < 4)
                        printf("  lookup-table mismatch n=%d m=%d: hw %d != %d\n",
                               n, m, (int)out[n * M + m], (int)e);
                    bad++;
                }
            }
    }
    if (bad) {
        printf("GEMM accelerator: LOOKUP-TABLE SELF-TEST FAILED (%d), GELU on the CPU\n", bad);
        accel_lut_ok = 0;
    } else {
        printf("GEMM accelerator: lookup-table self-test ok\n");
    }
    return 0;      /* the other paths are still good */
}

int dav2_accel_lut_ok(void) { return accel_ok && accel_lut_ok; }
int dav2_accel_ostats_ok(void) { return dav2_accel_init() && accel_ostats_ok; }
int dav2_accel_osums_ok(void) { return dav2_accel_init() && accel_ostats_ok && accel_osums_ok; }
int dav2_accel_onchip_ok(void) { return dav2_accel_init() && accel_onchip_ok; }

int dav2_accel_qgemm_onchip_async(const dav2_tensor_t *a, const dav2_qw_t *wt,
                                  dav2_accel_stats_t *st)
{
    st->v = 0; st->tiles = 0;
    if (!dav2_accel_init() || !accel_onchip_ok)
        return 0;
    const int N = a->n, K = a->c, M = wt->m;
    if (K < 4 || (K & 3) || (unsigned)K > accel_kmax || N < 1 || M < 1
        || N > (int)accel_nrows || (long)N * M > DAV2_ACCEL_CR_WORDS)
        return 0;
    if ((((uintptr_t)a->v) | ((uintptr_t)wt->w)) & 3u)
        return 0;
    /* the statistics are the only way back: without them, no result */
    int32_t *stats = (int32_t *)dav2_arena_alloc((size_t)M * 2 * sizeof(int32_t));
    if (!stats)
        return 0;
    st->v = stats; st->tiles = 1;
    accel_defer = 1;
    accel_gemm_ctrl = CTRL_ONCHIP;
    int r = accel_run(a->v, 0, wt->w, 0, 0, N, K, M, stats);
    accel_gemm_ctrl = 0;
    accel_defer = 0;
    if (!r) { st->v = 0; st->tiles = 0; }
    return r;
}

/* The result RAM (CTRL.onchip): a 20 x 64 x 8 GEMM left on chip, then its
 * requantisation read from there, against the CPU. */
static int accel_check_onchip(void)
{
    if (!accel_onchip_ok)
        return 0;
    enum { N = 20, K = 64, M = 8 };
    static int32_t par[3 * M] __attribute__((aligned(4)));
    int16_t *a   = chk_a;              /* N x K */
    int16_t *out = chk_a + N * K;      /* N x M */
    int8_t  *w   = (int8_t *)(chk_a + N * K + N * M);   /* M x K */
    uint32_t seed = 0x5bd1e995u;
    for (int i = 0; i < N * K; i++)
        a[i] = (int16_t)((int32_t)(chk_rand(&seed) % 16383u) - 8191);
    for (int i = 0; i < M * K; i++)
        w[i] = (int8_t)((int32_t)(chk_rand(&seed) % 255u) - 127);
    for (int m = 0; m < M; m++) {
        par[3 * m]     = (int32_t)(0x40000000u + chk_rand(&seed) % 0x3fffffffu);
        par[3 * m + 1] = 44 + (int32_t)(chk_rand(&seed) % 4u);
        par[3 * m + 2] = (int32_t)(chk_rand(&seed) % 2001u) - 1000;
    }
    dav2_tensor_t at = { a, XF_ONE, N, K, -1, 0, 0 };
    dav2_qw_t wq = { w, 0, 0, M, K };
    dav2_accel_stats_t st;
    dav2_rq_epi_t epi;
    memset(&epi, 0, sizeof epi);
    epi.onchip = 1;
    int32_t amax = 0;
    const size_t mark = dav2_arena_mark();
    int ok = dav2_accel_qgemm_onchip_async(&at, &wq, &st) && dav2_accel_finish()
          && dav2_accel_requant_rows_async(0, N, M, 0, M, par, out, &amax, &epi)
          && dav2_accel_finish();
    int bad = 0;
    for (int m = 0; ok && m < M; m++) {
        int32_t mx = -2147483647 - 1, mn = 2147483647;
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)a[n * K + k] * w[m * K + k];
            if (acc > mx) mx = acc;
            if (acc < mn) mn = acc;
            int32_t e = chk_sat(chk_apply(acc, par[3 * m], par[3 * m + 1]) + par[3 * m + 2]);
            if (out[n * M + m] != e) bad++;
        }
        if (st.v[2 * m] != mx || st.v[2 * m + 1] != mn) bad++;
    }
    dav2_arena_release(mark);
    if (!ok) {
        printf("GEMM accelerator: result-RAM self-test could not run\n");
        accel_onchip_ok = 0;
    } else if (bad) {
        printf("GEMM accelerator: RESULT-RAM SELF-TEST FAILED (%d), results via DDR3\n", bad);
        accel_onchip_ok = 0;
    } else {
        printf("GEMM accelerator: result-RAM self-test ok\n");
    }
    return 0;
}

/* Output row statistics (CTRL.ostats) on an int16-input job: 20 rows of
 * 8 columns, each row's {max, min} against the output itself. */
static int accel_check_ostats(void)
{
    if (!accel_ostats_ok)
        return 0;
    enum { N = 20, M = 8 };
    static int32_t par[3 * M] __attribute__((aligned(4)));
    static uint32_t st[N] __attribute__((aligned(4)));
    int16_t *in  = chk_a;              /* M x N int16 */
    int16_t *out = chk_a + 256;        /* N x M int16 */
    uint32_t seed = 0x7feb352du;
    for (int i = 0; i < M * N; i++)
        in[i] = (int16_t)((int32_t)(chk_rand(&seed) % 16383u) - 8191);
    for (int m = 0; m < M; m++) {
        par[3 * m]     = (int32_t)(0x40000000u + chk_rand(&seed) % 0x3fffffffu);
        par[3 * m + 1] = 30 + (int32_t)(chk_rand(&seed) % 4u);
        par[3 * m + 2] = (int32_t)(chk_rand(&seed) % 2001u) - 1000;
    }
    int32_t amax = 0;
    if (!dav2_accel_requant16(in, N, M, par, out, &amax, st)) {
        printf("GEMM accelerator: row-statistics self-test could not run\n");
        accel_ostats_ok = 0;
        return 0;
    }
    int bad = 0;
    for (int n = 0; n < N; n++) {
        int32_t mx = -32768, mn = 32767;
        for (int m = 0; m < M; m++) {
            if (out[n * M + m] > mx) mx = out[n * M + m];
            if (out[n * M + m] < mn) mn = out[n * M + m];
        }
        if ((int16_t)(st[n] >> 16) != mx || (int16_t)(st[n] & 0xffffu) != mn)
            bad++;
    }
    if (bad) {
        printf("GEMM accelerator: ROW-STATISTICS SELF-TEST FAILED (%d), ranges on the CPU\n", bad);
        accel_ostats_ok = 0;
    } else {
        printf("GEMM accelerator: row-statistics self-test ok\n");
    }
    return 0;
}

/* Output row sums (CTRL.osums): an int32 20 x 8 requantisation, each row's
 * {max, min}, sum and sum of squares against the output itself. */
static int accel_check_osums(void)
{
    if (!accel_ostats_ok || !accel_osums_ok) {
        accel_osums_ok = 0;
        return 0;
    }
    enum { N = 20, M = 8 };
    static int32_t par[3 * M] __attribute__((aligned(4)));
    static int32_t acc[M * N] __attribute__((aligned(4)));
    static uint32_t st[4 * N] __attribute__((aligned(4)));
    int16_t *out = chk_a;              /* N x M int16 */
    uint32_t seed = 0x2c1b3c6du;
    for (int i = 0; i < M * N; i++)
        acc[i] = (int32_t)(chk_rand(&seed) % 2000001u) - 1000000;
    for (int m = 0; m < M; m++) {
        par[3 * m]     = (int32_t)(0x40000000u + chk_rand(&seed) % 0x3fffffffu);
        par[3 * m + 1] = 38 + (int32_t)(chk_rand(&seed) % 3u);
        par[3 * m + 2] = (int32_t)(chk_rand(&seed) % 2001u) - 1000;
    }
    dav2_rq_epi_t epi;
    memset(&epi, 0, sizeof epi);
    epi.ostats = st;
    epi.osums = 1;
    int32_t amax = 0;
    if (!dav2_accel_requant_rows_async(acc, N, M, 0, M, par, out, &amax, &epi)
        || !dav2_accel_finish()) {
        printf("GEMM accelerator: row-sums self-test could not run\n");
        accel_osums_ok = 0;
        return 0;
    }
    int bad = 0;
    for (int n = 0; n < N; n++) {
        int32_t mx = -32768, mn = 32767, sm = 0;
        uint64_t sq = 0;
        for (int m = 0; m < M; m++) {
            int32_t v = out[n * M + m];
            if (v > mx) mx = v;
            if (v < mn) mn = v;
            sm += v;
            sq += (uint64_t)((int64_t)v * v);
        }
        const uint32_t *r = st + 4 * n;
        if ((int16_t)(r[0] >> 16) != mx || (int16_t)(r[0] & 0xffffu) != mn
            || (int32_t)r[1] != sm || (((uint64_t)r[3] << 32) | r[2]) != sq)
            bad++;
    }
    if (bad) {
        printf("GEMM accelerator: ROW-SUMS SELF-TEST FAILED (%d), sums on the CPU\n", bad);
        accel_osums_ok = 0;
    } else {
        printf("GEMM accelerator: row-sums self-test ok\n");
    }
    return 0;
}

/* int16 weights (CTRL.w16): 20 x 64 x 8 against the CPU, with statistics. */
static int accel_check_w16(void)
{
    if (!accel_w16_ok)
        return 0;                      /* an older bitstream: attention as before */
    enum { N = 20, K = 64, M = 8 };
    int16_t *a = chk_a;                /* N x K */
    int16_t *w = chk_a + N * K;        /* M x K */
    int32_t *acc = chk_sw;             /* M x N */
    static int32_t st[2 * M] __attribute__((aligned(4)));
    uint32_t seed = 0x1b873593u;
    for (int i = 0; i < N * K; i++)
        a[i] = (int16_t)((int32_t)(chk_rand(&seed) % 16383u) - 8191);
    for (int i = 0; i < M * K; i++)
        w[i] = (int16_t)((int32_t)(chk_rand(&seed) % 4095u) - 2047);
    int r = dav2_accel_gemm16_async(a, 0, w, 0, acc, N, K, M, st);
    if (r == 0 || !dav2_accel_finish()) {
        printf("GEMM accelerator: int16-weight self-test could not run\n");
        accel_w16_ok = 0;
        return 0;
    }
    int bad = 0;
    for (int m = 0; m < M; m++) {
        int32_t mx = -2147483647 - 1, mn = 2147483647;
        for (int n = 0; n < N; n++) {
            int32_t e = 0;
            for (int k = 0; k < K; k++)
                e += (int32_t)a[n * K + k] * w[m * K + k];
            if (e > mx) mx = e;
            if (e < mn) mn = e;
            if (acc[m * N + n] != e) {
                if (bad < 4)
                    printf("  int16-weight mismatch m=%d n=%d: hw %ld != %ld\n",
                           m, n, (long)acc[m * N + n], (long)e);
                bad++;
            }
        }
        if (st[2 * m] != mx || st[2 * m + 1] != mn)
            bad++;
    }
    if (bad) {
        printf("GEMM accelerator: INT16-WEIGHT SELF-TEST FAILED (%d), attention as before\n", bad);
        accel_w16_ok = 0;
    } else {
        printf("GEMM accelerator: int16-weight self-test ok\n");
    }
    return 0;
}

/* int16 input (CTRL.a16): 8 rows of 20 int16, negative multipliers too,
 * against the CPU's arithmetic. */
static int accel_check_a16(void)
{
    if (!accel_a16_ok)
        return 0;                      /* an older bitstream: LayerNorm on the CPU */
    enum { N = 20, M = 8 };
    static int32_t par[3 * M] __attribute__((aligned(4)));
    int16_t *in  = chk_a;              /* M x N int16 */
    int16_t *out = chk_a + 256;        /* N x M int16 */
    uint32_t seed = 0x3579bdfu;
    for (int i = 0; i < M * N; i++)
        in[i] = (int16_t)((int32_t)(chk_rand(&seed) % 16383u) - 8191);
    for (int m = 0; m < M; m++) {
        int32_t mul = (int32_t)(0x40000000u + chk_rand(&seed) % 0x3fffffffu);
        par[3 * m]     = (m & 1) ? -mul : mul;
        par[3 * m + 1] = 30 + (int32_t)(chk_rand(&seed) % 4u);
        par[3 * m + 2] = (int32_t)(chk_rand(&seed) % 2001u) - 1000;
    }
    int32_t amax = 0;
    int bad = 0;
    if (!dav2_accel_requant16(in, N, M, par, out, &amax, 0)) {
        printf("GEMM accelerator: int16-input self-test could not run\n");
        accel_a16_ok = 0;
        return 0;
    }
    for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++) {
            int32_t e = chk_sat(chk_apply(in[m * N + n], par[3 * m], par[3 * m + 1])
                                + par[3 * m + 2]);
            if (out[n * M + m] != e) {
                if (bad < 4)
                    printf("  int16-input mismatch n=%d m=%d: hw %d != %d\n",
                           n, m, (int)out[n * M + m], (int)e);
                bad++;
            }
        }
    if (bad) {
        printf("GEMM accelerator: INT16-INPUT SELF-TEST FAILED (%d), LayerNorm on the CPU\n", bad);
        accel_a16_ok = 0;
    } else {
        printf("GEMM accelerator: int16-input self-test ok\n");
    }
    return 0;
}

int dav2_accel_requant16(const int16_t *in, int N, int M, const int32_t *params,
                         int16_t *out, int32_t *amax_out, uint32_t *ostats)
{
    if (!dav2_accel_init() || !accel_a16_ok || (N & 1) || (M & 1) || N < 2 || M < 2)
        return 0;
    if (ostats && (!accel_ostats_ok || M > (int)(accel_kmax / 2u)))
        return 0;                          /* statistics need one chunk of rows */
    dav2_rq_epi_t epi;
    memset(&epi, 0, sizeof epi);
    epi.ostats = ostats;
    if ((((uintptr_t)in) | ((uintptr_t)params) | ((uintptr_t)out)) & 3u)
        return 0;
    const int mc_max = (int)(accel_kmax / 2u) & ~1;
    int32_t amax = 0;
    for (int m0 = 0; m0 < M; m0 += mc_max) {
        int mc = M - m0;
        if (mc > mc_max) mc = mc_max;
        if (!accel_requant_rows(in, N, M, m0, mc, params, out, &amax, ostats ? &epi : 0, 1))
            return 0;
    }
    if (!dav2_accel_finish())
        return 0;
    *amax_out = amax;
    return 1;
}

int dav2_accel_check(void)
{
    if (!dav2_accel_init())
        return 0;

    uint32_t seed = 0x1234567u;
    for (int i = 0; i < CHK_N * CHK_K; i++)
        chk_a[i] = (int16_t)((int32_t)(chk_rand(&seed) % 16383u) - 8191);
    for (int i = 0; i < CHK_M * CHK_K; i++)
        chk_w[i] = (int8_t)((int32_t)(chk_rand(&seed) % 255u) - 127);

    dav2_tensor_t a = { chk_a, XF_ONE, CHK_N, CHK_K, -1, 0, 0 };
    dav2_qw_t     w = { chk_w, 0, 0, CHK_M, CHK_K };

    dav2_qgemm_cpu(chk_a, chk_w, chk_sw, CHK_N, CHK_K, CHK_M);

    if (!dav2_accel_qgemm(&a, &w, chk_hw, 0)) {
        printf("GEMM accelerator: self-test could not run\n");
        return 0;
    }

    int bad = 0;
    for (int i = 0; i < CHK_M * CHK_N; i++) {
        if (chk_hw[i] != chk_sw[i]) {
            if (bad < 4)
                printf("  mismatch at %d: hw %d != sw %d\n",
                       i, (int)chk_hw[i], (int)chk_sw[i]);
            bad++;
        }
    }

    if (bad) {
        printf("GEMM accelerator: SELF-TEST FAILED (%d/%d), disabling\n",
               bad, CHK_M * CHK_N);
        accel_ok = 0;
    } else {
        printf("GEMM accelerator: self-test ok (%dx%dx%d)\n",
               CHK_N, CHK_K, CHK_M);
    }
    if (!bad)
        bad = accel_check_gather();
    if (!bad)
        bad = accel_check_epilogue();
    if (!bad)
        bad = accel_check_lut();
    if (!bad)
        bad = accel_check_a16();
    if (!bad)
        bad = accel_check_w16();
    if (!bad)
        bad = accel_check_ostats();
    if (!bad)
        bad = accel_check_osums();
    if (!bad)
        bad = accel_check_onchip();
    return bad;
}

#else /* !DAV2_ACCEL */

int  dav2_accel_init(void)    { return 0; }
int  dav2_accel_present(void) { return 0; }
void dav2_accel_report(void)  { }
int  dav2_accel_check(void)   { return 0; }
int  dav2_accel_lut_ok(void)  { return 0; }
int  dav2_accel_w16_ok(void)  { return 0; }
int  dav2_accel_gemm16_async(const int16_t *a, uint32_t a_stride, const int16_t *w,
                             uint32_t w_stride, int32_t *acc, int N, int K, int M,
                             int32_t *stats)
{ (void)a;(void)a_stride;(void)w;(void)w_stride;(void)acc;(void)N;(void)K;(void)M;(void)stats; return 0; }
int  dav2_accel_requant_stride(const int32_t *acc, int N, int M, const int32_t *params,
                               int16_t *out, int out_stride, int32_t *amax)
{ (void)acc;(void)N;(void)M;(void)params;(void)out;(void)out_stride;(void)amax; return 0; }
int  dav2_accel_requant16(const int16_t *in, int N, int M, const int32_t *params,
                          int16_t *out, int32_t *amax, uint32_t *ostats)
{ (void)in; (void)N; (void)M; (void)params; (void)out; (void)amax; (void)ostats; return 0; }
int  dav2_accel_ostats_ok(void) { return 0; }
int  dav2_accel_osums_ok(void) { return 0; }
int  dav2_accel_requant_lut_async(const int32_t *acc, int N, int M, const int32_t *params,
                                  int16_t *out, int out_stride, const int16_t *lut,
                                  int lut_load)
{ (void)acc;(void)N;(void)M;(void)params;(void)out;(void)out_stride;(void)lut;(void)lut_load;
  return 0; }
int  dav2_accel_onchip_ok(void) { return 0; }
int  dav2_accel_qgemm_onchip_async(const dav2_tensor_t *a, const dav2_qw_t *wt,
                                   dav2_accel_stats_t *st)
{ (void)a; (void)wt; st->v = 0; st->tiles = 0; return 0; }
int  dav2_accel_conv_onchip_async(const int16_t *img, int h, int w, int C, int k, int stride,
                                  int pad, const int8_t *wt, int M, dav2_accel_stats_t *st,
                                  int in_relu)
{ (void)img;(void)h;(void)w;(void)C;(void)k;(void)stride;(void)pad;(void)wt;(void)M;(void)in_relu;
  st->v = 0; st->tiles = 0; return 0; }

int dav2_accel_qgemm(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc,
                     dav2_accel_stats_t *st)
{
    (void)a; (void)wt; (void)acc;
    if (st) { st->v = 0; st->tiles = 0; }
    return 0;
}

int dav2_accel_requant(const int32_t *acc, int N, int M, const int32_t *params,
                       int16_t *out, int32_t *amax_out)
{
    (void)acc; (void)N; (void)M; (void)params; (void)out;
    if (amax_out) *amax_out = -1;
    return 0;
}

int dav2_accel_conv(const int16_t *img, int h, int w, int C, int k, int stride,
                    int pad, const int8_t *wt, int M, int32_t *acc,
                    dav2_accel_stats_t *st)
{
    (void)img; (void)h; (void)w; (void)C; (void)k; (void)stride; (void)pad;
    (void)wt; (void)M; (void)acc;
    if (st) { st->v = 0; st->tiles = 0; }
    return 0;
}

int dav2_accel_gemm_raw(const int16_t *a, uint32_t a_stride,
                        const int8_t *w, uint32_t w_stride,
                        int32_t *acc, int N, int K, int M)
{
    (void)a; (void)a_stride; (void)w; (void)w_stride; (void)acc;
    (void)N; (void)K; (void)M;
    return 0;
}

int dav2_accel_qgemm_async(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc,
                           dav2_accel_stats_t *st)
{
    return dav2_accel_qgemm(a, wt, acc, st);
}

int dav2_accel_conv_async(const int16_t *img, int h, int w, int C, int k, int stride,
                          int pad, const int8_t *wt, int M, int32_t *acc,
                          dav2_accel_stats_t *st, int in_relu)
{
    if (in_relu) { if (st) { st->v = 0; st->tiles = 0; } return 0; }
    return dav2_accel_conv(img, h, w, C, k, stride, pad, wt, M, acc, st);
}
int dav2_accel_grelu_ok(void) { return 0; }

int dav2_accel_gemm_raw_async(const int16_t *a, uint32_t a_stride,
                              const int8_t *w, uint32_t w_stride,
                              int32_t *acc, int N, int K, int M)
{
    return dav2_accel_gemm_raw(a, a_stride, w, w_stride, acc, N, K, M);
}

int dav2_accel_requant_rows_async(const int32_t *acc, int N, int M, int m0, int mc,
                                  const int32_t *params, int16_t *out, int32_t *amax,
                                  const dav2_rq_epi_t *epi)
{
    (void)acc; (void)N; (void)M; (void)m0; (void)mc; (void)params; (void)out; (void)amax;
    (void)epi;
    return 0;
}

int dav2_accel_finish(void) { return 1; }
int dav2_accel_busy(void)   { return 0; }

#endif /* DAV2_ACCEL */
