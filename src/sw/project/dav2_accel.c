/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Driver for the student_gemm int8 GEMM accelerator.
 *
 * On the host (and in any build without the block) every entry point reports
 * "not available" and the engine falls back to the software kernel, so the
 * numerical result is identical either way.
 */

#include "dav2_accel.h"

#if defined(__riscv) && !defined(DAV2_NO_ACCEL)
#define DAV2_ACCEL 1
#else
#define DAV2_ACCEL 0
#endif

#if DAV2_ACCEL

#include <stdio.h>
#include <rvlab.h>

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

#define STATUS_BUSY 0x1u
#define STATUS_DONE 0x2u
#define STATUS_ERR  0x4u

static uint32_t accel_mcycle(void)
{
    uint32_t v;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(v));
    return v;
}

/* Deliberately huge. An earlier 200M-cycle bound disabled a perfectly correct
 * accelerator: simulation against a slow, shallow memory shows the block is
 * latency-bound rather than stalled (0.116 MAC/cycle vs 12.4 against a fast
 * model), and real DDR3 here is slower still. This exists only to stop a true
 * hardware hang wedging the program forever, not to police throughput. */
#define ACCEL_TIMEOUT_CYCLES 2000000000u

static int      accel_probed;
static int      accel_ok;
static unsigned accel_nrows;
static unsigned accel_kmax;

/* Cumulative hardware cycles, for the boot log. */
static unsigned long accel_cycles;
static unsigned long accel_jobs;

int dav2_accel_init(void)
{
    if (accel_probed)
        return accel_ok;
    accel_probed = 1;

    uint32_t caps = REG32(GEMM_CAPS);
    accel_nrows = caps & 0xffu;
    accel_kmax  = (caps >> 8) & 0xffffu;

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
}

int dav2_accel_qgemm(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc)
{
    if (!dav2_accel_init())
        return 0;

    const int N = a->n, K = a->c, M = wt->m;

    /* Shape and alignment contract of student_gemm.hjson. Anything outside it
     * is handed back to the software kernel rather than approximated. */
    if (K < 4 || (K & 3) != 0 || (unsigned)K > accel_kmax)
        return 0;
    if (N < 1 || M < 1)
        return 0;
    if ((((uintptr_t)a->v) | ((uintptr_t)wt->w) | ((uintptr_t)acc)) & 3u)
        return 0;

    const int nrows = (int)accel_nrows;

    REG32(GEMM_W_ADDR)   = (uint32_t)(uintptr_t)wt->w;
    REG32(GEMM_C_STRIDE) = (uint32_t)N * 4u;
    REG32(GEMM_K_LEN)    = (uint32_t)K;
    REG32(GEMM_M_LEN)    = (uint32_t)M;

    for (int n0 = 0; n0 < N; n0 += nrows) {
        int nt = N - n0;
        if (nt > nrows)
            nt = nrows;

        REG32(GEMM_A_ADDR) = (uint32_t)(uintptr_t)(a->v + (size_t)n0 * K);
        REG32(GEMM_C_ADDR) = (uint32_t)(uintptr_t)(acc + n0);
        REG32(GEMM_N_ROWS) = (uint32_t)nt;
        REG32(GEMM_CTRL)   = 1u;

        /* Bounded wait. The block has only ever been exercised against BRAM
         * (dav2_accel_check runs from .bss); the model's tensors live in the
         * DDR3 arena, so a stall that never appears in simulation or at boot
         * is possible here. An unbounded poll would wedge the program with no
         * diagnostic at all. */
        {
            uint32_t t0 = accel_mcycle();
            while (REG32(GEMM_STATUS) & STATUS_BUSY) {
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
                           (unsigned long)(uintptr_t)(a->v + (size_t)n0 * K),
                           (unsigned long)(uintptr_t)wt->w,
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
                    printf("  bus: areq_valid=%lu a_ready=%lu d_valid=%lu "
                           "err=%lu addr=%08lx accepted=%lu responses=%lu\n",
                           (unsigned long)((d1 >> 12) & 1u),
                           (unsigned long)((d1 >> 13) & 1u),
                           (unsigned long)((d1 >> 14) & 1u),
                           (unsigned long)((d1 >> 15) & 1u),
                           (unsigned long)REG32(GEMM_DBG2),
                           (unsigned long)REG32(GEMM_DBG3),
                           (unsigned long)REG32(GEMM_DBG4));
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
        }
    }

    return 1;
}

/* Small fixed test case run once at start-up. Kept in .bss so it works before
 * the DDR3 arena exists, and deliberately sized so that N straddles a tile
 * boundary (20 = 16 + 4 for the default 16-row array). */
#define CHK_N 20
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
 * buffer in DDR3 rather than BRAM.
 *
 * The small check passes on hardware while the full inference produces a depth
 * map uncorrelated with the host build (r = 0.19), so the failure is one the
 * small case cannot reach: real N/K/M, real tile counts, and operands that
 * live in the aliasing DDR3 arena instead of on-chip memory. Returns the
 * number of mismatching accumulator words. */
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

    dav2_tensor_t at = { a, 1.0f, N, K };
    dav2_qw_t     wt = { w, 0, 0, M, K };

    dav2_qgemm_cpu(a, w, sw, N, K, M);
    if (!dav2_accel_qgemm(&at, &wt, hw)) {
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

    dav2_arena_release(mark);
    return bad;
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

    dav2_tensor_t a = { chk_a, 1.0f, CHK_N, CHK_K };
    dav2_qw_t     w = { chk_w, 0, 0, CHK_M, CHK_K };

    dav2_qgemm_cpu(chk_a, chk_w, chk_sw, CHK_N, CHK_K, CHK_M);

    if (!dav2_accel_qgemm(&a, &w, chk_hw)) {
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
    return bad;
}

#else /* !DAV2_ACCEL */

int  dav2_accel_init(void)    { return 0; }
int  dav2_accel_present(void) { return 0; }
void dav2_accel_report(void)  { }
int  dav2_accel_check(void)   { return 0; }

int dav2_accel_qgemm(const dav2_tensor_t *a, const dav2_qw_t *wt, int32_t *acc)
{
    (void)a; (void)wt; (void)acc;
    return 0;
}

#endif /* DAV2_ACCEL */
