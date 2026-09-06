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

#define STATUS_BUSY 0x1u
#define STATUS_DONE 0x2u
#define STATUS_ERR  0x4u

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

        while (REG32(GEMM_STATUS) & STATUS_BUSY)
            ;

        if (REG32(GEMM_STATUS) & STATUS_ERR) {
            /* A bus error means acc is partly garbage and the block cannot be
             * trusted for the rest of the run either. */
            printf("GEMM accelerator: bus error, falling back to the CPU\n");
            accel_ok = 0;
            return 0;
        }

        accel_cycles += REG32(GEMM_CYCLES);
        accel_jobs++;
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
