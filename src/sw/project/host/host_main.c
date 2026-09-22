/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Host-side harness: runs the exact same engine sources natively so the
 * quantised result can be compared against the PyTorch golden reference
 * without waiting for the FPGA.
 *
 * Because every inner loop is integer, the host and RISC-V builds are expected
 * to produce bit-identical output; comparing them is how the port is verified
 * on hardware.
 *
 * Build:  make -C src/sw/project/host
 * Run:    ./dav2_host build/dav2/dav2_weights.bin image.dav2img out.bin
 */

#include "../dav2.h"
#include "../dav2_accel.h"

#include <stdio.h>
#include <time.h>

#ifdef DAV2_ACCEL_EMU
#include <sys/mman.h>
/* The accelerator model is handed 32-bit addresses, as the real block is:
 * everything it may touch must live below 4 GB. */
static void *host_alloc(size_t n)
{
    void *p = mmap(0, n, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
    return p == MAP_FAILED ? 0 : p;
}
#else
#define host_alloc malloc
#endif
#include <stdlib.h>
#include <string.h>

void dav2_progress(const char *stage)
{
    printf("  [%s]\n", stage);
    fflush(stdout);
}

/* Nanoseconds; the profile is printed as "kcycles" but on the host they are
 * microseconds. Only the shares matter here -- and they differ from the
 * board's, which has no FPU and a far smaller cache. */
uint64_t dav2_cycles(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <dav2_weights.bin> <image.dav2img> <out.bin> [arena_mb]\n"
                        "  image.dav2img: size*size*3 int16 HWC followed by one float32 scale,\n"
                        "  as written by tools/dav2_image.py -- the same bytes the board receives.\n",
                argv[0]);
        return 2;
    }
    const size_t arena_mb = (argc > 4) ? (size_t)atoi(argv[4]) : 256;

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *blob = host_alloc((size_t)n);
    if (fread(blob, 1, (size_t)n, f) != (size_t)n) { perror("read"); return 1; }
    fclose(f);
    printf("loaded %ld bytes of weights\n", n);

    void *arena = host_alloc(arena_mb << 20);
    if (!arena) { fprintf(stderr, "arena alloc failed\n"); return 1; }

    dav2_blob_init(blob);
    dav2_arena_init(arena, arena_mb << 20);

    /* The image arrives separately from the weights, exactly as on the board:
     * int16 HWC pixels then a float32 scale. Reading the identical file the
     * runner sends over JTAG is what makes the host a bit-exact oracle. */
    {
        FILE *fi = fopen(argv[2], "rb");
        if (!fi) { perror(argv[2]); return 1; }
        fseek(fi, 0, SEEK_END);
        long ni = ftell(fi);
        fseek(fi, 0, SEEK_SET);
        uint8_t *img = host_alloc((size_t)ni);
        if (fread(img, 1, (size_t)ni, fi) != (size_t)ni) { perror("read image"); return 1; }
        fclose(fi);
        float scale;
        memcpy(&scale, img + ni - 4, 4);
        dav2_set_image((const int16_t *)img, scale);
        printf("loaded %ld-byte image, scale %g\n", ni, scale);
    }

    /* configuration comes from the generated header */
#include "dav2_blob_config.h"
    dav2_cfg_t cfg = { DAV2_INPUT_SIZE, DAV2_PATCH_GRID, DAV2_N_TOKENS };

    const int out_size = cfg.grid * DAV2_PATCH;
    float *depth = (float *)malloc((size_t)out_size * out_size * sizeof(float));

#ifdef DAV2_ACCEL_EMU
    /* the board's boot sequence: probe, then the GEMM and gather self-tests */
    dav2_accel_report();
    dav2_accel_check();
#endif
    printf("running inference at %dx%d (%d patches)\n",
           cfg.size, cfg.size, cfg.grid * cfg.grid);
    dav2_infer(&cfg, depth);

    printf("arena peak: %.2f MB\n", (double)dav2_arena_peak() / 1e6);
#ifdef DAV2_ACCEL_EMU
    dav2_accel_report();
#endif
#ifdef DAV2_TRACE
    {
        extern double dav2_mac_count, dav2_elem_count;
        printf("total MACs: %.3f G   requantised elements: %.3f M\n",
               dav2_mac_count / 1e9, dav2_elem_count / 1e6);
        /* CV32E40P is single-issue in-order: each MAC costs one lw (int16
         * activation from BRAM tile), one lb (int8 weight from DDR3), one mul
         * and one add, and the loop is unrolled by 4. */
        for (int cpm = 4; cpm <= 8; cpm += 2)
            printf("  at %d cycles/MAC and 100 MHz: %.1f s/frame\n",
                   cpm, dav2_mac_count * cpm / 1e8);
    }
#endif

    FILE *o = fopen(argv[3], "wb");
    fwrite(depth, sizeof(float), (size_t)out_size * out_size, o);
    fclose(o);
    printf("wrote %s (%dx%d floats)\n", argv[3], out_size, out_size);
    return 0;
}
