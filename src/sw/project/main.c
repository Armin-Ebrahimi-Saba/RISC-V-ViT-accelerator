/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Depth-Anything V2 Small on the rvlab SoC.
 *
 * DDR3 map (see docs/design_ref/memory_map.rst -- DDR3 starts at 0x80000000):
 *
 *   0x80000000  weight blob produced by tools/export_dav2.py (~25 MB)
 *   0x82000000  activation arena for the engine (peak ~16 MB at 126x126)
 *   0x8F000000  handshake word written by the host loader once the blob is in
 *
 * The CPU cannot pull 25 MB in by itself, so the flow loads the blob over JTAG
 * (see flow/tools/dav2_loader.py) while this program waits at the handshake.
 */

#include <stdio.h>
#include <stdint.h>
#include <rvlab.h>

#include "dav2.h"
#include "dav2_accel.h"
#include "dav2_blob_config.h"

void dav2_selftest_report(void);

#define BLOB_ADDR   0x80000000u
#define ARENA_ADDR  0x82000000u
#define ARENA_SIZE  (64u * 1024u * 1024u)
#define GO_ADDR     0x8F000000u
#define GO_MAGIC    0xD00DFEEDu

/* Handshake flag, polled by main() and written over JTAG by the host loader.
 * Must live in BRAM -- see the comment at its use below. */
volatile uint32_t dav2_go;

static uint32_t cycles_lo(void)
{
    uint32_t v;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(v));
    return v;
}

void dav2_progress(const char *stage)
{
    printf("  [%s] t=%u kcycles\n", stage, (unsigned)(cycles_lo() / 1000u));
}

/* Print the depth map as coarse ASCII art so the result is visible over the
 * hostio link without transferring the whole image. */
static void print_ascii_depth(const float *depth, int size, int cols)
{
    const char *ramp = " .:-=+*#%@";
    int step = size / cols;
    if (step < 1) step = 1;

    float lo = depth[0], hi = depth[0];
    for (int i = 0; i < size * size; i++) {
        if (depth[i] < lo) lo = depth[i];
        if (depth[i] > hi) hi = depth[i];
    }
    float span = (hi > lo) ? (hi - lo) : 1.0f;

    printf("depth map (near = bright), range %d..%d milli-units\n",
           (int)(lo * 1000.0f), (int)(hi * 1000.0f));
    for (int y = 0; y < size; y += step * 2) {   /* *2: characters are tall */
        for (int x = 0; x < size; x += step) {
            float v = (depth[y * size + x] - lo) / span;
            int idx = (int)(v * 9.0f + 0.5f);
            if (idx < 0) idx = 0;
            if (idx > 9) idx = 9;
            putchar(ramp[idx]);
        }
        putchar('\n');
    }
}

int main(void)
{
    printf("\n=== Depth-Anything V2 Small on CV32E40P ===\n");
    printf("input %dx%d, %d patches, %d tokens\n",
           DAV2_INPUT_SIZE, DAV2_INPUT_SIZE, DAV2_N_PATCHES, DAV2_N_TOKENS);

    /* Kernel self-test first: it needs no DDR3 and no weights, so it also
     * works in RTL simulation and isolates toolchain problems from model
     * problems. The host build prints the same number. */
    dav2_selftest_report();

    /* Probe the GEMM accelerator and cross-check it against the software
     * kernel before the model runs. Both need only BRAM, so this also works
     * in a DDR3-less simulation. If the check fails the block is disabled and
     * inference still produces the correct (slower) result. */
    dav2_accel_report();
    dav2_accel_check();

    if (ddr_init()) {
        printf("FATAL: DDR3 init failed; weights cannot be stored.\n");
        return 1;
    }

    /* Quick DDR3 sanity check before trusting 25 MB of it. */
    volatile uint32_t *probe = (volatile uint32_t *)(ARENA_ADDR);
    probe[0] = 0xa5a5a5a5u;
    probe[1] = 0x5a5a5a5au;
    if (probe[0] != 0xa5a5a5a5u || probe[1] != 0x5a5a5a5au) {
        printf("FATAL: DDR3 readback mismatch (%08x %08x)\n",
               (unsigned)probe[0], (unsigned)probe[1]);
        return 1;
    }
    printf("DDR3 ready.\n");

    /* Wait for the host to place the weight blob in DDR3.
     *
     * The flag lives in BRAM, not DDR3. A DDR3 handshake word does not work:
     * once the debugger has pushed the blob through system bus access, the
     * CPU stops observing debugger writes to DDR3 -- the core spins on a
     * stale value forever even though a debugger read of the same address
     * returns the new one. BRAM is not behind that cache, so a write here is
     * seen immediately. The host finds this variable by symbol name. */
    dav2_go = 0;
    printf("DAV2_WAITING_FOR_WEIGHTS\n");
    while (dav2_go != GO_MAGIC)
        ;
    printf("weights present (%u bytes expected)\n", (unsigned)DAV2_BLOB_BYTES);

    dav2_blob_init((const void *)BLOB_ADDR);
    dav2_arena_init((void *)ARENA_ADDR, ARENA_SIZE);

    const int out_size = DAV2_PATCH_GRID * DAV2_PATCH;
    float *depth = (float *)dav2_arena_alloc((size_t)out_size * out_size * sizeof(float));
    if (!depth) {
        printf("FATAL: arena too small for the output image\n");
        return 1;
    }

    dav2_cfg_t cfg = { DAV2_INPUT_SIZE, DAV2_PATCH_GRID, DAV2_N_TOKENS };

    uint32_t t0 = cycles_lo();
    dav2_infer(&cfg, depth);
    uint32_t t1 = cycles_lo();

    if (dav2_arena_failed) {
        printf("FATAL: activation arena exhausted -- increase ARENA_SIZE or "
               "lower DAV2_INPUT_SIZE\n");
        return 1;
    }

    printf("\ninference finished in %u kcycles\n", (unsigned)((t1 - t0) / 1000u));
    printf("arena peak %u KB\n", (unsigned)(dav2_arena_peak() / 1024u));

    print_ascii_depth(depth, out_size, 63);

    /* Emit the raw depth map so the host can compare it bit-for-bit against
     * the host build of the same engine. */
    printf("DAV2_RESULT_BEGIN %d\n", out_size * out_size);
    const uint8_t *raw = (const uint8_t *)depth;
    for (int i = 0; i < out_size * out_size * 4; i++) {
        printf("%02x", raw[i]);
        if ((i & 31) == 31)
            putchar('\n');
    }
    printf("\nDAV2_RESULT_END\n");
    return 0;
}
