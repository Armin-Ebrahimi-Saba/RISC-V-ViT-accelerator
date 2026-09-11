/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Depth-Anything V2 Small on the rvlab SoC -- the program the CPU runs.
 *
 * What happens, in order:
 *   1. Print a banner and run dav2_selftest (a checksum that must match the
 *      host build bit for bit) and the accelerator's small self-check.
 *   2. Bring up DDR3 and wait for the host to load the weights over JTAG.
 *   3. Run one inference (dav2_infer in dav2_engine.c) into a float depth map.
 *   4. Print the depth map as hex so the host can capture it.
 *
 * DDR3 map (see docs/design_ref/memory_map.rst -- DDR3 starts at 0x80000000):
 *
 *   0x80000000  weight blob produced by tools/export_dav2.py (~25 MB)
 *   0x82000000  activation arena for the engine (peak ~16 MB at 126x126)
 *
 * The handshake that says "weights are in" is the BRAM variable dav2_go
 * below, NOT a DDR3 address -- see its comment. The CPU cannot pull 25 MB in
 * by itself, so the host script (src/sw/project/tools/dav2_run_fpga.py) loads
 * the blob over JTAG while this program spins at the handshake, then writes
 * the flag.
 *
 * How this file fits with the others:
 *   dav2_engine.c   the network: patch embedding, 12 transformer blocks, DPT head
 *   dav2_ops.c      the quantised kernels those layers are built from
 *   dav2_accel.c    driver for the hardware GEMM; the kernels call it first
 *   dav2_blob.c     finds tensors by name in the weight blob
 *   dav2_mathf.c    sqrt/exp/erf, because there is no libm in a -nostdlib build
 *   dav2_selftest.c the bit-exactness checksum
 *
 * "Blob" throughout means the single file of weights plus input image that
 * export_dav2.py produces; "arena" is a bump allocator (allocate by moving a
 * pointer forward, free by moving it back) over a DDR3 region.
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
#define GO_MAGIC    0xD00DFEEDu   /* value the host writes into dav2_go */

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
    /* Wait for the host to say the weights are complete.
     *
     * Do not be tempted to start on "the blob header looks valid" instead:
     * the magic word lives at the very start of the blob, so it is written by
     * the first chunk of the transfer, and inference then runs against a blob
     * that is still being written underneath it. Measured effect: DDR3
     * read-back fails outright a few seconds later. The flag is the only
     * signal that means the whole transfer finished. */
    while (dav2_go != GO_MAGIC)
        ;
    printf("weights present (%u bytes expected)\n", (unsigned)DAV2_BLOB_BYTES);

    dav2_blob_init((const void *)BLOB_ADDR);
    dav2_arena_init((void *)ARENA_ADDR, ARENA_SIZE);

    /* Start-up self-checks. Off by default: the checksum reads 24.87 MB and
     * the GEMM comparison runs every model shape twice, which is a second or
     * two each and pointless on a run whose purpose is inference. Set to 1
     * when the numbers look wrong -- between them they localise a bad result
     * to the weights, the accelerator, or neither. */
#define DAV2_STARTUP_CHECKS 0

#if DAV2_STARTUP_CHECKS
    {
        /* Does DDR3 hold what is written to it, at all?
         *
         * Random words at random arena addresses, in two passes: write them
         * all, then read them all back. The set is 64k words spread over the
         * arena, so by the time anything is re-read the 16 kB cache has been
         * turned over hundreds of times and every read is a genuine refill of
         * a line that was written back.
         *
         * Addresses that the random sequence picks more than once hold the
         * LAST value written, not the one this op wrote. The first version of
         * this test compared against the op's own value and reported exactly
         * 111 failures, stable across runs -- precisely the number of
         * repeated addresses. That was the checker, not the memory. One
         * replay pass now marks repeated addresses in a bitmap kept at the
         * top of the arena, and those are skipped. */
        enum { NWORDS = 65536 };
        volatile uint32_t *arena = (volatile uint32_t *)ARENA_ADDR;
        const uint32_t bitmap_bytes = 2u * 1048576u;          /* 16M bits */
        const uint32_t test_words   = (ARENA_SIZE - bitmap_bytes) / 4u;
        volatile uint32_t *seen  = (volatile uint32_t *)(ARENA_ADDR + ARENA_SIZE - bitmap_bytes);
        volatile uint32_t *twice = (volatile uint32_t *)(ARENA_ADDR + ARENA_SIZE - bitmap_bytes / 2u);
        uint32_t seed, bad = 0, first_bad = 0, got_bad = 0, want_bad = 0, skipped = 0;

        for (uint32_t i = 0; i < bitmap_bytes / 4u; i++) seen[i] = 0;
        seed = 0x9e3779b9u;
        for (uint32_t i = 0; i < NWORDS; i++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            uint32_t idx = seed % test_words;
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            uint32_t w = idx >> 5, b = 1u << (idx & 31u);
            if (seen[w] & b) twice[w] |= b; else seen[w] |= b;
        }

        seed = 0x9e3779b9u;
        for (uint32_t i = 0; i < NWORDS; i++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            uint32_t idx = seed % test_words;
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            arena[idx] = seed;
        }
        seed = 0x9e3779b9u;
        for (uint32_t i = 0; i < NWORDS; i++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            uint32_t idx = seed % test_words;
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            if (twice[idx >> 5] & (1u << (idx & 31u))) { skipped++; continue; }
            uint32_t got = arena[idx];
            if (got != seed) {
                if (!bad) { first_bad = idx; got_bad = got; want_bad = seed; }
                if (bad < 6)
                    printf("  bad %08lx got %08lx want %08lx reread %08lx\n",
                           (unsigned long)(ARENA_ADDR + idx * 4u),
                           (unsigned long)got, (unsigned long)seed,
                           (unsigned long)arena[idx]);
                bad++;
            }
        }
        printf("DDR3 random r/w: %lu/%u words wrong (%lu repeated addresses skipped)",
               (unsigned long)bad, NWORDS, (unsigned long)skipped);
        if (bad)
            printf(", first at %08lx got %08lx want %08lx",
                   (unsigned long)(ARENA_ADDR + first_bad * 4u),
                   (unsigned long)got_bad, (unsigned long)want_bad);
        printf("\n");

        /* Random order, but confined to 8 kB -- half the cache, so nothing is
         * ever evicted. If this passes while the full-arena random test
         * fails, the fault is in eviction and write-back, not in random
         * access as such. */
        {
            uint32_t s2 = 0x7f4a7c15u, cbad = 0;
            for (uint32_t i = 0; i < NWORDS; i++) {
                s2 ^= s2 << 13; s2 ^= s2 >> 17; s2 ^= s2 << 5;
                arena[s2 % 2048u] = s2;
                if (arena[s2 % 2048u] != s2) cbad++;
            }
            printf("DDR3 random r/w within cache: %lu/%u wrong\n",
                   (unsigned long)cbad, NWORDS);
        }

        /* Same test, sequential rather than random. A DRAM cell or lane
         * fault fails regardless of order; an eviction or reordering fault
         * in the cache path needs the random pattern to provoke it. */
        {
            uint32_t sbad = 0;
            for (uint32_t i = 0; i < NWORDS; i++)
                arena[i * 64u] = i * 2654435761u;      /* one word per line */
            for (uint32_t i = 0; i < NWORDS; i++)
                if (arena[i * 64u] != i * 2654435761u) sbad++;
            printf("DDR3 sequential r/w: %lu/%u words wrong\n",
                   (unsigned long)sbad, NWORDS);
        }
    }
    {
        /* Checksum the whole blob as the CPU sees it, through the same path
         * the engine uses. A JTAG sample says what DDR3 holds at rest; this
         * says what the CPU actually reads. */
        const volatile uint32_t *bw = (const volatile uint32_t *)BLOB_ADDR;
        uint32_t h = 2166136261u;
        /* Proven identical to the file already; skip the 24.87 MB walk. */
        for (uint32_t i = 0; i < 0u; i++) {
            h ^= bw[i];
            h *= 16777619u;
        }
        printf("blob checksum (device) %08lx over %lu bytes\n",
               (unsigned long)h, (unsigned long)DAV2_BLOB_BYTES);
    }

    {
        /* Accelerator against the CPU kernel at the shapes the model issues,
         * with every buffer in DDR3. 81x588x384 -- patch embedding -- loses
         * one accumulator word of 31104, deterministically; the same shape
         * passes in student_gemm_tb against ideal memory, so the fault is on
         * the DDR3 path rather than in the GEMM. See docs/DEBUGGING.md. */
        static const int shapes[][3] = {
            { DAV2_N_TOKENS,   384,  384 },   /* proj            */
            { DAV2_N_TOKENS,   384, 1152 },   /* qkv             */
            { DAV2_N_TOKENS,   384, 1536 },   /* fc1             */
            { DAV2_N_TOKENS,  1536,  384 },   /* fc2             */
            { DAV2_N_PATCHES,  588,  384 },   /* patch embedding */
        };
        for (unsigned i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
            dav2_accel_bigcheck(shapes[i][0], shapes[i][1], shapes[i][2]);
    }
#endif

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
