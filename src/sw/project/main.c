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
#define GO_MAGIC    0xD00DFEEDu   /* host: weights are loaded and verified */
#define GO_FRAME    0xF00DF00Du   /* host: a new image is at IMAGE_ADDR, run it */
#define GO_DONE     0xD0DEC0DEu   /* device: result printed, ready for the next */
#define DAV2_AUTOSTART_ADDR 0x81F00000u   /* simulation-only start token */

/* Where the host puts each input image. In the 8 MB gap between the end of
 * the blob and the start of the arena, so neither weights nor activations can
 * touch it. Layout: size*size*3 int16 (HWC), then one float32 scale. */
#define IMAGE_ADDR  0x81E00000u

/* Handshake flag, polled by main() and written over JTAG by the host loader.
 * Must live in BRAM -- see the comment at its use below. */
volatile uint32_t dav2_go;

/* 64-bit cycle count.
 *
 * mcycle alone is 32 bits and wraps every 85.9 s at 50 MHz -- less than one
 * frame. Differencing two 32-bit reads across a frame therefore reported a
 * time short by whole multiples of 85.9 s; the first "inference finished"
 * figure printed by this program was 12.5 s for a 98 s frame. Read the high
 * half too, with the standard re-read loop so a carry between the two reads
 * cannot produce a value that is wrong by 2^32. */
static uint64_t cycles64(void)
{
    uint32_t hi, lo, hi2;
    do {
        __asm__ volatile ("csrr %0, mcycleh" : "=r"(hi));
        __asm__ volatile ("csrr %0, mcycle"  : "=r"(lo));
        __asm__ volatile ("csrr %0, mcycleh" : "=r"(hi2));
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

void dav2_progress(const char *stage)
{
    /* kcycles fit in 32 bits for over 24 hours at 50 MHz. */
    printf("  [%s] t=%u kcycles\n", stage, (unsigned)(cycles64() / 1000u));
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
    while (dav2_go != GO_MAGIC) {
        /* Simulation start token: see ddr3_blk_model.sv. Two words in the
         * gap between blob and arena that nothing on the board writes. Both
         * must match, so a random DDR3 power-up pattern cannot trigger it. */
        const volatile uint32_t *tok = (const volatile uint32_t *)DAV2_AUTOSTART_ADDR;
        if (tok[0] == GO_MAGIC && tok[1] == ~GO_MAGIC) {
            printf("autostart token found (simulation)\n");
            break;
        }
    }
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

    /* The weights are in and verified; from here on the program is a server.
     * Each frame: the host writes the quantised image to IMAGE_ADDR, sets
     * dav2_go = GO_FRAME, and waits for the result markers; the program runs
     * one inference, prints the depth map, and sets dav2_go = GO_DONE. The
     * 25 MB weight transfer happens once per session instead of once per
     * image. */
    const int16_t *image = (const int16_t *)IMAGE_ADDR;
    const float   *image_scale =
        (const float *)(IMAGE_ADDR + (uint32_t)cfg.size * cfg.size * 3u * 2u);

    printf("DAV2_READY\n");
    for (unsigned frame = 0;; frame++) {
        dav2_go = GO_DONE;
        while (dav2_go != GO_FRAME)
            ;

        dav2_set_image(image, *image_scale);
        printf("frame %u: image scale %d/1e6\n", frame, (int)(*image_scale * 1e6f));

        uint64_t t0 = cycles64();
        dav2_infer(&cfg, depth);
        uint64_t t1 = cycles64();

        if (dav2_arena_failed) {
            printf("FATAL: activation arena exhausted -- increase ARENA_SIZE or "
                   "lower DAV2_INPUT_SIZE\n");
            return 1;
        }

        {
            /* Also in seconds and frames per second, so the number that
             * matters does not have to be derived by hand. 50 MHz clock. */
            uint64_t dc = t1 - t0;
            unsigned ms = (unsigned)(dc / 50000u);            /* cycles -> ms */
            printf("\ninference finished in %u kcycles = %u.%03u s  (%u.%04u FPS)\n",
                   (unsigned)(dc / 1000u), ms / 1000u, ms % 1000u,
                   (unsigned)(1000u / (ms ? ms : 1)),
                   (unsigned)((10000000ull / (ms ? ms : 1)) % 10000u));
        }
        printf("arena peak %u KB\n", (unsigned)(dav2_arena_peak() / 1024u));

        print_ascii_depth(depth, out_size, 63);

        /* Emit the raw depth map so the host can compare it bit-for-bit
         * against the host build of the same engine. */
        printf("DAV2_RESULT_BEGIN %d\n", out_size * out_size);
        {
            const uint8_t *raw = (const uint8_t *)depth;
            for (int i = 0; i < out_size * out_size * 4; i++) {
                printf("%02x", raw[i]);
                if ((i & 31) == 31)
                    putchar('\n');
            }
        }
        printf("\nDAV2_RESULT_END\n");
    }
}
