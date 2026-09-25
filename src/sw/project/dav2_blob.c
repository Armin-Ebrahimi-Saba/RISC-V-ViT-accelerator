/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Access to the weight blob that export_dav2.py places in DDR3.
 *
 * The blob is self-describing: a header, a directory of named tensor records,
 * and 64-byte-aligned tensor data. Lookups happen once per layer during setup,
 * never inside an inner loop, so a linear scan of the directory is fine.
 */

#include "dav2.h"

#include <stdio.h>
#include <string.h>

static const uint8_t       *blob_base;
static const dav2_header_t *blob_hdr;
static const dav2_record_t *blob_dir;

void dav2_blob_init(const void *blob)
{
    blob_base = (const uint8_t *)blob;
    blob_hdr  = (const dav2_header_t *)blob;
    blob_dir  = (const dav2_record_t *)(blob_base + blob_hdr->dir_off);
}

int dav2_blob_check(void)
{
    if (blob_hdr->magic != DAV2_MAGIC) {
        printf("dav2: bad magic %08lx (expected %08lx) -- are the weights "
               "loaded into DDR3?\n",
               (unsigned long)blob_hdr->magic, (unsigned long)DAV2_MAGIC);
        return 1;
    }
    if (blob_hdr->version != DAV2_VERSION) {
        printf("dav2: blob version %lu, engine expects %lu%s\n",
               (unsigned long)blob_hdr->version, (unsigned long)DAV2_VERSION,
               blob_hdr->version < DAV2_VERSION
                   ? " -- convert the float blob with tools/dav2_blob_int.py" : "");
        return 1;
    }
    return 0;
}

const void *dav2_find(const char *name, uint32_t *nbytes)
{
    /* Retry a miss once -- a diagnostic that stays because it is cheap and
     * answers a question in one line.
     *
     * Every tensor named here does exist, so a miss means the directory scan
     * read something other than what DDR3 holds. A lookup that fails and
     * then succeeds would prove a *transient* bad read; one that fails twice
     * proves a *persistent* one. On hardware the misses were all persistent,
     * which pointed away from timing and at a logic fault: rvlab_ddr_prefetch
     * returning another region's cache line (now bypassed). With the
     * prefetcher out of the path there are no misses at all, and this loop
     * finds every tensor on its first pass. */
    for (int attempt = 0; attempt < 2; attempt++) {
        for (uint32_t i = 0; i < blob_hdr->n_tensors; i++) {
            if (strcmp(blob_dir[i].name, name) == 0) {
                if (nbytes) *nbytes = blob_dir[i].nbytes;
                if (attempt)
                    printf("dav2: '%s' found on retry (transient bad read)\n",
                           name);
                return blob_base + blob_dir[i].offset;
            }
        }
    }
    /* Report the header as the scan just saw it. n_tensors is re-read from
     * DDR3 on every call and used as the loop bound, so a wrong word here
     * would silently truncate the search. (Checked during debugging: it was
     * always the correct 299, which ruled the header out.) */
    printf("dav2: tensor '%s' not found in blob (magic=%08lx n_tensors=%lu)\n",
           name, (unsigned long)blob_hdr->magic,
           (unsigned long)blob_hdr->n_tensors);
    return 0;
}

/* Same lookup, but a miss is legitimate (e.g. layers without a bias). */
const void *dav2_find_quiet(const char *name)
{
    for (uint32_t i = 0; i < blob_hdr->n_tensors; i++)
        if (strcmp(blob_dir[i].name, name) == 0)
            return blob_base + blob_dir[i].offset;
    return 0;
}

const uint32_t *dav2_find_dims(const char *name)
{
    for (uint32_t i = 0; i < blob_hdr->n_tensors; i++)
        if (strcmp(blob_dir[i].name, name) == 0)
            return blob_dir[i].dims;
    return 0;
}

void dav2_qw(dav2_qw_t *out, const char *base, int expect_k)
{
    char buf[64];
    uint32_t nb = 0;

    strcpy(buf, base); strcat(buf, ".w");
    out->w = (const int8_t *)dav2_find(buf, &nb);
    const uint32_t *dims = dav2_find_dims(buf);

    strcpy(buf, base); strcat(buf, ".s");
    out->s = (const dav2_xf_t *)dav2_find(buf, 0);     /* (m, sh) pairs */

    strcpy(buf, base); strcat(buf, ".b");
    out->b = (const dav2_xf_t *)dav2_find_quiet(buf);

    if (dims) {
        out->m = (int)dims[0];
        out->k = (int)dims[1];
    } else {
        out->m = 0;
        out->k = 0;
    }
    if (expect_k > 0 && out->k != expect_k)
        printf("dav2: '%s' has k=%d, expected %d\n", base, out->k, expect_k);
}
