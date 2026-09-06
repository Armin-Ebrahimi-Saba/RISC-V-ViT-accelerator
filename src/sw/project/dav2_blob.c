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
        printf("dav2: blob version %lu, engine expects %lu\n",
               (unsigned long)blob_hdr->version, (unsigned long)DAV2_VERSION);
        return 1;
    }
    return 0;
}

const void *dav2_find(const char *name, uint32_t *nbytes)
{
    for (uint32_t i = 0; i < blob_hdr->n_tensors; i++) {
        if (strcmp(blob_dir[i].name, name) == 0) {
            if (nbytes) *nbytes = blob_dir[i].nbytes;
            return blob_base + blob_dir[i].offset;
        }
    }
    printf("dav2: tensor '%s' not found in blob\n", name);
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
    out->s = (const float *)dav2_find(buf, 0);

    strcpy(buf, base); strcat(buf, ".b");
    out->b = (const float *)dav2_find_quiet(buf);

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
