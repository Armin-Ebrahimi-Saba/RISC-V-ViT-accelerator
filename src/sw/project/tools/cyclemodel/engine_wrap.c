/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 */
#include "dav2_engine.c"
void bench_attention(const dav2_tensor_t *qkv, int n, dav2_tensor_t *ctx) { attention_all(qkv, n, ctx); }
