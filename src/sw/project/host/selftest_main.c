/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Host side of the kernel self-test. Prints the same checksum the RISC-V
 * build prints; the two must be identical.
 */
#include <stdint.h>
#include <stdio.h>
void dav2_selftest_report(void);
/* The operators time themselves; the self-test does not need the numbers. */
uint64_t dav2_cycles(void) { return 0; }
void dav2_progress(const char *stage) { (void)stage; }
int main(void) { dav2_selftest_report(); return 0; }
