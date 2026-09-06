/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Host side of the kernel self-test. Prints the same checksum the RISC-V
 * build prints; the two must be identical.
 */
#include <stdio.h>
void dav2_selftest_report(void);
int main(void) { dav2_selftest_report(); return 0; }
