# SPDX-License-Identifier: CC0-1.0
# SPDX-FileCopyrightText: 2026 RVLab Student Project
#
# Host build of the inference engine, for verification against the PyTorch
# reference. Uses the exact same engine sources as the RISC-V build.

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -Wno-unused-parameter -g
BUILD   ?= ../../../../build/dav2

SRCS = host_main.c host_stubs.c \
       ../dav2_ops.c ../dav2_engine.c ../dav2_blob.c \
       ../dav2_accel.c

all: dav2_host

dav2_host: $(SRCS) ../dav2.h ../dav2_blob_config.h
	$(CC) $(CFLAGS) -I.. -o $@ $(SRCS) -lm

run: dav2_host
	./dav2_host $(BUILD)/dav2_weights.bin $(BUILD)/host_depth.bin

# The same engine with the accelerator driver compiled in, programming a
# register-level C model of student_gemm (accel_emu.c) instead of hardware.
# Its output must be bit-identical to dav2_host's: that checks the driver and
# every accelerated code path without a board. Non-PIE, and the big buffers
# come from MAP_32BIT, because the block's registers hold 32-bit addresses.
REGGEN ?= ../../../../build/reggen/generate/include
dav2_host_emu: $(SRCS) accel_emu.c accel_emu.h ../dav2.h ../dav2_accel.h
	$(CC) $(CFLAGS) -DDAV2_ACCEL_EMU -no-pie -fno-pie -I.. -I$(REGGEN) \
	    -o $@ $(SRCS) accel_emu.c -lm

emu: dav2_host_emu

clean:
	rm -f dav2_host dav2_host_emu dav2_selftest

.PHONY: all run clean emu

selftest:
	$(CC) $(CFLAGS) -I.. -o dav2_selftest \
	    selftest_main.c ../dav2_selftest.c ../dav2_ops.c \
	    ../dav2_accel.c -lm
	./dav2_selftest
