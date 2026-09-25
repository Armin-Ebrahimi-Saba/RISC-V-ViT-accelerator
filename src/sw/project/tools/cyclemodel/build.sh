#!/bin/sh
# Build the three harness programs against the engine sources.
#   build.sh [engine source dir]      (default: this repository's)
# Needs the RISC-V GCC the flow uses. Run from this directory.
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-$HERE/../..}
GCC=$(command -v riscv-none-elf-gcc || ls ~/Public/xpack-riscv-none-elf-gcc-*/bin/riscv-none-elf-gcc | tail -1)
FL="-Wall -ffreestanding -fno-builtin -nostdlib -g -Os -mabi=ilp32 -march=rv32imc_zicsr -Wl,--no-warn-rwx-segments"
cd "$HERE"
$GCC $FL -T link.ld -o calib.elf start.S calib.c -lgcc
$GCC $FL -I"$SRC" -T link.ld -o bench.elf start.S bench.c engine_wrap.c "$SRC/dav2_ops.c" "$SRC/dav2_blob.c" -lgcc
$GCC $FL -I"$SRC" -T link.ld -o frame.elf start.S frame.c "$SRC/dav2_ops.c" "$SRC/dav2_engine.c" "$SRC/dav2_blob.c" -lgcc
