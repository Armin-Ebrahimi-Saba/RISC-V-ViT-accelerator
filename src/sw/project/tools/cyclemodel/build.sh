#!/bin/sh
# Build the three harness programs against the engine sources.
#   build.sh [engine source dir]      (default: this repository's)
# Needs the RISC-V GCC the flow uses. Run from this directory.
# FRAME_FLAGS="-DMACW=1 -DPIPE=2" models the block before round fifteen.
# FRAME_FLAGS=-DONCHIP_C=0 models it without the result RAM (before round sixteen).
# FRAME_FLAGS="-DREUSE_C=0 -DONCHIP_CONV=0" models round sixteen.
# FRAME_FLAGS=-DGRELU_C=0 models the block without the gather ReLU (round eighteen).
# FRAME_FLAGS=-DEXP_C=0 models the softmax on the CPU (round nineteen).
# FRAME_FLAGS=-DWSH_C=0 models the block without CTRL.wsh and CTRL.lutint (round twenty-one).
# FRAME_FLAGS=-DLERP_C=0 models the block without the LERP job (round twenty-six).
# FRAME_FLAGS=-DATTCR_C=0 models the attention's S and C in DDR3 (round twenty-seven).
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-$HERE/../..}
GCC=$(command -v riscv-none-elf-gcc || ls ~/Public/xpack-riscv-none-elf-gcc-*/bin/riscv-none-elf-gcc | tail -1)
FL="-Wall -ffreestanding -fno-builtin -nostdlib -g -Os -mabi=ilp32 -march=rv32imc_zicsr -Wl,--no-warn-rwx-segments"
cd "$HERE"
$GCC $FL -T link.ld -o calib.elf start.S calib.c -lgcc
$GCC $FL -I"$SRC" -T link.ld -o bench.elf start.S bench.c engine_wrap.c "$SRC/dav2_ops.c" "$SRC/dav2_blob.c" -lgcc
$GCC $FL $FRAME_FLAGS -I"$SRC" -T link.ld -o frame.elf start.S frame.c "$SRC/dav2_ops.c" "$SRC/dav2_engine.c" "$SRC/dav2_blob.c" -lgcc
