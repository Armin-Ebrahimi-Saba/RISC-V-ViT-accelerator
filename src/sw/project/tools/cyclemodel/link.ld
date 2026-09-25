ENTRY(_start)
MEMORY { ram (rwx) : ORIGIN = 0x00000000, LENGTH = 1M }
SECTIONS {
  .text : { *(.text.start) *(.text*) } > ram
  .rodata : { *(.rodata*) *(.srodata*) } > ram
  .data : { *(.data*) *(.sdata*) } > ram
  .bss : { __bss_start = .; *(.bss*) *(.sbss*) *(COMMON) __bss_end = .; } > ram
  __stack_top = ORIGIN(ram) + LENGTH(ram);
}
