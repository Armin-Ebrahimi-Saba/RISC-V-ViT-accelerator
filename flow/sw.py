# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2024 RVLab Contributors

from pydesignflow import Block, task, Result
from .tools.build_sw import build_sw, build_static_lib
from .tools.elf2mem import elfdelta
from .tools import openocd
from pathlib import Path
from .tools.overlay import filter_solutions_overlay

class Libsys(Block):
    """
    Shared system library including a small libc providing basic system functions such as printf, memcpy etc.
    """

    def setup(self):
        self.src_dir = self.flow.base_dir / "src"

    @task(requires={'reggen':'reggen.generate'},hidden=True)
    def build(self, cwd, reggen):
        """
        Builds library for static linking (.a).
        """
        r = Result()

        sw_dir = self.src_dir / "sw"
        sys_src_dir = sw_dir / "sys"
        sys_include_dir = sw_dir / "include"
        srcs = list(sys_src_dir.glob("*.c"))

        r.lib = cwd / "libsys.a"

        build_static_lib(cwd, srcs, r.lib,
            include_system=[
                sys_include_dir,
                reggen.c_include_dir,
            ],
            include_quote=[],
        )

        return r

class Program(Block):
    """Program for the RISC-V CPU"""

    def __init__(self, name, **kwargs):
        """
        Args:
            name: Name of src/sw/ subdirectory containing program-specific
                sources files.
        """
        super().__init__(**kwargs)
        self.name = name

    def setup(self):
        self.src_dir = self.flow.base_dir / "src"
        self.design_dir = self.src_dir / "design"

    @task(requires={
        'libsys':'libsys.build',
        'reggen':'reggen.generate',
        }, hidden=True, always_rebuild=True)
    def build(self, cwd, libsys, reggen):
        """
        Main program for simulation and later use on FPGA.
        """
        r = Result()

        sw_dir = self.src_dir / "sw"
        ldscript = str(sw_dir / "link.ld")
        main_dir = sw_dir / self.name
        sys_include_dir = sw_dir / "include"
        user_includes = sw_dir / "include"

        srcs = []
        srcs += list(main_dir.glob("*.S"))
        srcs += list(main_dir.glob("*.c"))

        # Shared source files in sw/ folder:
        srcs += list(sw_dir.glob("*.S")) # should be [crt0.S] at the moment
        srcs += list(sw_dir.glob("*.c")) # should be [hostio.c] at the moment

        srcs = filter_solutions_overlay(srcs, self.src_dir)

        r.elf = cwd / "sw.elf"
        r.mem = cwd / "sw.mem"
        r.disasm = cwd / "sw.disasm"

        # Optional extra compiler/linker flags for this program, one per
        # line, "#" starts a comment. Absent file: no extra flags.
        extra_flags = []
        flags_file = main_dir / "build_flags.txt"
        if flags_file.exists():
            for line in flags_file.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    extra_flags.append(line)

        build_sw(
            cwd=cwd,
            srcs=srcs,
            ldscript=ldscript,
            output_elf_filename=r.elf,
            output_disasm_filename=r.disasm,
            output_mem_filename=r.mem,
            static_libs=[libsys.lib],
            include_system=[
                sys_include_dir,
                reggen.c_include_dir,
            ],
            include_quote=[],
            extra_flags=extra_flags,
        )

        return r

    @task(requires={'build':'.build'})
    def run(self, cwd, build):
        """Run on FPGA via OpenOCD"""
        with openocd.start(self.design_dir / "openocd/fpga.cfg") as ocd:
            ocd.run_prog(build.elf)
            #input("Press enter to continue...")

    @task(requires={'build':'.build', 'ref_build':'ref.build'}, hidden=True, always_rebuild=True)
    def delta(self, cwd, build, ref_build):
        """Differential image for fast loading in simulator"""
        r = Result()
        r.deltafile = cwd / "delta"
        elfdelta(build.elf, ref_build.elf, r.deltafile)
        return r
