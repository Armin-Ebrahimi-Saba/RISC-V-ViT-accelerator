# Goal

My goal is to run smallest version of depth-anything v2 on this board
You should write in the student-related files
Read the docs/ to get yourself familiar with the structure of the project
Model weights should be stored on the DRAM (DDR3)

# Hardware

Our hardware is Xilinx Artix-7 XC7A200T.



# Run

All flow commands need `source .venv/bin/activate` first. `flow` with no
arguments lists every target and its status; `-R` forces a rebuild of
dependencies (needed whenever RTL changed but the flow thinks it is current).

    # 1. reference — whole model natively, ~4 s, the golden output
    make -C src/sw/project/host
    ./src/sw/project/host/dav2_host build/dav2/dav2_weights.bin out.bin

    # 2. simulation — module testbenches (seconds to minutes)
    flow <tb_name>.sim_rtl_xsim            # e.g. rvlab_ddr_dready_tb
    flow systb_project.sim_rtl_xsim        # whole SoC, no DDR3
    flow systb_project.sim_ddrmodel_xsim   # whole SoC, behavioural DDR3 back end

    # 3. software
    flow reggen.generate                   # after any *.hjson change
    flow libsys.build && flow sw_project.build   # BOTH, after reggen

    # 4. hardware
    flow rvlab_fpga_top.syn
    flow rvlab_fpga_top.pnr
    flow rvlab_fpga_top.bitstream          # builds syn+pnr if needed
    flow rvlab_fpga_top.program            # load onto the board

    # 5. run on the board (loads weights over JTAG, ~1 min)
    python -u src/sw/project/tools/dav2_run_fpga.py --timeout 300

Changing a register in `src/design/reggen/*.hjson` shifts the offsets of every
register after it, so step 3 must rebuild `libsys` as well as `sw_project`.
Software built against a stale register map fails in confusing ways.

After running bitstream, pnr and syn check the following files for warnings and fix them without changing third party libraries if possible:

- build/rvlab_fpga_top/bitstream/rvlab_fpga_top.io_report.txt
- build/rvlab_fpga_top/syn/rvlab_fpga_top.*.txt
- build/rvlab_fpga_top/pnr/rvlab_fpga_top.*.txt

DDR3 simulation is very slow, do not do it.

# Debug

Killing the runner to attach causes openocd.start to reset the core, so every PC reading is a freshly restarted program, not a stalled one

Use debugger for solving software issues.
As the program can hang consider checking the progress every 1 minute or 30 seconds.
You may define new registers that hold the debugging info needed.


# Git

Do not mention yourself as contributor.