# Hardware

Our hardware is Xilinx Artix-7 XC7A200T.


# Run 

After running bitstream, pnr and syn check the following files for warnings and fix them without changing third party libraries if possible:

- build/rvlab_fpga_top/bitstream/rvlab_fpga_top.io_report.txt
- build/rvlab_fpga_top/syn/rvlab_fpga_top.*.txt
- build/rvlab_fpga_top/pnr/rvlab_fpga_top.*.txt

DDR3 simulation is very slow, do not do it.
Use debugger for solving software issues.