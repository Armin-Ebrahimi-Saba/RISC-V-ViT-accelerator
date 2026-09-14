RISC-V ViT Accelerator
======================

Depth-Anything V2 Small — a vision transformer that turns a photo into a depth
map — running on a CV32E40P RISC-V core on an Artix-7 FPGA, with a custom int8
GEMM accelerator for the matrix multiplications.

No operating system, no inference framework, no floating-point unit. The
network is a hand-written freestanding C engine; the matmuls are hardware.

![Input photo, PyTorch depth, FPGA depth](img/result.png)

*Left: input. Middle: PyTorch reference. Right: the FPGA's output. Bright is
near.*

Results
-------

| | |
|---|---|
| FPGA output vs. the same C engine on a PC | **bit-exact**, 15876/15876 pixels, on every image tried |
| FPGA output vs. PyTorch reference | correlation **0.99984** |
| Frame time on the board | **93.6 s** (0.0107 FPS) at 50 MHz |
| Accelerator vs. CPU on the matmul | **~36×** |
| Timing | met, WNS **+0.287 ns** at 50 MHz, 0 failing endpoints |
| Resources | LUT 12.4 %, BRAM 23.0 %, DSP 3.1 % of an XC7A200T |

One frame at 126×126 is 2.4 G multiply-accumulates. The accelerator takes those
off the critical path; what remains is the CPU's element-wise work
(LayerNorm, softmax, GELU, requantisation) in software floating point, and the
25 MB of weights crossing DDR3 once per frame.

More of the images it has produced, input left and depth right. All are
bit-exact with the host build:

![road](img/road.png)
![pillars](img/pillars.png)
![sphere](img/sphere.png)
![corridor](img/corridor.png)

How it works
------------

![Accelerator architecture](img/accelerator.svg)

**Software** (`src/sw/project/`) — ~2000 lines of C, freestanding. Weights are
int8 with a per-output-channel scale, activations int16 at 14 bits, and each
scale is pre-baked into a gemmlowp-style `(multiplier, shift)` pair so the
device never divides. Memory comes from a bump allocator over a DDR3 arena;
`sqrt`, `exp` and `erf` are hand-written because the build links `-nostdlib`.

Every matmul in the network — patch embedding, attention projections, MLPs,
the DPT head — funnels through one function, `dav2_qgemm()`. That single choke
point is why the accelerator needed exactly one attachment point.

**Hardware** (`src/rtl/student/student_gemm.sv`) — a 16-wide
int8×int16 → int32 MAC array with its own TL-UL host port. A tile of 16
activation rows loads into on-chip BRAM once, then the weight matrix streams
past as one contiguous byte stream, each weight broadcasting to all 16
multipliers. Weights — the dominant memory traffic — cross the bus exactly
once. A retry timer re-issues any read the platform's cache fails to answer.

The software kernel never leaves: `dav2_accel_qgemm()` declines any shape the
hardware cannot take, and a boot-time cross-check disables the block on
mismatch rather than producing wrong answers.

What it took
------------

The accelerator was right early. What stood between it and a correct depth
map were three defects in the platform's DDR3 path — a request-accept
handshake taken from the wrong module, a cache that wrote back stale data
when a line was evicted the cycle after it was written, and a prefetcher that
returned the wrong address's data under collisions. Finding them needed a
hardware watchdog register (because a CPU wedged on a bus access cannot be
halted by a debugger) and a testbench driver rewritten to issue requests
back-to-back like a real CPU. The full account, including the wrong turns, is
in `docs/DEBUGGING.md`.

Documentation
-------------

    docs/ARCHITECTURE.md   the SoC, the memory map, the accelerator, the DDR3 path — with diagrams
    docs/DATAFLOW.md       timing diagrams: the bus handshake, one inference, the write-back bug
    docs/DEBUGGING.md      every defect: symptom, wrong theories, instrument, fix, verification
    docs/LESSONS.md        portable rules for the next project
    docs/HANDOFF.md        current state, what is bypassed, what is not done
    CLAUDE.md              build, simulate, and run commands

Layout
------

    src/rtl/student/student_gemm.sv        the accelerator
    src/rtl/student/student_tl_watch.sv    stalled-transaction watchdog register
    src/rtl/student/student.sv             bus integration
    src/rtl/ddr3/rvlab_tlul_ddr.sv         DDR3 path top (request mux fix, prefetch bypass)
    src/rtl/ddr3/rvlab_ddr_block_cache.sv  the cache (write-back fix)
    src/design/reggen/                     register maps
    src/sw/project/                        the inference engine
    src/sw/project/host/                   native build — the oracle
    src/sw/project/tools/                  weight export, board runner, image patcher
    src/tb/                                testbenches

Building and running
--------------------

    source .venv/bin/activate
    flow sw_project.build                       # RISC-V program
    flow student_gemm_tb.sim_rtl_xsim           # accelerator unit test
    flow rvlab_ddr_alias_tb.sim_rtl_xsim        # the cache and prefetcher tests
    flow rvlab_fpga_top.bitstream               # syn + pnr + bitstream
    flow rvlab_fpga_top.program                 # load onto the board
    python -u src/sw/project/tools/dav2_run_fpga.py --timeout 900

The same engine sources build natively, which is the fast way to check any
change and the reference every board result must match bit for bit:

    make -C src/sw/project/host
    ./src/sw/project/host/dav2_host build/dav2/dav2_weights.bin out.bin

To run a different image, `src/sw/project/tools/dav2_patch_image.py` rewrites
the image tensors in a copy of the blob — synthetic scenes built in, or any
binary P6 PPM — with no torch required.

Built on
--------

[tub-msc/rvlab](https://github.com/tub-msc/rvlab), the SoC + RISC-V Lab platform
from the [Mixed Signal Circuit Design group](http://tu.berlin/msc) at TU Berlin.
The platform's own documentation is at
[rvlab.readthedocs.io](https://rvlab.readthedocs.io/en/latest/); setup
instructions are in `docs/tutorials/setup/index.rst`.

Third-party sources under `src/rtl/` (CV32E40P, the TL-UL fabric, OpenTitan
primitives, the UberDDR3 controller) are unmodified. Two RVLab platform files
in `src/rtl/ddr3/` carry fixes, each documented at the change.
