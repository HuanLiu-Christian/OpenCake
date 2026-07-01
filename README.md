# WaferSim

`wafersim` is a Python/SimPy wafer-scale accelerator simulator focused on a
Cerebras GEMV9-style streaming dataflow. The repository keeps the application
example Python-only while preserving the same conceptual split as the SDK
example: layout, PE program, and host launcher.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python examples/gemv_09_streaming/run.py
```

Run the vector-vector dot product example:

```bash
python examples/vector_vector_dot/run.py
python examples/vector_vector_dot/run.py --vector-length 64 --kernel-cols 8
```

Generate a multi-PE vector-dot compute/reduction IR for Chisel replay:

```bash
python examples/vector_vector_dot/run.py --ir-out /tmp/vector_dot.jsonl
cd chisel
WAFERSIM_VECTOR_DOT_IR=/tmp/vector_dot.jsonl sbt test
```

Run the PE-local memcpy golden trace:

```bash
python examples/pe_local_memcpy/run.py
```

Run the PE-to-PE memcpy example:

```bash
python examples/pe_to_pe_memcpy/run.py
```

Generate a PE-local memcpy command IR for Chisel replay:

```bash
python examples/pe_local_memcpy/run.py --ir-out /tmp/pe_local_memcpy.jsonl
cd chisel
WAFERSIM_MEMCPY_IR=/tmp/pe_local_memcpy.jsonl sbt test
```

Generate a PE-to-PE memcpy command IR for Chisel replay:

```bash
python examples/pe_to_pe_memcpy/run.py --ir-out /tmp/pe_to_pe_memcpy.jsonl
cd chisel
WAFERSIM_P2P_MEMCPY_IR=/tmp/pe_to_pe_memcpy.jsonl sbt test
```

Run a custom GEMV problem:

```bash
python examples/gemv_09_streaming/run.py \
  --matrix-rows 1024 \
  --matrix-cols 1024 \
  --kernel-rows 32 \
  --kernel-cols 32
```

Generate a multi-PE GEMV command IR for Chisel replay:

```bash
python examples/gemv_09_streaming/run.py --ir-out /tmp/gemv_09_streaming.jsonl
cd chisel
WAFERSIM_GEMV_IR=/tmp/gemv_09_streaming.jsonl sbt test
```

Run the GEMM streaming example:

```bash
python examples/gemm_streaming/run.py
python examples/gemm_streaming/run.py --m 8 --n 8 --k 8 --kernel-rows 2 --kernel-cols 2
```

Generate a multi-PE GEMM command IR for Chisel replay:

```bash
python examples/gemm_streaming/run.py --ir-out /tmp/gemm_streaming.jsonl
cd chisel
WAFERSIM_GEMM_IR=/tmp/gemm_streaming.jsonl sbt test
```

Run tests:

```bash
python -m unittest discover -s tests
```

Run the first Chisel RTL mapping test:

```bash
cd chisel
sbt test
```

## Current Dataflow

The example computes:

```text
y = A x + b
```

The active kernel rectangle is a PE grid. Each PE owns a local A tile and a
local y buffer. The host injects x chunks into the top row. PE programs forward
x south through `x_color`, compute their local `A_tile * x_chunk`, and reduce
partial y chunks east through `ax_color_1` / `ax_color_2`. The east edge returns
the final y chunks to the host.

## Project Layout

```text
examples/
  pe_local_memcpy/
    run.py             builds PE-local memcpy API calls, emits IR, runs SimPy
  pe_to_pe_memcpy/
    run.py             builds color-routed PE-to-PE memcpy calls and runs SimPy
  gemv_09_streaming/
    layout.py          mesh size, active PE rectangle, color route directions
    pe_program.py      PE-local GEMV tasks, local buffers, compute, reduction
    run.py             host launcher, CLI args, SimPy execution, verification
  gemm_streaming/
    layout.py          GEMM active rectangle and A/B color route directions
    pe_program.py      PE-local GEMM tasks and tile parameters
    run.py             host launcher, IR generation, SimPy replay, verification
  vector_vector_dot/
    layout.py          1D PE row and eastward partial-sum color
    pe_program.py      PE-local vector chunk dot product and reduction
    run.py             host launcher, CLI args, SimPy execution, verification
src/wafersim/
  config.py            mesh and PE timing parameters
  program.py           Python-native layout, PE program, and problem specs
  csl/
    compat.py          Python mappings for common CSL-style builtins/libraries
  mapping/
    memcpy_ir.py       portable memcpy command IR and program builder API
    trace_io.py        JSONL serialization for mapping commands
    gemv_ir.py         portable multi-PE GEMV command IR
    gemv_trace_io.py   JSONL serialization for GEMV commands
    gemm_ir.py         portable multi-PE GEMM command IR
    gemm_trace_io.py   JSONL serialization for GEMM commands
    vector_dot_ir.py   portable multi-PE vector-dot command IR
    vector_dot_trace_io.py
                      JSONL serialization for vector-dot commands
  hardware/
    gemv.py            SimPy multi-PE GEMV golden model
    gemm.py            SimPy multi-PE GEMM golden model
    local_memcpy.py    SimPy PE SRAM memcpy golden model, including color p2p
    vector_dot.py      SimPy multi-PE vector-dot golden model
    pe.py              PE compute and memory timing model
    network.py         mesh, directed links, packet routing, packet metrics
  workloads/
    kernels.py         CPU reference kernels, currently GEMV, GEMM, and vector dot
docs/
  csl_compat_mapping.md
  chisel_memcpy_mapping.md
tests/
  fixtures/
    pe_local_memcpy.jsonl    shared SimPy/Chisel PE-local memcpy command IR
    pe_to_pe_memcpy.jsonl    shared SimPy/Chisel PE-to-PE memcpy command IR
    gemv_09_streaming.jsonl  shared SimPy/Chisel multi-PE GEMV IR
    gemm_streaming.jsonl     shared SimPy/Chisel multi-PE GEMM IR
    vector_dot.jsonl         shared SimPy/Chisel multi-PE vector-dot IR
    pe_local_memcpy.trace    legacy text trace, converted into command IR
  test_simulator.py
chisel/
  src/main/scala/wafersim/rtl/
    PeMemcpyEngine.scala     PE-local memcpy FSM and SRAM
    TwoPeMemcpyTop.scala     two-PE p2p memcpy controller and datapath
    GemvPeEngine.scala       PE-local GEMV tile compute engine
    GemvArrayTop.scala       multi-PE GEMV array with x south / y east commands
    GemmPeEngine.scala       PE-local GEMM tile compute engine
    GemmArrayTop.scala       multi-PE GEMM array with A east / B south commands
    VectorDotEngine.scala    PE-local vector dot FSM and SRAM
    VectorDotRowTop.scala    multi-PE row wrapper with east reduction
    ProcessingElement.scala  PE wrapper around the memcpy engine
  src/test/scala/wafersim/rtl/
    PeMemcpyEngineTest.scala Chisel tests plus SimPy trace replay
    GemvArrayTopTest.scala   Chisel multi-PE GEMV IR replay test
    GemmArrayTopTest.scala   Chisel multi-PE GEMM IR replay test
    PeToPeMemcpyTest.scala   Chisel two-PE p2p replay test
    VectorDotEngineTest.scala
                            Chisel PE-local and multi-PE vector-dot IR tests
```

## What Is Modeled

- A parameterized 2D PE mesh.
- Local PE SRAM capacity and memory timing.
- PE compute timing in cycles.
- Directional links with finite queues.
- Packet timing and utilization metrics.
- CSL-like color/runtime helpers used by Python examples.
- GEMV9-style streaming with PE-local tasks running concurrently in SimPy.
- Chisel RTL mappings for PE-local memcpy and a two-PE color-routed memcpy path
  using the same portable command IR as the SimPy golden model.
- A multi-PE GEMV mapping: SimPy emits load, x-south, compute, y-east reduce,
  and expect commands; Chisel replays the same JSONL IR against `GemvArrayTop`.
- A multi-PE GEMM mapping: SimPy emits A-panel load/east-forward, B-panel
  load/south-forward, local C-tile compute, and expect commands; Chisel replays
  the same JSONL IR against `GemmArrayTop`.
- A multi-PE vector dot mapping: PE-local dot engines plus eastward partial-sum
  reduction using the same SimPy/Chisel JSONL command IR.

This is not a CSL compiler and does not import Cerebras SDK libraries. It is a
Python simulator that mirrors the execution structure so future workloads can be
added without requiring users to learn CSL first.

## OpenCake Prototype Pipeline

The current OpenCake prototype implements this end-to-end flow:

```text
Chakra workload trace (.jsonl prototype or real .et protobuf)
    -> OpenCake Mapper
    -> Wafer-scale Spatial IR
    -> Astra-Sim-style wafer simulator
```

Files:

```text
examples/opencake_pipeline/
  chakra_gemm.jsonl  simplified Chakra-style GEMM workload trace
  run.py             full pipeline launcher

src/wafersim/opencake/
  chakra.py          reads JSONL traces or real Chakra ET protobuf traces
  workload.py        GEMM workload and wafer hardware spec
  mapper.py          GEMM tiling, PE-block placement, color route, SRAM layout
  ir.py              wafer-scale Spatial IR and JSONL export
  schema.py          formal Spatial IR version and validation
  astrasim.py        formal Astra-Sim experiment package exporter
  simulator.py       Astra-Sim-style wafer backend wrapper
  backend.py         rank-level compute/memory/network cycle model
  pipeline.py        connects trace -> mapper -> IR -> simulator
  formal.py          connects trace -> mapper -> validated IR -> package
```

The JSONL trace keeps the prototype easy to inspect. Real Chakra `.et` input is
also supported through Chakra's generated `et_def_pb2.py`; install the optional
protobuf dependency with `pip install -e ".[chakra]"` or run inside a
Chakra/Astra-Sim Python environment. If the generated protobuf file is outside
the default Astra-Sim checkout path, set `WAFERSIM_CHAKRA_ET_DEF_PB2` to the
full path of `et_def_pb2.py`.

Run the full flow:

```bash
.venv/bin/python examples/opencake_pipeline/run.py \
  --spatial-ir-out /tmp/opencake_spatial_ir.jsonl
```

Default example:

```text
GEMM = 1024 x 1024 x 1024
mesh = 32 x 32 PE
PE block = 8 x 8 PE
rank grid = 4 x 4 = 16 ranks
Spatial IR ops = 90
```

Generate the formal Astra-Sim handoff package:

```bash
.venv/bin/python examples/opencake_formal/run.py \
  --out-dir /tmp/opencake_formal_package
```

The formal package contains `manifest.json`, `system.json`, `network.yml`,
`workload/opencake_spatial_ir.jsonl`, and `reference_result.json`. The Python
reference result stays in the package so a future Astra-Sim C++ OpenCake backend
can be checked against the same Spatial IR.

Validate the same package with the standalone C++ backend:

```bash
.venv/bin/python examples/opencake_formal/run_cpp_backend.py \
  --out-dir /tmp/opencake_formal_package_cpp
```

Install and validate the backend inside an official Astra-Sim source tree:

```bash
.venv/bin/python examples/opencake_formal/run_astrasim_integrated.py \
  --use-docker \
  --out-dir /tmp/opencake_astrasim_integrated_package
```

This adds the Astra-Sim CMake target `AstraSim_OpenCake_Wafer`, runs it on the
formal package, and checks its result against the Python reference result.
