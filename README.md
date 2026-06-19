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
    vector_dot_ir.py   portable multi-PE vector-dot command IR
    vector_dot_trace_io.py
                      JSONL serialization for vector-dot commands
  hardware/
    local_memcpy.py    SimPy PE SRAM memcpy golden model, including color p2p
    vector_dot.py      SimPy multi-PE vector-dot golden model
    pe.py              PE compute and memory timing model
    network.py         mesh, directed links, packet routing, packet metrics
  workloads/
    kernels.py         CPU reference kernels, currently GEMV and vector dot
docs/
  csl_compat_mapping.md
  chisel_memcpy_mapping.md
tests/
  fixtures/
    pe_local_memcpy.jsonl    shared SimPy/Chisel PE-local memcpy command IR
    pe_to_pe_memcpy.jsonl    shared SimPy/Chisel PE-to-PE memcpy command IR
    vector_dot.jsonl         shared SimPy/Chisel multi-PE vector-dot IR
    pe_local_memcpy.trace    legacy text trace, converted into command IR
  test_simulator.py
chisel/
  src/main/scala/wafersim/rtl/
    PeMemcpyEngine.scala     PE-local memcpy FSM and SRAM
    TwoPeMemcpyTop.scala     two-PE p2p memcpy controller and datapath
    VectorDotEngine.scala    PE-local vector dot FSM and SRAM
    VectorDotRowTop.scala    multi-PE row wrapper with east reduction
    ProcessingElement.scala  PE wrapper around the memcpy engine
  src/test/scala/wafersim/rtl/
    PeMemcpyEngineTest.scala Chisel tests plus SimPy trace replay
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
- A multi-PE vector dot mapping: PE-local dot engines plus eastward partial-sum
  reduction using the same SimPy/Chisel JSONL command IR.

This is not a CSL compiler and does not import Cerebras SDK libraries. It is a
Python simulator that mirrors the execution structure so future workloads can be
added without requiring users to learn CSL first.
