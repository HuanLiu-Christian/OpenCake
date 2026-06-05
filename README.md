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
  gemv_09_streaming/
    layout.py          mesh size, active PE rectangle, color route directions
    pe_program.py      PE-local GEMV tasks, local buffers, compute, reduction
    run.py             host launcher, CLI args, SimPy execution, verification
src/wafersim/
  config.py            mesh and PE timing parameters
  program.py           Python-native layout, PE program, and problem specs
  csl/
    compat.py          Python mappings for common CSL-style builtins/libraries
  hardware/
    pe.py              PE compute and memory timing model
    network.py         mesh, directed links, packet routing, packet metrics
  workloads/
    kernels.py         CPU reference kernels, currently GEMV
docs/
  csl_compat_mapping.md
tests/
  test_simulator.py
```

## What Is Modeled

- A parameterized 2D PE mesh.
- Local PE SRAM capacity and memory timing.
- PE compute timing in cycles.
- Directional links with finite queues.
- Packet timing and utilization metrics.
- CSL-like color/runtime helpers used by Python examples.
- GEMV9-style streaming with PE-local tasks running concurrently in SimPy.

This is not a CSL compiler and does not import Cerebras SDK libraries. It is a
Python simulator that mirrors the execution structure so future workloads can be
added without requiring users to learn CSL first.
