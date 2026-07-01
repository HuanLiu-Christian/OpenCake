# OpenCake Formal Pipeline

This project keeps two OpenCake paths side by side:

```text
Python reference path
  Chakra-like JSONL
    -> OpenCake Mapper
    -> Wafer-scale Spatial IR
    -> Python reference backend

Formal Astra-Sim handoff path
  Chakra-like JSONL
    -> OpenCake Mapper
    -> validated Wafer-scale Spatial IR
    -> Astra-Sim experiment package
```

The Python backend remains a reference model. The formal path emits files that
a future Astra-Sim C++ OpenCake wafer backend can consume.

## Directory Layout

```text
src/wafersim/opencake/
  chakra.py      simplified Chakra-like JSONL reader
  workload.py    GEMM workload and wafer hardware spec
  mapper.py      GEMM tiling, rank placement, colors, SRAM layout
  ir.py          Spatial IR operation records and JSONL writer
  schema.py      formal Spatial IR schema version and validation
  backend.py     Python reference timing backend
  simulator.py   Astra-Sim-style Python reference wrapper
  astrasim.py    formal Astra-Sim experiment package exporter
  pipeline.py    prototype reference pipeline
  formal.py      formal package-generation pipeline

examples/opencake_pipeline/
  chakra_gemm.jsonl   input workload trace
  run.py              prototype trace -> mapper -> IR -> Python simulator

examples/opencake_formal/
  run.py              trace -> mapper -> validated IR -> formal package
  run_cpp_backend.py  formal package -> C++ backend -> reference check

cpp/opencake_wafer_backend/
  Makefile
  src/opencake_wafer_backend.cpp
                      standalone C++ backend that reads the formal package

integrations/astra_sim_opencake/
  install.py          installs the backend into an official Astra-Sim tree
```

## Formal Package Files

Running the formal launcher creates:

```text
manifest.json
  package metadata, schema version, file list, workload and hardware summary

system.json
  rank model, rank grid, SRAM capacity, compute and SRAM bandwidth

network.yml
  OpenCake 2D mesh topology, link bandwidth, router latency, queue capacity

workload/opencake_spatial_ir.jsonl
  validated Spatial IR operations: define_color, alloc, send, compute

reference_result.json
  Python reference result used later to compare against the C++ backend
```

## Run

```bash
.venv/bin/python examples/opencake_formal/run.py \
  --out-dir /tmp/opencake_formal_package
```

Default model:

```text
GEMM = 1024 x 1024 x 1024
mesh = 32 x 32 PE
PE block = 8 x 8 PE
rank grid = 4 x 4 = 16 ranks
Spatial IR schema = opencake.spatial_ir.v1
Spatial IR ops = 90
```

Reference result:

```text
sim_time_cycles = 32768
compute_cycles = 32768
memory_cycles = 2304
network_cycles = 24579
SRAM usage = 75%
max link utilization = 75.0%
```

## Standalone C++ Backend

The standalone backend reads:

```text
manifest.json
system.json
network.yml
workload/opencake_spatial_ir.jsonl
```

and matches `reference_result.json` for the same input package.

```bash
.venv/bin/python examples/opencake_formal/run_cpp_backend.py \
  --out-dir /tmp/opencake_formal_package_cpp
```

That command:

```text
1. generates the formal OpenCake package
2. builds cpp/opencake_wafer_backend
3. runs the C++ backend on the package
4. checks the C++ result against reference_result.json
```

## Official Astra-Sim Integration

The integrated flow installs the OpenCake backend into an official Astra-Sim
source tree and builds it as an Astra-Sim CMake target:

```text
AstraSim_OpenCake_Wafer
```

Run with the MICRO 2024 tutorial Docker image:

```bash
.venv/bin/python examples/opencake_formal/run_astrasim_integrated.py \
  --use-docker \
  --out-dir /tmp/opencake_astrasim_integrated_package
```

That command:

```text
1. generates the formal OpenCake package
2. copies the backend into astra-sim/astra-sim/network_frontend/opencake/
3. patches astra-sim/astra-sim/network_frontend/analytical/CMakeLists.txt
4. builds Astra-Sim target AstraSim_OpenCake_Wafer
5. runs the Astra-Sim-built binary on the package
6. checks the result against reference_result.json
```

Verified integrated output:

```text
target = AstraSim_OpenCake_Wafer
compute_cycles = 32768
memory_cycles = 2304
network_cycles = 24579
sim_time_cycles = 32768
compute_ops = 2147483648
SRAM usage = 75%
```
