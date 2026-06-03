# WaferSim

`wafersim` is a SimPy-based starting point for a medium-scale, Cerebras-like
wafer-scale accelerator simulator. It models a parameterized 2D PE mesh, local
PE memory, directional links with finite queues, deterministic XY routing, and
a memory-copy workload from the top-left PE to the bottom-right PE.

The goal is to get architecture feedback before writing RTL, then reuse the
same packet contracts and traces when moving toward Chisel.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python -m wafersim.cli --config configs/memory_copy.json
```

Write a packet trace for future analysis:

```bash
python -m wafersim.cli --config configs/memory_copy.json --trace-jsonl traces/memory_copy.jsonl
```

Run the memory copy from the top-left PE to the bottom-right PE:

```bash
python -m wafersim.cli --config configs/memory_copy.json
```

Run the tiled GEMM workload:

```bash
python -m wafersim.cli --config configs/gemm.json
```

Run the CSL-compatibility GEMM workload:

```bash
python -m wafersim.cli --workload csl-gemm --gemm-m 1024 --gemm-n 1024 --gemm-k 1024
```

Run the CSL GEMV 9 color-streaming pipeline:

```bash
python -m wafersim.cli --config configs/csl_gemv_streaming.json
```

Run the same GEMV 9 model from a Python example directory that mirrors the SDK
layout / PE program / host launcher flow:

```bash
python examples/gemv_09_streaming/run.py
python examples/gemv_09_streaming/run.py --matrix-rows 1024 --matrix-cols 1024 --kernel-rows 32 --kernel-cols 32
```

Run the CSL-topic concept demo:

```bash
python -m wafersim.cli --config configs/csl_topics.json
```

Run the later SDK examples concept demo:

```bash
python -m wafersim.cli --config configs/sdk_examples.json
```

Run the CSL builtins/libraries compatibility demo:

```bash
python -m wafersim.cli --config configs/csl_runtime.json
```

Run tests:

```bash
python -m unittest discover -s tests
```

## What The Default Model Means

The medium config uses:

- `32x32` PEs, so `1024` total processing elements.
- `48 KiB` local SRAM per PE with `128 bytes/cycle` local memory bandwidth.
- A `32 KiB` array copied from `PE(0,0)` to `PE(31,31)`.
- Source and destination local memory accesses are modeled.
- Directional nearest-neighbor mesh links.
- Store-and-forward packet timing with finite per-link queues.

This is not intended to reproduce proprietary Cerebras WSE details. It is a
clean open model that keeps the same class of architectural questions visible:
PE granularity, local compute/communication balance, link pressure, routing, and
eventual hardware interfaces.

## CSL Topic Coverage

`configs/csl_topics.json` runs concept-level models for the CSL examples shown
in the public Cerebras SDK tutorial list:

- memcpy-style streaming
- arrays and pointers
- library-like compute kernels
- streaming wavelet data
- sparse tensor wavelet filtering
- sentinels
- switches and control entrypoints
- filters
- FIFOs
- `@map`-style coordinate mapping
- collective broadcast/reduce traffic
- debug and simprint logs
- color swap
- WSE-3-style microthread concurrency
- output queue reuse

These are simulator abstractions, not compiled CSL programs. They are useful for
studying data movement, memory pressure, FIFO backpressure, and PE activity
before deciding whether to target CSL, Chisel RTL, or another hardware path.

## SDK Example Coverage

`configs/sdk_examples.json` runs concept-level models for later Cerebras SDK
examples:

- pipeline redirect, H2D FIFO, and artificial halo
- SDK layout routing, ports, and H2D/D2H streaming
- GEMV, checkerboard GEMV, collective GEMV, and collective GEMM
- residual, Cholesky, 25-point stencil, bandwidth test, and 3D 7-point SpMV
- power method, CG, PCG, BiCGSTAB, and single-tile matvec
- host-to-device broadcast, 3D FFT, and Conway's Game of Life

These examples are implemented with Python and SimPy only. Cerebras SDK
libraries such as `<memcpy>` or `<collectives_2d>` are represented by simulator
events instead of being imported or compiled.

## CSL Compatibility Layer

`src/wafersim/csl/compat.py` provides Python/SimPy mappings for common CSL
builtins and SDK libraries:

- builtins: rectangle layout, `@map`-style IDs, zero/constant buffers, pointer
  load/store, `sizeof`, bitcast, and module import names
- libraries: math, random, time, debug, simprint, layout, malloc, memcpy,
  collectives_2d, and tile_config-style FIFO/filter/switch helpers

`configs/csl_runtime.json` runs an integrated smoke workload through this
compatibility layer. This keeps application-level workloads from depending on
handwritten packet calls and gives a cleaner path toward later CSL or Chisel
mapping.

See `docs/csl_compat_mapping.md` for the implemented, partially implemented,
and intentionally omitted CSL builtin/library mappings.

## Python Example Layout

The simulator also supports a Cerebras SDK-like example structure without
requiring users to write or compile CSL:

```text
examples/gemv_09_streaming/
  layout.py        mesh rectangle, color names, and route directions
  pe_program.py    PE-local GEMV task declarations
  run.py           host-side launcher that converts Python specs into a
                   SimPy SimulationConfig
```

This path preserves the SDK example flow while keeping the simulator Python-only.
Application developers can define layout and PE tasks with dataclasses instead
of learning CSL syntax.

## Current Outputs

The CLI prints:

- simulated cycles
- PE count, compute utilization, memory utilization, and active utilization
- local SRAM footprint and memory capacity utilization
- packet count, bytes, mean/p95/max packet latency
- link count, max link utilization, and the busiest links

## Project Layout

```text
src/wafersim/
  config.py            simulator and workload parameters
  program.py           Python-native layout, PE program, and problem specs
  hardware/
    pe.py              processing element timing model
    network.py         mesh, links, routing, packet metrics
  csl/
    compat.py          CSL-like builtins, libraries, colors, and runtime
  workloads/
    __init__.py        memory-copy, GEMM, GEMV streaming, stencil, and demos
    kernels.py         pure GEMV/GEMM-style numerical kernels
  simulator.py         orchestration and trace export
  cli.py               command-line entry point
configs/
  memory_copy.json
  gemm.json
  csl_gemm.json
  csl_gemv_streaming.json
  csl_topics.json
  sdk_examples.json
  csl_runtime.json
  medium_stencil.json
docs/
  hardware_mapping.md
examples/
  gemv_09_streaming/
    layout.py
    pe_program.py
    run.py
tests/
  test_simulator.py
```

## Chisel Path

See `docs/hardware_mapping.md` for the mapping plan. The important early
contract is the packet shape plus ready/valid or credit-based link behavior.
Once those are stable, the SimPy packet traces can become Chisel test vectors.
