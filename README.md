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

Run the CSL-topic concept demo:

```bash
python -m wafersim.cli --config configs/csl_topics.json
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
  config.py      simulator parameters
  pe.py          processing element timing model
  network.py     mesh, links, routing, packet metrics
  workloads.py   memory-copy, GEMM, stencil, and CSL-topic workload generators
  simulator.py   orchestration and trace export
  cli.py         command-line entry point
configs/
  memory_copy.json
  gemm.json
  csl_topics.json
  medium_stencil.json
docs/
  hardware_mapping.md
tests/
  test_simulator.py
```

## Chisel Path

See `docs/hardware_mapping.md` for the mapping plan. The important early
contract is the packet shape plus ready/valid or credit-based link behavior.
Once those are stable, the SimPy packet traces can become Chisel test vectors.
