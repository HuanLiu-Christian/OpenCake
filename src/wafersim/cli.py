from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from wafersim.config import GemmWorkloadConfig, MemoryCopyWorkloadConfig
from wafersim.config import SimulationConfig
from wafersim.config import StencilWorkloadConfig
from wafersim.simulator import load_config, run_simulation, strip_internal_records
from wafersim.simulator import write_packet_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a SimPy wafer-scale accelerator simulation."
    )
    parser.add_argument("--config", type=Path, help="JSON config file")
    parser.add_argument(
        "--workload",
        choices=["stencil", "memory-copy", "array-stream", "gemm"],
        help="workload type to run",
    )
    parser.add_argument("--rows", type=int, help="mesh rows")
    parser.add_argument("--cols", type=int, help="mesh columns")
    parser.add_argument("--steps", type=int, help="stencil time steps")
    parser.add_argument("--tile-rows", type=int, help="rows per PE tile")
    parser.add_argument("--tile-cols", type=int, help="cols per PE tile")
    parser.add_argument("--ops-per-cell", type=int, help="compute ops per tile cell")
    parser.add_argument("--halo-bytes", type=int, help="halo packet bytes per edge")
    parser.add_argument(
        "--link-bandwidth",
        type=float,
        help="directional link bandwidth in bytes/cycle",
    )
    parser.add_argument(
        "--router-latency",
        type=float,
        help="per-hop router latency in cycles",
    )
    parser.add_argument(
        "--link-queue-capacity",
        type=int,
        help="per-direction link queue depth",
    )
    parser.add_argument(
        "--compute-throughput",
        type=float,
        help="PE compute throughput in ops/cycle",
    )
    parser.add_argument(
        "--local-sram-bytes",
        type=int,
        help="local SRAM capacity per PE in bytes",
    )
    parser.add_argument(
        "--memory-bandwidth",
        type=float,
        help="local memory bandwidth per PE in bytes/cycle",
    )
    parser.add_argument("--bytes-per-cell", type=int, help="stencil bytes per cell")
    parser.add_argument(
        "--memory-reads-per-cell",
        type=int,
        help="stencil memory reads per cell",
    )
    parser.add_argument(
        "--memory-writes-per-cell",
        type=int,
        help="stencil memory writes per cell",
    )
    parser.add_argument(
        "--single-buffer-tiles",
        action="store_true",
        help="store one tile buffer instead of old/new double buffers",
    )
    parser.add_argument("--gemm-m", type=int, help="GEMM M dimension")
    parser.add_argument("--gemm-n", type=int, help="GEMM N dimension")
    parser.add_argument("--gemm-k", type=int, help="GEMM K dimension")
    parser.add_argument("--tile-m", type=int, help="GEMM tile M per PE")
    parser.add_argument("--tile-n", type=int, help="GEMM tile N per PE")
    parser.add_argument("--tile-k", type=int, help="GEMM K tile")
    parser.add_argument("--ops-per-mac", type=int, help="ops counted per MAC")
    parser.add_argument(
        "--single-buffer-inputs",
        action="store_true",
        help="store one GEMM A/B input buffer instead of double buffering",
    )
    parser.add_argument(
        "--no-gemm-pipeline",
        action="store_true",
        help="disable GEMM overlap of communication, memory, and compute",
    )
    parser.add_argument("--array-elements", type=int, help="memory copy array elements")
    parser.add_argument(
        "--bytes-per-element",
        type=int,
        help="memory copy bytes per element",
    )
    parser.add_argument(
        "--packet-payload-bytes",
        type=int,
        help="memory copy packet payload bytes",
    )
    parser.add_argument("--source-row", type=int, help="memory copy source row")
    parser.add_argument("--source-col", type=int, help="memory copy source column")
    parser.add_argument("--dest-row", type=int, help="memory copy destination row")
    parser.add_argument("--dest-col", type=int, help="memory copy destination column")
    parser.add_argument(
        "--injection-gap-cycles",
        type=float,
        help="cycles between memory copy packet injections",
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        help="optional packet trace output path for later Chisel testbenches",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else SimulationConfig()
    config = apply_overrides(config, args)

    summary = run_simulation(config)
    if args.trace_jsonl:
        write_packet_trace(args.trace_jsonl, summary["_packet_records"])

    print(json.dumps(strip_internal_records(summary), indent=2, sort_keys=True))


def apply_overrides(config: SimulationConfig, args: argparse.Namespace) -> SimulationConfig:
    mesh = config.mesh
    workload = select_workload_config(config.workload, args.workload)

    mesh_updates = {
        "rows": args.rows,
        "cols": args.cols,
        "link_bandwidth_bytes_per_cycle": args.link_bandwidth,
        "router_latency_cycles": args.router_latency,
        "link_queue_capacity": args.link_queue_capacity,
        "compute_throughput_ops_per_cycle": args.compute_throughput,
        "local_sram_bytes_per_pe": args.local_sram_bytes,
        "memory_bandwidth_bytes_per_cycle": args.memory_bandwidth,
    }
    mesh = replace(
        mesh,
        **{key: value for key, value in mesh_updates.items() if value is not None},
    )
    workload = apply_workload_overrides(workload, args)
    return SimulationConfig(mesh=mesh, workload=workload)


def select_workload_config(
    workload: StencilWorkloadConfig | MemoryCopyWorkloadConfig | GemmWorkloadConfig,
    workload_type: str | None,
) -> StencilWorkloadConfig | MemoryCopyWorkloadConfig | GemmWorkloadConfig:
    if workload_type is None:
        return workload
    if workload_type == "gemm":
        return workload if isinstance(workload, GemmWorkloadConfig) else GemmWorkloadConfig()
    if workload_type in {"memory-copy", "array-stream"}:
        return (
            workload
            if isinstance(workload, MemoryCopyWorkloadConfig)
            else MemoryCopyWorkloadConfig()
        )
    return workload if isinstance(workload, StencilWorkloadConfig) else StencilWorkloadConfig()


def apply_workload_overrides(
    workload: StencilWorkloadConfig | MemoryCopyWorkloadConfig | GemmWorkloadConfig,
    args: argparse.Namespace,
) -> StencilWorkloadConfig | MemoryCopyWorkloadConfig | GemmWorkloadConfig:
    if isinstance(workload, GemmWorkloadConfig):
        updates = {
            "m": args.gemm_m,
            "n": args.gemm_n,
            "k": args.gemm_k,
            "tile_m": args.tile_m,
            "tile_n": args.tile_n,
            "tile_k": args.tile_k,
            "bytes_per_element": args.bytes_per_element,
            "ops_per_mac": args.ops_per_mac,
        }
        if args.single_buffer_inputs:
            updates["double_buffer_inputs"] = False
            updates["pipeline_k_tiles"] = False
        if args.no_gemm_pipeline:
            updates["pipeline_k_tiles"] = False
    elif isinstance(workload, MemoryCopyWorkloadConfig):
        updates = {
            "elements": args.array_elements,
            "bytes_per_element": args.bytes_per_element,
            "packet_payload_bytes": args.packet_payload_bytes,
            "source_row": args.source_row,
            "source_col": args.source_col,
            "dest_row": args.dest_row,
            "dest_col": args.dest_col,
            "injection_gap_cycles": args.injection_gap_cycles,
        }
    else:
        updates = {
            "steps": args.steps,
            "tile_rows": args.tile_rows,
            "tile_cols": args.tile_cols,
            "ops_per_cell": args.ops_per_cell,
            "halo_bytes_per_edge": args.halo_bytes,
            "bytes_per_cell": args.bytes_per_cell,
            "memory_reads_per_cell": args.memory_reads_per_cell,
            "memory_writes_per_cell": args.memory_writes_per_cell,
        }
        if args.single_buffer_tiles:
            updates["double_buffer_tiles"] = False
    return replace(
        workload,
        **{key: value for key, value in updates.items() if value is not None},
    )


if __name__ == "__main__":
    main()
