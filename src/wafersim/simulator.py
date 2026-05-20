from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import simpy

from wafersim.config import CslDemoWorkloadConfig, GemmWorkloadConfig
from wafersim.config import MemoryCopyWorkloadConfig
from wafersim.config import MeshConfig, SimulationConfig
from wafersim.config import StencilWorkloadConfig
from wafersim.network import PacketRecord, WaferMesh
from wafersim.workloads import CslDemoWorkload, GemmWorkload, MemoryCopyWorkload
from wafersim.workloads import StencilWorkload


def run_simulation(config: SimulationConfig) -> dict[str, Any]:
    config.validate()
    env = simpy.Environment()
    mesh = WaferMesh(env=env, config=config.mesh)
    workload = make_workload(config.workload)

    workload_process = env.process(workload.run(env, mesh))
    env.run(until=workload_process)

    elapsed = env.now
    summary = {
        "sim_time_cycles": elapsed,
        "mesh": {
            "rows": config.mesh.rows,
            "cols": config.mesh.cols,
            "pes": config.mesh.rows * config.mesh.cols,
        },
        "workload": workload.summary(mesh),
        **mesh.pe_summary(elapsed),
        **mesh.packet_summary(),
        **mesh.link_summary(elapsed),
        "_packet_records": mesh.packet_records,
    }
    if isinstance(workload, MemoryCopyWorkload | GemmWorkload):
        summary.update(workload.result_summary(mesh, elapsed))
    return summary


def load_config(path: Path) -> SimulationConfig:
    data = json.loads(path.read_text())
    mesh_data = data.get("mesh", {})
    workload_data = data.get("workload", {})
    workload_type = normalize_workload_type(workload_data.get("type", "stencil"))
    workload_data = {**workload_data, "type": workload_type}
    if workload_type == "csl_demo":
        workload = CslDemoWorkloadConfig(**workload_data)
    elif workload_type == "gemm":
        workload = GemmWorkloadConfig(**workload_data)
    elif workload_type in {"memory_copy", "array_stream"}:
        workload_data["type"] = "memory_copy"
        workload = MemoryCopyWorkloadConfig(**workload_data)
    elif workload_type == "stencil":
        workload = StencilWorkloadConfig(**workload_data)
    else:
        raise ValueError(f"unknown workload type: {workload_type}")
    return SimulationConfig(
        mesh=MeshConfig(**mesh_data),
        workload=workload,
    )


def normalize_workload_type(workload_type: str) -> str:
    return workload_type.replace("-", "_")


def make_workload(
    config: (
        StencilWorkloadConfig
        | MemoryCopyWorkloadConfig
        | GemmWorkloadConfig
        | CslDemoWorkloadConfig
    ),
) -> StencilWorkload | MemoryCopyWorkload | GemmWorkload | CslDemoWorkload:
    if isinstance(config, CslDemoWorkloadConfig):
        return CslDemoWorkload(config)
    if isinstance(config, GemmWorkloadConfig):
        return GemmWorkload(config)
    if isinstance(config, MemoryCopyWorkloadConfig):
        return MemoryCopyWorkload(config)
    return StencilWorkload(config)


def write_packet_trace(path: Path, records: list[PacketRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as trace_file:
        for record in records:
            trace_file.write(
                json.dumps(
                    {
                        "packet_id": record.packet_id,
                        "src": record.src,
                        "dst": record.dst,
                        "size_bytes": record.size_bytes,
                        "kind": record.kind,
                        "created_at": record.created_at,
                        "delivered_at": record.delivered_at,
                        "latency": record.latency,
                        "hops": record.hops,
                    },
                    separators=(",", ":"),
                )
            )
            trace_file.write("\n")


def strip_internal_records(summary: dict[str, Any]) -> dict[str, Any]:
    public_summary = dict(summary)
    public_summary.pop("_packet_records", None)
    return public_summary
