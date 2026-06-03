from __future__ import annotations

from dataclasses import dataclass, field

from wafersim.config import CslGemvStreamingWorkloadConfig, MeshConfig
from wafersim.config import SimulationConfig
from wafersim.csl import Direction


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    """Python-native layout description for simulator examples.

    This mirrors the role of a hardware layout file without requiring users to
    learn or write a separate accelerator language.
    """

    mesh_rows: int
    mesh_cols: int
    kernel_rows: int
    kernel_cols: int
    routes: dict[str, Direction | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PeTaskSpec:
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class PeProgramSpec:
    """Python-native PE program description for simulator examples."""

    kernel: str
    tasks: tuple[PeTaskSpec, ...]


@dataclass(frozen=True, slots=True)
class GemvProblemSpec:
    matrix_rows: int
    matrix_cols: int
    bytes_per_element: int = 4
    packet_payload_bytes: int = 64
    ops_per_mac: int = 2


def build_gemv_streaming_config(
    *,
    layout: LayoutSpec,
    problem: GemvProblemSpec,
) -> SimulationConfig:
    return SimulationConfig(
        mesh=MeshConfig(rows=layout.mesh_rows, cols=layout.mesh_cols),
        workload=CslGemvStreamingWorkloadConfig(
            matrix_rows=problem.matrix_rows,
            matrix_cols=problem.matrix_cols,
            kernel_rows=layout.kernel_rows,
            kernel_cols=layout.kernel_cols,
            bytes_per_element=problem.bytes_per_element,
            packet_payload_bytes=problem.packet_payload_bytes,
            ops_per_mac=problem.ops_per_mac,
        ),
    )
