from __future__ import annotations

from typing import Protocol

from wafersim.opencake.mapper.tiler import GemmTilePlan
from wafersim.opencake.workload import GemmWorkload, HardwareSpec


class GemmStreamingStrategyContract(Protocol):
    """Target-independent contract consumed by AIR and Spatial-IR lowering.

    The current producer is the Stream/TETRA full-mesh search. Compiler stages
    consume only its selected tile, directions, colors, objective, and active
    mesh, keeping strategy selection separate from materialization.
    """

    schema: str
    source: str
    solver: str
    target_mesh: tuple[int, int]
    tile_m: int
    tile_n: int
    tile_k: int
    ab_buffer_count: int
    a_color: str
    b_color: str
    a_direction: str
    b_direction: str
    objective_cycles: float
    optimality_gap: float

    @property
    def tile_sizes(self) -> tuple[int, int, int]: ...

    def tile_plan(self, workload: GemmWorkload, hardware: HardwareSpec) -> GemmTilePlan: ...

    def validate_target_mesh(self, rows: int, cols: int) -> None: ...

    def summary(self) -> dict[str, object]: ...
