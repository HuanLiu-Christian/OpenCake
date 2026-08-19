from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

from .specs import GemmSpec, WaferSpec


@dataclass(frozen=True, slots=True)
class GemmMappingContract:
    """Normalized mapping data parsed from the TimeLoop artifact pair."""

    gemm: GemmSpec
    wafer: WaferSpec
    active_rows: int
    active_cols: int
    tile_m: int
    tile_n: int
    tile_k: int
    mapper_mode: str
    map_path: Path
    stats_path: Path
    local_compute_cycles: int
    local_energy_uj: float
    # Real, per-SINGLE-instance Accelergy area (um^2), parsed from the same
    # real Docker run's ART.yaml (see timeloop.py's _pe_internal_area_um2).
    # NOT pre-multiplied by active_pes -- unlike local_energy_uj, TimeLoop's
    # own ART output does not consistently aggregate area across spatial
    # instances (confirmed against a real run: the `mac` level's stats.txt
    # entry does report an instance-multiplied "Area (total)", but the
    # `pe_scratchpad` level's entry does not -- ART.yaml itself is the one
    # place both are reliably single-instance), so multiplication by PE
    # count is done explicitly downstream (see wafer/area.py).
    mac_area_um2: float
    pe_scratchpad_area_um2: float

    @property
    def active_pes(self) -> int:
        return self.active_rows * self.active_cols

    @property
    def pe_internal_area_um2(self) -> float:
        """Real Accelergy-derived area of ONE PE's mac + pe_scratchpad."""
        return self.mac_area_um2 + self.pe_scratchpad_area_um2

    @property
    def k_phases(self) -> int:
        return ceil(self.gemm.k / self.tile_k)

    @property
    def c_tile_bytes(self) -> int:
        return self.tile_m * self.tile_n * self.gemm.bytes_per_element

    @property
    def a_phase_bytes(self) -> int:
        return self.tile_m * self.tile_k * self.gemm.bytes_per_element

    @property
    def b_phase_bytes(self) -> int:
        return self.tile_k * self.tile_n * self.gemm.bytes_per_element

    @property
    def data_sram_bytes_per_pe(self) -> int:
        # Single-buffered A/B (see wafer/simulator.py's simulate_gemm -- real
        # Cerebras CSL GEMM task synchronization does not prefetch a phase
        # ahead, so there is no second A/B slot to budget for here). FIFO
        # state is modeled as fabric buffering and deliberately excluded
        # from this budget.
        return self.c_tile_bytes + self.a_phase_bytes + self.b_phase_bytes

    @property
    def data_sram_utilization(self) -> float:
        return self.data_sram_bytes_per_pe / self.wafer.sram_bytes_per_pe

    def validate(self) -> None:
        self.gemm.validate()
        self.wafer.validate()
        if min(self.active_rows, self.active_cols, self.tile_m, self.tile_n, self.tile_k) <= 0:
            raise ValueError("mapping factors must be positive")
        if self.active_rows > self.wafer.mesh_rows or self.active_cols > self.wafer.mesh_cols:
            raise ValueError("active PE grid exceeds wafer mesh")
        if self.data_sram_bytes_per_pe > self.wafer.sram_bytes_per_pe:
            raise ValueError("GEMM data exceeds PE-local SRAM")

    def to_dict(self) -> dict[str, object]:
        return {
            "gemm": asdict(self.gemm),
            "wafer": asdict(self.wafer),
            "active_rows": self.active_rows,
            "active_cols": self.active_cols,
            "tile": {"m": self.tile_m, "n": self.tile_n, "k": self.tile_k},
            "active_pes": self.active_pes,
            "k_phases": self.k_phases,
            "mapper_mode": self.mapper_mode,
            "map_path": str(self.map_path),
            "stats_path": str(self.stats_path),
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "mac_area_um2": self.mac_area_um2,
            "pe_scratchpad_area_um2": self.pe_scratchpad_area_um2,
            "pe_internal_area_um2": self.pe_internal_area_um2,
            "data_sram_bytes_per_pe": self.data_sram_bytes_per_pe,
            "data_sram_utilization": self.data_sram_utilization,
        }
