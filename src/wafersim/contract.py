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
    # Real, per-single-instance Accelergy area (um^2) for the FIFO/router
    # buffers now declared as real SRAM-class components inside the PE's
    # own architecture.yaml subtree (fifo_buffer/router_buffer -- see
    # timeloop.py's generate_timeloop_inputs and _pe_internal_area_um2).
    # Parsed the same real, ART.yaml-derived way as mac_area_um2/
    # pe_scratchpad_area_um2 above -- router_area_um2 is only an
    # approximation (a router is not really a memory; see specs.py's
    # router_ports_per_pe/router_port_bytes comment for the citation basis
    # of that approximation), but fifo_area_um2 is a real SRAM area number
    # for a component that genuinely is a buffer.
    fifo_area_um2: float
    router_area_um2: float

    @property
    def active_pes(self) -> int:
        return self.active_rows * self.active_cols

    @property
    def pe_internal_area_um2(self) -> float:
        """Real Accelergy-derived area of ONE PE's mac + pe_scratchpad +
        fifo_buffer + router_buffer -- everything physically inside the PE.
        """
        return (
            self.mac_area_um2
            + self.pe_scratchpad_area_um2
            + self.fifo_area_um2
            + self.router_area_um2
        )

    # ---- compute / memory / network area classification -----------------
    # A non-overlapping re-bucketing of the same five area terms above (mac,
    # pe_scratchpad, fifo_buffer, router_buffer, link) into three functional
    # categories, instead of the "which Accelergy component estimated it"
    # split used by pe_internal_area_um2/interconnect_area_um2_per_pe. Each
    # of the five terms is assigned to EXACTLY one category below, so
    # compute + memory + network == total_area_um2 with no double-counting:
    #   - compute: mac only (the real Accelergy intmac area).
    #   - memory:  pe_scratchpad only (the real Accelergy SRAM that holds
    #     the GEMM A/B/C tiles -- NOT fifo_buffer/router_buffer, even though
    #     those are also Accelergy SRAM-class components: functionally they
    #     hold in-flight network packets, not GEMM data, so they belong in
    #     "network" below).
    #   - network: fifo_buffer + router_buffer (real Accelergy, SRAM-shaped
    #     approximations -- see specs.py's router_ports_per_pe comment) plus
    #     the inter-PE link/wire (wafer.link_area_um2_per_pe -- OpenCake's
    #     own reasoned estimate, not Accelergy-derived; see specs.py's
    #     link_area_um2_per_pe comment for its derivation and citations).
    @property
    def compute_area_um2(self) -> float:
        return self.mac_area_um2

    @property
    def memory_area_um2(self) -> float:
        return self.pe_scratchpad_area_um2

    @property
    def network_area_um2(self) -> float:
        return self.fifo_area_um2 + self.router_area_um2 + self.wafer.link_area_um2_per_pe

    @property
    def total_area_um2(self) -> float:
        """compute_area_um2 + memory_area_um2 + network_area_um2, for ONE
        PE -- equals pe_internal_area_um2 + wafer.link_area_um2_per_pe
        (the same five terms, just re-bucketed; link is the only term not
        already inside pe_internal_area_um2)."""
        return self.compute_area_um2 + self.memory_area_um2 + self.network_area_um2

    @property
    def compute_area_fraction(self) -> float:
        return self.compute_area_um2 / self.total_area_um2

    @property
    def memory_area_fraction(self) -> float:
        return self.memory_area_um2 / self.total_area_um2

    @property
    def network_area_fraction(self) -> float:
        return self.network_area_um2 / self.total_area_um2

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
        # and router storage now really occupy part of the PE's declared
        # SRAM budget (wafer.fifo_router_bytes_per_pe -- see specs.py), so
        # the GEMM-data-only figure below is added to that reserved amount
        # to report what's actually consumed out of wafer.sram_bytes_per_pe.
        return (
            self.c_tile_bytes
            + self.a_phase_bytes
            + self.b_phase_bytes
            + self.wafer.fifo_router_bytes_per_pe
        )

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
            "fifo_area_um2": self.fifo_area_um2,
            "router_area_um2": self.router_area_um2,
            "pe_internal_area_um2": self.pe_internal_area_um2,
            "data_sram_bytes_per_pe": self.data_sram_bytes_per_pe,
            "data_sram_utilization": self.data_sram_utilization,
            # Non-overlapping compute/memory/network re-bucketing of the
            # same five area terms above -- see the property comments
            # above for exactly which term goes in which category. The
            # three *_area_fraction values always sum to 1.0.
            "area_breakdown": {
                "compute_area_um2": self.compute_area_um2,
                "memory_area_um2": self.memory_area_um2,
                "network_area_um2": self.network_area_um2,
                "total_area_um2": self.total_area_um2,
                "compute_area_fraction": self.compute_area_fraction,
                "memory_area_fraction": self.memory_area_fraction,
                "network_area_fraction": self.network_area_fraction,
            },
        }
