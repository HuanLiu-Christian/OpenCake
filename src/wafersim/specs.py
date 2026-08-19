from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GemmSpec:
    """Problem C[M, N] += A[M, K] * B[K, N]."""

    m: int
    n: int
    k: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.m, self.n, self.k, self.bytes_per_element) <= 0:
            raise ValueError("GEMM dimensions and element size must be positive")


@dataclass(frozen=True, slots=True)
class WaferSpec:
    """The subset of wafer resources needed by this performance model."""

    mesh_rows: int = 32
    mesh_cols: int = 32
    sram_kib_per_pe: int = 48
    link_bandwidth_bytes_per_cycle: int = 32
    router_latency_cycles: int = 1
    packet_payload_bytes: int = 256
    fifo_capacity_packets: int = 8
    receive_dequeue_latency_cycles: int = 1
    clock_ghz: float = 1.0

    # Process node passed to the real TimeLoop/Accelergy architecture model
    # (compound components require an explicit `technology` attribute).
    technology_node: str = "45nm"

    # There used to be a `macs_per_cycle_per_pe: int = 32` config default
    # here, used to convert a MAC count into a PE-internal cycle count. It
    # has been REMOVED: that number was never derived from anything real --
    # it silently stood in for TimeLoop's own real, Accelergy-timed Cycles
    # result (GemmMappingContract.local_compute_cycles), and in fact
    # disagreed with it by 32x (this project's real architecture.yaml
    # declares exactly one scalar `mac` component per PE, i.e. a real
    # 1-cycle-per-MAC rate, confirmed against a real Docker run -- not a
    # 32-MAC/cycle vector engine). wafer/simulator.py's simulate_gemm now
    # derives the real cycles/MAC rate directly from
    # GemmMappingContract.local_compute_cycles instead, the same real
    # per-mapping Cycles number a real Docker run reports -- see that
    # module for the derivation. PE-internal compute (MAC) + local
    # scratchpad SRAM access energy, and PE-internal (mac + pe_scratchpad)
    # area, are likewise no longer configurable coefficients here -- they
    # come from TimeLoop/Accelergy's own real, technology-calibrated
    # GemmMappingContract (local_energy_uj / pe_internal_area_um2; see
    # wafer/energy.py and wafer/area.py). Only the network/static terms
    # below remain simple configurable coefficients, since TimeLoop has no
    # concept of inter-PE network cost or wafer-wide static leakage.
    fifo_energy_pj_per_byte: float = 0.005
    router_energy_pj_per_byte_hop: float = 0.02
    wire_energy_pj_per_byte_hop: float = 0.01
    static_power_mw_per_pe: float = 0.005

    # Router/link/FIFO interconnect area stays a configurable OpenCake
    # coefficient, same reasoning as the network energy terms above: real
    # Accelergy has no confirmed calibrated area estimator plug-in for
    # NoC routers/links/FIFOs (checked Aladdin/Cacti/mcpat/library plug-ins;
    # none declares support for those primitive classes as of this writing).
    # These default to 0.0 -- NOT an arbitrary nonzero guess like the energy
    # coefficients above -- so the interconnect term contributes exactly
    # nothing to a reported total area until you explicitly supply a real or
    # cited value (e.g. from a DSENT/Orion NoC area model, or a published
    # paper at your technology node). router/link are flat per-PE area
    # (a router+its links don't get bigger just because FIFOs are deeper);
    # fifo_area_um2_per_byte multiplies against each PE's real *configured*
    # FIFO byte capacity (fifo_capacity_packets * packet_payload_bytes,
    # x8 for the 4-direction x send/receive queues -- see network.py), so at
    # least that term tracks real config rather than being a flat guess too.
    router_area_um2_per_pe: float = 0.0
    link_area_um2_per_pe: float = 0.0
    fifo_area_um2_per_byte: float = 0.0

    @property
    def sram_bytes_per_pe(self) -> int:
        return self.sram_kib_per_pe * 1024

    def validate(self) -> None:
        integer_fields = (
            self.mesh_rows,
            self.mesh_cols,
            self.sram_kib_per_pe,
            self.link_bandwidth_bytes_per_cycle,
            self.router_latency_cycles,
            self.packet_payload_bytes,
            self.fifo_capacity_packets,
            self.receive_dequeue_latency_cycles,
        )
        if min(integer_fields) <= 0:
            raise ValueError("wafer integer resources must be positive")
        if self.clock_ghz <= 0:
            raise ValueError("clock_ghz must be positive")
        area_coefficients = (
            self.router_area_um2_per_pe,
            self.link_area_um2_per_pe,
            self.fifo_area_um2_per_byte,
        )
        if min(area_coefficients) < 0:
            raise ValueError("interconnect area coefficients cannot be negative")
