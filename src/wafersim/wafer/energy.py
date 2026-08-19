"""Transparent energy accounting for the wafer execution model.

Design (per the 2026-08 TimeLoop/OpenCake division-of-labor redesign):
intra-PE compute (MAC) + local scratchpad SRAM access energy is no longer a
made-up mac_energy_pj/sram_energy_pj_per_byte coefficient -- it is TimeLoop/
Accelergy's own real, technology-calibrated number (see
GemmMappingContract.local_energy_uj, parsed from a real timeloop-model run's
stats.txt "Summary Stats" section). TimeLoop's architecture.yaml declares
the PE array as a spatial subtree (`PE[0..active_pes-1]`), and TimeLoop's
own Energy accounting (unlike its Cycles critical-path number) is already
the aggregate across every one of those instances -- so this module does
NOT multiply it again by active_pes.

fifo_uj/router_uj/wire_uj/static_uj remain OpenCake's own analytical
domain: TimeLoop's `spatial` dimension has no concept of inter-PE distance,
routing, or queueing (see wafer/network.py's docstring), and it has no
notion of a wafer-wide static leakage budget either, so those stay
configurable via WaferSpec's own coefficients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..specs import WaferSpec


@dataclass(frozen=True, slots=True)
class EnergyBreakdown:
    pe_internal_uj: float
    fifo_uj: float
    router_uj: float
    wire_uj: float
    static_uj: float

    @property
    def total_uj(self) -> float:
        return sum(asdict(self).values())

    def to_dict(self) -> dict[str, float]:
        values = asdict(self)
        values["total_uj"] = self.total_uj
        return values


def estimate_energy(
    wafer: WaferSpec,
    *,
    active_pes: int,
    sim_time_cycles: float,
    pe_internal_energy_uj: float,
    network: dict[str, float | int],
) -> EnergyBreakdown:
    """Compute an explicit energy accounting: real TimeLoop PE-internal
    energy plus OpenCake's own analytical network/static terms.

    ``pe_internal_energy_uj`` should be ``GemmMappingContract.local_energy_uj``
    -- a real accelergy-derived number, already aggregated across all active
    PEs by TimeLoop's own spatial-instance accounting. Passing anything else
    (e.g. a placeholder 0.0, as this project's unit tests do for a synthetic
    contract) is valid but will simply report zero PE-internal energy.
    """
    static_pj = (
        wafer.static_power_mw_per_pe * active_pes * sim_time_cycles / wafer.clock_ghz
    )
    return EnergyBreakdown(
        pe_internal_uj=pe_internal_energy_uj,
        fifo_uj=float(network["fifo_enqueued_bytes"]) * wafer.fifo_energy_pj_per_byte / 1_000_000.0,
        router_uj=float(network["router_byte_hops"]) * wafer.router_energy_pj_per_byte_hop / 1_000_000.0,
        wire_uj=float(network["wire_byte_hops"]) * wafer.wire_energy_pj_per_byte_hop / 1_000_000.0,
        static_uj=static_pj / 1_000_000.0,
    )
