"""Transparent on-chip area accounting for the wafer execution model.

Mirrors wafer/energy.py's division of labor: PE-internal (mac +
pe_scratchpad) area is no longer a made-up coefficient -- it is TimeLoop/
Accelergy's own real, technology-calibrated per-instance number (see
GemmMappingContract.pe_internal_area_um2, parsed from a real timeloop run's
ART.yaml). Router/link/FIFO interconnect area remains OpenCake's own
analytical domain, via WaferSpec's *_area_um2_per_pe / fifo_area_um2_per_byte
coefficients -- these default to 0.0 (see specs.py) because, as of this
writing, no installed Accelergy plug-in (Aladdin/Cacti/mcpat/library) has a
confirmed calibrated area estimator for NoC routers, links, or FIFOs, so
there is no real number to report by default; a total computed with the
defaults left at 0.0 is PE-internal area only, still 100% real.

Two PE counts are reported side by side, not one blended number:
  - active_pes: PEs actually participating in THIS GEMM run (contract.
    active_pes) -- "how much silicon did this problem actually use".
  - mesh_pes: every PE physically present on the wafer (wafer.mesh_rows *
    wafer.mesh_cols) -- a chip's die area is fixed at fabrication time and
    does not shrink because a given problem leaves some PEs idle, so this is
    the number that answers "how big is this chip".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..specs import WaferSpec

# Every PE has one independent send + one independent receive FIFO per
# direction (east/west/south/north) -- see wafer/network.py's
# PacketNetwork.__init__ and its module docstring (mirrors Cerebras CSL's
# per-direction fabric queues). 4 directions x 2 (send + receive) = 8 FIFOs.
_FIFOS_PER_PE = 8


@dataclass(frozen=True, slots=True)
class AreaBreakdown:
    pe_internal_area_um2: float
    interconnect_area_um2_per_pe: float
    active_pes: int
    mesh_pes: int

    @property
    def area_um2_per_pe(self) -> float:
        return self.pe_internal_area_um2 + self.interconnect_area_um2_per_pe

    @property
    def active_pe_array_area_um2(self) -> float:
        """Total area of just the PEs this GEMM run actually activated."""
        return self.area_um2_per_pe * self.active_pes

    @property
    def die_pe_array_area_um2(self) -> float:
        """Total area of every PE physically present on the wafer mesh."""
        return self.area_um2_per_pe * self.mesh_pes

    def to_dict(self) -> dict[str, float]:
        values = asdict(self)
        values["area_um2_per_pe"] = self.area_um2_per_pe
        values["active_pe_array_area_um2"] = self.active_pe_array_area_um2
        values["die_pe_array_area_um2"] = self.die_pe_array_area_um2
        return values


def estimate_area(
    wafer: WaferSpec,
    *,
    active_pes: int,
    pe_internal_area_um2: float,
) -> AreaBreakdown:
    """Compute an explicit area accounting: real TimeLoop PE-internal area
    plus OpenCake's own analytical interconnect area, reported per-PE and
    rolled up over both the active PE count and the full physical mesh.

    ``pe_internal_area_um2`` should be
    ``GemmMappingContract.pe_internal_area_um2`` -- a real accelergy-derived
    number for ONE PE's mac + pe_scratchpad. Passing anything else (e.g. a
    placeholder 0.0, as this project's unit tests do for a synthetic
    contract) is valid but will simply report zero PE-internal area.
    """
    fifo_bytes_per_pe = (
        _FIFOS_PER_PE * wafer.fifo_capacity_packets * wafer.packet_payload_bytes
    )
    interconnect_area_um2_per_pe = (
        wafer.router_area_um2_per_pe
        + wafer.link_area_um2_per_pe
        + fifo_bytes_per_pe * wafer.fifo_area_um2_per_byte
    )
    return AreaBreakdown(
        pe_internal_area_um2=pe_internal_area_um2,
        interconnect_area_um2_per_pe=interconnect_area_um2_per_pe,
        active_pes=active_pes,
        mesh_pes=wafer.mesh_rows * wafer.mesh_cols,
    )
