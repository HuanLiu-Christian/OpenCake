"""Transparent on-chip area accounting for the wafer execution model.

Mirrors wafer/energy.py's division of labor, updated for the FIFO/router-
in-PE redesign: PE-internal area is no longer just mac + pe_scratchpad --
FIFO and router are now ALSO real SRAM-class Accelergy components declared
inside the PE's own architecture.yaml subtree (fifo_buffer/router_buffer --
see timeloop.py's generate_timeloop_inputs), so their area comes from the
same real, technology-calibrated per-instance ART.yaml numbers as mac/
pe_scratchpad (see GemmMappingContract.pe_internal_area_um2, which now sums
all four). router_area_um2 is only an approximation -- a router is not
really a memory -- but it is a real Accelergy-computed number, not an
assumed coefficient (see specs.py's router_ports_per_pe comment for the
citation basis).

Only the wire LINK between two adjacent PEs' routers has no such path: it
is not owned by a single PE's architecture subtree, so there is nothing for
Accelergy to estimate per-PE. It remains OpenCake's own analytical
coefficient, wafer.link_area_um2_per_pe -- defaults to 0.0 (see specs.py)
because no sufficiently precise real per-PE-attributable wire-area citation
was found; a total computed with that default left at 0.0 is PE-internal
area only (mac + pe_scratchpad + fifo_buffer + router_buffer), still 100%
real.

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
    number for ONE PE's mac + pe_scratchpad + fifo_buffer + router_buffer.
    Passing anything else (e.g. a placeholder 0.0, as this project's unit
    tests do for a synthetic contract) is valid but will simply report zero
    PE-internal area. ``interconnect_area_um2_per_pe`` below is now ONLY the
    inter-PE link/wire term -- FIFO/router area is already folded into
    ``pe_internal_area_um2`` (see this module's docstring).
    """
    interconnect_area_um2_per_pe = wafer.link_area_um2_per_pe
    return AreaBreakdown(
        pe_internal_area_um2=pe_internal_area_um2,
        interconnect_area_um2_per_pe=interconnect_area_um2_per_pe,
        active_pes=active_pes,
        mesh_pes=wafer.mesh_rows * wafer.mesh_cols,
    )
