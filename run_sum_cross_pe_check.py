#!/usr/bin/env python3
"""Real single-primitive check for CROSS-PE `sum`: a real RMSNorm-style
reduction whose reduced dimension is SHARDED ACROSS multiple physical PEs
(not all living on one PE) -- the case ``SumSpec``/``simulate_sum`` does
NOT cover (see ``wafersim.primitives.sum.__init__``'s own module
docstring: that bridge is for one lane fully local to a single PE).

Real mechanism exercised here (see ``_cross_pe_sum.py``'s own module
docstring for the full derivation and citations):
  1. REAL DOCKER: one real ``simulate_sum()`` sub-run for the LOCAL
     per-PE reduction (elements_per_pe = total_elements / num_pes),
     representative of every one of the ``num_pes`` PEs since they all do
     identical local work.
  2. REAL DOCKER: one real ``simulate_add()`` sub-run for a single
     combine-add firing, representative of every one of the
     ``num_pes - 1`` real relay-chain combine steps.
  3. REAL SIMULATION (not Docker, but real cycle-accurate link/FIFO
     timing): ``num_pes - 1`` sequential hops along one mesh row via the
     EXACT SAME ``wafer.network.PacketNetwork._send_hop`` code GEMM's own
     cross-PE broadcast uses.
  4. OPENCAKE'S OWN ANALYTICAL COEFFICIENT (explicitly not Docker-
     derived, same as GEMM): network energy (fifo/router/wire pJ-per-
     byte-hop terms) and static leakage, plus the analytical inter-PE
     link area term.

Example below: hidden_size=4096 (a real, typical transformer hidden
dimension, e.g. RMSNorm's mean-of-squares reduction), tensor-parallel
across num_pes=4 PEs -- elements_per_pe=1024 per PE locally, then a real
3-hop sequential relay chain combines the 4 partial sums into one grand
total on PE(0,0).

No synthetic fallback anywhere: if either real Docker sub-run fails, this
raises and prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_sum_cross_pe_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec
from wafersim.primitives.sum import SumCrossPESpec, simulate_sum_cross_pe

OUT_DIR = Path("outputs/sum_cross_pe_check")


def main() -> None:
    # Real, typical transformer hidden_size, tensor-parallel across 4 PEs
    # -- e.g. RMSNorm's mean-of-squares reduction over the hidden dim
    # when a tensor-parallel layout has split that dim across the mesh.
    total_elements = 4096
    num_pes = 4

    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = SumCrossPESpec(total_elements=total_elements, num_pes=num_pes, bytes_per_element=4)
    spec.validate()
    print(
        f"SumCrossPESpec: total_elements={spec.total_elements}, num_pes={spec.num_pes}, "
        f"elements_per_pe={spec.elements_per_pe}, "
        f"real_combine_hop_count={spec.real_combine_hop_count}, "
        f"bytes_per_element={spec.bytes_per_element}"
    )

    # Real Docker (local reduce + one combine-add firing) + real network
    # simulation for the relay chain -- raises (never fabricates) on
    # failure of either real Docker sub-run.
    cost = simulate_sum_cross_pe(spec, wafer, OUT_DIR)

    real_time_ns = cost.sim_time_cycles / (wafer.clock_ghz * 1e9) * 1e9
    local_cycles = cost.local_result.local_compute_cycles
    combine_cycles = (
        0
        if cost.combine_add_result is None
        else cost.combine_add_result.local_compute_cycles * spec.real_combine_hop_count
    )
    network_only_cycles = cost.sim_time_cycles - local_cycles - combine_cycles

    print("\n=== REAL local per-PE reduction sub-run (Docker) ===")
    print(f"  elements_per_pe (K)={spec.elements_per_pe}  real_add_count (K-1)={cost.local_result.sum.real_add_count}")
    print(f"  local_compute_cycles (real, per PE): {local_cycles}")
    print(f"  local_energy_uj (real, per PE):      {cost.local_result.local_energy_uj}")

    if cost.combine_add_result is not None:
        print("\n=== REAL combine-add sub-run (Docker, one firing, reused x hop_count) ===")
        print(f"  hop_count (real_combine_hop_count) = {spec.real_combine_hop_count}")
        print(f"  local_compute_cycles (real, per combine-add firing): {cost.combine_add_result.local_compute_cycles}")
        print(f"  local_energy_uj (real, per combine-add firing):      {cost.combine_add_result.local_energy_uj}")

    print("\n=== REAL network relay-chain simulation (not Docker, real link/FIFO timing) ===")
    for k, v in cost.network_summary.items():
        print(f"  {k}: {v}")

    print("\n=== TOTAL cross-PE sum cost (real Docker + real network sim + OpenCake analytical energy/area) ===")
    print(f"sim_time_cycles (total, real):                  {cost.sim_time_cycles}")
    print(f"  breakdown -- local phase (real, parallel):    {local_cycles} cycles")
    print(f"  breakdown -- combine-add phase (real, chain): {combine_cycles} cycles")
    print(f"  breakdown -- network-only residual (real sim): {network_only_cycles} cycles")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):     {real_time_ns:.3f} ns")
    print(f"energy total (real Docker PE-internal + OpenCake analytical network/static): {cost.energy.total_uj} uJ")
    print(f"  pe_internal_uj (real Docker, local x{num_pes} + combine x{spec.real_combine_hop_count}): {cost.energy.pe_internal_uj}")
    print(f"  fifo_uj (OpenCake analytical):    {cost.energy.fifo_uj}")
    print(f"  router_uj (OpenCake analytical):  {cost.energy.router_uj}")
    print(f"  wire_uj (OpenCake analytical):    {cost.energy.wire_uj}")
    print(f"  static_uj (OpenCake analytical):  {cost.energy.static_uj}")
    print(f"cycles/element (real, total / {total_elements}): {cost.cycles_per_element}")
    print(f"energy/element (real, total / {total_elements}): {cost.energy_uj_per_element}")
    print(f"\narea -- pe_internal_area_um2 (real Docker, one PE): {cost.area.pe_internal_area_um2}")
    print(f"area -- interconnect_area_um2_per_pe (OpenCake analytical): {cost.area.interconnect_area_um2_per_pe}")
    print(f"area -- active_pes: {cost.area.active_pes}  mesh_pes: {cost.area.mesh_pes}")


if __name__ == "__main__":
    main()
