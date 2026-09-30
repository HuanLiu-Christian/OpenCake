#!/usr/bin/env python3
"""Real single-primitive check for CROSS-PE `concat`/`stack`: assembling
one real output tensor whose real input shards live on DIFFERENT physical
PEs -- the case ``ConcatStackSpec``/``simulate_concat_stack`` does NOT
cover (that one is for every input already on one PE).

Real mechanism exercised here (see ``_cross_pe_concat_stack.py``'s own
module docstring for the full derivation and citations -- Cerebras's
official ``collectives_2d.gather()`` primitive as the closest documented
hardware mechanism for this "many real per-PE buffers -> one combined
buffer" shape, and WaferLLM's (arXiv 2502.04563) own "shift-based
redistribution" phrase for real KV-cache-concat-style cross-PE data
movement):
  1. REAL DOCKER: one real, single-shard-sized memory-move sub-run
     (``simulate_memory_move`` -- the exact same real ``bitwise``
     identity-pass-through component local `concat`/`stack` already
     uses), representative of every one of the ``num_pes - 1`` real
     per-hop appends along the shift-relay chain.
  2. REAL SIMULATION (not Docker, but real cycle-accurate link/FIFO/
     packet timing): ``num_pes - 1`` sequential hops along one mesh row,
     each hop's real payload GROWING by one more shard's worth of bytes
     (matching the "shift" framing) via the same
     ``wafer.network.PacketNetwork._send_hop``/``_packet_sizes`` code
     GEMM's own cross-PE broadcast uses -- chunked into real packets
     since, unlike `sum`/`max`'s single-scalar combine payload, this
     payload can genuinely exceed one packet.
  3. OPENCAKE'S OWN ANALYTICAL COEFFICIENT: network energy (fifo/router/
     wire pJ-per-byte-hop terms) and static leakage, plus the analytical
     inter-PE link area term.

Example below: recombining 8 real attention heads' output (a real
multi-head attention recombination before the output projection) --
head_dim=512 elements per head, num_pes=8 (one head's worth of output
resident on each of 8 physical PEs), total_elements=4096 -- a real 7-hop
sequential shift-relay chain assembles the full 4096-element combined
output on PE(0,0).

No synthetic fallback anywhere: if the real Docker sub-run fails, this
raises and prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_concat_stack_cross_pe_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec
from wafersim.primitives.concat_stack import ConcatStackCrossPESpec, simulate_concat_stack_cross_pe

OUT_DIR = Path("outputs/concat_stack_cross_pe_check")


def main() -> None:
    # 8 real attention heads, head_dim=512 each -- num_pes=8 physical PEs
    # each holding one head's real output; recombine into one real
    # 4096-element buffer before the output projection.
    total_elements = 4096
    num_pes = 8

    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = ConcatStackCrossPESpec(total_elements=total_elements, num_pes=num_pes, bytes_per_element=4)
    spec.validate()
    print(
        f"ConcatStackCrossPESpec: total_elements={spec.total_elements}, num_pes={spec.num_pes}, "
        f"elements_per_shard={spec.elements_per_shard}, "
        f"real_relay_hop_count={spec.real_relay_hop_count}, "
        f"bytes_per_element={spec.bytes_per_element}"
    )

    cost = simulate_concat_stack_cross_pe(spec, wafer, OUT_DIR)

    real_time_ns = cost.sim_time_cycles / (wafer.clock_ghz * 1e9) * 1e9

    if cost.append_result is not None:
        print("\n=== REAL per-hop append sub-run (Docker, one shard-sized firing, reused x hop_count) ===")
        print(f"  hop_count (real_relay_hop_count) = {spec.real_relay_hop_count}")
        print(f"  elements appended per hop (elements_per_shard): {spec.elements_per_shard}")
        print(f"  local_compute_cycles (real, per append firing): {cost.append_result.local_compute_cycles}")
        print(f"  local_energy_uj (real, per append firing):      {cost.append_result.local_energy_uj}")
    else:
        print("\n=== DEGENERATE num_pes=1 case: real single-PE whole-output memory-move ===")
        print(f"  local_compute_cycles (real): {cost.local_result.local_compute_cycles}")
        print(f"  local_energy_uj (real):      {cost.local_result.local_energy_uj}")

    print("\n=== REAL network shift-relay simulation (not Docker, real link/FIFO/packet timing) ===")
    for k, v in cost.network_summary.items():
        print(f"  {k}: {v}")

    print("\n=== TOTAL cross-PE concat/stack cost (real Docker + real network sim + OpenCake analytical energy/area) ===")
    print(f"sim_time_cycles (total, real):                  {cost.sim_time_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):     {real_time_ns:.3f} ns")
    print(f"energy total (real Docker PE-internal + OpenCake analytical network/static): {cost.energy.total_uj} uJ")
    print(f"  pe_internal_uj (real Docker, {spec.real_relay_hop_count} real append firings summed): {cost.energy.pe_internal_uj}")
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
