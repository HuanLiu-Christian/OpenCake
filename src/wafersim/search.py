"""Design-space search over intra-PE tile_k, using real per-candidate
TimeLoop numbers plus OpenCake's own full-system (compute + network) cost.

This implements the two ideas confirmed for the TimeLoop/OpenCake division
of labor:

  1. OpenCake consumes TimeLoop's real per-candidate PE-internal
     performance/energy numbers (from a real `timeloop-model` Docker run,
     not a Python estimate) and adds its own communication/network/static
     cost on top -- see wafer/energy.py's `estimate_energy`.
  2. TimeLoop can evaluate every SRAM-legal tile_k split for a given
     workload (see `timeloop.feasible_tile_k_candidates`), so OpenCake
     enumerates all of them -- one real `timeloop-model` run per candidate
     -- and picks the actual global optimum by full-system cost, which no
     single TimeLoop run alone can see (TimeLoop has zero visibility into
     inter-PE network cost; see wafer/network.py's docstring).

There is no synthetic short-cut here: every candidate's Cycles/Energy come
from a real Docker `timeloop-model` run. A candidate whose Docker run fails
raises immediately (see `timeloop.run_timeloop_model`) -- it is not skipped
or silently substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .specs import GemmSpec, WaferSpec
from .timeloop import (
    MODEL_MODE,
    TimeLoopArtifacts,
    _largest_divisor_leq,
    feasible_tile_k_candidates,
    generate_timeloop_inputs,
    parse_mapping_contract,
    run_timeloop_model,
)
from .wafer.simulator import WaferGemmResult, simulate_gemm


@dataclass(frozen=True, slots=True)
class TileKCandidate:
    tile_k: int
    artifacts: TimeLoopArtifacts
    result: WaferGemmResult

    def to_dict(self) -> dict[str, object]:
        return {
            "tile_k": self.tile_k,
            "k_phases": self.result.contract.k_phases,
            "local_compute_cycles": self.result.contract.local_compute_cycles,
            "local_energy_uj": self.result.contract.local_energy_uj,
            "data_sram_utilization": self.result.contract.data_sram_utilization,
            "sim_time_cycles": self.result.sim_time_cycles,
            "mean_pe_utilization": self.result.mean_pe_utilization,
            "packet_hops": self.result.network["packet_hops"],
            "max_link_utilization": self.result.network["max_link_utilization"],
            "total_energy_uj": self.result.energy.total_uj,
        }


def search_tile_k(
    gemm: GemmSpec,
    wafer: WaferSpec,
    out_dir: Path,
    *,
    image: str = "timeloopaccelergy/accelergy-timeloop-infrastructure:latest",
) -> list[TileKCandidate]:
    """Evaluate every SRAM-feasible tile_k with a real timeloop-model run
    each, then run the full wafer simulator (PE compute + network) on each
    candidate's real TimeLoop contract.

    Each candidate gets its own artifact subdirectory
    (``out_dir/tile_k_<value>/timeloop/``) so the real Docker outputs for
    every candidate are preserved side by side, not overwritten in place.
    Returns all candidates in ``feasible_tile_k_candidates`` order (largest
    tile_k / fewest K phases first) so callers can inspect the whole
    trade-off, not just the winner -- see `best_by_sim_time`.
    """
    gemm.validate()
    wafer.validate()
    # REAL FIX (this session): must match generate_timeloop_inputs()'s own
    # active_rows/active_cols exactly (both now real true-divisor searches
    # via timeloop._largest_divisor_leq, not a plain min()+ceil()) -- see
    # that function's own inline comment and _largest_divisor_leq's module
    # docstring for the full real-Docker-crash root cause this fixes. This
    # local tile_m/tile_n is only used to pick SRAM-feasible tile_k
    # candidates (feasible_tile_k_candidates); generate_timeloop_inputs()
    # below independently recomputes the identical active_rows/active_cols/
    # tile_m/tile_n from the same gemm/wafer, so the two stay consistent.
    active_rows = _largest_divisor_leq(gemm.m, wafer.mesh_rows)
    active_cols = _largest_divisor_leq(gemm.n, wafer.mesh_cols)
    tile_m = gemm.m // active_rows  # exact: active_rows is a true divisor of gemm.m
    tile_n = gemm.n // active_cols  # exact: active_cols is a true divisor of gemm.n
    candidates = feasible_tile_k_candidates(gemm, wafer, tile_m, tile_n)

    evaluated: list[TileKCandidate] = []
    for tile_k in candidates:
        candidate_dir = out_dir / f"tile_k_{tile_k}"
        artifacts = generate_timeloop_inputs(gemm, wafer, candidate_dir, tile_k=tile_k)
        run_timeloop_model(artifacts, image=image)
        contract = parse_mapping_contract(artifacts, gemm, wafer, mode=MODEL_MODE)
        result = simulate_gemm(contract)
        evaluated.append(TileKCandidate(tile_k=tile_k, artifacts=artifacts, result=result))
    return evaluated


def best_by_sim_time(candidates: list[TileKCandidate]) -> TileKCandidate:
    """Pick the candidate with the lowest full-system sim_time_cycles,
    breaking ties by lower total energy.

    This is OpenCake's decision, not TimeLoop's: it uses full-system cost
    (compute *and* the network cost TimeLoop cannot see), not TimeLoop's
    own per-candidate Cycles number alone.
    """
    if not candidates:
        raise ValueError("no tile_k candidates to choose from")
    return min(
        candidates,
        key=lambda c: (c.result.sim_time_cycles, c.result.energy.total_uj),
    )
