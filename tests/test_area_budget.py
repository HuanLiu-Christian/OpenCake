"""Covers the pure-Python control flow in area_budget.py: AreaBudget's
validation/derived-budget math, mesh derivation, and the generic binary
search helper.

Deliberately does NOT call derive_wafer_spec_from_area_budget or any of the
_search_max_*/_probe_contract functions: those always invoke a real
`timeloop-model` Docker run per candidate (see area_budget.py's module
docstring) -- there is no offline/synthetic fallback -- so this unit test
suite doesn't assume Docker is available. _binary_search_max_fit is tested
here with an injected fake `probe` callable instead of the real Docker-backed
one, exactly so the SEARCH ALGORITHM itself (not real area numbers) can be
verified quickly and deterministically; production code never uses a fake
probe.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from wafersim.area_budget import (
    _FIFO_CAPACITY_PACKETS_MAX,
    _SRAM_KIB_PER_PE_MAX,
    AreaBudget,
    AreaBudgetResult,
    _binary_search_max_fit,
    _derive_mesh,
    _divisors,
)
from wafersim.specs import GemmSpec, WaferSpec


class DivisorsTest(unittest.TestCase):
    def test_lists_all_divisors_ascending(self) -> None:
        self.assertEqual(_divisors(12), [1, 2, 3, 4, 6, 12])

    def test_prime_has_only_1_and_itself(self) -> None:
        self.assertEqual(_divisors(13), [1, 13])

    def test_perfect_square_does_not_duplicate_the_square_root(self) -> None:
        self.assertEqual(_divisors(9), [1, 3, 9])


class AreaBudgetValidationTest(unittest.TestCase):
    def test_derived_budgets_split_the_total_by_fraction(self) -> None:
        budget = AreaBudget(
            wafer_area_budget_um2=200_000_000.0,
            compute_area_fraction=0.05,
            memory_area_fraction=0.60,
            network_area_fraction=0.35,
        )
        budget.validate()
        self.assertEqual(budget.compute_area_budget_um2, 10_000_000.0)
        self.assertEqual(budget.memory_area_budget_um2, 120_000_000.0)
        self.assertEqual(budget.network_area_budget_um2, 70_000_000.0)
        self.assertEqual(
            budget.compute_area_budget_um2
            + budget.memory_area_budget_um2
            + budget.network_area_budget_um2,
            200_000_000.0,
        )

    def test_fractions_summing_under_100_percent_are_allowed(self) -> None:
        # Leaves 10% of the die unaccounted for (e.g. I/O, unmodeled
        # overhead) -- not an error, just an unused remainder.
        AreaBudget(
            wafer_area_budget_um2=100.0,
            compute_area_fraction=0.3,
            memory_area_fraction=0.3,
            network_area_fraction=0.3,
        ).validate()

    def test_fractions_summing_over_100_percent_raise(self) -> None:
        budget = AreaBudget(
            wafer_area_budget_um2=100.0,
            compute_area_fraction=0.5,
            memory_area_fraction=0.5,
            network_area_fraction=0.1,
        )
        with self.assertRaises(ValueError):
            budget.validate()

    def test_negative_fraction_raises(self) -> None:
        budget = AreaBudget(
            wafer_area_budget_um2=100.0,
            compute_area_fraction=-0.1,
            memory_area_fraction=0.5,
            network_area_fraction=0.5,
        )
        with self.assertRaises(ValueError):
            budget.validate()

    def test_nonpositive_wafer_area_budget_raises(self) -> None:
        budget = AreaBudget(
            wafer_area_budget_um2=0.0,
            compute_area_fraction=0.3,
            memory_area_fraction=0.3,
            network_area_fraction=0.3,
        )
        with self.assertRaises(ValueError):
            budget.validate()


class DeriveMeshTest(unittest.TestCase):
    """mesh_rows/mesh_cols must be true divisors (>=2) of gemm_m/gemm_n --
    a real Docker run confirmed TimeLoop's own mapping parser rejects a
    mesh that does not evenly divide the GEMM ("product of all factors of
    dimension M is 1026, which is not equal to the dimension size of the
    workload 1024"), so _derive_mesh only ever returns divisor pairs, never
    an arbitrary floor(sqrt(...)) mesh."""

    def test_picks_the_most_pes_among_valid_divisor_pairs(self) -> None:
        # gemm_m=gemm_n=12, divisors>=2: {2,3,4,6,12}. max_pes=10 (100/10).
        # No divisor pair multiplies to exactly 10; the best fitting <=10
        # is 3x3=9 (also already square, so the tie-break never triggers).
        rows, cols = _derive_mesh(
            compute_area_budget_um2=100.0, mac_area_um2=10.0, gemm_m=12, gemm_n=12
        )
        self.assertEqual((rows, cols), (3, 3))

    def test_asymmetric_gemm_shape_uses_divisors_of_each_dimension_separately(self) -> None:
        # gemm_m=8 (divisors>=2: 2,4,8), gemm_n=27 (divisors>=2: 3,9,27).
        # max_pes=50. Every valid (rows, cols) product <=50: 2x3=6, 2x9=18,
        # 4x3=12, 4x9=36, 8x3=24 -- the largest is 4x9=36 (8x9=72 and
        # 2x27/4x27/8x27 all exceed 50).
        rows, cols = _derive_mesh(
            compute_area_budget_um2=50.0, mac_area_um2=1.0, gemm_m=8, gemm_n=27
        )
        self.assertEqual((rows, cols), (4, 9))

    def test_budget_smaller_than_one_mac_raises(self) -> None:
        with self.assertRaises(ValueError):
            _derive_mesh(compute_area_budget_um2=5.0, mac_area_um2=10.0, gemm_m=1024, gemm_n=1024)

    def test_no_fitting_divisor_pair_raises(self) -> None:
        # gemm_m=gemm_n=3 -> the only divisor >=2 is 3 itself, so the
        # smallest possible mesh is 3x3=9 PEs. A budget covering only 1 PE
        # cannot fit that, and must raise rather than silently returning
        # something smaller (there is nothing smaller that is still a
        # divisor pair).
        with self.assertRaises(ValueError):
            _derive_mesh(compute_area_budget_um2=100.0, mac_area_um2=100.0, gemm_m=3, gemm_n=3)

    def test_smallest_divisor_pair_is_2x2_when_available(self) -> None:
        # gemm_m=gemm_n=4 -> divisors>=2: {2,4}. A budget for exactly 4 PEs
        # picks 2x2 (4x4=16 does not fit).
        rows, cols = _derive_mesh(
            compute_area_budget_um2=40.0, mac_area_um2=10.0, gemm_m=4, gemm_n=4
        )
        self.assertEqual((rows, cols), (2, 2))


class BinarySearchMaxFitTest(unittest.TestCase):
    def test_finds_the_largest_candidate_within_budget(self) -> None:
        # Fake monotonic "area" = candidate * 2.0 -- no Docker involved.
        def probe(candidate: int):
            return candidate * 2.0, candidate

        best, extra, exhausted = _binary_search_max_fit(1, 100, 24, probe, budget_um2=50.0)
        self.assertEqual(best, 25)  # 25*2=50 <= 50, 26*2=52 > 50
        self.assertEqual(extra, 25)
        self.assertFalse(exhausted)  # fully converged well within 24 iterations

    def test_none_probe_result_is_treated_as_too_small_not_over_budget(self) -> None:
        # Candidates 1-4 are "structurally infeasible" (None); from 5 up,
        # area = candidate (i.e. already over the budget of 3 for every
        # feasible candidate) -- so nothing should ever be accepted, and the
        # search must not mistake "None" for "fits".
        def probe(candidate: int):
            if candidate < 5:
                return None
            return float(candidate), candidate

        best, extra, exhausted = _binary_search_max_fit(1, 100, 24, probe, budget_um2=3.0)
        self.assertIsNone(best)
        self.assertIsNone(extra)
        self.assertFalse(exhausted)  # genuinely infeasible, not a step-count problem

    def test_infeasible_candidates_below_a_fitting_range_are_skipped(self) -> None:
        # Candidates 1-9 infeasible (None), 10-100 have area = candidate;
        # budget of 42 should land on the largest feasible candidate <= 42.
        def probe(candidate: int):
            if candidate < 10:
                return None
            return float(candidate), candidate

        best, extra, exhausted = _binary_search_max_fit(1, 100, 24, probe, budget_um2=42.0)
        self.assertEqual(best, 42)
        self.assertEqual(extra, 42)
        self.assertFalse(exhausted)

    def test_too_few_iterations_can_undershoot_but_never_exceeds_budget(self) -> None:
        def probe(candidate: int):
            return float(candidate), candidate

        best, _, exhausted = _binary_search_max_fit(1, 1000, 1, probe, budget_um2=500.0)
        self.assertIsNotNone(best)
        self.assertLessEqual(best, 500)
        self.assertTrue(exhausted)  # only 1 of the ~10 iterations a [1,1000] range needs

    def test_too_few_iterations_over_a_wide_range_reports_exhausted_not_infeasible(self) -> None:
        # Reproduces the exact real failure mode this flag exists for: a
        # fitting candidate genuinely exists (area=candidate, budget=3 fits
        # candidate=1,2,3), but a [1, 1_000_000] range needs ~20 halvings to
        # ever reach it, and only 3 iterations are given -- best must come
        # back None (not enough steps to get there), but `exhausted` must
        # say so, distinguishing this from "no fit exists at all".
        def probe(candidate: int):
            return float(candidate), candidate

        best, extra, exhausted = _binary_search_max_fit(1, 1_000_000, 3, probe, budget_um2=3.0)
        self.assertIsNone(best)
        self.assertIsNone(extra)
        self.assertTrue(exhausted)


def _fake_area_budget_result(*, sram_saturated: bool, fifo_saturated: bool) -> AreaBudgetResult:
    """Builds a structurally-valid AreaBudgetResult with dummy (not
    Docker-derived) contract numbers -- saturation_warning depends only on
    the two saturated booleans, never on the contracts' actual area
    numbers, so any valid GemmMappingContract works here; this stays
    entirely pure-Python, consistent with this module's no-Docker
    philosophy (see module docstring)."""
    gemm = GemmSpec(m=1024, n=1024, k=1024, bytes_per_element=4)
    wafer = WaferSpec(mesh_rows=2, mesh_cols=2)
    from wafersim.contract import GemmMappingContract

    dummy_contract = GemmMappingContract(
        gemm=gemm,
        wafer=wafer,
        active_rows=2,
        active_cols=2,
        tile_m=512,
        tile_n=512,
        tile_k=1,
        mapper_mode="model",
        map_path=Path("map.yaml"),
        stats_path=Path("stats.txt"),
        local_compute_cycles=1,
        local_energy_uj=1.0,
        mac_area_um2=1.0,
        pe_scratchpad_area_um2=1.0,
        fifo_area_um2=1.0,
        router_area_um2=1.0,
    )
    budget = AreaBudget(
        wafer_area_budget_um2=100.0,
        compute_area_fraction=0.3,
        memory_area_fraction=0.3,
        network_area_fraction=0.3,
    )
    return AreaBudgetResult(
        wafer=wafer,
        budget=budget,
        mac_area_um2=1.0,
        compute_pes=4,
        compute_area_used_um2=4.0,
        memory_contract=dummy_contract,
        network_contract=dummy_contract,
        sram_search_saturated=sram_saturated,
        fifo_search_saturated=fifo_saturated,
    )


class AreaBudgetResultSaturationWarningTest(unittest.TestCase):
    """A search stage's returned value landing exactly on its own artificial
    ceiling (_SRAM_KIB_PER_PE_MAX / _FIFO_CAPACITY_PACKETS_MAX) must be
    surfaced as an explicit caveat, not silently presented as a trustworthy
    budget-driven optimum -- see area_budget.py's module docstring, stage 3,
    for the real Docker run that motivated this (fifo_capacity_packets
    landing on 1,048,576 turned out to be an Accelergy/CACTI silent
    near-zero-area failure, not a real optimum)."""

    def test_neither_saturated_is_none(self) -> None:
        result = _fake_area_budget_result(sram_saturated=False, fifo_saturated=False)
        self.assertIsNone(result.saturation_warning)
        self.assertIsNone(result.to_dict()["saturation_warning"])
        self.assertFalse(result.to_dict()["sram_search_saturated"])
        self.assertFalse(result.to_dict()["fifo_search_saturated"])

    def test_fifo_saturated_mentions_the_fifo_ceiling(self) -> None:
        result = _fake_area_budget_result(sram_saturated=False, fifo_saturated=True)
        self.assertIsNotNone(result.saturation_warning)
        self.assertIn(str(_FIFO_CAPACITY_PACKETS_MAX), result.saturation_warning)
        # Note: _SRAM_KIB_PER_PE_MAX and _FIFO_CAPACITY_PACKETS_MAX currently
        # happen to share the same numeric value (1,048,576), so this can't
        # assert the SRAM ceiling's *number* is absent -- assert the SRAM
        # caveat's own wording is absent instead.
        self.assertNotIn("sram_kib_per_pe landed", result.saturation_warning)

    def test_sram_saturated_mentions_the_sram_ceiling(self) -> None:
        result = _fake_area_budget_result(sram_saturated=True, fifo_saturated=False)
        self.assertIsNotNone(result.saturation_warning)
        self.assertIn(str(_SRAM_KIB_PER_PE_MAX), result.saturation_warning)

    def test_both_saturated_mentions_both(self) -> None:
        result = _fake_area_budget_result(sram_saturated=True, fifo_saturated=True)
        self.assertIn(str(_SRAM_KIB_PER_PE_MAX), result.saturation_warning)
        self.assertIn(str(_FIFO_CAPACITY_PACKETS_MAX), result.saturation_warning)


if __name__ == "__main__":
    unittest.main()
