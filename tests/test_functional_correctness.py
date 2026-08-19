from __future__ import annotations

import unittest

from wafersim.specs import GemmSpec
from wafersim.wafer.functional import simulate_gemm_functional


class SummaFunctionalCorrectnessTest(unittest.TestCase):
    """Numerical (not timing) checks for the rotating-owner SUMMA schedule.

    These exercise the same tiling/phase-accumulation logic simulate_gemm()
    uses, but with real numpy data, and compare against a plain `A @ B`
    reference -- see wafer/functional.py's module docstring for exactly
    what this does and doesn't cover.
    """

    def test_matches_reference_when_evenly_divisible(self) -> None:
        gemm = GemmSpec(m=64, n=64, k=128)
        result = simulate_gemm_functional(
            gemm, active_rows=4, active_cols=4, tile_m=16, tile_n=16, tile_k=32
        )
        self.assertTrue(result.passed, f"max_abs_error={result.max_abs_error}")
        self.assertEqual(result.k_phases, 4)
        self.assertLess(result.max_abs_error, 1e-6)

    def test_matches_reference_with_boundary_tiles(self) -> None:
        # M/N/K deliberately don't divide evenly by mesh*tile or tile_k --
        # this is the scenario _extent() exists to handle (last row/col/
        # phase gets a smaller-than-tile slice). If the boundary clipping
        # were wrong, this is where it would show up as a numeric mismatch,
        # not just a shape error.
        gemm = GemmSpec(m=50, n=44, k=37)
        active_rows, active_cols = 4, 3
        tile_m, tile_n, tile_k = 13, 15, 8  # 4*13=52>50, 3*15=45>44, ceil(37/8)=5 phases
        result = simulate_gemm_functional(
            gemm, active_rows=active_rows, active_cols=active_cols,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        )
        self.assertTrue(result.passed, f"max_abs_error={result.max_abs_error}")
        self.assertEqual(result.k_phases, 5)
        self.assertLess(result.max_abs_error, 1e-6)

    def test_owner_rotation_covers_full_mesh(self) -> None:
        # With enough K phases, the rotating owner_col/owner_row should
        # cycle through every column/row at least once -- a regression
        # here (e.g. a stuck owner=0) wouldn't break this test's numeric
        # result (see module docstring: owner doesn't affect *what* data a
        # PE receives), but it would mean the schedule no longer resembles
        # the real v10 architecture change this project actually made.
        gemm = GemmSpec(m=32, n=32, k=64)
        result = simulate_gemm_functional(
            gemm, active_rows=4, active_cols=4, tile_m=8, tile_n=8, tile_k=8
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.k_phases, 8)
        self.assertEqual(result.owner_cols_seen, 4)
        self.assertEqual(result.owner_rows_seen, 4)

    def test_matches_reference_for_real_v11_best_candidate(self) -> None:
        # The exact tiling OpenCake picked as the real full-system optimum
        # for the 1024^3 GEMM / 32x32 mesh example (outputs/gemm1024_v11):
        # tile_m=tile_n=32, tile_k=16, k_phases=64. Runs at full scale
        # (1024x1024x1024) to confirm the schedule that was actually
        # selected is numerically correct, not just a small toy case.
        gemm = GemmSpec(m=1024, n=1024, k=1024)
        result = simulate_gemm_functional(
            gemm, active_rows=32, active_cols=32, tile_m=32, tile_n=32, tile_k=16
        )
        self.assertTrue(result.passed, f"max_abs_error={result.max_abs_error}")
        self.assertEqual(result.k_phases, 64)
        self.assertEqual(result.owner_cols_seen, 32)
        self.assertEqual(result.owner_rows_seen, 32)

    def test_rejects_nonpositive_dimensions(self) -> None:
        gemm = GemmSpec(m=8, n=8, k=8)
        with self.assertRaises(ValueError):
            simulate_gemm_functional(gemm, active_rows=0, active_cols=1, tile_m=1, tile_n=1, tile_k=1)


if __name__ == "__main__":
    unittest.main()
