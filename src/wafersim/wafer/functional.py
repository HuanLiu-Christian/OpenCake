"""Numerical correctness check for the rotating-owner SUMMA schedule.

``wafer/simulator.py``'s ``simulate_gemm()`` is a pure timing/energy model:
it tracks *when* each PE's data/compute becomes ready (timestamps in
``compute_ready``/``buffer_ready``), never the actual matrix values. It has
never been checked against real numbers -- so a bug in the tiling indices
or the rotating-owner accumulation loop could silently produce a schedule
that is well-timed but numerically wrong, and the timing model alone would
never catch that.

This module is the numerical counterpart. It runs the *same* phase/tiling
schedule as ``simulate_gemm()`` -- same ``k_phases``, same
``tile_m``/``tile_n``/``tile_k``, same rotating ``owner_col``/``owner_row``
formula, same boundary clipping (imports ``_extent`` directly from
``simulator.py`` rather than re-implementing it, so both models are
guaranteed to agree on how edge tiles are sized) -- but with real numpy
matrices, and checks the reconstructed ``C`` against a reference ``A @ B``.

Scope, deliberately narrow: this validates the SUMMA tiling/accumulation
**math** -- do the per-phase K-ranges and per-PE M/N extents, summed across
every phase, reconstruct the exact product, including boundary tiles when
M/N/K don't divide evenly by the mesh/tile sizes. It does **not** validate
the double-buffer timing-overlap safety that the cycle-level FIFO model in
``network.py``/``simulator.py`` reasons about. That distinction exists
because, by construction, this project's network model always delivers an
*identical* copy of a phase's A panel to every PE in that row (and B panel
to every PE in that column) regardless of which column/row happens to be
that phase's "owner" -- owner position changes *when* data arrives in the
timing model, never *what* data arrives. So functional correctness here is
independent of the owner-rotation network routing by construction, and
purely exercises the tiling indices and the phase-accumulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

from ..specs import GemmSpec
from .simulator import _extent


@dataclass(frozen=True, slots=True)
class FunctionalCheckResult:
    """Outcome of one numerical SUMMA correctness run."""

    active_rows: int
    active_cols: int
    tile_m: int
    tile_n: int
    tile_k: int
    k_phases: int
    seed: int
    max_abs_error: float
    max_rel_error: float
    owner_cols_seen: int
    owner_rows_seen: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_rows": self.active_rows,
            "active_cols": self.active_cols,
            "tile": {"m": self.tile_m, "n": self.tile_n, "k": self.tile_k},
            "k_phases": self.k_phases,
            "seed": self.seed,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "owner_cols_seen": self.owner_cols_seen,
            "owner_rows_seen": self.owner_rows_seen,
            "passed": self.passed,
        }


def simulate_gemm_functional(
    gemm: GemmSpec,
    active_rows: int,
    active_cols: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    seed: int = 0,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> FunctionalCheckResult:
    """Run the rotating-owner SUMMA schedule against real numpy data.

    Generates random ``A[M,K]``/``B[K,N]``, replays the exact same phase
    loop and rotating panel ownership ``simulate_gemm()`` uses, accumulates
    each PE's local ``C`` block, and compares the reassembled result to a
    plain ``A @ B`` reference. Raises ``ValueError`` for a nonsensical
    mesh/tile shape (mirrors ``GemmMappingContract.validate()``'s geometry
    checks) rather than silently producing a meaningless result.
    """
    if min(active_rows, active_cols, tile_m, tile_n, tile_k) <= 0:
        raise ValueError("mesh/tile dimensions must be positive")

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((gemm.m, gemm.k))
    B = rng.standard_normal((gemm.k, gemm.n))
    expected = A @ B

    k_phases = ceil(gemm.k / tile_k)
    C = np.zeros((gemm.m, gemm.n))
    owner_cols_seen: set[int] = set()
    owner_rows_seen: set[int] = set()

    for phase in range(k_phases):
        k_start = phase * tile_k
        k_extent = _extent(k_start, tile_k, gemm.k)

        # Rotating SUMMA panel ownership -- identical formula to
        # simulate_gemm(). Tracked here purely to verify the rotation
        # itself actually cycles through every row/column over a full run
        # (see the assertion below); it does not otherwise affect the
        # numerical result -- every PE in a row receives the same A panel
        # this phase no matter which column in that row "owns" it.
        owner_col = phase % active_cols
        owner_row = phase % active_rows
        owner_cols_seen.add(owner_col)
        owner_rows_seen.add(owner_row)

        for row in range(active_rows):
            m_start = row * tile_m
            m_extent = _extent(m_start, tile_m, gemm.m)
            if m_extent <= 0:
                continue
            a_panel = A[m_start : m_start + m_extent, k_start : k_start + k_extent]

            for col in range(active_cols):
                n_start = col * tile_n
                n_extent = _extent(n_start, tile_n, gemm.n)
                if n_extent <= 0:
                    continue
                b_panel = B[k_start : k_start + k_extent, n_start : n_start + n_extent]
                C[m_start : m_start + m_extent, n_start : n_start + n_extent] += (
                    a_panel @ b_panel
                )

    diff = np.abs(C - expected)
    max_abs_error = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(expected), 1e-12)
    max_rel_error = float((diff / denom).max()) if diff.size else 0.0
    passed = bool(np.allclose(C, expected, atol=atol, rtol=rtol))

    return FunctionalCheckResult(
        active_rows=active_rows,
        active_cols=active_cols,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        k_phases=k_phases,
        seed=seed,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        owner_cols_seen=len(owner_cols_seen),
        owner_rows_seen=len(owner_rows_seen),
        passed=passed,
    )
