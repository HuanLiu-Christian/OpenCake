"""Energy precision of ``_parse_summary_stats``.

TimeLoop's headline ``Energy: X uJ`` has 2 decimals (0.01 uJ floor); the same
Summary Stats block's ``Computes`` x ``fJ/Compute Total`` carries the exact
value. These tests pin: the precise path is used, it agrees with the headline
to within the headline's rounding, the fallback still works, and a genuine
disagreement is raised, not guessed.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest

from wafersim.timeloop import _parse_summary_stats

LOG = Path("/nonexistent/timeloop-mapper.log")


def _block(cycles=16384, headline="9.39 uJ", computes=65536, total_fj="143349.72", table=True):
    text = (
        "Summary Stats\n-------------\n"
        "GFLOPs (@1GHz): 4.00\nUtilization: 100.00%\n"
        f"Cycles: {cycles}\nEnergy: {headline}\nEDP(J*cycle): 1.54e-01\nArea: 0.00 mm^2\n\n"
    )
    if table:
        text += (
            f"Computes = {computes}\nfJ/Compute\n"
            "    bitwise_gate                    = 579.20\n"
            f"    Total                           = {total_fj}\n"
        )
    return "preamble\n" + text


def test_uses_precise_value_not_two_decimal_headline():
    cycles, energy = _parse_summary_stats(_block(), LOG)
    assert cycles == 16384
    assert energy == pytest.approx(65536 * 143349.72 * 1e-9, rel=1e-12)
    assert energy != 9.39  # not the quantized headline


def test_large_gemm_keeps_tighter_headline():
    # Real GEMM from outputs: 1,073,741,824 computes x 25885.66 fJ; the table's
    # 0.01 fJ x 1e9 is looser than the headline's 0.01 uJ, so headline is kept
    # (and the two must still agree within the summed rounding).
    _, energy = _parse_summary_stats(
        _block(cycles=131072, headline="27794.51 uJ", computes=1073741824, total_fj="25885.66"), LOG
    )
    assert energy == pytest.approx(27794.51)


def test_tiny_run_below_headline_floor_is_resolved():
    # Headline rounds ~1e-6 uJ to 0.00; the table keeps it.
    # 4 computes x 350.5 fJ = 1402 fJ = 1.402e-6 uJ
    cycles, energy = _parse_summary_stats(
        _block(cycles=4, headline="0.00 uJ", computes=4, total_fj="350.50"), LOG
    )
    assert energy == pytest.approx(1.402e-6, rel=1e-9)


def test_falls_back_to_headline_when_table_absent():
    _, energy = _parse_summary_stats(_block(table=False), LOG)
    assert energy == pytest.approx(9.39)


def test_headline_in_other_units_is_cross_checked():
    # 2.34 mJ headline == 2340 uJ; precise 2340.0004 uJ is within half of 0.01 mJ.
    _, energy = _parse_summary_stats(
        _block(headline="2.34 mJ", computes=1000, total_fj="2340000000.4"), LOG
    )
    assert energy == pytest.approx(2340.0000004, rel=1e-9)


def test_disagreement_beyond_headline_rounding_raises():
    with pytest.raises(RuntimeError, match="disagrees"):
        _parse_summary_stats(_block(headline="9.39 uJ", total_fj="200000.00"), LOG)


def test_missing_summary_still_raises():
    with pytest.raises(RuntimeError, match="Summary Stats"):
        _parse_summary_stats("nothing here", LOG)


def test_missing_cycles_or_energy_still_raises():
    with pytest.raises(RuntimeError, match="Cycles and"):
        _parse_summary_stats("Summary Stats\n-----\nCycles: 5\n", LOG)


_REAL = sorted(
    f
    for f in glob.glob(str(Path(__file__).resolve().parents[1] / "outputs" / "**" / "*.stats.txt"), recursive=True)
    if os.path.getsize(f) < 400_000
)[:3000:7]


@pytest.mark.skipif(not _REAL, reason="no real TimeLoop stats files under outputs/")
def test_real_stats_files_parse_and_agree_with_headline():
    for f in _REAL:
        text = Path(f).read_text(errors="ignore")
        _parse_summary_stats(text, Path(f))  # would raise on any inconsistency
