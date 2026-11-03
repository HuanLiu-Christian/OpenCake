"""Tests for experiment_qwen3_slice_region/region_decode.py (weight-stationary, K-split decode on per-layer PE regions)."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("region_decode_script", ROOT / "experiment_qwen3_slice_region/region_decode.py")
RD = importlib.util.module_from_spec(_spec)
sys.modules["region_decode_script"] = RD
_spec.loader.exec_module(RD)


def _region(R, C, lanes=8, **ov):
    w = dataclasses.replace(RD.make_wafer(R, C, lanes, 16 if lanes == 8 else 0), **ov)
    return RD.Region(w, R, C)


def test_single_pe_region_has_no_communication_cost():
    reg = _region(1, 1)
    t = reg.gemm(0.0, 1, 64, 128, "x")
    assert t == ceil(1 * 64 * 128 / 8)                       # pure compute, nothing to scatter or reduce


def test_more_pes_shorten_the_compute_phase():
    small, big = _region(2, 2), _region(8, 8)
    small.gemm(0.0, 1, 256, 256, "x")
    big.gemm(0.0, 1, 256, 256, "x")
    assert big.parts["x:compute"] < small.parts["x:compute"]
    assert big.parts["x:compute"] == ceil(1 * ceil(256 / 8) * ceil(256 / 8) / 8)      # nt * kt / lanes


def test_reduce_and_scatter_cost_grow_with_region_size():
    a, b = _region(2, 2), _region(14, 18)
    a.gemm(0.0, 1, 512, 512, "x")
    b.gemm(0.0, 1, 512, 512, "x")
    assert b.parts["x:scatter"] > a.parts["x:scatter"] and b.parts["x:reduce"] > a.parts["x:reduce"]


def test_slower_links_make_the_step_slower():
    fast = RD.layer_cycles(RD.Cfg(), RD.make_wafer(14, 18, 8, 16), 14, 18, 1535)[0]
    slow_w = dataclasses.replace(RD.make_wafer(14, 18, 8, 16), link_bandwidth_bytes_per_cycle=8)
    slow = RD.layer_cycles(RD.Cfg(), slow_w, 14, 18, 1535)[0]
    assert slow > fast


def test_default_layout_fits_the_9000_pe_mesh_and_the_sram():
    assert 32 * 14 * 18 + 24 * 39 == 9000
    R, C, d, qd, kv, ffn = 14, 18, 460, 704, 64, 2300
    pe = lambda n, k: ceil(n / C) * ceil(k / R) * 2
    w = pe(qd + 2 * kv, d) + pe(d, qd) + pe(2 * ffn, d) + pe(d, ffn)
    kvb = ceil(2047 / (R * C)) * 2 * kv * 2
    assert 48 * 1024 - w - kvb - 10260 >= 5000               # >= 5 KB left for activations / buffers


def test_default_run_is_between_the_summa_one_row_run_and_the_ideal():
    out = RD.run()
    assert 1e5 < out["itl_cycles"] < 5.2e5                   # SUMMA one-row mapping: 521,447 cycles
    assert abs(out["decode_tok_s"] - 1e9 / out["itl_cycles"]) < 1e-6
    assert out["samples"][2046]["itl"] > out["samples"][1024]["itl"]     # longer cache -> slower
