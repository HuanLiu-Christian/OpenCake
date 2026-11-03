"""Tests for experiment_llama33_slice/llama33_slice.py: one CS-3's 20-layer share of Llama-3.3-70B at hidden 704 on a
9,000-PE (90x100) x 48 KiB wafer with weights and KV cache in SRAM."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiment_dense_llm_throughput"))
import dense_llm_throughput as D  # noqa: E402

_spec = importlib.util.spec_from_file_location("llama33_slice_script", ROOT / "experiment_llama33_slice/llama33_slice.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["llama33_slice_script"] = S
_spec.loader.exec_module(S)

from wafersim.specs import WaferSpec  # noqa: E402

CFG = S.LLAMA33_SLICE
W9000 = dataclasses.replace(WaferSpec(), **{**D.PINNED_WAFER, "sram_kib_per_pe": 48, "mesh_rows": 90, "mesh_cols": 100})
FREE_KIB = (W9000.sram_bytes_per_pe - W9000.fifo_router_bytes_per_pe) / 1024


def test_shape_is_20_layers_hidden_704_llama_style():
    assert (CFG.dim, CFG.n_layers, CFG.n_heads, CFG.n_kv_heads, CFG.head_dim, CFG.ffn_hidden, CFG.vocab) == (704, 20, 11, 1, 64, 2464, 12800)
    assert CFG.dim == CFG.n_heads * CFG.head_dim and CFG.q_dim == CFG.dim and CFG.tied_embeddings
    assert not CFG.qk_norm and not CFG.rope_interleaved and CFG.rope_theta == 500000.0 and CFG.norm_eps == 1e-5
    assert CFG.ffn_hidden == pytest.approx(3.5 * CFG.dim, abs=1)


def test_parameter_and_kv_sizes_are_the_hand_computed_ones():
    per_layer = 704 * 704 + 2 * 704 * 64 + 704 * 704 + 3 * 704 * 2464 + 2 * 704
    assert CFG.n_params() == 20 * per_layer + 12800 * 704 + 704
    assert CFG.n_params() / 1e6 == pytest.approx(134.7, abs=0.2)
    assert CFG.kv_bytes_per_token(2) == 2 * 20 * 64 * 2 == 5 * 1024


def test_block_parameters_follow_one_quarter_of_70b_over_100_within_a_factor_of_two_of_the_pe_ratio():
    # one wafer holds 1/4 of the 70B blocks; 9,000 PEs is 1/100 of 900,000, so ~1/400 of the 70B blocks at equal density.
    blocks70 = 80 * (8192 * 8192 * 2 + 2 * 8192 * 1024 + 3 * 8192 * 28672)
    ours = 20 * (704 * 704 * 2 + 2 * 704 * 64 + 3 * 704 * 2464)
    assert 1 / 4 * blocks70 / 100 / ours == pytest.approx(1.4, abs=0.15)   # ~30% less dense than the real wafer, on purpose


def test_weights_and_kv_fit_and_leave_room_for_tiles():
    w = D.weights_carve_kib(CFG, W9000)
    assert w == 30                                             # 29.2 KiB/PE rounded up
    kv = D.kv_carve_kib(CFG, W9000, 1024, 1024, 1)
    assert kv <= 2
    assert FREE_KIB - w - kv >= 5                              # >= 5 KiB of tiles
    assert D.choose_kv_placement(CFG, W9000, 1024, 1024, 1, 2, "auto", True) == "sram"


def test_the_exact_proportional_density_does_not_fit_but_this_does():
    exact_kib = 0.3525e9 / 9000 / 1024                         # 38.25 KiB/PE (35.25 GB / 4 / 100)
    assert exact_kib > FREE_KIB                                 # why hidden 704 and not ~820
    assert D.weights_carve_kib(CFG, W9000) < FREE_KIB


def test_script_defaults_and_mesh_override():
    a = S.build_parser().parse_args([])
    assert (a.sram_kib, a.weights_dram, a.carve_weights, a.kv_placement, a.batches, a.mesh) == (48, False, True, "auto", "1", "90x100")
    assert a.in_len == 1024 and a.out_len == 1024 and a.mac_lanes == 1
    w = D.apply_wafer_overrides(dataclasses.replace(WaferSpec(), **D.PINNED_WAFER), a)
    assert (w.mesh_rows, w.mesh_cols, w.sram_kib_per_pe) == (90, 100, 48)


def test_dry_run_fits_on_9000_pes(capsys):
    args = S.build_parser().parse_args(["--dry-run", "--skip-numeric", "--modes", "optimistic"])
    D.run_model(args, CFG, "llama33_slice_d704", {}, lambda: None, note=S.NOTE)
    out = capsys.readouterr().out
    assert "WEIGHTS resident in SRAM: 30 KiB/PE" in out and "MESH OVERRIDE: 90x100 = 9,000 PEs" in out
    assert "b1=SRAM" in out and "DRY RUN finished" in out
