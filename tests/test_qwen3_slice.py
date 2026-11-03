"""Tests for experiment_qwen3_slice/qwen3_slice.py: one CS-3's 32-layer share of Qwen3-32B at hidden 460 on a
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

_spec = importlib.util.spec_from_file_location("qwen3_slice_script", ROOT / "experiment_qwen3_slice/qwen3_slice.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["qwen3_slice_script"] = S
_spec.loader.exec_module(S)

from wafersim.specs import WaferSpec  # noqa: E402

CFG = S.QWEN3_SLICE
Q32 = S.Q32
W9000 = dataclasses.replace(WaferSpec(), **{**D.PINNED_WAFER, "sram_kib_per_pe": 48, "mesh_rows": 90, "mesh_cols": 100})
FREE_B = W9000.sram_bytes_per_pe - W9000.fifo_router_bytes_per_pe


def test_shape_is_32_layers_hidden_460_qwen3_style():
    assert (CFG.dim, CFG.n_layers, CFG.n_heads, CFG.n_kv_heads, CFG.head_dim, CFG.ffn_hidden, CFG.vocab) == (460, 32, 11, 1, 64, 2300, 15200)
    assert CFG.q_dim == 704 and CFG.q_dim != CFG.dim                 # Qwen3: Q width is not the hidden size
    assert CFG.qk_norm and not CFG.rope_interleaved and not CFG.tied_embeddings and CFG.sliding_window == 0
    assert CFG.rope_theta == 1_000_000.0 and CFG.norm_eps == 1e-6 and CFG.ffn_hidden == 5 * CFG.dim


def test_parameter_and_kv_sizes_are_the_hand_computed_ones():
    per_layer = 460 * 704 + 2 * 460 * 64 + 704 * 460 + 3 * 460 * 2300 + 2 * 460 + 2 * 64
    assert CFG.n_params() == 32 * per_layer + 2 * 15200 * 460 + 460
    assert CFG.n_params() / 1e6 == pytest.approx(138.2, abs=0.1)
    assert CFG.n_params() * 2 / 1e9 == pytest.approx(0.276, abs=0.001)
    assert CFG.kv_bytes_per_token(2) == 2 * 32 * 64 * 2 == 8 * 1024


def test_real_qwen3_32b_shape_gives_32_8b_parameters():
    # official config: hidden 5120, 64 layers, 64 Q / 8 KV heads of 128, FFN 25600, vocab 151936, untied -> 32.8 B
    assert Q32.QWEN3_32B.n_params() / 1e9 == pytest.approx(32.76, abs=0.01)
    assert Q32.QWEN3_32B.n_params() * 2 / 1e9 == pytest.approx(65.5, abs=0.1)


def test_weights_and_kv_fit_and_leave_6124_bytes_for_tiles():
    w = D.weights_carve_kib(CFG, W9000)
    assert w == 30
    kv = D.kv_carve_kib(CFG, W9000, 1024, 1024, 1)
    assert kv <= 2
    assert FREE_B - (w + kv) * 1024 == 6124
    assert D.choose_kv_placement(CFG, W9000, 1024, 1024, 1, 2, "auto", True) == "sram"


def test_the_proportional_target_does_not_fit_but_this_does():
    target_params = 32.76e9 / 2 / 100                               # 0.164 B on one of 2 wafers at 1/100
    assert target_params * 2 / 9000 / 1024 + 2 > FREE_B / 1024 - 5  # ~35.6 KiB weights + KV leaves < 5 KiB for tiles
    assert CFG.n_params() < target_params                           # ~16 % smaller, on purpose
    assert D.weights_carve_kib(CFG, W9000) * 1024 < FREE_B


def test_script_defaults_and_mesh_override():
    a = S.build_parser().parse_args([])
    assert (a.sram_kib, a.weights_dram, a.carve_weights, a.kv_placement, a.batches, a.mesh) == (48, False, True, "auto", "1", "90x100")
    assert a.in_len == 1024 and a.out_len == 1024 and a.mac_lanes == 1
    w = D.apply_wafer_overrides(dataclasses.replace(WaferSpec(), **D.PINNED_WAFER), a)
    assert (w.mesh_rows, w.mesh_cols, w.sram_kib_per_pe) == (90, 100, 48)


def test_dry_run_fits_on_9000_pes(capsys):
    args = S.build_parser().parse_args(["--dry-run", "--skip-numeric", "--modes", "optimistic"])
    D.run_model(args, CFG, "qwen3_slice_d460", {}, lambda: None, note=S.NOTE)
    out = capsys.readouterr().out
    assert "WEIGHTS resident in SRAM: 30 KiB/PE" in out and "MESH OVERRIDE: 90x100 = 9,000 PEs" in out
    assert "b1=SRAM" in out and "DRY RUN finished" in out
