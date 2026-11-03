"""Tests for experiment_llama33_scaled/llama33_70b_scaled.py: the 163.3x-smaller Llama-3.3-70B (80 layers, hidden 640,
real vocab, tied head), its placement of the KV cache on a 12,800-PE x 48 KiB wafer, and the HF-layout numeric check."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiment_dense_llm_throughput"))
import dense_llm_throughput as D  # noqa: E402

_spec = importlib.util.spec_from_file_location("llama33_scaled_script", ROOT / "experiment_llama33_scaled/llama33_70b_scaled.py")
L = importlib.util.module_from_spec(_spec)
sys.modules["llama33_scaled_script"] = L
_spec.loader.exec_module(L)

from wafersim.specs import WaferSpec  # noqa: E402

PINNED48 = dataclasses.replace(WaferSpec(), **{**D.PINNED_WAFER, "sram_kib_per_pe": 48})
BIG, S = L.LLAMA33_70B, L.LLAMA33_70B_S163


def test_the_full_model_has_the_published_70b_parameter_count():
    assert (BIG.dim, BIG.n_layers, BIG.n_heads, BIG.n_kv_heads, BIG.head_dim, BIG.ffn_hidden, BIG.vocab) == (8192, 80, 64, 8, 128, 28672, 128256)
    assert BIG.n_params() == 70_553_706_496  # 70.55 B
    assert BIG.kv_bytes_per_token(2) == 2 * 80 * 1024 * 2 == 327_680  # 320 KiB


def test_the_scaled_model_keeps_depth_and_cuts_the_block_parameters_by_about_163():
    assert (S.dim, S.n_layers, S.n_heads, S.n_kv_heads, S.head_dim, S.ffn_hidden, S.vocab) == (640, 80, 10, 2, 64, 2240, 128256)
    assert S.dim == S.n_heads * S.head_dim and S.q_dim == S.dim and S.tied_embeddings
    blocks = lambda c: c.n_layers * c.per_layer_params()
    assert blocks(BIG) / blocks(S) == pytest.approx(L.SCALE, rel=0.01)  # 161.9 vs 163.3
    assert blocks(S) == 80 * (640 * 640 + 2 * 640 * 128 + 640 * 640 + 3 * 640 * 2240 + 2 * 640)
    assert S.n_params() == 128256 * 640 + blocks(S) + 640  # tied: the table counted once
    assert 139 < BIG.n_params() / S.n_params() < 141  # the unscaled vocab keeps the TOTAL ratio at ~139.7
    assert S.kv_bytes_per_token(2) == 2 * 80 * 128 * 2 == 40 * 1024


def test_no_qk_norm_and_hf_layout_like_modeling_llama():
    assert not S.qk_norm and not S.rope_interleaved and S.rope_theta == 500000.0 and S.norm_eps == 1e-5 and S.sliding_window == 0


def test_weights_exceed_the_sram_so_they_must_stream_from_dram():
    rep = D.capacity_report(S, PINNED48, 1024, 1024, [1], kv_offchip={1: False}, weights_offchip=True)
    assert rep["weights_over_total_sram"] == pytest.approx(1.61, abs=0.01)
    assert rep["per_batch"][0]["need_bytes"] == rep["per_batch"][0]["kv_bytes"] and rep["all_fit_total_sram"]
    assert not D.capacity_report(S, PINNED48, 1024, 1024, [1])["all_fit_total_sram"]  # weights on the wafer: no


def test_auto_placement_keeps_batch_1_and_2_on_the_wafer_and_3_plus_in_dram():
    place = {b: D.choose_kv_placement(S, PINNED48, 1024, 1024, b, 2, "auto", True) for b in (1, 2, 3, 16, 64)}
    assert place == {1: "sram", 2: "sram", 3: "dram", 16: "dram", 64: "dram"}
    assert D.kv_carve_kib(S, PINNED48, 1024, 1024, 1) == 7 and D.kv_carve_kib(S, PINNED48, 1024, 1024, 2) == 13


def test_every_gemm_tiles_for_the_batches_we_run_with_the_kv_carved_out():
    for b in (1, 2):
        eff = PINNED48.sram_kib_per_pe - D.kv_carve_kib(S, PINNED48, 1024, 1024, b)
        w = dataclasses.replace(PINNED48, sram_kib_per_pe=eff)
        for t in D.decode_points(1024, 1024):
            assert D.step_fits(S, b, 1, t, w, 2) == []
        assert D.prefill_chunk_size(S, b, 1024, w, 2) >= 1
    for b in (16, 32, 64):
        for t in D.decode_points(1024, 1024):
            assert D.step_fits(S, b, 1, t, PINNED48, 2) == []
        assert D.prefill_chunk_size(S, b, 1024, PINNED48, 2) >= 1


def test_weight_stream_bytes_of_the_scaled_model():
    blocks = 80 * S.per_layer_params() + 640
    assert S.weight_stream_bytes(1, 2, tail=True) == (blocks + 640 + 128256 * 640) * 2
    assert S.weight_stream_bytes(1, 2, tail=True) == S.n_params() * 2 + 640 * 2  # whole stored model + 1 embedding row


def test_script_defaults():
    args = L.build_parser().parse_args([])
    assert (args.sram_kib, args.weights_dram, args.kv_placement, args.max_tile_k_candidates) == (48, True, "auto", 3)
    assert (args.in_len, args.out_len, args.batches) == (1024, 1024, "1,16,32,64")


def test_numeric_check_against_the_hf_layout_reference_passes(capsys):
    L.numeric_check()
    out = capsys.readouterr().out
    assert "numeric check PASSED" in out and "toy-llama-hf-gqa" in out


def test_dry_run_end_to_end(capsys):
    args = L.build_parser().parse_args(["--dry-run", "--skip-numeric", "--batches", "1,3,16", "--modes", "optimistic,conservative"])
    D.run_model(args, S, "llama33_70b_s163", {}, lambda: None, note=L.NOTE)
    out = capsys.readouterr().out
    assert "weights: in DRAM" in out and "b1=SRAM (-7 KiB/PE for tiles), b3=DRAM, b16=DRAM" in out
    assert "DRY RUN finished" in out
