"""Tests for experiment_qwen3_32b_dram_kv/qwen3_32b.py and the pieces it added to the LLaMA script:
an independent head_dim, per-head q/k RMSNorm, HF-style (half-split) RoPE, the off-chip DRAM KV cache model, the
SRAM override and the tile_k candidate cap."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiment_dense_llm_throughput"))
import dense_llm_throughput as D  # noqa: E402

_spec = importlib.util.spec_from_file_location("qwen3_32b_script", ROOT / "experiment_qwen3_32b_dram_kv/qwen3_32b.py")
Q = importlib.util.module_from_spec(_spec)
sys.modules["qwen3_32b_script"] = Q
_spec.loader.exec_module(Q)

from wafersim.specs import WaferSpec  # noqa: E402

PINNED48 = dataclasses.replace(WaferSpec(), **{**D.PINNED_WAFER, "sram_kib_per_pe": 48})
TOYS = [Q.TOY_QWEN3, Q.TOY_QWEN3_MHA]


# ------------------------------------------------------------------ the model


def test_qwen3_32b_matches_the_published_config_and_parameter_count():
    c = Q.QWEN3_32B
    assert (c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.ffn_hidden, c.vocab) == (5120, 64, 64, 8, 128, 25600, 151936)
    assert c.q_dim == 8192 != c.dim  # head_dim is NOT dim // n_heads (= 80)
    assert c.kv_dim == 1024 and c.q_mult == 8 and c.sliding_window == 0
    assert c.qk_norm and not c.rope_interleaved
    assert c.n_params() == 32_762_123_264  # the published 32.8 B (untied embedding + LM head)
    assert c.kv_bytes_per_token(2) == 2 * 64 * 8 * 128 * 2 == 262_144


def test_llama_configs_are_unchanged_by_the_new_fields():
    for c in (D.LLAMA2_7B, D.LLAMA3_8B):
        assert c.q_dim == c.dim and c.head_dim == c.dim // c.n_heads and not c.qk_norm and c.rope_interleaved
    assert abs(D.LLAMA2_7B.n_params() - 6.738415616e9) < 1


def _setup(cfg, seed=0):
    rng = np.random.default_rng(seed)
    ref = D.random_reference_weights(cfg, rng, 0.2)
    return rng, ref, D.ModelWeights.from_reference(ref, cfg)


@pytest.mark.parametrize("cfg", TOYS, ids=lambda c: c.name)
@pytest.mark.parametrize("b", [1, 2])
def test_forward_matches_a_numpy_transcription_of_hf_qwen3(cfg, b):
    rng, ref, w = _setup(cfg, seed=b)
    pre, steps = 6, 4
    toks = rng.integers(0, cfg.vocab, size=(b, pre + steps))
    rc, caches = D.new_reference_cache(cfg, b, pre + steps), D.new_caches(cfg, b)
    got = D.model_forward(toks[:, :pre], w, cfg, caches, None).data
    assert np.abs(got - D.reference_forward(toks[:, :pre], 0, ref, cfg, rc)[:, -1]).max() < 1e-4
    for i in range(steps):
        pos = pre + i
        got = D.model_forward(toks[:, pos:pos + 1], w, cfg, caches, None).data
        assert np.abs(got - D.reference_forward(toks[:, pos:pos + 1], pos, ref, cfg, rc)[:, -1]).max() < 1e-4


def test_q_and_k_norm_and_the_rope_pairing_actually_matter():
    cfg = Q.TOY_QWEN3
    rng, ref, w = _setup(cfg)
    toks = rng.integers(0, cfg.vocab, size=(1, 6))
    base = D.reference_forward(toks, 0, ref, cfg, D.new_reference_cache(cfg, 1, 6))
    no_norm = D.reference_forward(toks, 0, ref, dataclasses.replace(cfg, qk_norm=False), D.new_reference_cache(cfg, 1, 6))
    adjacent = D.reference_forward(toks, 0, ref, dataclasses.replace(cfg, rope_interleaved=True), D.new_reference_cache(cfg, 1, 6))
    assert np.abs(base - no_norm).max() > 1e-3
    assert np.abs(base - adjacent).max() > 1e-3
    with pytest.raises(ValueError):  # q/k norm weights would need the de-interleave too
        D.ModelWeights.from_reference(ref, dataclasses.replace(cfg, rope_interleaved=True))


def test_chunked_prefill_equals_one_shot_with_qk_norm():
    cfg = Q.TOY_QWEN3
    rng, ref, w = _setup(cfg)
    toks = rng.integers(0, cfg.vocab, size=(2, 8))
    one = D.model_forward(toks, w, cfg, D.new_caches(cfg, 2), None).data
    caches = D.new_caches(cfg, 2)
    for a in range(0, 8, 4):
        part = D.model_forward(toks[:, a:a + 4], w, cfg, caches, None).data
    assert np.abs(part - one).max() < 1e-5


def _numeric_log(cfg, b, s, t_old):
    rng, ref, w = _setup(cfg)
    toks = rng.integers(0, cfg.vocab, size=(b, t_old + s))
    caches = D.new_caches(cfg, b)
    if t_old:
        D.model_forward(toks[:, :t_old], w, cfg, caches, None)
    cost = D.DenseCostTracker(dry_run=True)
    D.model_forward(toks[:, t_old:], w, cfg, caches, cost)
    return [(e["op"], e["key"]) for e in cost.log]


def _emitted_log(cfg, b, s, t_old):
    cost = D.DenseCostTracker(dry_run=True)
    D.emit_step(cost, cfg, b, s, t_old, 4, D.SectionRecorder(cost), layers=cfg.n_layers, expand=True, sampling=False)
    return [(e["op"], e["key"]) for e in cost.log]


@pytest.mark.parametrize("cfg", TOYS, ids=lambda c: c.name)
@pytest.mark.parametrize("b,s,t_old", [(1, 1, 5), (2, 1, 7), (1, 4, 0), (2, 3, 5)])
def test_cost_only_path_emits_the_same_ops_as_the_numeric_path(cfg, b, s, t_old):
    assert _numeric_log(cfg, b, s, t_old) == _emitted_log(cfg, b, s, t_old)


def test_the_qk_norm_ops_are_really_in_the_cost_path():
    cost = D.DenseCostTracker(dry_run=True)
    D.emit_step(cost, Q.TOY_QWEN3, 1, 1, 5, 4, D.SectionRecorder(cost), layers=1, expand=False)
    plain = D.DenseCostTracker(dry_run=True)
    D.emit_step(plain, dataclasses.replace(Q.TOY_QWEN3, qk_norm=False), 1, 1, 5, 4, D.SectionRecorder(plain), layers=1)
    # two extra RMSNorms (7 element-wise ops each) + two extra shape-only reshapes (q and k are viewed as (m*heads, d) and back)
    assert len(cost.log) == len(plain.log) + 2 * 7 + 2
    gemms = [e["key"][1:4] for e in cost.log if e["op"] == "gemm"]
    cfg = Q.TOY_QWEN3
    assert (1, cfg.q_dim, cfg.dim) in gemms and (1, cfg.dim, cfg.q_dim) in gemms  # Q is 32 -> 48, o_proj is 48 -> 32


# ------------------------------------------------- tileability at 12,800 PEs x 48 KiB


@pytest.mark.parametrize("b", [1, 16, 32, 64])
def test_every_costed_qwen3_32b_step_is_tileable_on_the_48kib_wafer(b):
    cfg, n_in = Q.QWEN3_32B, 1024
    c = D.prefill_chunk_size(cfg, b, n_in, PINNED48, 2)
    starts = D.chunk_starts(n_in, c)
    for t0 in starts:
        assert D.step_fits(cfg, b, c, t0, PINNED48, 2, tail=(t0 == starts[-1])) == []
    for t0 in D.decode_points(n_in, 1024):
        assert D.step_fits(cfg, b, 1, t0, PINNED48, 2) == []


def test_48kib_lets_the_prefill_run_in_far_fewer_passes_than_13kib():
    roomy = D.prefill_chunk_size(Q.QWEN3_32B, 1, 1024, PINNED48, 2)
    tight = D.prefill_chunk_size(Q.QWEN3_32B, 1, 1024, dataclasses.replace(PINNED48, sram_kib_per_pe=13), 2)
    assert roomy == 1024 and tight < roomy


def test_sram_override_changes_only_the_sram():
    class A:  # argparse stand-in
        sram_kib = 48
    w = dataclasses.replace(WaferSpec(), **D.PINNED_WAFER)
    w2 = D.apply_wafer_overrides(w, A())
    assert (w2.mesh_rows, w2.mesh_cols, w2.sram_kib_per_pe) == (10, 1280, 48) and w.sram_kib_per_pe == 13

    class B:
        sram_kib = 0
    assert D.apply_wafer_overrides(w, B()) is w


# ------------------------------------------------------------- the DRAM KV model


def _off(bw=1000.0, lat=100.0, pj=40.0, clock=1.0, cap=0.0):
    return D.OffchipKV.make(bw, lat, pj, clock, cap)


def test_traffic_is_hand_computed_for_qwen3_32b():
    per_token = 2 * 1024 * 2  # K and V of one layer, one token, fp16
    read, write = D.OffchipKV.traffic(Q.QWEN3_32B, 4, 1, 1000, 2)
    assert read == 64 * 4 * 1000 * per_token and write == 64 * 4 * 1 * per_token
    read, write = D.OffchipKV.traffic(Q.QWEN3_32B, 2, 512, 0, 2)  # first prefill chunk: nothing to read yet
    assert read == 0 and write == 64 * 2 * 512 * per_token


def test_traffic_respects_a_sliding_window():
    cfg = dataclasses.replace(Q.QWEN3_32B, sliding_window=100)
    read, _ = D.OffchipKV.traffic(cfg, 1, 1, 1000, 2)
    assert read == 64 * 99 * 2 * 1024 * 2  # window - 1 past tokens (the new one is on the wafer)


def test_time_and_energy_follow_bandwidth_latency_and_pj_per_byte():
    off = _off(bw=500.0, lat=100.0, pj=40.0, clock=2.0)
    assert off.bytes_per_cycle == 250.0 and off.latency_cycles == 200.0
    read, write = D.OffchipKV.traffic(Q.QWEN3_32B, 1, 1, 1023, 2)
    cycles, uj = off.step(Q.QWEN3_32B, 1, 1, 1023, 2)
    assert cycles == pytest.approx((read + write) / 250.0 + 64 * 200.0)
    assert uj == pytest.approx((read + write) * 40.0 * 1e-6)
    assert _off(bw=2000.0).step(Q.QWEN3_32B, 1, 1, 1023, 2)[0] < _off(bw=1000.0).step(Q.QWEN3_32B, 1, 1, 1023, 2)[0]
    with pytest.raises(ValueError):
        D.OffchipKV.make(0.0, 100.0, 40.0, 1.0)


def test_run_step_overlaps_or_adds_the_dram_time(monkeypatch):
    fixed = D.Section(cycles=1000.0, energy_uj=5.0)
    monkeypatch.setattr(D, "step_totals", lambda secs, cfg, b, concurrent: D.combine([(fixed, 1, 1)]))
    cfg, off = Q.TOY_QWEN3, _off(bw=1.0, lat=0.0)  # 1 byte/cycle: DRAM time == bytes
    dram_cycles, dram_uj = off.step(cfg, 1, 1, 50, 4)
    assert dram_cycles > 1000  # slower than compute here
    cost = D.DenseCostTracker(dry_run=True)
    over = D.run_step(cfg, 1, 1, 50, cost, 4, True, offchip=off, kv_overlap=True)
    assert over.cycles == pytest.approx(dram_cycles)  # max(compute, DRAM)
    assert over.by_op["kv_dram"][0] == pytest.approx(dram_cycles - 1000.0)  # only the EXPOSED part is attributed
    added = D.run_step(cfg, 1, 1, 50, cost, 4, True, offchip=off, kv_overlap=False)
    assert added.cycles == pytest.approx(1000.0 + dram_cycles)
    assert added.energy_uj == pytest.approx(5.0 + dram_uj)
    # when compute dominates, overlap hides the DRAM completely
    fast = D.run_step(cfg, 1, 1, 50, cost, 4, True, offchip=_off(bw=1e9, lat=0.0), kv_overlap=True)
    assert fast.cycles == pytest.approx(1000.0)
    # no off-chip model => nothing changes
    assert D.run_step(cfg, 1, 1, 50, cost, 4, True).cycles == pytest.approx(1000.0)


def test_modes_overlap_in_optimistic_and_serialise_in_conservative():
    assert {m.name: m.kv_overlap for m in D.MODES} == {"optimistic": True, "conservative": False}


def test_capacity_report_with_the_kv_in_dram():
    rep = D.capacity_report(Q.QWEN3_32B, PINNED48, 1024, 1024, [1, 64], kv_offchip=True, dram_capacity_bytes=20 * 2 ** 30)
    assert rep["total_sram_bytes"] == 12800 * 48 * 1024 and rep["data_sram_bytes"] == 12800 * (48 * 1024 - 10260)
    for r in rep["per_batch"]:
        assert r["need_bytes"] == rep["weights_bytes"]  # the KV is not on the wafer
    b1, b64 = rep["per_batch"]
    assert b1["kv_in_dram_bytes"] == 2047 * 262144 and b1["fits_dram"]
    assert b64["kv_in_dram_bytes"] == 64 * 2047 * 262144 and not b64["fits_dram"]  # 31.98 GiB > 20 GiB
    on = D.capacity_report(Q.QWEN3_32B, PINNED48, 1024, 1024, [1], kv_offchip=False)
    assert on["per_batch"][0]["need_bytes"] == on["weights_bytes"] + on["per_batch"][0]["kv_bytes"]


# ------------------------------------------------------- tile_k candidate cap


def test_max_candidates_keeps_the_largest_tile_ks_and_none_keeps_all(monkeypatch, tmp_path):
    from wafersim import search

    seen = []

    class _R:
        sim_time_cycles, energy = 1.0, type("E", (), {"total_uj": 1.0})()

    monkeypatch.setattr(search, "feasible_tile_k_candidates", lambda g, w, tm, tn: [64, 32, 16, 8, 4, 2, 1])
    monkeypatch.setattr(search, "generate_timeloop_inputs", lambda g, w, d, tile_k: seen.append(tile_k) or tile_k)
    monkeypatch.setattr(search, "run_timeloop_model", lambda a, image=None: None)
    monkeypatch.setattr(search, "parse_mapping_contract", lambda a, g, w, mode=None: a)
    monkeypatch.setattr(search, "simulate_gemm", lambda c: _R())
    gemm, wafer = D.GemmSpec(m=8, n=8, k=64, bytes_per_element=2), PINNED48
    assert len(search.search_tile_k(gemm, wafer, tmp_path)) == 7
    seen.clear()
    out = search.search_tile_k(gemm, wafer, tmp_path, max_candidates=3)
    assert seen == [64, 32, 16] and [c.tile_k for c in out] == [64, 32, 16]
    with pytest.raises(ValueError):
        search.search_tile_k(gemm, wafer, tmp_path, max_candidates=0)


def test_the_cost_tracker_passes_its_cap_to_the_search(monkeypatch, tmp_path):
    got = {}

    def fake_search(spec, wafer, d, max_candidates=None):
        got["cap"] = max_candidates
        return [type("C", (), {"result": type("R", (), {"sim_time_cycles": 7.0, "energy": type("E", (), {"total_uj": 2.0})()})()})()]

    monkeypatch.setattr(D.moe, "search_tile_k", fake_search)
    monkeypatch.setattr(D.moe, "best_by_sim_time", lambda c: c[0])
    cost = D.DenseCostTracker(wafer=PINNED48, out_dir=tmp_path, tiling=D.Tiling(), max_tile_k_candidates=3)
    cost.op_gemm(8, 8, 64, 2)
    assert got["cap"] == 3 and cost.log[-1]["op"] == "gemm" and cost.log[-1]["cycles"] == 7.0


# ------------------------------------------------------------------ the script


def test_script_defaults_to_48kib_dram_kv_and_a_candidate_cap():
    a = Q.build_parser().parse_args([])
    assert (a.sram_kib, a.kv_dram, a.max_tile_k_candidates, a.in_len, a.out_len, a.model) == (48, True, 3, 1024, 1024, "qwen3-32b")
    b = Q.build_parser().parse_args(["--kv-on-chip", "--sram-kib", "13", "--dram-bw-gbs", "2000"])
    assert (b.sram_kib, b.kv_dram, b.dram_bw_gbs) == (13, False, 2000.0)


def test_qwen3_32b_dry_run_end_to_end(capsys):
    args = Q.build_parser().parse_args(["--dry-run", "--skip-numeric", "--batches", "1,64", "--modes", "optimistic,conservative"])
    D.run_model(args, Q.QWEN3_32B, "qwen3_32b", {}, lambda: None, note=Q.NOTE)
    out = capsys.readouterr().out
    assert "DRAM: 1000 GB/s" in out and "10x1280 PEs, 48 KiB/PE" in out
    assert "fits DRAM?" in out and "weights are NOT charged" in out
    assert "DRY RUN finished" in out and "prefill 1 pass(es) of 1024 tokens" in out
