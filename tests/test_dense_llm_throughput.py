"""Tests for experiment_dense_llm_throughput/dense_llm_throughput.py (LLaMA-2-7B).

Three kinds of guarantee:
  1. NUMERIC: the stacked-batch + KV-cache forward equals a literal numpy
     transcription of Meta's llama/model.py (complex-number RoPE, cache arrays,
     repeat_kv), MHA and GQA, batch > 1, prefill / chunked prefill / decode.
  2. SYNC: the numeric path and the cost-only path emit IDENTICAL op logs (dry-run
     tracker), so the real-dim cost pass cannot drift from the verified numerics.
  3. ACCOUNTING: batch/instance multipliers, data-move chunking, throughput
     arithmetic, Simpson averaging -- checked against hand-computed values.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_dense_llm_throughput"))
import dense_llm_throughput as D  # noqa: E402
from wafersim.tiling import Tiling  # noqa: E402

TOY_WINDOW = dataclasses.replace(D.TOY_GQA, name="toy-window", sliding_window=3)
NUMERIC_CFGS = [D.TOY_MHA, D.TOY_GQA, D.TOY_GQA_L3]


def _setup(cfg, b, seed=0, scale=0.2):
    rng = np.random.default_rng(seed)
    ref = D.random_reference_weights(cfg, rng, scale)
    return rng, ref, D.ModelWeights.from_reference(ref, cfg)


# ---------------------------------------------------------------- 1. numeric


@pytest.mark.parametrize("cfg", NUMERIC_CFGS, ids=lambda c: c.name)
@pytest.mark.parametrize("b", [1, 2])
def test_prefill_then_decode_matches_reference(cfg, b):
    rng, ref, w = _setup(cfg, b)
    pre, steps = 6, 4
    toks = rng.integers(0, cfg.vocab, size=(b, pre + steps))
    rc, caches = D.new_reference_cache(cfg, b, pre + steps), D.new_caches(cfg, b)
    got = D.model_forward(toks[:, :pre], w, cfg, caches, None).data
    np.testing.assert_allclose(got, D.reference_forward(toks[:, :pre], 0, ref, cfg, rc)[:, -1], rtol=1e-4, atol=1e-5)
    for i in range(steps):
        pos = pre + i
        got = D.model_forward(toks[:, pos:pos + 1], w, cfg, caches, None).data
        np.testing.assert_allclose(got, D.reference_forward(toks[:, pos:pos + 1], pos, ref, cfg, rc)[:, -1],
                                   rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("cfg", NUMERIC_CFGS, ids=lambda c: c.name)
def test_chunked_prefill_matches_reference(cfg):
    rng, ref, w = _setup(cfg, 2)
    toks = rng.integers(0, cfg.vocab, size=(2, 9))
    rc, caches = D.new_reference_cache(cfg, 2, 9), D.new_caches(cfg, 2)
    D.model_forward(toks[:, :4], w, cfg, caches, None)
    got = D.model_forward(toks[:, 4:9], w, cfg, caches, None).data  # 5 new tokens on a 4-token cache
    full = D.reference_forward(toks, 0, ref, cfg, rc)
    np.testing.assert_allclose(got, full[:, -1], rtol=1e-4, atol=1e-5)


def test_logits_depend_on_history_and_batch_rows_are_independent():
    cfg = D.TOY_GQA
    rng, ref, w = _setup(cfg, 2)
    toks = rng.integers(0, cfg.vocab, size=(2, 5))
    both = D.model_forward(toks, w, cfg, D.new_caches(cfg, 2), None).data
    alone = D.model_forward(toks[1:2], w, cfg, D.new_caches(cfg, 1), None).data
    np.testing.assert_allclose(both[1], alone[0], rtol=1e-5, atol=1e-6)
    changed = toks.copy()
    changed[1, 0] = (changed[1, 0] + 1) % cfg.vocab  # change an EARLY token of sequence 1
    out2 = D.model_forward(changed, w, cfg, D.new_caches(cfg, 2), None).data
    assert not np.allclose(out2[1], both[1])  # history matters
    np.testing.assert_allclose(out2[0], both[0], rtol=1e-6, atol=1e-7)  # other sequence untouched


def test_adjacent_pair_rope_equals_half_split_after_column_permutation():
    """The reason wq/wk columns are de-interleaved: reference rotation == ours after permuting."""
    rng = np.random.default_rng(3)
    nh, d, s = 3, 8, 5
    x = rng.standard_normal((1, s, nh, d)).astype(np.float32)
    fc = D._ref_freqs_cis(d, s, 10000.0)
    expect = D._ref_rope(x, fc)
    perm = D.deinterleave_perm(nh, d)
    xp = x.reshape(1, s, nh * d)[..., perm].reshape(1, s, nh, d)
    cos, sin = D.rope_tables(np.arange(s), d, 10000.0)
    x1, x2 = xp[..., : d // 2], xp[..., d // 2:]
    c, sn = cos[None, :, None, :], sin[None, :, None, :]
    rot = np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], axis=-1)
    back = np.empty_like(rot).reshape(1, s, nh * d)
    back[..., perm] = rot.reshape(1, s, nh * d)
    np.testing.assert_allclose(back.reshape(1, s, nh, d), expect, rtol=1e-5, atol=1e-6)


def test_deinterleave_perm_is_a_permutation_per_head():
    p = D.deinterleave_perm(3, 8)
    assert sorted(p.tolist()) == list(range(24))
    assert p[:8].tolist() == [0, 2, 4, 6, 1, 3, 5, 7] and p[8:16].tolist() == [8, 10, 12, 14, 9, 11, 13, 15]


def test_config_arithmetic_matches_the_reference_formulas():
    assert D.LLAMA2_7B.ffn_hidden == 11008
    assert D.llama_ffn_hidden(4096, 1024, 1.3) == 14336  # LLaMA-3-8B params
    assert D.llama_ffn_hidden(8192, 4096, 1.3) == 28672  # LLaMA-2-70B params
    assert D.LLAMA2_7B.head_dim == 128 and D.LLAMA2_7B.q_mult == 1 and D.LLAMA2_7B.kv_dim == 4096
    assert abs(D.LLAMA2_7B.n_params() - 6.738415616e9) < 1  # the published 6.74 B


# ------------------------------------------------------------------- 2. sync


def _numeric_log(cfg, b, s, t_old):
    rng, ref, w = _setup(cfg, b)
    toks = rng.integers(0, cfg.vocab, size=(b, t_old + s))
    caches = D.new_caches(cfg, b)
    if t_old:
        D.model_forward(toks[:, :t_old], w, cfg, caches, None)  # fill caches, no cost
    cost = D.DenseCostTracker(dry_run=True)
    D.model_forward(toks[:, t_old:], w, cfg, caches, cost)
    return [(e["op"], e["key"]) for e in cost.log]


def _emitted_log(cfg, b, s, t_old, *, tracker_kw=None):
    cost = D.DenseCostTracker(dry_run=True, **(tracker_kw or {}))
    D.emit_step(cost, cfg, b, s, t_old, 4, D.SectionRecorder(cost), layers=cfg.n_layers, expand=True, sampling=False)
    return [(e["op"], e["key"]) for e in cost.log]


@pytest.mark.parametrize("cfg", [D.TOY_MHA, D.TOY_GQA, D.TOY_GQA_L3, TOY_WINDOW], ids=lambda c: c.name)
@pytest.mark.parametrize("b,s,t_old", [(1, 4, 0), (2, 4, 0), (1, 1, 5), (2, 1, 5), (2, 3, 6), (3, 1, 0)])
def test_cost_only_path_emits_exactly_what_the_numeric_path_does(cfg, b, s, t_old):
    assert _emitted_log(cfg, b, s, t_old) == _numeric_log(cfg, b, s, t_old)


def test_sync_test_is_not_vacuous():
    log = _numeric_log(D.TOY_GQA, 2, 3, 6)
    ops = {op for op, _ in log}
    assert {"gemm", "mul", "add", "sub", "exp", "div", "sum", "max", "rsqrt", "silu", "concat", "gather", "reshape"} <= ops
    assert len(log) > 100


@pytest.mark.parametrize("s,t_old,window", [(1, 0, 0), (1, 9, 0), (4, 0, 0), (3, 6, 0), (1, 9, 4), (5, 2, 3), (1, 2, 3)])
def test_decode_needs_no_mask_prefill_always_does(s, t_old, window):
    t = t_old + s
    key_start = max(0, t_old - window + 1) if window else 0
    needs = D.build_mask(s, 1, t_old, key_start, t - key_start, window) is not None
    assert needs == (s > 1)


@pytest.mark.parametrize("s", [1, 3])
@pytest.mark.parametrize("nkv,qm", [(1, 1), (1, 4), (2, 1), (2, 4), (4, 2)])
def test_group_reshape_copy_rule_matches_real_numpy(s, nkv, qm):
    nh, d = nkv * qm, 4
    q = np.zeros((s, nh, d))
    g = 0 if nkv == 1 else nkv - 1  # any group; contiguity is the same for all
    sl = q[:, g * qm:(g + 1) * qm, :]
    out = sl.reshape(s * qm, d)
    assert (not np.shares_memory(out, sl)) == D.q_group_reshape_copied(s, qm, nkv)


# ------------------------------------------------------------ 3. accounting


def _sec(cycles, energy, **by_op):
    sec = D.Section(cycles, energy)
    for op, (c, e) in by_op.items():
        sec.by_op[op] = [c, e]
    return sec


def test_instance_multipliers_serial_vs_concurrent():
    cfg = dataclasses.replace(D.TOY_GQA, n_layers=3)  # 2 KV heads
    secs = {"embed": _sec(10, 1), "tail": _sec(20, 2), "stacked": _sec(100, 5), "core": _sec(7, 0.5), "perseq": _sec(3, 0.25)}
    b, L, inst = 4, 3, 4 * 2
    serial = D.step_totals(secs, cfg, b, concurrent=False)
    conc = D.step_totals(secs, cfg, b, concurrent=True)
    assert serial.cycles == 10 + 20 + L * 100 + L * inst * 7 + L * b * 3
    assert conc.cycles == 10 + 20 + L * 100 + L * 7 + L * 3
    expect_energy = 1 + 2 + L * 5 + L * inst * 0.5 + L * b * 0.25  # every instance burns energy in BOTH modes
    assert serial.energy_uj == pytest.approx(expect_energy) and conc.energy_uj == pytest.approx(expect_energy)


def test_by_op_split_sums_to_the_totals():
    cfg = dataclasses.replace(D.TOY_MHA, n_layers=2)
    secs = {"stacked": _sec(100, 4, gemm=(80, 3), mul=(20, 1)), "core": _sec(10, 1, gemm=(6, 0.6), exp=(4, 0.4))}
    tot = D.step_totals(secs, cfg, 2, concurrent=False)
    assert sum(c for c, _ in tot.by_op.values()) == pytest.approx(tot.cycles)
    assert sum(e for _, e in tot.by_op.values()) == pytest.approx(tot.energy_uj)


def test_concurrent_is_never_slower_than_serial():
    cfg = D.TOY_GQA
    secs = {"stacked": _sec(100, 1), "core": _sec(9, 1), "perseq": _sec(2, 1)}
    for b in (1, 3, 16):
        assert D.step_totals(secs, cfg, b, True).cycles <= D.step_totals(secs, cfg, b, False).cycles


def test_metrics_follow_the_paper_definition():
    m = D.metrics(_sec(2e9, 1e6), _sec(1e6, 1e3), b=4, n_in=1024, n_out=1024, clock_ghz=1.0)
    assert m["ttft_s"] == pytest.approx(2.0) and m["itl_s"] == pytest.approx(1e-3)
    e2e = 2.0 + 1023 * 1e-3
    assert m["e2e_s"] == pytest.approx(e2e)
    assert m["paper_throughput_tok_s"] == pytest.approx(4 * 2048 / e2e)  # batch*(in+out)/latency
    assert m["output_throughput_tok_s"] == pytest.approx(4 * 1024 / e2e)
    assert m["decode_throughput_tok_s"] == pytest.approx(4 / 1e-3)
    assert m["energy_j"] == pytest.approx((1e6 + 1023 * 1e3) * 1e-6)
    m2 = D.metrics(_sec(2e9, 0), _sec(1e6, 0), 4, 1024, 1024, clock_ghz=2.0)  # twice the clock -> half the time
    assert m2["e2e_s"] == pytest.approx(e2e / 2)


def test_decode_points_and_simpson_are_exact_for_linear_cost():
    a, mid, z = D.decode_points(1024, 1024)
    assert (a, mid, z) == (1024, 1535, 2046)  # n_out-1 = 1023 decode steps, cache 1024..2046
    f = lambda T: _sec(5.0 + 3.0 * T, 2.0 * T)
    mean = D.simpson_mean(f(a), f(mid), f(z))
    steps = np.arange(a, z + 1)
    assert mean.cycles == pytest.approx(np.mean(5.0 + 3.0 * steps))  # mean over the actual steps
    assert mean.energy_uj == pytest.approx(np.mean(2.0 * steps))


def test_move_chunking_serial_and_mesh():
    ser = D.DenseCostTracker(dry_run=True, move_tiling=Tiling.serial())
    n = 4096 * 4096
    ser.op_concat(D.phantom((4096, 4096), 2))
    e = ser.log[-1]
    assert e["cycle_repeats"] == math.ceil(n / D.MAX_MOVE_ELEMS) == 16
    assert e["key"][1] <= D.MAX_MOVE_ELEMS and e["key"][1] * e["cycle_repeats"] >= n
    assert e["energy_repeats"] == pytest.approx(16.0)  # all the work is on one PE: energy = chunks x chunk energy
    mesh = D.DenseCostTracker(dry_run=True, move_tiling=Tiling())
    big = 65536 * 4096
    mesh.op_gather(D.phantom((65536, 4096), 2))
    e = mesh.log[-1]
    per_pe = math.ceil(big / mesh.total_pes)
    assert e["cycle_repeats"] == 1 and e["key"][1] == per_pe
    assert e["energy_repeats"] == pytest.approx(big / per_pe)  # total energy conserved across PEs


def test_zero_element_move_is_a_free_view_entry():
    c = D.DenseCostTracker(dry_run=True)
    c.op_reshape_flag(0, 2, False)
    e = c.log[-1]
    assert e["op"] == "reshape" and e["key"][1] == 0 and e["cycle_repeats"] == 1


def test_phantom_tensor_costs_no_memory():
    t = D.phantom((1 << 20, 1 << 12), 2)
    assert t.size == 1 << 32 and t.bytes_per_element == 2
    assert D.phantom((4, 4), 4).bytes_per_element == 4


def test_real_dim_decode_step_has_the_expected_gemms():
    cfg = D.LLAMA2_7B
    cost = D.DenseCostTracker(dry_run=True)
    rec = D.SectionRecorder(cost)
    D.emit_step(cost, cfg, 16, 1, 1024, 2, rec, layers=1, expand=False, sampling=True)
    gemms = [e["key"] for e in cost.log if e["op"] == "gemm"]
    H, F, V = cfg.dim, cfg.ffn_hidden, cfg.vocab
    assert gemms.count(("gemm", 16, H, H, 2)) == 4  # wq, wk, wv, wo (MHA: kv_dim == dim)
    assert gemms.count(("gemm", 16, F, H, 2)) == 2  # w1, w3
    assert gemms.count(("gemm", 16, H, F, 2)) == 1  # w2
    assert gemms.count(("gemm", 1, 1024 + 1, 128, 2)) == 1 and gemms.count(("gemm", 1, 128, 1025, 2)) == 1  # one instance
    assert gemms.count(("gemm", 16, V, H, 2)) == 1  # LM head on 16 rows, not 16*S
    assert len(gemms) == 7 + 2 + 1


def test_real_dim_prefill_emits_for_the_largest_batch_without_error():
    cfg = D.LLAMA2_7B
    cost = D.DenseCostTracker(dry_run=True)  # serial moves: the case that needs chunking
    D.emit_step(cost, cfg, 64, 1024, 0, 2, D.SectionRecorder(cost), layers=1, expand=False, sampling=True)
    assert all(e["key"][1] <= D.MAX_MOVE_ELEMS for e in cost.log if e["op"] in ("concat", "gather", "reshape"))
    assert any(e["cycle_repeats"] > 1 for e in cost.log if e["op"] == "concat")
    rows = [e["key"] for e in cost.log if e["op"] == "gemm"]
    assert ("gemm", 64 * 1024, cfg.ffn_hidden, cfg.dim, 2) in rows
    assert ("gemm", 64, cfg.vocab, cfg.dim, 2) in rows  # LM head: last row per sequence only


def test_embedding_and_last_row_gather_only_when_needed():
    cost = D.DenseCostTracker(dry_run=True)
    D.emit_step(cost, D.TOY_MHA, 2, 1, 5, 4, D.SectionRecorder(cost), layers=0)
    assert [e["op"] for e in cost.log].count("gather") == 1  # decode: embedding only
    cost = D.DenseCostTracker(dry_run=True)
    D.emit_step(cost, D.TOY_MHA, 2, 4, 0, 4, D.SectionRecorder(cost), layers=0)
    assert [e["op"] for e in cost.log].count("gather") == 2  # prefill: embedding + last-row select


def test_sampling_adds_one_vocab_wide_max_per_sequence():
    base = D.DenseCostTracker(dry_run=True)
    D.emit_step(base, D.TOY_MHA, 3, 1, 5, 4, D.SectionRecorder(base), layers=0, sampling=False)
    samp = D.DenseCostTracker(dry_run=True)
    D.emit_step(samp, D.TOY_MHA, 3, 1, 5, 4, D.SectionRecorder(samp), layers=0, sampling=True)
    extra = samp.log[len(base.log):]
    assert [(e["op"], e["key"]) for e in extra] == [("max", ("max", D.TOY_MHA.vocab, 4))]
    assert extra[0]["cycle_repeats"] == 3 or extra[0]["lanes"] == 3


def test_simpson_is_exact_for_a_quadratic_cost_and_distinguishes_equal_weights():
    a, mid, z = D.decode_points(1024, 1024)
    f = lambda T: _sec(float(T) ** 2, 0.0)
    exact_mean = (z ** 3 - a ** 3) / (3 * (z - a))  # continuous average of T^2 over [a, z]
    assert D.simpson_mean(f(a), f(mid), f(z)).cycles == pytest.approx(exact_mean, rel=1e-12)
    assert (a ** 2 + mid ** 2 + z ** 2) / 3 != pytest.approx(exact_mean, rel=1e-6)  # the wrong weights would fail


# ----------------------------------------------------------------- LLaMA-3-8B


def test_llama3_8b_config_matches_the_published_architecture():
    c = D.LLAMA3_8B
    assert (c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab) == (4096, 32, 32, 8, 128256)
    assert c.ffn_hidden == 14336 and c.rope_theta == 500000.0 and c.norm_eps == 1e-5
    assert c.head_dim == 128 and c.q_mult == 4 and c.kv_dim == 1024
    assert abs(c.n_params() - 8.030261248e9) < 1  # the published 8.03 B
    assert c.kv_bytes_per_token(2) == 2 * 32 * 1024 * 2  # 4x smaller KV per token than LLaMA-2-7B (MHA)
    assert D.LLAMA2_7B.kv_bytes_per_token(2) == 4 * c.kv_bytes_per_token(2)


def test_model_registry_and_paper_reference_values():
    assert set(D.MODELS) == {"llama2-7b", "llama3-8b"}
    for key, (cfg, slug, paper) in D.MODELS.items():
        assert slug in {"llama2_7b", "llama3_8b"} and len(paper) == 7 and all(v > 0 for v in paper.values())
    assert D.MODELS["llama3-8b"][0] is D.LLAMA3_8B and D.MODELS["llama2-7b"][0] is D.LLAMA2_7B
    assert D.MODELS["llama3-8b"][2]["8x SN40L (SambaFlow)"] > D.MODELS["llama2-7b"][2]["8x SN40L (SambaFlow)"]


def test_rope_theta_is_used_not_hardcoded():
    """toy-gqa and toy-gqa-l3 differ ONLY in theta; their logits must differ, and each matches its own reference."""
    rng = np.random.default_rng(5)
    toks = rng.integers(0, 50, size=(1, 6))
    outs = []
    for cfg in (D.TOY_GQA, D.TOY_GQA_L3):
        ref = D.random_reference_weights(cfg, np.random.default_rng(9))
        w = D.ModelWeights.from_reference(ref, cfg)
        outs.append(D.model_forward(toks, w, cfg, D.new_caches(cfg, 1), None).data)
    assert not np.allclose(outs[0], outs[1], atol=1e-3)


def test_llama3_real_dim_decode_gemm_shapes_are_grouped_query():
    cfg = D.LLAMA3_8B
    cost = D.DenseCostTracker(dry_run=True)
    D.emit_step(cost, cfg, 32, 1, 1024, 2, D.SectionRecorder(cost), layers=1, expand=False, sampling=True)
    g = [e["key"] for e in cost.log if e["op"] == "gemm"]
    H, kv, F, V = cfg.dim, cfg.kv_dim, cfg.ffn_hidden, cfg.vocab
    assert g.count(("gemm", 32, H, H, 2)) == 2  # wq, wo
    assert g.count(("gemm", 32, kv, H, 2)) == 2  # wk, wv are 4x narrower (8 KV heads)
    assert g.count(("gemm", 4, 1025, 128, 2)) == 1 and g.count(("gemm", 4, 128, 1025, 2)) == 1  # q_mult=4 rows per group
    assert g.count(("gemm", 32, F, H, 2)) == 2 and g.count(("gemm", 32, H, F, 2)) == 1
    assert g.count(("gemm", 32, V, H, 2)) == 1


def test_llama3_prefill_copies_each_query_group_but_decode_does_not():
    cfg = D.LLAMA3_8B
    pre, dec = D.DenseCostTracker(dry_run=True), D.DenseCostTracker(dry_run=True)
    D.emit_core(pre, cfg, 1024, 1024, True, 2, D.SectionRecorder(pre))
    D.emit_core(dec, cfg, 1, 1025, False, 2, D.SectionRecorder(dec))
    copied = lambda c: [e["key"] for e in c.log if e["op"] == "reshape" and e["key"][3]]
    assert copied(pre) == [("reshape", 1024 * 4 * 128, 2, True)]  # (s*q_mult, d) rows copied once per instance
    assert copied(dec) == []


# -------------------------------------------------------------- wafer / machine


def test_wafer_is_derived_from_the_gptoss_geometry_not_the_models_own_gemm():
    """Deriving with (10,4096,4096) at 2 bytes made a 40,960-PE mesh with no feasible SRAM; the machine is pinned."""
    assert D.REP_GEMM == (10, 5120, 2880) and D.WAFER_DERIVE_BPE == 4


def test_make_wafer_dry_run_does_not_derive(tmp_path):
    calls = []
    w = D.make_wafer(True, tmp_path, derive=lambda *a: calls.append(a))
    assert calls == [] and (w.mesh_rows, w.mesh_cols) == (32, 32) and not (tmp_path / "wafer_derived.json").exists()


def _stub_derive(calls):
    from types import SimpleNamespace
    from wafersim.primitives import WaferSpec

    def derive(budget, rep, base, out_dir):
        calls.append((budget, rep, out_dir))
        return SimpleNamespace(wafer=WaferSpec(mesh_rows=10, mesh_cols=1280, sram_kib_per_pe=13, fifo_capacity_packets=5))
    return derive


def test_make_wafer_derives_once_then_reuses_the_cache(tmp_path):
    calls = []
    w1 = D.make_wafer(False, tmp_path, derive=_stub_derive(calls))
    w2 = D.make_wafer(False, tmp_path, derive=_stub_derive(calls))
    assert len(calls) == 1  # the second call came from wafer_derived.json
    assert (w1.mesh_rows, w1.mesh_cols, w1.sram_kib_per_pe, w1.fifo_capacity_packets) == (10, 1280, 13, 5)
    assert w1 == w2
    budget, rep, out_dir = calls[0]
    assert (rep.m, rep.n, rep.k, rep.bytes_per_element) == (10, 5120, 2880, 4)  # NOT the 2-byte cost precision
    assert budget.wafer_area_budget_um2 == 800_000_000.0 and out_dir == tmp_path / "area_budget_derivation_fp32"


def test_make_wafer_rederives_when_the_settings_changed(tmp_path, monkeypatch):
    calls = []
    D.make_wafer(False, tmp_path, derive=_stub_derive(calls))
    monkeypatch.setattr(D, "REP_GEMM", (10, 4096, 4096))
    D.make_wafer(False, tmp_path, derive=_stub_derive(calls))
    assert len(calls) == 2


# ------------------------------------------------- 4. chunked prefill / tiling fit

from wafersim.primitives import WaferSpec  # noqa: E402

PINNED = WaferSpec(mesh_rows=10, mesh_cols=1280, sram_kib_per_pe=13, fifo_capacity_packets=5)


def test_fit_probe_reproduces_the_real_failure_and_its_fix():
    # the exact GEMM from the user's traceback: (1024, 11008, 4096) fp16 on the pinned 10x1280 / 13 KiB wafer
    probe = D.FitProbe(PINNED)
    probe.op_gemm(1024, 11008, 4096, 2)
    assert probe.bad == [(1024, 11008, 4096)]
    ok = D.FitProbe(PINNED)
    ok.op_gemm(512, 11008, 4096, 2)
    assert ok.bad == []
    ok.op_concat(None)  # every non-GEMM op is ignored
    with pytest.raises(AttributeError):
        ok.something_else


def test_fit_probe_agrees_with_the_real_tile_search_rule():
    from wafersim.primitives import GemmSpec
    from wafersim.timeloop import _largest_divisor_leq, feasible_tile_k_candidates
    for m in (1, 7, 16, 63, 64, 100, 512, 1024):
        for n, k in ((4096, 4096), (11008, 4096), (1024, 4096), (32000, 4096)):
            g = GemmSpec(m=m, n=n, k=k, bytes_per_element=2)
            try:
                feasible_tile_k_candidates(g, PINNED, m // _largest_divisor_leq(m, 10), n // _largest_divisor_leq(n, 1280))
                want = []
            except ValueError:
                want = [(m, n, k)]
            p = D.FitProbe(PINNED)
            p.op_gemm(m, n, k, 2)
            assert p.bad == want


@pytest.mark.parametrize("cfg", [D.LLAMA2_7B, D.LLAMA3_8B], ids=lambda c: c.name)
@pytest.mark.parametrize("b,expect", [(1, 512), (16, 32), (32, 16), (64, 8)])
def test_prefill_chunk_on_the_pinned_wafer(cfg, b, expect):
    c = D.prefill_chunk_size(cfg, b, 1024, PINNED, 2)
    assert c == expect and 1024 % c == 0
    starts = D.chunk_starts(1024, c)
    for t0 in starts:  # every costed chunk (and the decode steps at the real cache lengths) is tileable
        assert D.step_fits(cfg, b, c, t0, PINNED, 2, tail=(t0 == starts[-1])) == []
    for t0 in D.decode_points(1024, 1024):
        assert D.step_fits(cfg, b, 1, t0, PINNED, 2) == []
    # and it is the LARGEST: the next bigger divisor fails somewhere
    nxt = min(x for x in range(c + 1, 1025) if 1024 % x == 0)
    st = D.chunk_starts(1024, nxt)
    assert any(D.step_fits(cfg, b, nxt, t0, PINNED, 2, tail=(t0 == st[-1])) for t0 in st)


def test_a_roomy_wafer_needs_no_chunking():
    assert D.prefill_chunk_size(D.TOY_MHA, 2, 16, WaferSpec(), 4) == 16


def test_chunk_starts():
    assert D.chunk_starts(1024, 1024) == (0,)
    assert D.chunk_starts(1024, 512) == (0, 512)
    assert D.chunk_starts(1024, 8) == (0, 63 * 8, 127 * 8)
    assert D.chunk_starts(12, 4) == (0, 4, 8)


class _StubCost:
    """Every op costs 1 cycle / 1 uJ, so section sums are op COUNTS."""

    def __init__(self):
        self.log = []

    def __getattr__(self, name):
        if name.startswith("op_"):
            return lambda *a, **kw: self.log.append(
                {"op": name, "cycles": 1.0, "energy_uj": 1.0, "cycle_repeats": 1, "energy_repeats": 1})
        raise AttributeError(name)


@pytest.mark.parametrize("n_chunks", [1, 2, 3, 4])
def test_run_prefill_bills_the_lm_head_once_and_the_body_per_pass(n_chunks):
    cfg, b, n_in = D.TOY_GQA, 2, 12
    chunk = n_in // n_chunks
    cost = _StubCost()
    pre = D.run_prefill(cfg, b, n_in, chunk, cost, 4, False)
    body = D.run_step(cfg, b, chunk, 0, cost, 4, False, tail=False)
    tail = D.run_tail(cfg, b, chunk, cost, 4)
    assert pre.cycles == pytest.approx(n_chunks * body.cycles + tail.cycles)
    assert pre.energy_uj == pytest.approx(n_chunks * body.energy_uj + tail.energy_uj)
    full = D.run_step(cfg, b, chunk, 0, cost, 4, False, tail=True)  # body + tail == the old whole-step accounting
    assert full.cycles == pytest.approx(body.cycles + tail.cycles)
    if n_chunks == 1:  # one pass == the old single-step prefill
        one = D.run_step(cfg, b, n_in, 0, cost, 4, False, tail=True)
        assert pre.cycles == pytest.approx(one.cycles)


def test_run_prefill_rejects_a_chunk_that_does_not_divide():
    with pytest.raises(ValueError):
        D.run_prefill(D.TOY_MHA, 1, 10, 4, _StubCost(), 4, False)


# ------------------------------------------------------- 4. capacity: does it fit on the wafer?


def _pinned_wafer():
    from wafersim.specs import WaferSpec
    return dataclasses.replace(WaferSpec(), **D.PINNED_WAFER)


def test_capacity_report_matches_hand_computed_bytes():
    w = _pinned_wafer()
    rep = D.capacity_report(D.LLAMA2_7B, w, 1024, 1024, [1, 64])
    pes = 10 * 1280
    assert rep["pes"] == pes
    assert rep["total_sram_bytes"] == pes * 13 * 1024
    assert rep["data_sram_bytes"] == pes * (13 * 1024 - w.fifo_router_bytes_per_pe)
    assert w.fifo_router_bytes_per_pe == 10260 and rep["data_sram_bytes"] == pes * 3052
    assert rep["weights_bytes"] == D.LLAMA2_7B.n_params() * 2
    assert rep["kv_bytes_per_token"] == 2 * 32 * 4096 * 2  # 512 KiB/token (MHA)
    assert rep["peak_cache_tokens"] == 1024 + 1024 - 1  # the last generated token is never fed back
    b64 = rep["per_batch"][1]
    assert b64["kv_bytes"] == 64 * 2047 * 524288
    assert b64["need_bytes"] == rep["weights_bytes"] + b64["kv_bytes"]


def test_real_models_do_not_fit_the_pinned_wafer():
    w = _pinned_wafer()
    for cfg in (D.LLAMA2_7B, D.LLAMA3_8B):
        rep = D.capacity_report(cfg, w, 1024, 1024, [1, 16, 32, 64])
        assert not rep["all_fit_total_sram"] and not rep["all_fit_data_sram"]
        assert rep["weights_over_total_sram"] > 50  # weights alone are tens of times the SRAM
        assert rep["kv_tokens_that_fit_total_after_weights"] == 0  # no room left for any KV
        assert all(not r["fits_total_sram"] for r in rep["per_batch"])


def test_a_small_model_on_a_roomy_wafer_fits_and_the_data_budget_is_the_tighter_one():
    roomy = dataclasses.replace(_pinned_wafer(), mesh_rows=32, mesh_cols=32, sram_kib_per_pe=1024)
    rep = D.capacity_report(D.TOY_GQA, roomy, 8, 8, [1, 2])
    assert rep["all_fit_total_sram"] and rep["all_fit_data_sram"]
    assert all(r["need_over_data_sram"] > r["need_over_total_sram"] for r in rep["per_batch"])
    # exactly one byte over the total SRAM flips the verdict
    w = _pinned_wafer()
    tiny = dataclasses.replace(D.TOY_GQA, name="x")
    rep = D.capacity_report(tiny, w, 8, 8, [1])
    kv_fit = rep["kv_tokens_that_fit_total_after_weights"]
    assert kv_fit * rep["kv_bytes_per_token"] + rep["weights_bytes"] <= rep["total_sram_bytes"]
    assert (kv_fit + 1) * rep["kv_bytes_per_token"] + rep["weights_bytes"] > rep["total_sram_bytes"]


def test_capacity_scales_linearly_with_batch_and_is_json_serialisable():
    import json
    rep = D.capacity_report(D.LLAMA3_8B, _pinned_wafer(), 512, 512, [1, 2, 4])
    kv = [r["kv_bytes"] for r in rep["per_batch"]]
    assert kv[1] == 2 * kv[0] and kv[2] == 4 * kv[0]
    json.dumps(rep)  # goes into summary.json


def test_print_capacity_warns_when_the_model_does_not_fit(capsys):
    rep = D.print_footprint(D.LLAMA2_7B, _pinned_wafer(), 1024, 1024, [1])
    out = capsys.readouterr().out
    assert "WARNING" in out and "NO" in out and "upper bound" in out
    assert rep["all_fit_total_sram"] is False
