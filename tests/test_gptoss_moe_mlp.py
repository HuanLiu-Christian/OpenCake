"""Tests for the expert-grouped gpt-oss MoE MLPBlock
(experiment_gptoss_moe_mlp/gptoss_moe_mlp.py).

Numeric tests compare against a numpy transcription of the reference
``MLPBlock`` (literal per-token weight gather + einsums, interleaved
SwiGLU). Cost-accounting tests use ``CostTracker(dry_run=True)``, which
records every primitive call's shape with zero cost and never touches
Docker -- so this whole file runs without Docker, like
test_primitives_numeric.py. Real Docker numbers come from running the
experiment script itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from wafersim.primitives import Tensor

_SCRIPT = Path(__file__).resolve().parent.parent / "experiment_gptoss_moe_mlp" / "gptoss_moe_mlp.py"
_spec = importlib.util.spec_from_file_location("gptoss_moe_mlp", _SCRIPT)
moe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = moe  # dataclasses need the module registered
_spec.loader.exec_module(moe)

TOY = moe.TOY


def _run(seed: int, s: int, cfg=TOY, scale: float = 0.2):
    rng = np.random.default_rng(seed)
    ref = moe.random_reference_weights(cfg, rng, scale=scale)
    xs = rng.standard_normal((s, cfg.hidden_size)).astype(np.float32)
    return ref, xs


# ---- numeric correctness ---------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("s", [1, 3, 10])
def test_expert_grouped_matches_reference(seed, s):
    ref, xs = _run(seed, s)
    out, _ = moe.moe_mlp_block(Tensor.from_values(xs), moe.MoEWeights.from_reference(ref), TOY, None)
    np.testing.assert_allclose(out.data, moe.reference_moe_numpy(xs, ref, TOY), rtol=1e-4, atol=1e-5)


def test_matches_reference_when_swiglu_clamps_actually_fire():
    # large weights/activations push glu above +7 and linear beyond +-7
    ref, xs = _run(7, 6, scale=1.5)
    t1 = np.abs(xs).max()
    out, _ = moe.moe_mlp_block(Tensor.from_values(xs * 3), moe.MoEWeights.from_reference(ref), TOY, None)
    expected = moe.reference_moe_numpy(xs * 3, ref, TOY)
    np.testing.assert_allclose(out.data, expected, rtol=1e-3, atol=1e-3)
    assert t1 > 0


def test_clamp_regime_is_really_exercised():
    """Guard for the test above: confirm clamps change the result."""
    ref, xs = _run(7, 6, scale=1.5)
    cfg_loose = moe.MoEConfig(**{**TOY.__dict__, "swiglu_limit": 1e9})
    a = moe.reference_moe_numpy(xs * 3, ref, TOY)
    b = moe.reference_moe_numpy(xs * 3, ref, cfg_loose)
    assert not np.allclose(a, b)


def test_output_is_residual_plus_moe():
    ref, xs = _run(0, 4)
    out, _ = moe.moe_mlp_block(Tensor.from_values(xs), moe.MoEWeights.from_reference(ref), TOY, None)
    assert out.shape == xs.shape and np.isfinite(out.data).all()
    assert not np.allclose(out.data, xs)


# ---- routing bookkeeping ---------------------------------------------

def test_routing_is_a_consistent_expert_grouping():
    rng = np.random.default_rng(0)
    s, k, e = 9, 4, 16
    idx = np.stack([rng.permutation(e)[:k] for _ in range(s)]).astype(np.int64)
    r = moe.build_routing(idx, e)
    assert r.counts.sum() == s * k and r.counts.shape == (e,)
    flat = idx.reshape(-1)
    sorted_experts = flat[r.order]
    assert np.all(np.diff(sorted_experts) >= 0)  # grouped by expert
    for ex in range(e):  # each expert's block holds exactly its tokens
        rows = r.token_of_sorted[r.offsets[ex]: r.offsets[ex] + r.counts[ex]]
        assert sorted(rows.tolist()) == sorted(np.nonzero((idx == ex).any(axis=1))[0].tolist())
    np.testing.assert_array_equal(r.order[r.inverse], np.arange(s * k))  # true inverse


# ---- cost accounting (dry run, no Docker) -----------------------------

def _logs(cost):
    return [(e["op"], e["key"]) for e in cost.log]


@pytest.mark.parametrize("seed,s", [(0, 1), (1, 4), (2, 10)])
def test_cost_only_path_logs_identical_ops_to_full_numeric_path(seed, s):
    """The cost-only path (used for the real 120b dims, no expert weights)
    must record EXACTLY the ops the full numeric path records."""
    ref, xs = _run(seed, s)
    w_full = moe.MoEWeights.from_reference(ref)
    full = moe.CostTracker(dry_run=True)
    moe.moe_mlp_block(Tensor.from_values(xs), w_full, TOY, full)
    w_gate = moe.MoEWeights(w_full.norm_scale, w_full.gate_w, w_full.gate_b)  # no experts
    fast = moe.CostTracker(dry_run=True)
    moe.moe_mlp_block(Tensor.from_values(xs), w_gate, TOY, fast, cost_only_experts=True)
    assert _logs(full) == _logs(fast)


def test_expert_gemm_shapes_and_counts_follow_routing():
    ref, xs = _run(1, 10)
    cost = moe.CostTracker(dry_run=True)
    _, r = moe.moe_mlp_block(Tensor.from_values(xs), moe.MoEWeights.from_reference(ref), TOY, cost)
    h, i = TOY.hidden_size, TOY.intermediate_size
    gemms = [e["key"] for e in cost.log if e["op"] == "gemm"]
    active = [(ex, int(n)) for ex, n in enumerate(r.counts) if n > 0]
    assert len(gemms) == 1 + 2 * len(active)  # gate + (mlp1, mlp2) per active expert
    assert gemms[0][1:4] == (10, TOY.num_experts, h)  # gate GEMM (S,E,H)
    expect = []
    for _, n in active:
        expect += [("gemm", n, 2 * i, h, 4), ("gemm", n, h, i, 4)]
    assert gemms[1:] == expect
    assert sum(n for _, n in active) == 10 * TOY.experts_per_token  # no row lost/duplicated


def test_swiglu_clamps_are_costed_one_and_two_sided_per_expert():
    ref, xs = _run(0, 5)
    cost = moe.CostTracker(dry_run=True)
    _, r = moe.moe_mlp_block(Tensor.from_values(xs), moe.MoEWeights.from_reference(ref), TOY, cost)
    clamps = [e["key"] for e in cost.log if e["op"] == "clamp"]
    n_active = int((r.counts > 0).sum())
    assert len(clamps) == 2 * n_active
    assert sorted({c[2] for c in clamps}) == [1, 2]  # one-sided glu, two-sided linear


def test_router_topk_is_costed_with_real_shape():
    ref, xs = _run(0, 7)
    for tiling, per_pe in ((moe.Tiling.serial(), 7), (moe.Tiling(), 1)):
        cost = moe.CostTracker(dry_run=True, tiling=tiling)
        moe.moe_mlp_block(Tensor.from_values(xs), moe.MoEWeights.from_reference(ref), TOY, cost)
        topks = [e for e in cost.log if e["op"] == "topk"]
        assert len(topks) == 1 and topks[0]["rows"] == 7  # logical rows always recorded
        assert topks[0]["key"] == ("topk", per_pe, TOY.num_experts, TOY.experts_per_token, 4)


def test_no_per_token_weight_gather_is_charged():
    """The reference's weight gather must NOT appear: the only gathers are
    the token dispatch and the combine un-permute, both (S*k, H)-sized."""
    ref, xs = _run(0, 6)
    cost = moe.CostTracker(dry_run=True)
    moe.moe_mlp_block(Tensor.from_values(xs), moe.MoEWeights.from_reference(ref), TOY, cost)
    gathers = [e["key"] for e in cost.log if e["op"] == "gather"]
    assert len(gathers) == 2
    assert all(g[1] == 6 * TOY.experts_per_token * TOY.hidden_size for g in gathers)


def test_memoization_reuses_identical_shapes():
    ref, xs = _run(2, 10)
    cost = moe.CostTracker(dry_run=True)
    moe.moe_mlp_block(Tensor.from_values(xs), moe.MoEWeights.from_reference(ref), TOY, cost)
    assert cost.memoized_reuses > 0
    assert cost.unique_op_shapes + cost.memoized_reuses == len(cost.log)


def test_real_tracker_requires_wafer_and_outdir():
    with pytest.raises(ValueError):
        moe.CostTracker()


# ---- real gpt-oss-120b dims, shapes only (no weights, no Docker) -------

def test_real_dims_decode_routes_to_exactly_k_experts_with_one_row_each():
    cfg = moe.GPT_OSS_120B
    rng = np.random.default_rng(0)
    cost = moe.CostTracker(dry_run=True)
    x = Tensor.from_values(rng.standard_normal((1, cfg.hidden_size)).astype(np.float32))
    _, r = moe.moe_mlp_block(x, moe.gate_only_weights(cfg, rng), cfg, cost, cost_only_experts=True)
    assert int((r.counts > 0).sum()) == 4 and int(r.counts.sum()) == 4
    gemms = [e["key"][1:4] for e in cost.log if e["op"] == "gemm"]
    assert gemms[0] == (1, 128, 2880)
    assert gemms[1:] == [(1, 5760, 2880), (1, 2880, 2880)] * 4


def test_real_dims_prefill_row_conservation():
    cfg = moe.GPT_OSS_120B
    rng = np.random.default_rng(0)
    cost = moe.CostTracker(dry_run=True)
    x = Tensor.from_values(rng.standard_normal((10, cfg.hidden_size)).astype(np.float32))
    _, r = moe.moe_mlp_block(x, moe.gate_only_weights(cfg, rng), cfg, cost, cost_only_experts=True)
    assert int(r.counts.sum()) == 40  # 10 tokens * top-4
    ns = [e["key"][1] for e in cost.log if e["op"] == "gemm"][1::2]
    assert sorted(ns) == sorted(int(n) for n in r.counts if n > 0)


# ---- tiling accounting (stubbed simulators: no Docker) -----------------
#
# Stub per-PE simulators that are exactly LINEAR in the work they are given
# (cycles = 3/element, energy = 0.5 uJ/element), so what the CostTracker does
# with tiling can be checked in closed form.

from types import SimpleNamespace  # noqa: E402

FAKE_WAFER = SimpleNamespace(mesh_rows=120, mesh_cols=120)  # 14,400 PEs
CYC, EN = 3, 0.5


def _fake_sim(spec, wafer, out_dir):
    if hasattr(spec, "rows_per_pe"):  # TopKSpec
        n = spec.rows_per_pe * spec.num_elements
    else:
        n = spec.elements_per_pe  # may legitimately be 0 (zero-cost view reshape)
    return SimpleNamespace(local_compute_cycles=CYC * n, local_energy_uj=EN * n)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    for name in ("add", "mul", "div", "sub", "exp", "rsqrt", "silu", "sum", "max",
                 "clamp", "topk", "gather", "concat_stack", "reshape"):
        monkeypatch.setattr(moe, f"simulate_{name}", _fake_sim, raising=False)
    cand = SimpleNamespace(result=SimpleNamespace(sim_time_cycles=1000, energy=SimpleNamespace(total_uj=7.0)))
    monkeypatch.setattr(moe, "search_tile_k", lambda *a, **k: [cand])
    monkeypatch.setattr(moe, "best_by_sim_time", lambda c: c[0])

    def make(tiling, cache=None):
        kw = {} if cache is None else {"_cache": cache}
        return moe.CostTracker(wafer=FAKE_WAFER, out_dir=tmp_path / str(id(tiling)), tiling=tiling, **kw)
    return make


def test_serial_tiling_reproduces_old_accounting(stubbed):
    c = stubbed(moe.Tiling.serial())
    c.op_add(126_720)
    c.op_sum(2880, lanes=10)
    assert c.total_cycles == CYC * 126_720 + CYC * 2880 * 10  # whole tensor on one PE; lanes serial
    assert c.total_energy_uj == pytest.approx(EN * 126_720 + EN * 2880 * 10)


def test_mesh_tiling_cuts_latency_but_conserves_energy(stubbed):
    ser, mesh = stubbed(moe.Tiling.serial()), stubbed(moe.Tiling())
    for c in (ser, mesh):
        c.op_silu(126_720)
        c.op_clamp(126_720, 2)
        c.op_sum(2880, lanes=10)
        c.op_max(128, lanes=31_702)
        c.op_topk(10, 128, 4)
    assert mesh.total_cycles < ser.total_cycles / 8
    assert mesh.total_energy_uj == pytest.approx(ser.total_energy_uj, rel=1e-9)


def test_mesh_elementwise_latency_is_the_busiest_pe_slice(stubbed):
    c = stubbed(moe.Tiling())
    c.op_add(126_720)  # ceil(126720/14400) = 9 elements on the busiest PE
    assert c.total_cycles == CYC * 9
    e = c.log[0]
    assert (e["elements"], e["per_pe"]) == (126_720, 9)
    assert c.total_energy_uj == pytest.approx(EN * 126_720)


def test_reduction_lanes_parallelize_but_one_lane_stays_serial(stubbed):
    c = stubbed(moe.Tiling())
    c.op_sum(2880, lanes=10)  # 10 lanes on 10 PEs -> latency of ONE lane, not 10
    assert c.total_cycles == CYC * 2880
    c2 = stubbed(moe.Tiling())
    c2.op_sum(2880, lanes=1)
    assert c2.total_cycles == CYC * 2880  # a single long reduction is not split


def test_gemm_and_data_movement_ignore_tiling(stubbed):
    ref, xs = _run(0, 6)
    totals = {}
    for label, tiling in (("serial", moe.Tiling.serial()), ("mesh", moe.Tiling())):
        c = stubbed(tiling)
        moe.moe_mlp_block(Tensor.from_values(xs), moe.MoEWeights.from_reference(ref), TOY, c)
        totals[label] = {op: sum(e["cycles"] * e["cycle_repeats"] for e in c.log if e["op"] == op)
                         for op in ("gemm", "gather", "concat", "reshape")}
    assert totals["serial"] == totals["mesh"]


def test_full_block_mesh_vs_serial_bracket(stubbed):
    ref, xs = _run(1, 10)
    res = {}
    for label, tiling in (("serial", moe.Tiling.serial()), ("mesh", moe.Tiling())):
        c = stubbed(tiling)
        moe.moe_mlp_block(Tensor.from_values(xs), moe.MoEWeights.from_reference(ref), TOY, c)
        res[label] = c
    assert res["mesh"].total_cycles < res["serial"].total_cycles
    assert res["mesh"].total_energy_uj == pytest.approx(res["serial"].total_energy_uj, rel=1e-9)


def test_shared_cache_between_tilings_does_not_change_results(stubbed):
    ref, xs = _run(2, 8)
    w = moe.MoEWeights.from_reference(ref)
    shared = {}
    a = stubbed(moe.Tiling(), cache=shared)
    b = stubbed(moe.Tiling.serial(), cache=shared)
    moe.moe_mlp_block(Tensor.from_values(xs), w, TOY, a)
    moe.moe_mlp_block(Tensor.from_values(xs), w, TOY, b)
    b_alone = stubbed(moe.Tiling.serial())
    moe.moe_mlp_block(Tensor.from_values(xs), w, TOY, b_alone)
    assert b.total_cycles == b_alone.total_cycles
    assert b.total_energy_uj == pytest.approx(b_alone.total_energy_uj)


def test_cost_only_path_still_matches_full_path_under_tiling(stubbed):
    ref, xs = _run(3, 5)
    w = moe.MoEWeights.from_reference(ref)
    gate_only = moe.MoEWeights(w.norm_scale, w.gate_w, w.gate_b)
    for tiling in (moe.Tiling.serial(), moe.Tiling()):
        a, b = stubbed(tiling), stubbed(tiling)
        moe.moe_mlp_block(Tensor.from_values(xs), w, TOY, a)
        moe.moe_mlp_block(Tensor.from_values(xs), gate_only, TOY, b, cost_only_experts=True)
        assert _logs(a) == _logs(b)
        assert a.total_cycles == b.total_cycles
        assert a.total_energy_uj == pytest.approx(b.total_energy_uj)


@pytest.mark.parametrize("op,args", [
    ("op_add", ()), ("op_mul", ()), ("op_div", ()), ("op_sub", ()), ("op_exp", ()),
    ("op_rsqrt", ()), ("op_silu", ()), ("op_clamp", (1,)), ("op_clamp", (2,)),
])
def test_every_elementwise_op_is_tiled_by_the_same_rule(stubbed, op, args):
    ser, mesh = stubbed(moe.Tiling.serial()), stubbed(moe.Tiling())
    getattr(ser, op)(126_720, *args)
    getattr(mesh, op)(126_720, *args)
    assert ser.total_cycles == CYC * 126_720
    assert mesh.total_cycles == CYC * 9  # ceil(126720 / 14400)
    assert mesh.total_energy_uj == pytest.approx(ser.total_energy_uj)


def test_topk_rows_are_tiled_and_energy_conserved(stubbed):
    ser, mesh = stubbed(moe.Tiling.serial()), stubbed(moe.Tiling())
    ser.op_topk(10, 128, 4)
    mesh.op_topk(10, 128, 4)
    assert ser.total_cycles == CYC * 10 * 128 and mesh.total_cycles == CYC * 1 * 128
    assert mesh.total_energy_uj == pytest.approx(ser.total_energy_uj)
