"""Tests for gpt-oss attention + KV cache
(experiment_gptoss_attention_kvcache/gptoss_attention_kvcache.py).

Numeric tests compare against a literal numpy transcription of the reference
(no cache, -inf mask, sink concatenated then dropped). Cost-accounting tests use
``CostTracker(dry_run=True)`` (zero cost, never touches Docker), so this file
runs without Docker like the other test files.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from wafersim.primitives import Tensor

_SCRIPT = Path(__file__).resolve().parent.parent / "experiment_gptoss_attention_kvcache" / "gptoss_attention_kvcache.py"
_spec = importlib.util.spec_from_file_location("gptoss_attention_kvcache", _SCRIPT)
att = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = att
_spec.loader.exec_module(att)

TOY = att.TOY
FULL, SLIDING = att.FULL, att.SLIDING


def _setup(seed=0, total=12, cfg=TOY, scale=0.2):
    rng = np.random.default_rng(seed)
    ref = att.random_reference_weights(cfg, rng, scale=scale)
    xs = rng.standard_normal((total, cfg.hidden_size)).astype(np.float32)
    return ref, att.AttnWeights.from_reference(ref), xs


def _run_cached(xs, w, cfg, kind, split, cost=None):
    """prefill xs[:split] then decode the rest one token at a time."""
    cache = att.LayerKVCache()
    outs = [att.attention_block(Tensor.from_values(xs[:split]), w, cfg, cache, kind, cost).data]
    for i in range(split, len(xs)):
        outs.append(att.attention_block(Tensor.from_values(xs[i:i + 1]), w, cfg, cache, kind, cost).data)
    return np.concatenate(outs, axis=0), cache


# ---- numeric correctness ---------------------------------------------

@pytest.mark.parametrize("kind", [FULL, SLIDING])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("s", [1, 5, 12])
def test_no_cache_prefill_matches_reference(kind, seed, s):
    ref, w, xs = _setup(seed, total=s)
    out = att.attention_block(Tensor.from_values(xs), w, TOY, att.LayerKVCache(), kind, None).data
    np.testing.assert_allclose(out, att.reference_attention_numpy(xs, ref, TOY, att.window_for(TOY, kind)), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kind", [FULL, SLIDING])
@pytest.mark.parametrize("split", [1, 3, 7, 11])
def test_prefill_plus_cached_decode_equals_reference_on_whole_sequence(kind, split):
    # total 12 > 2x the toy window (4): the sliding window really binds in decode
    ref, w, xs = _setup(3, total=12)
    got, cache = _run_cached(xs, w, TOY, kind, split)
    np.testing.assert_allclose(got, att.reference_attention_numpy(xs, ref, TOY, att.window_for(TOY, kind)), rtol=1e-4, atol=1e-5)
    assert cache.length == 12


def test_chunked_prefill_equals_single_prefill():
    ref, w, xs = _setup(4, total=12)
    cache = att.LayerKVCache()
    parts = [att.attention_block(Tensor.from_values(xs[a:b]), w, TOY, cache, SLIDING, None).data
             for a, b in ((0, 5), (5, 9), (9, 12))]
    np.testing.assert_allclose(np.concatenate(parts), att.reference_attention_numpy(xs, ref, TOY, TOY.sliding_window), rtol=1e-4, atol=1e-5)


def test_sliding_window_actually_changes_the_result():
    ref, w, xs = _setup(5, total=12)
    full, _ = _run_cached(xs, w, TOY, FULL, 7)
    sliding, _ = _run_cached(xs, w, TOY, SLIDING, 7)
    # first `window` tokens see identical context; later ones must differ
    np.testing.assert_allclose(full[:TOY.sliding_window], sliding[:TOY.sliding_window], rtol=1e-5, atol=1e-6)
    assert np.abs(full[-1] - sliding[-1]).max() > 1e-3


def test_sink_matters_and_matches_literal_concat_form():
    ref, w, xs = _setup(6, total=6)
    ref_big = dict(ref, sinks=ref["sinks"] + 5.0)
    out_a = att.attention_block(Tensor.from_values(xs), w, TOY, att.LayerKVCache(), FULL, None).data
    out_b = att.attention_block(Tensor.from_values(xs), att.AttnWeights.from_reference(ref_big), TOY, att.LayerKVCache(), FULL, None).data
    np.testing.assert_allclose(out_b, att.reference_attention_numpy(xs, ref_big, TOY, 0), rtol=1e-4, atol=1e-5)
    assert np.abs(out_a - out_b).max() > 1e-3  # a large sink really absorbs attention


def test_sink_is_in_the_softmax_denominator():
    # one query, one key, huge sink: attention weight on the key must vanish,
    # so the block output is ~ x + out_bias (not x + out_proj(v))
    rng = np.random.default_rng(11)
    ref = att.random_reference_weights(TOY, rng)
    ref["sinks"] = np.full(TOY.num_attention_heads, 50.0, dtype=np.float32)
    x = rng.standard_normal((1, TOY.hidden_size)).astype(np.float32)
    out = att.attention_block(Tensor.from_values(x), att.AttnWeights.from_reference(ref), TOY, att.LayerKVCache(), FULL, None).data
    np.testing.assert_allclose(out, x + ref["out_bias"], atol=1e-4)


def test_cache_holds_roped_keys_not_raw():
    ref, w, xs = _setup(7, total=4)
    cache = att.LayerKVCache()
    att.attention_block(Tensor.from_values(xs), w, TOY, cache, FULL, None)
    # position 0 has angle 0 -> rope is identity there; later rows must differ from raw projections
    t = xs * (1.0 / np.sqrt(np.mean(xs**2, axis=-1, keepdims=True) + TOY.rms_norm_eps)) * ref["norm_scale"]
    qkv = t @ ref["qkv_weight"].T + ref["qkv_bias"]
    raw_k = qkv[:, TOY.q_dim: TOY.q_dim + TOY.kv_dim]
    np.testing.assert_allclose(cache.k.data[0], raw_k[0], rtol=1e-4, atol=1e-5)
    assert np.abs(cache.k.data[3] - raw_k[3]).max() > 1e-3


# ---- cost accounting -------------------------------------------------

def _dry(cfg, kind, s, t0, charge=None, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    ref = att.random_reference_weights(cfg, rng, scale=0.02)
    w = att.AttnWeights.from_reference(ref)
    cache = att.LayerKVCache.synthetic(cfg, t0, rng) if t0 else att.LayerKVCache()
    cost = att.CostTracker(dry_run=True)
    att.attention_block(Tensor.from_values(rng.standard_normal((s, cfg.hidden_size)).astype(np.float32)),
                        w, cfg, cache, kind, cost, append_charge=charge)
    return cost


def _gemms(cost):
    return [e["key"][1:4] for e in cost.log if e["op"] == "gemm"]  # (m, n, k)


def test_decode_gqa_by_grouping_no_repeat_and_one_gemm_pair_per_kv_head():
    c = _dry(TOY, FULL, 1, 20)
    assert not [e for e in c.log if e["op"] == "repeat"]
    gemms = _gemms(c)
    qk = [g for g in gemms if g == (TOY.q_mult, 21, TOY.head_dim)]
    pv = [g for g in gemms if g == (TOY.q_mult, TOY.head_dim, 21)]
    assert len(qk) == len(pv) == TOY.num_key_value_heads
    assert len(gemms) == 2 * TOY.num_key_value_heads + 2  # + fused qkv + out proj


def test_fused_qkv_and_out_gemm_shapes():
    c = _dry(TOY, FULL, 3, 0)
    gemms = _gemms(c)
    assert (3, TOY.q_dim + 2 * TOY.kv_dim, TOY.hidden_size) in gemms
    assert (3, TOY.hidden_size, TOY.q_dim) in gemms


def test_sliding_decode_reads_only_window_keys_full_reads_all():
    t0 = 30
    sliding = _gemms(_dry(TOY, SLIDING, 1, t0))
    full = _gemms(_dry(TOY, FULL, 1, t0))
    assert (TOY.q_mult, TOY.sliding_window, TOY.head_dim) in sliding
    assert not any(g[1] == t0 + 1 for g in sliding if g[2] == TOY.head_dim)
    assert (TOY.q_mult, t0 + 1, TOY.head_dim) in full


def test_sliding_cost_is_flat_in_context_but_full_cost_grows():
    keys = lambda c: sorted(e["key"] for e in c.log if e["op"] in ("gemm", "exp", "mul", "sum", "max"))
    assert keys(_dry(TOY, SLIDING, 1, 20)) == keys(_dry(TOY, SLIDING, 1, 60))
    assert keys(_dry(TOY, FULL, 1, 20)) != keys(_dry(TOY, FULL, 1, 60))


def test_append_charges_only_new_rows_by_default_full_copy_on_request():
    def concat_elems(c):
        return [e["key"][1] for e in c.log if e["op"] == "concat"]
    new_rows = concat_elems(_dry(TOY, FULL, 1, 40, "new_rows"))
    full_copy = concat_elems(_dry(TOY, FULL, 1, 40, "full"))
    assert max(new_rows) < max(full_copy)
    assert sum(1 for _ in new_rows) == sum(1 for _ in full_copy)  # same number of concat calls


def test_default_append_charge_is_new_rows_only():
    assert att.APPEND_CHARGE == "new_rows"
    kv_new_row = TOY.kv_dim  # one decode token's K (or V) row
    concat_k_v = [e for e in _dry(TOY, FULL, 1, 40).log if e["op"] == "concat" and e["key"][1] == kv_new_row]
    assert len(concat_k_v) >= 2  # the K and V appends are billed at exactly one row each
    default = sorted(e["key"] for e in _dry(TOY, FULL, 1, 40).log if e["op"] == "concat")
    assert default == sorted(e["key"] for e in _dry(TOY, FULL, 1, 40, "new_rows").log if e["op"] == "concat")
    assert default != sorted(e["key"] for e in _dry(TOY, FULL, 1, 40, "full").log if e["op"] == "concat")


def test_bad_append_charge_rejected():
    with pytest.raises(ValueError):
        att.append_kv(att.LayerKVCache(), Tensor.zeros((1, 4)), Tensor.zeros((1, 4)), None, charge="bogus")


def test_mask_added_only_when_something_is_masked():
    rows_t = lambda s, t: s * TOY.q_mult * t
    scores_adds = lambda c, n: [e for e in c.log if e["op"] == "add" and e["key"][1] * e.get("energy_repeats", 1) >= 0 and e.get("elements") == n]
    prefill = _dry(TOY, FULL, 4, 0)
    decode = _dry(TOY, FULL, 1, 10)
    assert len(scores_adds(prefill, rows_t(4, 4))) == TOY.num_key_value_heads  # causal mask add per kv group
    assert not scores_adds(decode, rows_t(1, 11))  # nothing masked -> no add


def test_build_mask_semantics():
    assert att.build_mask(1, 2, 10, 0, 11, 0) is None
    m = att.build_mask(3, 2, 0, 0, 3, 0)  # prefill causal, rows ordered (token, mult)
    N = att.NEG
    expected = np.array([[0, N, N], [0, N, N], [0, 0, N], [0, 0, N], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(m, expected)
    # decode, sliding window 4, keys 7..10 are exactly the last 4 -> nothing masked
    assert att.build_mask(1, 1, 10, 7, 4, 4) is None
    # sliding prefill: token at abs pos 5 with window 3 sees keys 3,4,5 only
    sw = att.build_mask(1, 1, 5, 0, 6, 3)
    np.testing.assert_array_equal(sw[0], np.array([N, N, N, 0, 0, 0], dtype=np.float32))


def test_real_dims_decode_shapes_dry_run():
    cfg = att.GPT_OSS_120B
    rng = np.random.default_rng(0)
    w = att.AttnWeights(
        norm_scale=Tensor.from_values(np.ones(cfg.hidden_size, np.float32)),
        wqkv=Tensor.zeros((cfg.hidden_size, cfg.q_dim + 2 * cfg.kv_dim)),
        bqkv=Tensor.zeros((cfg.q_dim + 2 * cfg.kv_dim,)),
        wo=Tensor.zeros((cfg.q_dim, cfg.hidden_size)), bo=Tensor.zeros((cfg.hidden_size,)),
        sinks=np.zeros(cfg.num_attention_heads, np.float32))
    for kind, t_eff in ((SLIDING, 128), (FULL, 301)):
        cache = att.LayerKVCache.synthetic(cfg, 300, rng)
        cost = att.CostTracker(dry_run=True)
        att.attention_block(Tensor.from_values(np.ones((1, cfg.hidden_size), np.float32)), w, cfg, cache, kind, cost)
        gemms = _gemms(cost)
        assert (1, 5120, 2880) in gemms and (1, 2880, 4096) in gemms
        assert gemms.count((8, t_eff, 64)) == 8 and gemms.count((8, 64, t_eff)) == 8
