#!/usr/bin/env python3
"""gpt-oss-120b AttentionBlock + a real per-layer KV cache, built ENTIRELY from
this project's ``wafersim.primitives`` (cost tracker and primitive wrappers are
reused from ``experiment_gptoss_moe_mlp/gptoss_moe_mlp.py``).

Reference being modeled: ``AttentionBlock`` + ``sdpa`` in openai/gpt-oss
``gpt_oss/torch/model.py``. Dims (from its ``ModelConfig``): hidden 2880,
64 query heads, 8 KV heads (q_mult = 8 query heads share a KV head), head_dim
64, sliding_window 128 on every other layer (full attention on the rest),
one learned attention-sink logit per query head, RoPE (YaRN in the real model).
The reference has NO KV cache: ``TokenGenerator`` re-runs the whole sequence
every token. This file adds the cache the way real inference does it.

BLOCK (one layer):
  1. rms_norm(x)
  2. ONE fused qkv GEMM (S,H)x(H,(64+8+8)*64) + bias, sliced into q, k, v
  3. RoPE (half-split rotation) on q and k only
  4. KV CACHE APPEND: k, v rows of the S new tokens are written to the layer's
     cache. Only the rows being written are charged (a real cache is a
     preallocated buffer; history is not copied). ``APPEND_CHARGE="full"``
     charges a whole-cache copy instead, as the Qwen3 script's concat does.
  5. attention PER KV HEAD g (8 of them), all q_mult query heads of a group
     batched into the GEMM's M dimension:
        scores = Q_g (S*8, 64) x K_g^T (64, T_eff)         one real GEMM
        scores *= 1/sqrt(64);  (+ mask, only if any entry is masked)
        sink-aware softmax;  out_g = probs (S*8,T_eff) x V_g (T_eff, 64)
     GQA is therefore handled by GROUPING, never by repeating K/V 8x
     (the reference's ``K[:, :, None, :].expand`` is a view, not traffic).
  6. concat the 8 groups -> (S, 4096), out GEMM (4096 -> 2880) + bias, residual.

SLIDING WINDOW (every other layer): a decode token only needs the last
``sliding_window`` cached rows (keys older than that get weight exactly 0 in
the reference's -inf mask), so those layers read ``T_eff = min(T, 128)`` rows
no matter how long the context is; full layers read all T. This is exact, not
an approximation. Prefill of S new tokens needs keys from
``max(0, T_old - window + 1)`` on and is masked inside that range.

ATTENTION SINK: the reference concatenates the sink logit as an extra softmax
column then drops its probability column. Costing that literally copies the
whole score matrix for nothing, so the sink is folded into the softmax
exactly:  m = max(rowmax, sink);  p = exp(s-m) / (sum exp(s-m) + exp(sink-m)).
The numeric test checks this against the literal concat form.

NUMERIC pass: real numpy through Tensor, checked against a literal numpy
transcription of the reference (full recompute, no cache, -inf mask, sink
concat), including prefill + token-by-token decode == reference on the whole
sequence. COST pass: real Docker, memoized per unique op shape, real model
dims, decode on a synthetic cache of random K/V of length T0 (the cache's
VALUES never change any cost; only its length does).

KNOWN LIMITATIONS (named, not buried):
  - Cost is a SERIAL sum over ops and over the 8 KV groups (which are
    independent and could run on different PE regions). Same convention as
    the MoE script; absolute cycles are an upper-bound style number.
  - gather/concat/repeat/reshape are not PE-tiled (cross-PE movement needs a
    network model); the cache append and head concat are charged serially.
  - K/V are assumed stored head-contiguous on hardware, so per-head column
    slices of the (T, 8*64) cache tensor are free views.
  - Prefill is costed at the FULL S x T score matrix (like the reference);
    a banded/flash-style mapping would be cheaper for long sliding-window
    prefill.
  - RoPE tables here are plain RoPE (theta=150000). Real gpt-oss uses YaRN
    scaling, which changes table VALUES only; shapes, op counts and cost are
    identical.
  - fp32 / 4 bytes everywhere (gpt-oss runs bf16); area fractions carried
    over from the MoE/Qwen3 runs.
  - Cache read bandwidth is only what the two attention GEMMs already charge
    for their K/V operands; no separate cache-read term.

Docker/TimeLoop failures RAISE -- no fallback to a fabricated cost number.

Usage (project root, venv active, Docker Desktop open):
  python3 experiment_gptoss_attention_kvcache/gptoss_attention_kvcache.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from wafersim.primitives import (
    AreaBudget,
    GemmSpec,
    Tensor,
    WaferSpec,
    derive_wafer_spec_from_area_budget,
    slice_ as p_slice,
    transpose as p_transpose,
    concat as p_concat,
)
from wafersim.tiling import Tiling

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuse the MoE script's CostTracker + costed primitive wrappers (it is a
# script, not a package, so load it by path and register it first -- the
# dataclasses inside need their module in sys.modules).
_MOE_PATH = REPO_ROOT / "experiment_gptoss_moe_mlp" / "gptoss_moe_mlp.py"
_spec = importlib.util.spec_from_file_location("gptoss_moe_mlp", _MOE_PATH)
moe = sys.modules.get("gptoss_moe_mlp") or importlib.util.module_from_spec(_spec)
if "gptoss_moe_mlp" not in sys.modules:
    sys.modules["gptoss_moe_mlp"] = moe
    _spec.loader.exec_module(moe)

CostTracker = moe.CostTracker
DTYPE = moe.DTYPE
BYTES_PER_ELEMENT = moe.BYTES_PER_ELEMENT
op_add, op_mul, op_sub, op_exp, op_div = moe.op_add, moe.op_mul, moe.op_sub, moe.op_exp, moe.op_div
op_sum, op_max, op_concat, op_reshape, op_matmul = moe.op_sum, moe.op_max, moe.op_concat, moe.op_reshape, moe.op_matmul
rms_norm, _const = moe.rms_norm, moe._const

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 0
RUN_HARDWARE_COST = True
APPEND_CHARGE = "new_rows"  # or "full" (charge a whole-cache copy per append)
NEG = -1e30  # "-inf" for the additive mask; exp(NEG - m) == 0 in fp32 and stays finite

PREFILL_TOKENS = 10
DECODE_CONTEXTS = (10, 128, 1024, 4096)  # cache length T0 BEFORE the new decode token
SERIAL_CONTEXT_LIMIT = 128  # serial (one-PE) bracket only where one PE's slice still fits its SRAM

FIXED_AREA_BUDGET_UM2 = moe.FIXED_AREA_BUDGET_UM2
AREA_COMPUTE_FRACTION = moe.AREA_COMPUTE_FRACTION
AREA_MEMORY_FRACTION = moe.AREA_MEMORY_FRACTION
AREA_NETWORK_FRACTION = moe.AREA_NETWORK_FRACTION
MESH_TILING = moe.MESH_TILING
SERIAL_TILING = moe.SERIAL_TILING
OUT_ROOT = REPO_ROOT / "outputs" / "gptoss_attention_kvcache"


@dataclass(frozen=True)
class AttnConfig:
    hidden_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 150000.0

    @property
    def q_mult(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim


GPT_OSS_120B = AttnConfig()
TOY = AttnConfig(hidden_size=32, head_dim=8, num_attention_heads=8, num_key_value_heads=2, sliding_window=4)

FULL, SLIDING = "full", "sliding"  # gpt-oss alternates: layer_idx % 2 == 0 is sliding in the reference


def window_for(cfg: AttnConfig, kind: str) -> int:
    return cfg.sliding_window if kind == SLIDING else 0


# --------------------------------------------------------------------------
# Weights. Reference layout (model.py): qkv.weight (80*64, H), qkv.bias,
# out.weight (H, 64*64), out.bias, sinks (64,), norm scale (H,).
# Hardware layout: matmul-ready x @ W.
# --------------------------------------------------------------------------


@dataclass
class AttnWeights:
    norm_scale: Tensor  # (H,)
    wqkv: Tensor  # (H, q_dim + 2*kv_dim)
    bqkv: Tensor
    wo: Tensor  # (q_dim, H)
    bo: Tensor  # (H,)
    sinks: np.ndarray  # (num_attention_heads,) head index = kv_group*q_mult + m

    @classmethod
    def from_reference(cls, ref: dict) -> "AttnWeights":
        f = lambda a: Tensor.from_values(np.ascontiguousarray(a, dtype=DTYPE))
        return cls(f(ref["norm_scale"]), f(ref["qkv_weight"].T), f(ref["qkv_bias"]),
                   f(ref["out_weight"].T), f(ref["out_bias"]), np.asarray(ref["sinks"], dtype=DTYPE))


def random_reference_weights(cfg: AttnConfig, rng: np.random.Generator, scale: float = 0.2) -> dict:
    h, qkv = cfg.hidden_size, cfg.q_dim + 2 * cfg.kv_dim
    r = lambda *s: (rng.standard_normal(s) * scale).astype(DTYPE)
    return {"norm_scale": (1.0 + 0.1 * rng.standard_normal(h)).astype(DTYPE),
            "qkv_weight": r(qkv, h), "qkv_bias": r(qkv),
            "out_weight": r(h, cfg.q_dim), "out_bias": r(h),
            "sinks": r(cfg.num_attention_heads)}


# --------------------------------------------------------------------------
# KV cache
# --------------------------------------------------------------------------


@dataclass
class LayerKVCache:
    """One layer's cache: K and V as (T, num_kv_heads*head_dim); KV head g owns
    columns [g*d, (g+1)*d). ``None`` until the first append."""
    k: Optional[Tensor] = None
    v: Optional[Tensor] = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[0]

    @classmethod
    def synthetic(cls, cfg: AttnConfig, length: int, rng: np.random.Generator) -> "LayerKVCache":
        """A cache already holding ``length`` random rows. Cost depends on the
        cache's LENGTH only, never its values."""
        mk = lambda: Tensor.from_values(rng.standard_normal((length, cfg.kv_dim)).astype(DTYPE))
        return cls(mk(), mk())


def append_kv(cache: LayerKVCache, k_new: Tensor, v_new: Tensor, cost, charge: Optional[str] = None) -> None:
    """Write the new rows into the cache. ``charge="new_rows"`` bills only the
    rows written (preallocated buffer); ``"full"`` bills the whole-cache copy.
    ``None`` -> the module-level ``APPEND_CHARGE`` read at call time."""
    charge = APPEND_CHARGE if charge is None else charge
    if charge not in ("new_rows", "full"):
        raise ValueError(f"charge must be 'new_rows' or 'full', got {charge!r}")
    if cache.k is None:
        k_all, v_all = k_new, v_new
    else:
        k_all, v_all = p_concat([cache.k, k_new], 0), p_concat([cache.v, v_new], 0)
    if cost is not None:
        billed_k, billed_v = (k_new, v_new) if charge == "new_rows" else (k_all, v_all)
        cost.op_concat(billed_k)
        cost.op_concat(billed_v)
    cache.k, cache.v = k_all, v_all


# --------------------------------------------------------------------------
# RoPE tables and application
# --------------------------------------------------------------------------


def rope_tables(positions: np.ndarray, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """(S, head_dim/2) cos/sin. Plain RoPE; real gpt-oss folds YaRN scaling
    into these same-shaped tables (values differ, cost does not)."""
    inv_freq = theta ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    ang = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    return np.cos(ang).astype(DTYPE), np.sin(ang).astype(DTYPE)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, cost) -> Tensor:
    """x (S, H, d); cos/sin (S, 1, d/2). Half-split rotation exactly as the
    reference: o1 = x1*cos - x2*sin ; o2 = x2*cos + x1*sin."""
    half = x.shape[-1] // 2
    x1, x2 = p_slice(x, -1, 0, half), p_slice(x, -1, half, 2 * half)
    o1 = op_sub(op_mul(x1, cos, cost), op_mul(x2, sin, cost), cost)
    o2 = op_add(op_mul(x2, cos, cost), op_mul(x1, sin, cost), cost)
    return op_concat([o1, o2], -1, cost)


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------


def build_mask(s_new: int, q_mult: int, t_old: int, key_start: int, t_eff: int, window: int) -> Optional[np.ndarray]:
    """Additive mask (S*q_mult, t_eff) for keys at absolute positions
    [key_start, key_start+t_eff), rows ordered (token, mult). Query token i sits
    at absolute position t_old+i: it may see key j iff j <= t_old+i and, for
    sliding layers, t_old+i-j < window. Returns None if nothing is masked."""
    qpos = (t_old + np.arange(s_new))[:, None]
    kpos = (key_start + np.arange(t_eff))[None, :]
    allowed = kpos <= qpos
    if window > 0:
        allowed &= (qpos - kpos) < window
    if allowed.all():
        return None
    return np.repeat(np.where(allowed, 0.0, NEG).astype(DTYPE), q_mult, axis=0)


def softmax_with_sink(scores: Tensor, sink_col: Tensor, cost) -> Tensor:
    """softmax over [scores, sink] returning only the score columns, folded so
    the sink is never concatenated onto the (large) score matrix. Exact."""
    row_max = op_max(scores, -1, cost)  # (R,1)
    m = op_max(op_concat([row_max, sink_col], 1, cost), 1, cost)  # (R,1)
    e = op_exp(op_sub(scores, m, cost), cost)
    e_sink = op_exp(op_sub(sink_col, m, cost), cost)
    denom = op_add(op_sum(e, -1, cost), e_sink, cost)
    return op_div(e, denom, cost)


def attention_block(x: Tensor, w: AttnWeights, cfg: AttnConfig, cache: LayerKVCache, kind: str,
                    cost: Optional[CostTracker], append_charge: Optional[str] = None) -> Tensor:
    """``x + Attn(rms_norm(x))`` for S new tokens appended after
    ``cache.length`` cached ones. ``cache`` is updated in place. S=T_new for
    prefill on an empty cache; S=1 for one decode step."""
    s = x.shape[0]
    nh, nkv, d, qm = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.q_mult
    window = window_for(cfg, kind)
    t_old = cache.length

    normed = rms_norm(x, w.norm_scale, cfg.rms_norm_eps, cost)
    qkv = op_add(op_matmul(normed, w.wqkv, cost), w.bqkv, cost)
    q = p_slice(qkv, 1, 0, cfg.q_dim)
    k_new = p_slice(qkv, 1, cfg.q_dim, cfg.q_dim + cfg.kv_dim)
    v_new = p_slice(qkv, 1, cfg.q_dim + cfg.kv_dim, cfg.q_dim + 2 * cfg.kv_dim)

    cos_np, sin_np = rope_tables(t_old + np.arange(s), d, cfg.rope_theta)
    cos = Tensor.from_values(cos_np[:, None, :])
    sin = Tensor.from_values(sin_np[:, None, :])
    q = apply_rope(op_reshape(q, (s, nh, d), cost), cos, sin, cost)  # (S, nh, d)
    k_rot = apply_rope(op_reshape(k_new, (s, nkv, d), cost), cos, sin, cost)
    k_rows = op_reshape(k_rot, (s, cfg.kv_dim), cost)

    append_kv(cache, k_rows, v_new, cost, append_charge)

    t = cache.length
    key_start = max(0, t_old - window + 1) if window > 0 else 0
    t_eff = t - key_start
    k_win = p_slice(cache.k, 0, key_start, t)  # (T_eff, nkv*d), a view
    v_win = p_slice(cache.v, 0, key_start, t)

    rows = s * qm
    mask_np = build_mask(s, qm, t_old, key_start, t_eff, window)
    mask = None if mask_np is None else Tensor.from_values(mask_np)
    scale = _const((rows, t_eff), 1.0 / math.sqrt(d))

    outs = []
    for g in range(nkv):
        qg = op_reshape(p_slice(q, 1, g * qm, (g + 1) * qm), (rows, d), cost)  # rows ordered (token, mult)
        kg = p_slice(k_win, 1, g * d, (g + 1) * d)  # (T_eff, d)
        vg = p_slice(v_win, 1, g * d, (g + 1) * d)
        scores = op_mul(op_matmul(qg, p_transpose(kg, 0, 1), cost), scale, cost)  # (rows, T_eff)
        if mask is not None:
            scores = op_add(scores, mask, cost)
        sink_col = Tensor.from_values(np.tile(w.sinks[g * qm:(g + 1) * qm], s)[:, None])  # (rows,1)
        probs = softmax_with_sink(scores, sink_col, cost)
        outs.append(op_reshape(op_matmul(probs, vg, cost), (s, qm * d), cost))

    attn = op_concat(outs, 1, cost)  # (S, nh*d), head order g*qm+m
    return op_add(x, op_add(op_matmul(attn, w.wo, cost), w.bo, cost), cost)


# --------------------------------------------------------------------------
# Literal numpy transcription of the reference (full recompute, no cache)
# --------------------------------------------------------------------------


def reference_attention_numpy(x: np.ndarray, ref: dict, cfg: AttnConfig, window: int) -> np.ndarray:
    """gpt_oss/torch/model.py AttentionBlock.forward + sdpa, literally: no
    cache, -inf mask, sink concatenated as an extra column then dropped."""
    s = x.shape[0]
    nh, nkv, d, qm = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.q_mult
    x = x.astype(np.float32)
    t = (x * (1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + cfg.rms_norm_eps)) * ref["norm_scale"]).astype(np.float32)
    qkv = t @ ref["qkv_weight"].T + ref["qkv_bias"]
    q = qkv[:, : nh * d].reshape(s, nkv, qm, d)
    k = qkv[:, nh * d: (nh + nkv) * d].reshape(s, nkv, d)
    v = qkv[:, (nh + nkv) * d:].reshape(s, nkv, d)
    cos, sin = rope_tables(np.arange(s), d, cfg.rope_theta)

    def rope(a):  # a (..., d); cos/sin (s, d/2) broadcast over the middle dims
        a1, a2 = a[..., : d // 2], a[..., d // 2:]
        shape = (s,) + (1,) * (a.ndim - 2) + (d // 2,)
        c, sn = cos.reshape(shape), sin.reshape(shape)
        return np.concatenate([a1 * c - a2 * sn, a2 * c + a1 * sn], axis=-1)

    q, k = rope(q), rope(k)
    qk = np.einsum("qhmd,khd->hmqk", q, k) * (1.0 / math.sqrt(d))
    mask = np.triu(np.full((s, s), -np.inf, dtype=np.float32), k=1)
    if window > 0:
        mask = mask + np.tril(np.full((s, s), -np.inf, dtype=np.float32), k=-window)
    qk = qk + mask[None, None]
    sink = np.broadcast_to(ref["sinks"].reshape(nkv, qm, 1, 1), (nkv, qm, s, 1))
    full = np.concatenate([qk, sink], axis=-1)
    full = np.exp(full - full.max(axis=-1, keepdims=True))
    wts = (full / full.sum(axis=-1, keepdims=True))[..., :-1]
    attn = np.einsum("hmqk,khd->qhmd", wts, v).reshape(s, nh * d)
    return (x + attn @ ref["out_weight"].T + ref["out_bias"]).astype(np.float32)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _breakdown(cost: CostTracker, c0_idx: int) -> dict:
    by_op: dict = defaultdict(lambda: [0, 0.0, 0.0])
    for e in cost.log[c0_idx:]:
        a = by_op[e["op"]]
        a[0] += 1
        a[1] += e["cycles"] * e["cycle_repeats"]
        a[2] += e["energy_uj"] * e["energy_repeats"]
    return by_op


def _kv_bytes(cfg: AttnConfig, length: int) -> int:
    return 2 * length * cfg.kv_dim * BYTES_PER_ELEMENT


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("STEP 1: NUMERIC check at toy dims vs. literal numpy reference (no cache)")
    print("=" * 78)
    ref = random_reference_weights(TOY, rng)
    w = AttnWeights.from_reference(ref)
    total = 12  # > 2x the toy window of 4, so the window really binds
    xs = rng.standard_normal((total, TOY.hidden_size)).astype(DTYPE)
    for kind in (FULL, SLIDING):
        expected = reference_attention_numpy(xs, ref, TOY, window_for(TOY, kind))
        cache = LayerKVCache()
        pre = attention_block(Tensor.from_values(xs[:7]), w, TOY, cache, kind, None).data
        steps = [attention_block(Tensor.from_values(xs[i:i + 1]), w, TOY, cache, kind, None).data for i in range(7, total)]
        got = np.concatenate([pre] + steps, axis=0)
        err = float(np.max(np.abs(got - expected)))
        print(f"{kind:<8} prefill(7)+decode(5) with KV cache vs reference on all 12 tokens: max err = {err:.3e}")
        if not np.allclose(got, expected, rtol=1e-4, atol=1e-5):
            raise RuntimeError(f"{kind}: cached attention disagrees with the reference -- composition bug")
    print("numeric check PASSED")

    if not RUN_HARDWARE_COST:
        print("RUN_HARDWARE_COST is False -- skipping the real Docker hardware-cost pass.")
        return

    print()
    print("=" * 78)
    print("STEP 2: HARDWARE-COST pass at REAL gpt-oss-120b dims (real Docker/TimeLoop)")
    print("=" * 78)
    cfg = GPT_OSS_120B
    print(f"config: {cfg}   append charge: {APPEND_CHARGE}")
    budget = AreaBudget(wafer_area_budget_um2=FIXED_AREA_BUDGET_UM2, compute_area_fraction=AREA_COMPUTE_FRACTION,
                        memory_area_fraction=AREA_MEMORY_FRACTION, network_area_fraction=AREA_NETWORK_FRACTION)
    rep = GemmSpec(m=PREFILL_TOKENS, n=cfg.q_dim + 2 * cfg.kv_dim, k=cfg.hidden_size, bytes_per_element=BYTES_PER_ELEMENT)
    print(f"deriving WaferSpec from area budget {FIXED_AREA_BUDGET_UM2:,.0f} um^2 (representative GEMM m={rep.m} n={rep.n} k={rep.k}) ...")
    wafer = derive_wafer_spec_from_area_budget(budget, rep, WaferSpec(), OUT_ROOT / "area_budget_derivation").wafer
    print(f"derived wafer: mesh {wafer.mesh_rows}x{wafer.mesh_cols}, sram_kib_per_pe={wafer.sram_kib_per_pe}")

    weights = AttnWeights(
        norm_scale=Tensor.from_values(np.ones(cfg.hidden_size, dtype=DTYPE)),
        wqkv=Tensor.from_values((rng.standard_normal((cfg.hidden_size, cfg.q_dim + 2 * cfg.kv_dim)) * 0.02).astype(DTYPE)),
        bqkv=Tensor.from_values(np.zeros(cfg.q_dim + 2 * cfg.kv_dim, dtype=DTYPE)),
        wo=Tensor.from_values((rng.standard_normal((cfg.q_dim, cfg.hidden_size)) * 0.02).astype(DTYPE)),
        bo=Tensor.from_values(np.zeros(cfg.hidden_size, dtype=DTYPE)),
        sinks=(rng.standard_normal(cfg.num_attention_heads) * 0.5).astype(DTYPE))

    # scenario = (label, new tokens S, cache length T0 before them)
    scenarios = [("prefill", PREFILL_TOKENS, 0)] + [(f"decode@{t0}", 1, t0) for t0 in DECODE_CONTEXTS]
    inputs = {name: Tensor.from_values(rng.standard_normal((s, cfg.hidden_size)).astype(DTYPE)) for name, s, _ in scenarios}
    shared_cache: dict = {}
    table: dict = {}
    breakdowns: dict = {}
    for label, tiling in (("mesh", MESH_TILING), ("serial", SERIAL_TILING)):
        cost = CostTracker(wafer=wafer, out_dir=OUT_ROOT / "primitive_cost_traces" / label, tiling=tiling, _cache=shared_cache)
        print(f"\n##### tiling = {tiling.describe(cost.total_pes)}")
        for name, s, t0 in scenarios:
            if label == "serial" and max(s, t0) > SERIAL_CONTEXT_LIMIT:
                continue
            for kind in (FULL, SLIDING):
                cache = LayerKVCache.synthetic(cfg, t0, rng) if t0 else LayerKVCache()
                c0, e0, i0, w0 = cost.total_cycles, cost.total_energy_uj, len(cost.log), time.time()
                attention_block(inputs[name], weights, cfg, cache, kind, cost)
                table[(label, name, kind)] = (cost.total_cycles - c0, cost.total_energy_uj - e0, cache.length)
                breakdowns[(label, name, kind)] = _breakdown(cost, i0)
                print(f"[{name:<12} {kind:<8}] cycles {table[(label, name, kind)][0]:>14,.0f}  energy {table[(label, name, kind)][1]:>12.2f} uJ  wall {time.time() - w0:.1f}s")

    print()
    print("=" * 78)
    print("ONE ATTENTION LAYER (mesh tiling; serial bracket shown where it fits). KV cache is per layer.")
    print("=" * 78)
    print(f"{'scenario':<13} {'layer':<8} {'keys read':>9} {'cycles (mesh)':>15} {'energy uJ':>12} {'cycles (serial)':>16} {'KV cache':>10}")
    for name, s, t0 in scenarios:
        for kind in (FULL, SLIDING):
            mc, me, tl = table[("mesh", name, kind)]
            sc = table.get(("serial", name, kind))
            window = window_for(cfg, kind)
            keys = tl - (max(0, t0 - window + 1) if window else 0)
            kv = f"{_kv_bytes(cfg, tl) / 2**20:.2f} MiB"
            print(f"{name:<13} {kind:<8} {keys:>9} {mc:>15,.0f} {me:>12.2f} {(f'{sc[0]:,.0f}' if sc else 'n/a'):>16} {kv:>10}")

    print("\nDECODE-STEP ATTENTION for all 36 layers (18 full + 18 sliding, mesh tiling) -- attention only, no MoE:")
    for name, s, t0 in scenarios[1:]:
        f, sl = table[("mesh", name, FULL)][0], table[("mesh", name, SLIDING)][0]
        print(f"  {name:<13} {18 * f + 18 * sl:>16,.0f} cycles  (full layers {18 * f:,.0f} + sliding {18 * sl:,.0f})")

    for key in (("mesh", "prefill", FULL), ("mesh", f"decode@{DECODE_CONTEXTS[-1]}", FULL), ("mesh", f"decode@{DECODE_CONTEXTS[-1]}", SLIDING)):
        by_op = breakdowns[key]
        tot = sum(v[1] for v in by_op.values())
        print(f"\n--- per-op breakdown: {key[1]} / {key[2]} layer ---")
        print(f"{'op':<10} {'calls':>7} {'total_cycles':>16} {'share':>7} {'total_energy_uj':>18}")
        for op_name, (calls, cyc, en) in sorted(by_op.items(), key=lambda kv: -kv[1][1]):
            print(f"{op_name:<10} {calls:>7} {cyc:>16,.0f} {cyc / tot:>7.1%} {en:>18.4f}")
    print("\nNOT tiled: gather/concat/reshape (serial) -- the cache append, group concat and rope concat are charged serially.")
    print(f"real artifact tree: {OUT_ROOT}")


if __name__ == "__main__":
    main()
