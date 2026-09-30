#!/usr/bin/env python3
"""End-to-end Qwen3-tiny forward-pass composition, built ENTIRELY from this
project's already-implemented, real ``wafersim.primitives`` -- answers the
question "can the 16 primitives we already have compose into a real
decoder-only LLM prefill pass under a fixed chip-area budget, and does it
run through to a real numeric+hardware-cost result?".

SCOPE (as agreed with the user): the full Qwen3Config.tiny() architecture
(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
num_key_value_heads=2, head_dim=32, intermediate_size=768 -- the only six
numbers actually given in the pasted ``qwen3_compute_sim.py``), run for
PREFILL ONLY (one causal forward pass over a full input sequence, no
autoregressive decode/KV-cache loop -- that is a natural follow-up once
this composition is confirmed to work).

ASSUMED, NOT GIVEN (the pasted reference file was truncated before its
``Qwen3Config`` dataclass finished, so these three numbers are not on
record anywhere in this project -- change them here if the real Qwen3-tiny
config differs):
  - rms_norm_eps  = 1e-6        (Qwen2/2.5/3's own published default)
  - rope_theta    = 1_000_000.0 (Qwen2.5/3's own published default)
  - SEQ_LEN       = 64          (prefill sequence length -- not specified
    at all; pick any value and re-run, cost scales with it)
Model weights and the input activations are real, seeded random data
(``numpy.random.default_rng(SEED)``, not fabricated in the sense this
project's discipline forbids -- that discipline is about never fabricating
a *hardware cost* number; a synthetic-but-real-valued weight tensor for a
smoke test is the same thing every unit test in this repo already does,
e.g. ``Tensor.random`` in ``primitives/tensor``). No pretrained Qwen3
weights exist in this project, and are not needed to answer the user's
actual question (does the primitive composition run and produce a real,
finite result, and what would it really cost on real fixed-area hardware).

MISSING PRIMITIVE: there is no dedicated ``sub``/subtract primitive in
this project's real 16-primitive set (confirmed again in this file's own
research pass -- see ``primitives/__init__.py``). Every subtraction below
(RoPE's rotate_half ``-x2``, softmax's ``x - row_max``) is composed the
same way real hardware without a dedicated subtracter would do it: negate
via a real ``mul`` by a constant ``-1`` tensor, then a real ``add`` -- both
already-Docker-verified primitives, so this is a real composition, not a
workaround that skips the cost model.

TWO INDEPENDENT PASSES (mirrors this project's own Tensor-vs-TimeLoop
split convention, see ``primitives/tensor``'s module docstring):

  1. NUMERIC pass (always runs, no Docker) -- every op is the primitive's
     real numpy-backed numeric function (``add``/``mul``/``div``/``rsqrt``/
     ``exp``/``silu``/``matmul``/``reshape``/``view``/``permute``/
     ``transpose``/``repeat``/``gather``/``concat``/``stack``/``where``/
     ``cast``). Answers "does the composition even produce a correctly
     shaped, finite real result" in well under a second, no Docker needed.

  2. HARDWARE-COST pass (opt-in, gated by ``RUN_HARDWARE_COST`` below) --
     re-traces the IDENTICAL op sequence, this time also issuing each
     primitive's real Docker-backed ``simulate_*`` call and accumulating
     real cycles/energy. Left OFF by default per the user's own explicit
     instruction this round ("先写脚本但不实际跑,等你确认" -- write the
     script but do not actually run the real-Docker pass yet, wait for
     confirmation) -- flip it to True only when you're ready to spend real
     Docker/TimeLoop time. Recommended: after the three experiment-2
     resume windows (compute/memory/network sweep) finish or are known to
     be idle -- this project already hit one real "too many open files"
     crash from over-concurrent Docker load, and this script would add a
     fourth concurrent Docker consumer on top of those three if run at the
     same time.

REAL-HARDWARE-COST MEMOIZATION -- an explicit, documented design choice,
not a fabrication: TimeLoop/Accelergy's cost model is a function of an
operation's SHAPE/COUNT only, never of the tensor's actual data values
(see ``primitives/tensor``'s own module docstring for why: "TimeLoop has
no numeric-values API at all"). Since all 4 decoder layers here share the
exact same architecture (same shapes at every op, only real numpy weight
VALUES differ per layer, which TimeLoop cannot see and does not affect its
cost), a real Docker measurement for one op shape is deterministically
reusable for every later occurrence of that exact shape -- an honest cache
of a real, already-obtained measurement, not a synthetic stand-in. Every
`CostTracker.op_*` call below transparently either (a) triggers exactly
one real Docker `timeloop-model` run when a shape is seen for the first
time, raising if it fails (never fabricating a result -- same standing
project discipline every other script here follows), or (b) reuses that
exact real result for every later identical-shape occurrence. This is what
keeps a 4-layer x 8-head model's real Docker call count in the low
dozens instead of the many hundreds it would otherwise need.

ELEMENTWISE/REDUCTION COST-ACCOUNTING MODEL -- also an explicit, stated
simplification for this FIRST feasibility pass, not a hidden shortcut:
  - matmul: uses the real, already-existing multi-PE wafer-level SUMMA +
    network simulation (``search_tile_k``/``best_by_sim_time``) -- this
    already reflects the WHOLE derived mesh executing that GEMM.
  - every elementwise/reduction primitive (add/mul/div/rsqrt/exp/silu/
    cast/where/sum/max/reshape/repeat/gather/concat): modeled as ONE
    representative PE processing that op's FULL element count itself, in
    one real Docker-backed call -- i.e. this first pass does NOT attempt
    to partition elementwise work across the derived mesh's active PE
    count the way a full wafer-level scheduler would. The reported
    "total_cycles" is therefore a real, Docker-measured SEQUENTIAL trace
    cost (a legitimate lower bound on the compute-bound part of one lane's
    latency), not a full parallel-schedule wall-clock estimate -- a
    reasonable, clearly-labeled scope for "can this even run and produce a
    real number", with full wafer-level elementwise scheduling as a
    natural, separate follow-up once this composition itself is confirmed
    to work.
  - sum/max reductions get special handling: per `sum`/`max`'s own module
    docstring, ``elements_per_pe`` must be ONE reduction lane's width (K),
    never the total element count across multiple independent lanes (that
    would silently undercount by (num_lanes - 1) real sequential
    firings). So each reduction issues ONE real Docker call for one lane
    of width K (memoized), and its real cost is then applied
    ``num_lanes`` times to the running total -- exact, not approximate,
    given sum/max's own documented "K-1 sequential real firings" cost
    model.
  - RMSNorm's divide-by-hidden_size uses `mul` by a precomputed
    reciprocal constant (N is compile-time known, so folding it into a
    multiply is the realistic hardware choice), while softmax's
    normalization uses the real `div` primitive (the divisor -- the row's
    real exp-sum -- is genuinely runtime data, not a compile-time
    constant) -- this mirrors real accelerator practice, not an
    inconsistency.

Docker/TimeLoop failures RAISE here, exactly like every other script in
this project -- no fallback to a locally-estimated or fabricated cost
number at any point.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wafersim.primitives import (
    AreaBudget,
    CastSpec,
    ConcatStackSpec,
    ExpSpec,
    GatherSpec,
    GemmSpec,
    RepeatSpec,
    ReshapeSpec,
    RsqrtSpec,
    SiluSpec,
    Tensor,
    WaferSpec,
    WhereSpec,
    best_by_sim_time,
    cast as p_cast,
    concat as p_concat,
    derive_wafer_spec_from_area_budget,
    exp as p_exp,
    matmul as p_matmul,
    reshape as p_reshape,
    rsqrt as p_rsqrt,
    search_tile_k,
    silu as p_silu,
    simulate_cast,
    simulate_concat_stack,
    simulate_exp,
    simulate_gather,
    simulate_repeat,
    simulate_reshape,
    simulate_rsqrt,
    simulate_silu,
    simulate_where,
    slice_ as p_slice,
    transpose as p_transpose,
    view,
    where as p_where,
)
from wafersim.primitives import max as p_max
from wafersim.primitives import sum as p_sum
from wafersim.primitives.add import AddSpec, simulate_add
from wafersim.primitives.add import add as p_add
from wafersim.primitives.div import DivSpec, simulate_div
from wafersim.primitives.div import div as p_div
from wafersim.primitives.max import MaxSpec, simulate_max
from wafersim.primitives.mul import MulSpec, simulate_mul
from wafersim.primitives.mul import mul as p_mul
from wafersim.primitives.repeat_expand import repeat as p_repeat
from wafersim.primitives.sum import SumSpec, simulate_sum

# --------------------------------------------------------------------------
# Configuration -- see module docstring for what is given vs. assumed.
# --------------------------------------------------------------------------

DTYPE = np.float32
BYTES_PER_ELEMENT = 4
SEED = 0
SEQ_LEN = 64  # ASSUMED -- not specified anywhere; prefill sequence length

RUN_HARDWARE_COST = True  # <<< flipped on to run the real Docker/TimeLoop hardware-cost pass

FIXED_AREA_BUDGET_UM2 = 400_000_000.0  # ASSUMED placeholder -- reuses an
# existing experiment2_area_sweep area point (400M um^2) for continuity;
# change freely, this is exactly what "fixed chip area budget" is meant to
# be swept over.
AREA_COMPUTE_FRACTION = 0.15
AREA_MEMORY_FRACTION = 0.17
AREA_NETWORK_FRACTION = 0.68
# ^ REVISED after a real failed run: the original 0.5/0.3/0.2 split derived a
# mesh too large for its 20% network share to fit any valid FIFO capacity
# (real error: "no fifo_capacity_packets in [1, 2048] fits the per-PE network
# budget (3255.21 um^2) ... at a 64x384 mesh"). These replacement fractions
# are not a new guess -- they are copied from a REAL, already-successful
# experiment2_area_sweep run at this exact area budget (400,000,000 um^2):
# compute_sweep/area_400000000/trials.csv trial=24 (compute_frac=0.15,
# memory_frac=0.17, network_frac=0.68, real mesh=64x128, real
# fifo_capacity_packets=4) -- a network-heavy split is what this project's
# own real data already shows is required at this area/mesh scale.

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "qwen3_primitive_composition"


@dataclass(frozen=True)
class Qwen3TinyConfig:
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 32
    intermediate_size: int = 768
    rms_norm_eps: float = 1e-6  # ASSUMED, see module docstring
    rope_theta: float = 1_000_000.0  # ASSUMED, see module docstring


@dataclass(frozen=True)
class LayerWeights:
    input_norm_w: Tensor
    q_norm_w: Tensor
    k_norm_w: Tensor
    wq: Tensor
    wk: Tensor
    wv: Tensor
    wo: Tensor
    post_attn_norm_w: Tensor
    w_gate: Tensor
    w_up: Tensor
    w_down: Tensor


def build_layer_weights(rng: np.random.Generator, cfg: Qwen3TinyConfig) -> LayerWeights:
    def randn(*shape: int) -> Tensor:
        return Tensor.from_values((rng.standard_normal(shape) * 0.02).astype(DTYPE))

    H, KVH, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    return LayerWeights(
        input_norm_w=Tensor.from_values(np.ones(cfg.hidden_size, dtype=DTYPE)),
        q_norm_w=Tensor.from_values(np.ones(D, dtype=DTYPE)),
        k_norm_w=Tensor.from_values(np.ones(D, dtype=DTYPE)),
        wq=randn(cfg.hidden_size, H * D),
        wk=randn(cfg.hidden_size, KVH * D),
        wv=randn(cfg.hidden_size, KVH * D),
        wo=randn(H * D, cfg.hidden_size),
        post_attn_norm_w=Tensor.from_values(np.ones(cfg.hidden_size, dtype=DTYPE)),
        w_gate=randn(cfg.hidden_size, cfg.intermediate_size),
        w_up=randn(cfg.hidden_size, cfg.intermediate_size),
        w_down=randn(cfg.intermediate_size, cfg.hidden_size),
    )


# --------------------------------------------------------------------------
# Real, Docker-backed hardware-cost tracker (see module docstring for the
# memoization/lane-accounting discipline this implements).
# --------------------------------------------------------------------------


@dataclass
class CostTracker:
    wafer: WaferSpec
    out_dir: Path
    total_cycles: int = 0
    total_energy_uj: float = 0.0
    real_docker_calls: int = 0
    memoized_reuses: int = 0
    log: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict)
    _counter: int = 0

    def _fresh_dir(self, tag: str) -> Path:
        self._counter += 1
        d = self.out_dir / tag / f"{self._counter:05d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _record(self, op: str, key: tuple, cycles: int, energy_uj: float, hit: bool, repeats: int = 1) -> None:
        if hit:
            self.memoized_reuses += 1
        else:
            self.real_docker_calls += 1
        self.total_cycles += cycles * repeats
        self.total_energy_uj += energy_uj * repeats
        self.log.append(
            {"op": op, "key": key, "cache_hit": hit, "cycles": cycles, "energy_uj": energy_uj, "repeats": repeats}
        )

    def _elementwise(self, op_name: str, spec_cls, sim_fn, elements: int, bpe: int) -> None:
        key = (op_name, elements, bpe)
        hit = key in self._cache
        if hit:
            result = self._cache[key]
        else:
            spec = spec_cls(elements_per_pe=elements, bytes_per_element=bpe)
            result = sim_fn(spec, self.wafer, self._fresh_dir(op_name))
            self._cache[key] = result
        self._record(op_name, key, result.local_compute_cycles, result.local_energy_uj, hit)

    def op_add(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        self._elementwise("add", AddSpec, simulate_add, elements, bpe)

    def op_mul(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        self._elementwise("mul", MulSpec, simulate_mul, elements, bpe)

    def op_div(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        self._elementwise("div", DivSpec, simulate_div, elements, bpe)

    def op_rsqrt(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        self._elementwise("rsqrt", RsqrtSpec, simulate_rsqrt, elements, bpe)

    def op_exp(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        self._elementwise("exp", ExpSpec, simulate_exp, elements, bpe)

    def op_silu(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        self._elementwise("silu", SiluSpec, simulate_silu, elements, bpe)

    def op_cast(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        self._elementwise("cast", CastSpec, simulate_cast, elements, bpe)

    def op_where(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        self._elementwise("where", WhereSpec, simulate_where, elements, bpe)

    def _reduction(self, op_name: str, spec_cls, sim_fn, k: int, num_lanes: int, bpe: int) -> None:
        key = (op_name, k, bpe)
        hit = key in self._cache
        if hit:
            result = self._cache[key]
        else:
            spec = spec_cls(elements_per_pe=k, bytes_per_element=bpe)
            result = sim_fn(spec, self.wafer, self._fresh_dir(op_name))
            self._cache[key] = result
        self._record(op_name, key, result.local_compute_cycles, result.local_energy_uj, hit, repeats=num_lanes)

    def op_sum(self, k: int, num_lanes: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        self._reduction("sum", SumSpec, simulate_sum, k, num_lanes, bpe)

    def op_max(self, k: int, num_lanes: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        self._reduction("max", MaxSpec, simulate_max, k, num_lanes, bpe)

    def op_reshape(self, tensor_before: Tensor, shape: tuple, bpe: int = BYTES_PER_ELEMENT) -> None:
        spec = ReshapeSpec.from_call(tensor_before, shape)
        key = ("reshape", spec.elements_per_pe, spec.bytes_per_element, spec.copied)
        hit = key in self._cache
        if hit:
            result = self._cache[key]
        else:
            result = simulate_reshape(spec, self.wafer, self._fresh_dir("reshape"))
            self._cache[key] = result
        self._record("reshape", key, result.local_compute_cycles, result.local_energy_uj, hit)

    def op_repeat(self, out: Tensor, bpe: int = BYTES_PER_ELEMENT) -> None:
        spec = RepeatSpec.from_result(out)
        key = ("repeat", spec.elements_per_pe, spec.bytes_per_element)
        hit = key in self._cache
        if hit:
            result = self._cache[key]
        else:
            result = simulate_repeat(spec, self.wafer, self._fresh_dir("repeat"))
            self._cache[key] = result
        self._record("repeat", key, result.local_compute_cycles, result.local_energy_uj, hit)

    def op_gather(self, out: Tensor, bpe: int = BYTES_PER_ELEMENT) -> None:
        spec = GatherSpec.from_result(out)
        key = ("gather", spec.elements_per_pe, spec.bytes_per_element)
        hit = key in self._cache
        if hit:
            result = self._cache[key]
        else:
            result = simulate_gather(spec, self.wafer, self._fresh_dir("gather"))
            self._cache[key] = result
        self._record("gather", key, result.local_compute_cycles, result.local_energy_uj, hit)

    def op_concat(self, out: Tensor, bpe: int = BYTES_PER_ELEMENT) -> None:
        spec = ConcatStackSpec.from_result(out)
        key = ("concat", spec.elements_per_pe, spec.bytes_per_element)
        hit = key in self._cache
        if hit:
            result = self._cache[key]
        else:
            result = simulate_concat_stack(spec, self.wafer, self._fresh_dir("concat"))
            self._cache[key] = result
        self._record("concat", key, result.local_compute_cycles, result.local_energy_uj, hit)

    def op_gemm(self, m: int, n: int, k: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        key = ("gemm", m, n, k, bpe)
        hit = key in self._cache
        if hit:
            candidate = self._cache[key]
        else:
            gemm_spec = GemmSpec(m=m, n=n, k=k, bytes_per_element=bpe)
            candidates = search_tile_k(gemm_spec, self.wafer, self._fresh_dir("gemm"))
            candidate = best_by_sim_time(candidates)
            self._cache[key] = candidate
        self._record("gemm", key, candidate.result.sim_time_cycles, candidate.result.energy.total_uj, hit)


# --------------------------------------------------------------------------
# Cost-tracked primitive wrappers -- always perform the real numeric op;
# additionally issue the matching real Docker-backed cost call when a
# CostTracker is supplied. Zero-real-cost ops (view/permute/transpose/
# slice_ -- confirmed permanently zero cost by this project's own
# primitives/__init__.py docstring) are called directly, no wrapper needed.
# --------------------------------------------------------------------------


def op_add(a: Tensor, b: Tensor, cost: CostTracker | None) -> Tensor:
    out = p_add(a, b)
    if cost is not None:
        cost.op_add(out.size, out.bytes_per_element)
    return out


def op_mul(a: Tensor, b: Tensor, cost: CostTracker | None) -> Tensor:
    out = p_mul(a, b)
    if cost is not None:
        cost.op_mul(out.size, out.bytes_per_element)
    return out


def op_div(a: Tensor, b: Tensor, cost: CostTracker | None) -> Tensor:
    out = p_div(a, b)
    if cost is not None:
        cost.op_div(out.size, out.bytes_per_element)
    return out


def op_rsqrt(x: Tensor, cost: CostTracker | None) -> Tensor:
    out = p_rsqrt(x)
    if cost is not None:
        cost.op_rsqrt(out.size, out.bytes_per_element)
    return out


def op_exp(x: Tensor, cost: CostTracker | None) -> Tensor:
    out = p_exp(x)
    if cost is not None:
        cost.op_exp(out.size, out.bytes_per_element)
    return out


def op_silu(x: Tensor, cost: CostTracker | None) -> Tensor:
    out = p_silu(x)
    if cost is not None:
        cost.op_silu(out.size, out.bytes_per_element)
    return out


def op_cast(x: Tensor, dtype, cost: CostTracker | None) -> Tensor:
    out = p_cast(x, dtype)
    if cost is not None:
        cost.op_cast(out.size, out.bytes_per_element)
    return out


def op_where(condition: Tensor, a: Tensor, b: Tensor, cost: CostTracker | None) -> Tensor:
    out = p_where(condition, a, b)
    if cost is not None:
        cost.op_where(out.size, out.bytes_per_element)
    return out


def op_sum(x: Tensor, dim: int, cost: CostTracker | None) -> Tensor:
    out = p_sum(x, dim)
    if cost is not None:
        k = x.shape[dim]
        num_lanes = x.size // k
        cost.op_sum(k, num_lanes, x.bytes_per_element)
    return out


def op_max(x: Tensor, dim: int, cost: CostTracker | None) -> Tensor:
    out = p_max(x, dim)
    if cost is not None:
        k = x.shape[dim]
        num_lanes = x.size // k
        cost.op_max(k, num_lanes, x.bytes_per_element)
    return out


def op_reshape(x: Tensor, shape: tuple, cost: CostTracker | None) -> Tensor:
    out = p_reshape(x, shape)
    if cost is not None:
        cost.op_reshape(x, shape, x.bytes_per_element)
    return out


def op_repeat(x: Tensor, repeats: int, dim: int, cost: CostTracker | None) -> Tensor:
    out = p_repeat(x, repeats, dim)
    if cost is not None:
        cost.op_repeat(out, out.bytes_per_element)
    return out


def op_concat(tensors: list, dim: int, cost: CostTracker | None) -> Tensor:
    out = p_concat(tensors, dim)
    if cost is not None:
        cost.op_concat(out, out.bytes_per_element)
    return out


def op_matmul(a: Tensor, b: Tensor, cost: CostTracker | None) -> Tensor:
    out = p_matmul(a, b)
    if cost is not None:
        m, k = a.shape[-2], a.shape[-1]
        n = b.shape[-1]
        cost.op_gemm(m, n, k, a.bytes_per_element)
    return out


def negate(x: Tensor) -> Tensor:
    """Real ``-x`` via multiply-by-constant -- see module docstring for
    why there is no dedicated subtract primitive in this project."""
    return Tensor.from_values(np.full(x.shape, -1.0, dtype=x.dtype))


# --------------------------------------------------------------------------
# Model building blocks
# --------------------------------------------------------------------------


def rms_norm(x: Tensor, weight: Tensor, eps: float, cost: CostTracker | None) -> Tensor:
    xx = op_mul(x, x, cost)
    ss = op_sum(xx, dim=-1, cost=cost)
    n = x.shape[-1]
    recip_n = Tensor.from_values(np.full(ss.shape, 1.0 / n, dtype=x.dtype))
    mean = op_mul(ss, recip_n, cost)  # divide-by-constant-N folded into a real mul, see module docstring
    eps_t = Tensor.from_values(np.full(mean.shape, eps, dtype=x.dtype))
    var_eps = op_add(mean, eps_t, cost)
    inv = op_rsqrt(var_eps, cost)
    normed = op_mul(x, inv, cost)
    return op_mul(normed, weight, cost)


def build_rope_tables(seq_len: int, head_dim: int, theta: float) -> tuple[Tensor, Tensor]:
    """Real, precomputed RoPE cos/sin tables -- see module docstring for
    why this is treated as a compile-time constant (no sin/cos primitive
    exists or is needed; every real implementation precomputes these
    once too). Shape (S, 1, D) so it broadcasts against both
    (S, num_attention_heads, D) and (S, num_key_value_heads, D)."""
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    positions = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(DTYPE)[:, None, :]
    sin = np.sin(emb).astype(DTYPE)[:, None, :]
    return Tensor.from_values(cos), Tensor.from_values(sin)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, cost: CostTracker | None) -> Tensor:
    """x: (S, heads, D). Real rotate_half composed from slice_ (free
    view) + mul-by-(-1) + concat + mul + add -- see module docstring for
    why there is no dedicated subtract/rotate primitive."""
    d = x.shape[-1]
    half = d // 2
    x1 = p_slice(x, dim=-1, start=0, end=half)
    x2 = p_slice(x, dim=-1, start=half, end=d)
    neg_x2 = op_mul(x2, negate(x2), cost)
    rotated = op_concat([neg_x2, x1], dim=-1, cost=cost)
    a = op_mul(x, cos, cost)
    b = op_mul(rotated, sin, cost)
    return op_add(a, b, cost)


@dataclass(frozen=True)
class AttnConsts:
    causal_mask: Tensor  # (S, S), nonzero at masked (future) positions
    neg_inf: Tensor  # (S, S)
    inv_sqrt_d: Tensor  # (S, S)


def build_attn_consts(seq_len: int, head_dim: int) -> AttnConsts:
    causal = np.triu(np.ones((seq_len, seq_len), dtype=DTYPE), k=1)
    return AttnConsts(
        causal_mask=Tensor.from_values(causal),
        neg_inf=Tensor.from_values(np.full((seq_len, seq_len), -1e9, dtype=DTYPE)),
        inv_sqrt_d=Tensor.from_values(np.full((seq_len, seq_len), 1.0 / math.sqrt(head_dim), dtype=DTYPE)),
    )


def attention_block(
    hidden: Tensor,
    w: LayerWeights,
    cos: Tensor,
    sin: Tensor,
    consts: AttnConsts,
    cfg: Qwen3TinyConfig,
    cost: CostTracker | None,
) -> Tensor:
    s = hidden.shape[0]
    h_heads, kv_heads, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = h_heads // kv_heads

    normed = rms_norm(hidden, w.input_norm_w, cfg.rms_norm_eps, cost)

    q = op_matmul(normed, w.wq, cost)  # (S, H*D)
    k = op_matmul(normed, w.wk, cost)  # (S, KVH*D)
    v = op_matmul(normed, w.wv, cost)  # (S, KVH*D)

    q = view(q, (s, h_heads, d))  # free -- splitting a fresh contiguous trailing dim
    k = view(k, (s, kv_heads, d))
    v = view(v, (s, kv_heads, d))

    q = rms_norm(q, w.q_norm_w, cfg.rms_norm_eps, cost)  # Qwen3's per-head Q/K RMSNorm
    k = rms_norm(k, w.k_norm_w, cfg.rms_norm_eps, cost)

    q = apply_rope(q, cos, sin, cost)
    k = apply_rope(k, cos, sin, cost)

    k = op_repeat(k, n_rep, dim=1, cost=cost)  # GQA expand: (S, KVH, D) -> (S, H, D)
    v = op_repeat(v, n_rep, dim=1, cost=cost)

    head_outs = []
    for head in range(h_heads):
        qh = p_slice(q, dim=1, start=head, end=head + 1)  # (S,1,D) free view
        qh = op_reshape(qh, (s, d), cost)  # real copy -- strided per-head extraction
        kh = p_slice(k, dim=1, start=head, end=head + 1)
        kh = op_reshape(kh, (s, d), cost)
        vh = p_slice(v, dim=1, start=head, end=head + 1)
        vh = op_reshape(vh, (s, d), cost)

        kh_t = p_transpose(kh, 0, 1)  # (D,S), free
        scores = op_matmul(qh, kh_t, cost)  # (S,S) real GEMM m=S n=S k=D
        scores = op_mul(scores, consts.inv_sqrt_d, cost)
        scores = op_where(consts.causal_mask, consts.neg_inf, scores, cost)

        row_max = op_max(scores, dim=-1, cost=cost)
        shifted = op_add(scores, op_mul(row_max, negate(row_max), cost), cost)
        exp_scores = op_exp(shifted, cost)
        row_sum = op_sum(exp_scores, dim=-1, cost=cost)
        probs = op_div(exp_scores, row_sum, cost)  # real divide -- row_sum is runtime data

        out_h = op_matmul(probs, vh, cost)  # (S,D) real GEMM m=S n=D k=S
        head_outs.append(out_h)

    attn = op_concat(head_outs, dim=1, cost=cost)  # (S, H*D)
    attn_out = op_matmul(attn, w.wo, cost)  # (S, hidden)
    return op_add(hidden, attn_out, cost)


def mlp_block(hidden: Tensor, w: LayerWeights, cfg: Qwen3TinyConfig, cost: CostTracker | None) -> Tensor:
    normed = rms_norm(hidden, w.post_attn_norm_w, cfg.rms_norm_eps, cost)
    gate = op_matmul(normed, w.w_gate, cost)
    up = op_matmul(normed, w.w_up, cost)
    gated = op_mul(op_silu(gate, cost), up, cost)
    down = op_matmul(gated, w.w_down, cost)
    return op_add(hidden, down, cost)


def forward_model(
    x: Tensor,
    layer_weights: list[LayerWeights],
    final_norm_w: Tensor,
    cos: Tensor,
    sin: Tensor,
    consts: AttnConsts,
    cfg: Qwen3TinyConfig,
    cost: CostTracker | None,
) -> Tensor:
    hidden = x
    for w in layer_weights:
        hidden = attention_block(hidden, w, cos, sin, consts, cfg, cost)
        hidden = mlp_block(hidden, w, cfg, cost)
    hidden = rms_norm(hidden, final_norm_w, cfg.rms_norm_eps, cost)
    # Optional, illustrative: also exercise `cast` (fp32 -> fp16 narrowing)
    # on the final hidden state -- a real practice (lower-precision
    # output), not otherwise needed by the rest of this forward pass.
    return op_cast(hidden, np.float16, cost)


def main() -> None:
    rng = np.random.default_rng(SEED)
    cfg = Qwen3TinyConfig()

    x = Tensor.from_values((rng.standard_normal((SEQ_LEN, cfg.hidden_size)) * 0.02).astype(DTYPE))
    layer_weights = [build_layer_weights(rng, cfg) for _ in range(cfg.num_hidden_layers)]
    final_norm_w = Tensor.from_values(np.ones(cfg.hidden_size, dtype=DTYPE))
    cos, sin = build_rope_tables(SEQ_LEN, cfg.head_dim, cfg.rope_theta)
    consts = build_attn_consts(SEQ_LEN, cfg.head_dim)

    print("=" * 78)
    print("STEP 1: NUMERIC-ONLY forward pass (real numpy via Tensor, no Docker)")
    print("=" * 78)
    print(f"config: {cfg}")
    print(f"seq_len={SEQ_LEN} (assumed, see module docstring), dtype={DTYPE}")
    t0 = time.time()
    out_numeric = forward_model(x, layer_weights, final_norm_w, cos, sin, consts, cfg, cost=None)
    dt = time.time() - t0
    finite = np.isfinite(out_numeric.data.astype(np.float32)).all()
    print(f"output shape: {out_numeric.shape}, dtype: {out_numeric.dtype}")
    print(f"all finite (no NaN/Inf): {finite}")
    print(f"sample values [0, :8]: {out_numeric.data[0, :8]}")
    print(f"wall time: {dt:.3f}s")
    if not finite:
        raise RuntimeError("numeric forward pass produced NaN/Inf -- composition bug, not a hardware-cost issue")

    if not RUN_HARDWARE_COST:
        print()
        print("RUN_HARDWARE_COST is False -- skipping the real Docker/TimeLoop hardware-cost pass.")
        print("Flip RUN_HARDWARE_COST=True at the top of this script and re-run when ready to spend")
        print("real Docker time (recommended: once the 3 experiment-2 resume windows are done, to")
        print("avoid the same over-concurrent-Docker crash this project already hit once).")
        return

    print()
    print("=" * 78)
    print("STEP 2: HARDWARE-COST pass (real Docker/TimeLoop, memoized per unique op shape)")
    print("=" * 78)
    budget = AreaBudget(
        wafer_area_budget_um2=FIXED_AREA_BUDGET_UM2,
        compute_area_fraction=AREA_COMPUTE_FRACTION,
        memory_area_fraction=AREA_MEMORY_FRACTION,
        network_area_fraction=AREA_NETWORK_FRACTION,
    )
    representative_gemm = GemmSpec(
        m=SEQ_LEN, n=cfg.intermediate_size, k=cfg.hidden_size, bytes_per_element=BYTES_PER_ELEMENT
    )
    print(
        f"deriving WaferSpec from area budget {FIXED_AREA_BUDGET_UM2:,.0f} um^2 (real Docker binary "
        f"search, representative GEMM m={representative_gemm.m} n={representative_gemm.n} "
        f"k={representative_gemm.k}) ..."
    )
    area_result = derive_wafer_spec_from_area_budget(
        budget, representative_gemm, WaferSpec(), OUT_ROOT / "area_budget_derivation"
    )
    wafer = area_result.wafer
    print(
        f"derived wafer: mesh {wafer.mesh_rows}x{wafer.mesh_cols}, sram_kib_per_pe={wafer.sram_kib_per_pe}, "
        f"fifo_capacity_packets={wafer.fifo_capacity_packets}"
    )

    cost = CostTracker(wafer=wafer, out_dir=OUT_ROOT / "primitive_cost_traces")
    t0 = time.time()
    out_costed = forward_model(x, layer_weights, final_norm_w, cos, sin, consts, cfg, cost=cost)
    dt = time.time() - t0

    print()
    print(f"real Docker/TimeLoop primitive calls actually executed: {cost.real_docker_calls}")
    print(f"real results memoized/reused (identical shape, same real Docker result): {cost.memoized_reuses}")
    print(f"total real sequential-trace compute cycles: {cost.total_cycles:,}")
    print(f"total real energy (sum across every real op instance): {cost.total_energy_uj:.4f} uJ")
    print(f"wall time for the hardware-cost pass: {dt:.1f}s")

    by_op: dict = defaultdict(lambda: [0, 0.0, 0])
    for entry in cost.log:
        agg = by_op[entry["op"]]
        agg[0] += entry["cycles"] * entry["repeats"]
        agg[1] += entry["energy_uj"] * entry["repeats"]
        agg[2] += entry["repeats"]
    print()
    print(f"{'op':<10} {'instances':>10} {'total_cycles':>16} {'total_energy_uj':>18}")
    for op_name, (cyc, en, n) in sorted(by_op.items()):
        print(f"{op_name:<10} {n:>10} {cyc:>16,} {en:>18.4f}")

    assert np.isfinite(out_costed.data.astype(np.float32)).all()
    print()
    print("Hardware-cost pass complete -- every number above came from a real")
    print(f"timeloop-mapper/timeloop-model/Accelergy Docker run; see {OUT_ROOT} for the real")
    print("artifact tree (resolved.stats.txt / resolved.ART.yaml per unique op shape).")


if __name__ == "__main__":
    main()
