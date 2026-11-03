#!/usr/bin/env python3
"""gpt-oss-120b MoE MLPBlock, rewritten as EXPERT-GROUPED GEMMs and built
ENTIRELY from this project's ``wafersim.primitives`` (including the new
``topk`` and ``clamp``).

Reference being modeled: ``MLPBlock`` in openai/gpt-oss
``gpt_oss/torch/model.py``. Real gpt-oss-120b numbers (all read from that
file's own ``ModelConfig``, not assumed): hidden_size=2880,
intermediate_size=2880, num_experts=128, experts_per_token=4,
swiglu_limit=7.0 (alpha=1.702 is a literal in ``swiglu``), RMSNorm eps=1e-5.

WHY NOT COPY THE REFERENCE LITERALLY. The reference does
``self.mlp1_weight[expert_indices, ...]`` -- it COPIES the 4 chosen experts'
weights out for every token, then runs two batched einsums. That is a
PyTorch convenience, not something real hardware does (weights stay put in
their PEs; tokens move to them). Costing it literally would charge a
nonexistent weight gather and a per-token GEMV instead of real GEMMs. So
this file restructures the block the way a MoE kernel actually runs:

  1. rms_norm(x)                                  -- same as reference
  2. gate GEMM (S,H)x(H,E) + bias                 -- router scores
  3. topk(k) over E, softmax over the k scores    -- routing decision
  4. DISPATCH: sort the S*k (token, expert) pairs by expert and ``gather``
     the normed rows into one expert-contiguous buffer (S*k, H). This
     replaces the reference's weight gather: TOKENS move, weights don't.
  5. per expert e with n_e > 0 routed rows: ONE real GEMM per matrix --
        (n_e,H)x(H,2I) -> +bias -> SwiGLU -> (n_e,I)x(I,H) -> +bias
     n_e is the REAL per-expert token count produced by step 3's real
     topk indices, so GEMM shapes are data-dependent exactly like on real
     hardware (average S*k/E, but uneven in practice).
  6. COMBINE: ``concat`` expert outputs, ``gather`` with the inverse
     permutation back to (token, slot) order, multiply by the softmax
     weights, ``sum`` over the k slots, add the residual.

SwiGLU, reference form: ``x_glu*sigmoid(alpha*x_glu)*(x_linear+1)`` with
x_glu clamped above at limit and x_linear clamped to [-limit, limit], the
two halves INTERLEAVED in the mlp1 output (``x[..., ::2]``/``x[..., 1::2]``).
  - clamp: the new ``clamp`` primitive (1 bound for glu, 2 for linear).
  - sigmoid: no sigmoid primitive; uses the exact identity
    ``x*sigmoid(a*x) == silu(a*x) * (1/a)`` -> mul, silu, mul.
  - interleave: ``slice_`` has no step, so ``MoEWeights.from_reference``
    de-interleaves mlp1's output columns ONCE at weight-load time (glu
    columns first, then linear columns), the same layout transform a real
    compiler does offline. It costs nothing at runtime, and the test suite
    checks the result against the interleaved reference.

TWO-PASS STRUCTURE (same convention as the Qwen3 script):
  - NUMERIC pass: real numpy through Tensor, no Docker. Checked against a
    numpy transcription of the reference in ``reference_moe_numpy``.
  - COST pass: ``CostTracker`` records every primitive call's real Docker
    cost (memoized per unique op shape). For the real 120b dims the 128
    experts' weights (~8.5 GB fp32 for mlp1 alone) are never materialized:
    GEMM cost depends only on shapes, and shapes depend only on routing.
    ``moe_mlp_block(..., cost_only_experts=True)`` runs the REAL router on
    real (random) gate weights, and for each expert calls
    ``cost_expert_ffn(n_e)`` instead of computing it. The test suite
    asserts this logs the IDENTICAL op sequence as the full numeric path.
  - ``CostTracker(dry_run=True)`` records shapes with zero cost and never
    touches Docker -- used by the tests (and handy for sanity checks).

KNOWN LIMITATIONS (named, not buried):
  - The routing bookkeeping (argsort of the S*k expert ids, bincount,
    offsets, inverse permutation) is done in plain numpy and is NOT costed.
    It is control-plane work (on hardware: building dispatch descriptors),
    small next to the GEMMs, but nonzero.
  - Elementwise ops are modeled un-tiled on one representative PE, like
    every other script here, so silu/mul/clamp over (n_e, I) dominate more
    than they would on a tiled mapping.
  - One MoE block only (no attention, no 36-layer loop). Real routing
    differs per layer, so multiplying one block by 36 is only a rough
    extrapolation.
  - fp32 / 4 bytes everywhere. gpt-oss runs bf16, so absolute numbers are
    not comparable with hardware until BYTES_PER_ELEMENT is revisited.
  - The area-budget fractions below are carried over from the Qwen3 run,
    NOT re-verified for these GEMM shapes; expect to revise them if the
    real Docker run reports an infeasible mapping.

Docker/TimeLoop failures RAISE -- no fallback to a fabricated cost number.

Usage (project root, venv active, Docker Desktop open):
  python3 experiment_gptoss_moe_mlp/gptoss_moe_mlp.py
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from wafersim.primitives import (
    AreaBudget,
    ClampSpec,
    ConcatStackSpec,
    ExpSpec,
    GatherSpec,
    GemmSpec,
    ReshapeSpec,
    RsqrtSpec,
    SiluSpec,
    SubSpec,
    Tensor,
    TopKSpec,
    WaferSpec,
    best_by_sim_time,
    clamp as p_clamp,
    concat as p_concat,
    derive_wafer_spec_from_area_budget,
    exp as p_exp,
    gather as p_gather,
    matmul as p_matmul,
    reshape as p_reshape,
    rsqrt as p_rsqrt,
    search_tile_k,
    silu as p_silu,
    simulate_clamp,
    simulate_concat_stack,
    simulate_exp,
    simulate_gather,
    simulate_reshape,
    simulate_rsqrt,
    simulate_silu,
    simulate_sub,
    simulate_topk,
    slice_ as p_slice,
    sub as p_sub,
    topk as p_topk,
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
from wafersim.primitives.sum import SumSpec, simulate_sum
from wafersim.tiling import Tiling

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DTYPE = np.float32
BYTES_PER_ELEMENT = 4
SEED = 0
RUN_HARDWARE_COST = True

# (name, number of tokens S). "decode" = one new token (S=1, so exactly k
# experts each see n_e=1); "prefill" = a short prompt. S=10 matches the
# Qwen3 script's PREFILL_SEQ_LEN, which was chosen to stay under the
# single-PE elementwise SRAM ceiling.
SCENARIOS = (("prefill", 10), ("decode", 1))

FIXED_AREA_BUDGET_UM2 = 800_000_000.0
AREA_COMPUTE_FRACTION = 0.15
AREA_MEMORY_FRACTION = 0.17
AREA_NETWORK_FRACTION = 0.68

# Elementwise/reduction PE tiling (see wafersim/tiling.py). main() runs BOTH
# modes on the same inputs so the result is a bracket, not a single guess:
# "mesh" (optimistic parallelism) and "serial" (the old one-PE behavior).
MESH_TILING = Tiling()
SERIAL_TILING = Tiling.serial()

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "gptoss_moe_mlp"


@dataclass(frozen=True)
class MoEConfig:
    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_experts: int = 128
    experts_per_token: int = 4
    swiglu_limit: float = 7.0
    swiglu_alpha: float = 1.702
    rms_norm_eps: float = 1e-5


GPT_OSS_120B = MoEConfig()
TOY = MoEConfig(hidden_size=16, intermediate_size=24, num_experts=8, experts_per_token=2)


# --------------------------------------------------------------------------
# Weights. "Reference layout" = exactly what model.py stores:
#   gate_weight (E,H), gate_bias (E,), mlp1_weight (E,2I,H), mlp1_bias (E,2I),
#   mlp2_weight (E,H,I), mlp2_bias (E,H), norm scale (H,).
# "Hardware layout" (MoEWeights) = matmul-ready (x @ W), de-interleaved.
# --------------------------------------------------------------------------


@dataclass
class MoEWeights:
    norm_scale: Tensor  # (H,)
    gate_w: Tensor  # (H, E)
    gate_b: Tensor  # (E,)
    w1: Optional[list] = None  # E x Tensor (H, 2I): cols [0:I]=glu, [I:2I]=linear
    b1: Optional[list] = None  # E x Tensor (2I,), same column order
    w2: Optional[list] = None  # E x Tensor (I, H)
    b2: Optional[list] = None  # E x Tensor (H,)

    @classmethod
    def from_reference(cls, ref: dict) -> "MoEWeights":
        f = lambda a: Tensor.from_values(np.ascontiguousarray(a, dtype=DTYPE))
        n_exp = ref["mlp1_weight"].shape[0]
        # de-interleave mlp1's 2I output columns: even -> glu, odd -> linear
        perm = np.concatenate([np.arange(0, ref["mlp1_weight"].shape[1], 2),
                               np.arange(1, ref["mlp1_weight"].shape[1], 2)])
        return cls(
            norm_scale=f(ref["norm_scale"]),
            gate_w=f(ref["gate_weight"].T),
            gate_b=f(ref["gate_bias"]),
            w1=[f(ref["mlp1_weight"][e][perm, :].T) for e in range(n_exp)],
            b1=[f(ref["mlp1_bias"][e][perm]) for e in range(n_exp)],
            w2=[f(ref["mlp2_weight"][e].T) for e in range(n_exp)],
            b2=[f(ref["mlp2_bias"][e]) for e in range(n_exp)],
        )


def random_reference_weights(cfg: MoEConfig, rng: np.random.Generator, scale: float = 0.2) -> dict:
    h, i, e = cfg.hidden_size, cfg.intermediate_size, cfg.num_experts
    r = lambda *s: (rng.standard_normal(s) * scale).astype(DTYPE)
    return {
        "norm_scale": (1.0 + 0.1 * rng.standard_normal(h)).astype(DTYPE),
        "gate_weight": r(e, h), "gate_bias": r(e),
        "mlp1_weight": r(e, 2 * i, h), "mlp1_bias": r(e, 2 * i),
        "mlp2_weight": r(e, h, i), "mlp2_bias": r(e, h),
    }


def gate_only_weights(cfg: MoEConfig, rng: np.random.Generator) -> MoEWeights:
    """Weights for cost-only runs: real gate + norm, NO expert weights."""
    h, e = cfg.hidden_size, cfg.num_experts
    return MoEWeights(
        norm_scale=Tensor.from_values(np.ones(h, dtype=DTYPE)),
        gate_w=Tensor.from_values((rng.standard_normal((h, e)) * 0.05).astype(DTYPE)),
        gate_b=Tensor.from_values(np.zeros(e, dtype=DTYPE)),
    )


# --------------------------------------------------------------------------
# Cost tracking (same memoize-per-unique-shape idea as the Qwen3 script)
# --------------------------------------------------------------------------

_DRY_ELEMENTWISE = SimpleNamespace(local_compute_cycles=0, local_energy_uj=0.0)
_DRY_GEMM = SimpleNamespace(result=SimpleNamespace(sim_time_cycles=0, energy=SimpleNamespace(total_uj=0.0)))


@dataclass
class CostTracker:
    """Memoized cost recorder. ``tiling`` decides how LOCAL parallel work
    (elementwise ops, clamp, reduction lanes, top-k rows) is split across
    PEs; see wafersim/tiling.py. Memory-movement ops (gather/concat/reshape)
    and GEMM are NOT tiled here: GEMM already maps over the mesh, and the
    data-movement ops cross PEs and need a network model (open item)."""

    wafer: Optional[WaferSpec] = None
    out_dir: Optional[Path] = None
    dry_run: bool = False  # True: record op shapes, zero cost, never touch Docker
    tiling: Tiling = field(default_factory=Tiling)
    dry_run_pes: int = 14_400  # mesh size assumed when there is no wafer (dry run only)
    total_cycles: int = 0
    total_energy_uj: float = 0.0
    unique_op_shapes: int = 0
    memoized_reuses: int = 0
    log: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict)
    _counter: int = 0

    def __post_init__(self) -> None:
        if not self.dry_run and (self.wafer is None or self.out_dir is None):
            raise ValueError("a real (non-dry-run) CostTracker needs wafer and out_dir")

    @property
    def total_pes(self) -> int:
        if self.wafer is not None:
            return self.wafer.mesh_rows * self.wafer.mesh_cols
        return self.dry_run_pes

    def _fresh_dir(self, tag: str) -> Path:
        self._counter += 1
        d = self.out_dir / tag / f"{self._counter:05d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _lookup(self, key: tuple, tag: str, build, dry):
        hit = key in self._cache
        if not hit:
            self._cache[key] = dry if self.dry_run else build(self._fresh_dir(tag))
        return self._cache[key], hit

    def _record(self, op: str, key: tuple, cycles, energy_uj, hit: bool,
                cycle_repeats: int = 1, energy_repeats: float = 1.0, **meta) -> None:
        """cycles = LATENCY of one call (busiest PE); energy_uj = energy of one
        simulated PE-slice. Totals: cycles*cycle_repeats, energy*energy_repeats."""
        if hit:
            self.memoized_reuses += 1
        else:
            self.unique_op_shapes += 1
        self.total_cycles += cycles * cycle_repeats
        self.total_energy_uj += energy_uj * energy_repeats
        self.log.append({"op": op, "key": key, "cache_hit": hit, "cycles": cycles,
                         "energy_uj": energy_uj, "cycle_repeats": cycle_repeats,
                         "energy_repeats": energy_repeats, **meta})

    def _elementwise(self, name, spec_cls, sim_fn, elements: int, bpe: int) -> None:
        sh = self.tiling.elements(elements, self.total_pes)
        key = (name, sh.per_pe, bpe)
        res, hit = self._lookup(
            key, name,
            lambda d: sim_fn(spec_cls(elements_per_pe=sh.per_pe, bytes_per_element=bpe), self.wafer, d),
            _DRY_ELEMENTWISE,
        )
        self._record(name, key, res.local_compute_cycles, res.local_energy_uj, hit,
                     energy_repeats=sh.energy_multiplier, elements=elements, per_pe=sh.per_pe, pes=sh.pes)

    def op_add(self, n, bpe=BYTES_PER_ELEMENT): self._elementwise("add", AddSpec, simulate_add, n, bpe)
    def op_mul(self, n, bpe=BYTES_PER_ELEMENT): self._elementwise("mul", MulSpec, simulate_mul, n, bpe)
    def op_div(self, n, bpe=BYTES_PER_ELEMENT): self._elementwise("div", DivSpec, simulate_div, n, bpe)
    def op_sub(self, n, bpe=BYTES_PER_ELEMENT): self._elementwise("sub", SubSpec, simulate_sub, n, bpe)
    def op_exp(self, n, bpe=BYTES_PER_ELEMENT): self._elementwise("exp", ExpSpec, simulate_exp, n, bpe)
    def op_rsqrt(self, n, bpe=BYTES_PER_ELEMENT): self._elementwise("rsqrt", RsqrtSpec, simulate_rsqrt, n, bpe)
    def op_silu(self, n, bpe=BYTES_PER_ELEMENT): self._elementwise("silu", SiluSpec, simulate_silu, n, bpe)

    def _reduction(self, name, spec_cls, sim_fn, k: int, lanes: int, bpe: int) -> None:
        """``lanes`` independent reductions over k elements each. Lanes run in
        parallel across PEs (latency = lanes_per_pe * one lane); a single lane
        of k elements stays serial on one PE."""
        sh = self.tiling.lanes(lanes, self.total_pes)
        key = (name, k, bpe)
        res, hit = self._lookup(
            key, name,
            lambda d: sim_fn(spec_cls(elements_per_pe=k, bytes_per_element=bpe), self.wafer, d),
            _DRY_ELEMENTWISE,
        )
        self._record(name, key, res.local_compute_cycles, res.local_energy_uj, hit,
                     cycle_repeats=sh.per_pe, energy_repeats=lanes, lanes=lanes, pes=sh.pes)

    def op_sum(self, k, lanes, bpe=BYTES_PER_ELEMENT): self._reduction("sum", SumSpec, simulate_sum, k, lanes, bpe)
    def op_max(self, k, lanes, bpe=BYTES_PER_ELEMENT): self._reduction("max", MaxSpec, simulate_max, k, lanes, bpe)

    def op_clamp(self, elements: int, num_bounds: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        sh = self.tiling.elements(elements, self.total_pes)
        key = ("clamp", sh.per_pe, num_bounds, bpe)
        res, hit = self._lookup(
            key, "clamp",
            lambda d: simulate_clamp(ClampSpec(sh.per_pe, num_bounds, bpe), self.wafer, d),
            _DRY_ELEMENTWISE,
        )
        self._record("clamp", key, res.local_compute_cycles, res.local_energy_uj, hit,
                     energy_repeats=sh.energy_multiplier, elements=elements, per_pe=sh.per_pe, pes=sh.pes)

    def op_topk(self, rows: int, num_elements: int, k: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        sh = self.tiling.lanes(rows, self.total_pes)  # rows are independent
        key = ("topk", sh.per_pe, num_elements, k, bpe)
        res, hit = self._lookup(
            key, "topk",
            lambda d: simulate_topk(TopKSpec(sh.per_pe, num_elements, k, bpe), self.wafer, d),
            _DRY_ELEMENTWISE,
        )
        self._record("topk", key, res.local_compute_cycles, res.local_energy_uj, hit,
                     energy_repeats=sh.energy_multiplier, rows=rows, per_pe=sh.per_pe, pes=sh.pes)

    # ---- NOT tiled (serial, conservative; see class docstring) ----------
    def op_reshape(self, tensor_before: Tensor, shape: tuple) -> None:
        spec = ReshapeSpec.from_call(tensor_before, shape)
        key = ("reshape", spec.elements_per_pe, spec.bytes_per_element, spec.copied)
        res, hit = self._lookup(key, "reshape", lambda d: simulate_reshape(spec, self.wafer, d), _DRY_ELEMENTWISE)
        self._record("reshape", key, res.local_compute_cycles, res.local_energy_uj, hit)

    def op_gather(self, out: Tensor) -> None:
        spec = GatherSpec.from_result(out)
        key = ("gather", spec.elements_per_pe, spec.bytes_per_element)
        res, hit = self._lookup(key, "gather", lambda d: simulate_gather(spec, self.wafer, d), _DRY_ELEMENTWISE)
        self._record("gather", key, res.local_compute_cycles, res.local_energy_uj, hit)

    def op_concat(self, out: Tensor) -> None:
        spec = ConcatStackSpec.from_result(out)
        key = ("concat", spec.elements_per_pe, spec.bytes_per_element)
        res, hit = self._lookup(key, "concat", lambda d: simulate_concat_stack(spec, self.wafer, d), _DRY_ELEMENTWISE)
        self._record("concat", key, res.local_compute_cycles, res.local_energy_uj, hit)

    def op_gemm(self, m: int, n: int, k: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        key = ("gemm", m, n, k, bpe)

        def build(d):
            return best_by_sim_time(search_tile_k(GemmSpec(m=m, n=n, k=k, bytes_per_element=bpe), self.wafer, d))

        cand, hit = self._lookup(key, "gemm", build, _DRY_GEMM)
        self._record("gemm", key, cand.result.sim_time_cycles, cand.result.energy.total_uj, hit)


# --------------------------------------------------------------------------
# Cost-tracked primitive wrappers (cost=None -> numeric only)
# --------------------------------------------------------------------------


def _bpe(t: Tensor) -> int:
    return t.bytes_per_element


def op_add(a, b, cost):
    out = p_add(a, b)
    if cost is not None: cost.op_add(out.size, _bpe(out))
    return out


def op_mul(a, b, cost):
    out = p_mul(a, b)
    if cost is not None: cost.op_mul(out.size, _bpe(out))
    return out


def op_div(a, b, cost):
    out = p_div(a, b)
    if cost is not None: cost.op_div(out.size, _bpe(out))
    return out


def op_sub(a, b, cost):
    out = p_sub(a, b)
    if cost is not None: cost.op_sub(out.size, _bpe(out))
    return out


def op_exp(x, cost):
    out = p_exp(x)
    if cost is not None: cost.op_exp(out.size, _bpe(out))
    return out


def op_rsqrt(x, cost):
    out = p_rsqrt(x)
    if cost is not None: cost.op_rsqrt(out.size, _bpe(out))
    return out


def op_silu(x, cost):
    out = p_silu(x)
    if cost is not None: cost.op_silu(out.size, _bpe(out))
    return out


def op_sum(x, dim, cost):
    out = p_sum(x, dim)
    if cost is not None:
        k = x.shape[dim]
        cost.op_sum(k, x.size // k, _bpe(x))
    return out


def op_max(x, dim, cost):
    out = p_max(x, dim)
    if cost is not None:
        k = x.shape[dim]
        cost.op_max(k, x.size // k, _bpe(x))
    return out


def op_clamp(x, cost, lo=None, hi=None):
    out = p_clamp(x, min=lo, max=hi)
    if cost is not None:
        cost.op_clamp(out.size, int(lo is not None) + int(hi is not None), _bpe(out))
    return out


def op_topk(x, k, cost):
    vals, idx = p_topk(x, k, dim=-1)
    if cost is not None:
        cost.op_topk(x.size // x.shape[-1], x.shape[-1], k, _bpe(x))
    return vals, idx


def op_reshape(x, shape, cost):
    out = p_reshape(x, shape)
    if cost is not None: cost.op_reshape(x, shape)
    return out


def op_gather(table, dim, index, cost):
    out = p_gather(table, dim, index)
    if cost is not None: cost.op_gather(out)
    return out


def op_concat(tensors, dim, cost):
    out = p_concat(tensors, dim)
    if cost is not None: cost.op_concat(out)
    return out


def op_matmul(a, b, cost):
    out = p_matmul(a, b)
    if cost is not None:
        cost.op_gemm(a.shape[-2], b.shape[-1], a.shape[-1], _bpe(a))
    return out


def _const(shape, value) -> Tensor:
    return Tensor.from_values(np.full(shape, value, dtype=DTYPE))


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def rms_norm(x: Tensor, weight: Tensor, eps: float, cost) -> Tensor:
    xx = op_mul(x, x, cost)
    ss = op_sum(xx, -1, cost)
    mean = op_mul(ss, _const(ss.shape, 1.0 / x.shape[-1]), cost)
    inv = op_rsqrt(op_add(mean, _const(mean.shape, eps), cost), cost)
    return op_mul(op_mul(x, inv, cost), weight, cost)


def softmax_lastdim(x: Tensor, cost) -> Tensor:
    shifted = op_sub(x, op_max(x, -1, cost), cost)
    e = op_exp(shifted, cost)
    return op_div(e, op_sum(e, -1, cost), cost)


@dataclass(frozen=True)
class Routing:
    """Expert-grouped view of a (S, k) top-k index tensor. Pure bookkeeping
    (see module docstring KNOWN LIMITATIONS: not costed)."""

    order: np.ndarray  # flat (token,slot) positions, stable-sorted by expert id
    token_of_sorted: np.ndarray  # token id of each expert-sorted row
    inverse: np.ndarray  # inverse permutation: sorted row of each flat position
    counts: np.ndarray  # (E,) routed rows per expert -- the real n_e
    offsets: np.ndarray  # (E,) first sorted row of each expert


def build_routing(indices: np.ndarray, num_experts: int) -> Routing:
    s, k = indices.shape
    flat = indices.reshape(-1)
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=num_experts)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return Routing(order, order // k, inverse, counts, offsets)


def expert_ffn(x_e: Tensor, w: MoEWeights, e: int, cfg: MoEConfig, cost) -> Tensor:
    """One expert's FFN on its n_e routed rows: (n_e,H) -> (n_e,H)."""
    i = cfg.intermediate_size
    h = op_add(op_matmul(x_e, w.w1[e], cost), w.b1[e], cost)  # (n_e, 2I)
    glu = op_clamp(p_slice(h, 1, 0, i), cost, hi=cfg.swiglu_limit)
    lin = op_clamp(p_slice(h, 1, i, 2 * i), cost, lo=-cfg.swiglu_limit, hi=cfg.swiglu_limit)
    scaled = op_mul(glu, _const(glu.shape, cfg.swiglu_alpha), cost)
    out_glu = op_mul(op_silu(scaled, cost), _const(glu.shape, 1.0 / cfg.swiglu_alpha), cost)
    act = op_mul(out_glu, op_add(lin, _const(lin.shape, 1.0), cost), cost)
    return op_add(op_matmul(act, w.w2[e], cost), w.b2[e], cost)


def cost_expert_ffn(n: int, cfg: MoEConfig, cost: CostTracker) -> None:
    """Record EXACTLY the cost ops ``expert_ffn`` would, from shapes alone
    (no expert weights needed). Kept in the same order; a test asserts the
    two logs are identical."""
    h, i = cfg.hidden_size, cfg.intermediate_size
    cost.op_gemm(n, 2 * i, h)
    cost.op_add(n * 2 * i)
    cost.op_clamp(n * i, 1)
    cost.op_clamp(n * i, 2)
    cost.op_mul(n * i)
    cost.op_silu(n * i)
    cost.op_mul(n * i)
    cost.op_add(n * i)
    cost.op_mul(n * i)
    cost.op_gemm(n, h, i)
    cost.op_add(n * h)


def moe_mlp_block(
    x: Tensor, w: MoEWeights, cfg: MoEConfig, cost: Optional[CostTracker], cost_only_experts: bool = False
) -> tuple[Tensor, Routing]:
    """Expert-grouped gpt-oss MLPBlock: ``x + MoE(rms_norm(x))``.

    ``cost_only_experts=True`` runs the real router but charges each expert
    via ``cost_expert_ffn`` instead of computing it (needs ``cost``; the
    returned tensor is then NOT a meaningful numeric result)."""
    if cost_only_experts and cost is None:
        raise ValueError("cost_only_experts=True requires a CostTracker")
    s, h, k = x.shape[0], cfg.hidden_size, cfg.experts_per_token

    normed = rms_norm(x, w.norm_scale, cfg.rms_norm_eps, cost)
    logits = op_add(op_matmul(normed, w.gate_w, cost), w.gate_b, cost)  # (S, E)
    values, indices = op_topk(logits, k, cost)  # (S, k) each
    weights = softmax_lastdim(values, cost)  # (S, k)

    routing = build_routing(indices.data, cfg.num_experts)
    x_sorted = op_gather(normed, 0, routing.token_of_sorted, cost)  # (S*k, H) DISPATCH

    ys = []
    for e in range(cfg.num_experts):
        n = int(routing.counts[e])
        if n == 0:
            continue
        if cost_only_experts:
            cost_expert_ffn(n, cfg, cost)
            ys.append(Tensor.from_values(np.zeros((n, h), dtype=DTYPE)))
        else:
            off = int(routing.offsets[e])
            ys.append(expert_ffn(p_slice(x_sorted, 0, off, off + n), w, e, cfg, cost))

    y_sorted = op_concat(ys, 0, cost)  # (S*k, H)
    y_flat = op_gather(y_sorted, 0, routing.inverse, cost)  # back to (token, slot) order
    y3 = op_reshape(y_flat, (s, k, h), cost)
    w3 = op_reshape(weights, (s, k, 1), cost)
    combined = op_reshape(op_sum(op_mul(y3, w3, cost), 1, cost), (s, h), cost)  # COMBINE
    return op_add(x, combined, cost), routing


# --------------------------------------------------------------------------
# Plain-numpy transcription of the reference MLPBlock (for correctness checks)
# --------------------------------------------------------------------------


def reference_moe_numpy(x: np.ndarray, ref: dict, cfg: MoEConfig) -> np.ndarray:
    """Line-by-line numpy port of gpt_oss/torch/model.py ``MLPBlock.forward``
    (literal per-token weight gather and einsums, interleaved SwiGLU)."""
    x = x.astype(np.float32)
    t = x * (1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + cfg.rms_norm_eps))
    t = (t * ref["norm_scale"]).astype(np.float32)
    g = t @ ref["gate_weight"].T + ref["gate_bias"]
    idx = np.argsort(-g, axis=-1, kind="stable")[:, : cfg.experts_per_token]
    vals = np.take_along_axis(g, idx, axis=-1)
    ev = np.exp(vals - vals.max(axis=1, keepdims=True))
    wts = ev / ev.sum(axis=1, keepdims=True)

    t1 = np.einsum("beck,bk->bec", ref["mlp1_weight"][idx], t) + ref["mlp1_bias"][idx]
    x_glu, x_lin = t1[..., ::2], t1[..., 1::2]
    x_glu = np.minimum(x_glu, cfg.swiglu_limit)
    x_lin = np.clip(x_lin, -cfg.swiglu_limit, cfg.swiglu_limit)
    act = x_glu * (1.0 / (1.0 + np.exp(-cfg.swiglu_alpha * x_glu))) * (x_lin + 1.0)
    t2 = np.einsum("beck,bek->bec", ref["mlp2_weight"][idx], act) + ref["mlp2_bias"][idx]
    return (x + np.einsum("bec,be->bc", t2, wts)).astype(np.float32)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _print_routing(routing: Routing, cfg: MoEConfig) -> None:
    active = routing.counts[routing.counts > 0]
    print(f"  routing: {len(active)}/{cfg.num_experts} experts active, rows/expert "
          f"min={active.min()} max={active.max()} mean={active.mean():.2f} "
          f"(total rows {int(routing.counts.sum())})")


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("STEP 1: NUMERIC check at toy dims vs. numpy transcription of the reference MLPBlock")
    print("=" * 78)
    ref = random_reference_weights(TOY, rng)
    xs = rng.standard_normal((10, TOY.hidden_size)).astype(DTYPE)
    out, routing = moe_mlp_block(Tensor.from_values(xs), MoEWeights.from_reference(ref), TOY, None)
    expected = reference_moe_numpy(xs, ref, TOY)
    err = float(np.max(np.abs(out.data - expected)))
    print(f"toy config: {TOY}")
    _print_routing(routing, TOY)
    print(f"max |expert-grouped - reference| = {err:.3e}")
    if not (np.isfinite(out.data).all() and np.allclose(out.data, expected, rtol=1e-4, atol=1e-5)):
        raise RuntimeError("expert-grouped MoE disagrees with the reference -- composition bug")
    print("numeric check PASSED")

    if not RUN_HARDWARE_COST:
        print("RUN_HARDWARE_COST is False -- skipping the real Docker hardware-cost pass.")
        return

    print()
    print("=" * 78)
    print("STEP 2: HARDWARE-COST pass at REAL gpt-oss-120b dims (real Docker/TimeLoop)")
    print("=" * 78)
    cfg = GPT_OSS_120B
    print(f"config: {cfg}")
    budget = AreaBudget(
        wafer_area_budget_um2=FIXED_AREA_BUDGET_UM2,
        compute_area_fraction=AREA_COMPUTE_FRACTION,
        memory_area_fraction=AREA_MEMORY_FRACTION,
        network_area_fraction=AREA_NETWORK_FRACTION,
    )
    largest_gemm = GemmSpec(m=SCENARIOS[0][1], n=2 * cfg.intermediate_size, k=cfg.hidden_size,
                            bytes_per_element=BYTES_PER_ELEMENT)
    print(f"deriving WaferSpec from area budget {FIXED_AREA_BUDGET_UM2:,.0f} um^2 (representative GEMM "
          f"m={largest_gemm.m} n={largest_gemm.n} k={largest_gemm.k}) ...")
    area_result = derive_wafer_spec_from_area_budget(budget, largest_gemm, WaferSpec(), OUT_ROOT / "area_budget_derivation")
    wafer = area_result.wafer
    print(f"derived wafer: mesh {wafer.mesh_rows}x{wafer.mesh_cols}, sram_kib_per_pe={wafer.sram_kib_per_pe}, "
          f"fifo_capacity_packets={wafer.fifo_capacity_packets}")

    weights = gate_only_weights(cfg, rng)  # real gate + norm; expert weights never materialized
    # Same inputs (hence the same real routing) for BOTH tilings, so the two
    # runs differ ONLY in how local parallel work is mapped onto PEs.
    inputs = {name: Tensor.from_values(rng.standard_normal((s, cfg.hidden_size)).astype(DTYPE))
              for name, s in SCENARIOS}
    shared_cache: dict = {}  # GEMMs / data-movement ops are tiling-independent -> simulate once
    results = {}
    for label, tiling in (("mesh", MESH_TILING), ("serial", SERIAL_TILING)):
        cost = CostTracker(wafer=wafer, out_dir=OUT_ROOT / "primitive_cost_traces" / label,
                           tiling=tiling, _cache=shared_cache)
        print(f"\n##### tiling = {tiling.describe(cost.total_pes)}")
        per_scn = {}
        for name, s in SCENARIOS:
            c0, e0, t0 = cost.total_cycles, cost.total_energy_uj, time.time()
            _, routing = moe_mlp_block(inputs[name], weights, cfg, cost, cost_only_experts=True)
            per_scn[name] = (cost.total_cycles - c0, cost.total_energy_uj - e0)
            print(f"[{name}: S={s}] ", end="")
            _print_routing(routing, cfg)
            print(f"    cycles: {per_scn[name][0]:,.0f}   energy: {per_scn[name][1]:.4f} uJ   wall: {time.time() - t0:.1f}s")
        results[label] = (cost, per_scn)

    print()
    print("=" * 78)
    print("BRACKET: cycles/energy of ONE MoE block (the truth lies between the two modes)")
    print("=" * 78)
    print(f"{'scenario':<10} {'mesh cycles':>16} {'serial cycles':>16} {'mesh energy uJ':>16} {'serial energy uJ':>18}")
    for name, _ in SCENARIOS:
        (mc, me), (sc, se) = results["mesh"][1][name], results["serial"][1][name]
        print(f"{name:<10} {mc:>16,.0f} {sc:>16,.0f} {me:>16.2f} {se:>18.2f}")
    for label in ("mesh", "serial"):
        cost = results[label][0]
        print(f"\n--- per-op breakdown, tiling={label} ---")
        by_op: dict = defaultdict(lambda: [0, 0.0, 0.0])
        for e in cost.log:
            agg = by_op[e["op"]]
            agg[0] += 1
            agg[1] += e["cycles"] * e["cycle_repeats"]
            agg[2] += e["energy_uj"] * e["energy_repeats"]
        total_cyc = sum(v[1] for v in by_op.values())
        print(f"{'op':<10} {'calls':>7} {'total_cycles':>16} {'share':>7} {'total_energy_uj':>18}")
        for op_name, (calls, cyc, en) in sorted(by_op.items(), key=lambda kv: -kv[1][1]):
            print(f"{op_name:<10} {calls:>7} {cyc:>16,.0f} {cyc / total_cyc:>7.1%} {en:>18.4f}")
    print("\nNOT tiled in either mode (serial on one PE): gather, concat, reshape -- they cross PEs and need a")
    print("network model; in 'mesh' mode they are therefore over-represented in the cycle share.")
    print(f"\nreal artifact tree: {OUT_ROOT}")


if __name__ == "__main__":
    main()
