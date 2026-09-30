#!/usr/bin/env python3
"""Full Qwen3-8B forward-pass composition (PREFILL + autoregressive DECODE
with real per-layer KV-cache), built ENTIRELY from this project's already-
implemented, real ``wafersim.primitives``.

This is a direct follow-up to ``experiment_qwen3_primitive_composition/
qwen3_primitive_composition.py`` (the tiny-config, prefill-only version),
now driven by the user's FULL, complete reference file (uploaded this
round -- ``qwen3_compute_sim.py``, not truncated this time). Reading that
file end-to-end surfaced three things the tiny version deliberately left
out, all now implemented here:
  1. Real token embedding lookup (``embed_tokens``, an ``nn.Embedding`` in
     the reference) -- done here via the real ``gather`` primitive.
  2. Real LM head (``lm_head``, a Linear to vocab_size) -- done here via
     the real ``matmul`` primitive, producing real vocab-dimension logits.
  3. Real autoregressive DECODE with a real per-layer KV-cache (the
     reference's ``LayerKVCache``: prefill seeds the cache, each decode
     step appends exactly one new (k,v) pair and re-attends over the
     whole cache, no causal mask needed for the single new query token).
     Implemented here via the real ``concat`` primitive for the cache
     append and the real ``repeat`` primitive for the GQA expand, exactly
     mirroring the reference's ``cache.append()`` / ``repeat_kv()``.

SCOPE decided with the user this round (via AskUserQuestion):
  - decode + KV-cache: YES (previous round deferred this; user asked for
    it explicitly this time).
  - model scale: the REAL Qwen3Config.qwen3_8b() from the now-complete
    reference file -- hidden_size=4096, intermediate_size=12288,
    num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8,
    head_dim=128, vocab_size=151936, rms_norm_eps=1e-6,
    rope_theta=1_000_000.0. Every one of these numbers is now REAL and
    GIVEN (not assumed) -- confirmed straight from the uploaded file's own
    ``Qwen3Config`` dataclass, which was truncated in the earlier partial
    paste this project worked from before.
  - input scale: user chose the smaller of the two offered options --
    BATCH=1 (the reference supports batch, but every primitive/Tensor op
    in this project is 2D/3D-shaped, no batch axis modeled -- staying at
    an effective batch of 1 avoids inventing a 4th tensor axis nothing
    else in this project's primitive set expects), PREFILL_SEQ_LEN=10
    (originally 64/unchanged-from-the-tiny-run, REVISED after a real
    Docker crash -- see the "REAL DOCKER CRASH" note below),
    DECODE_STEPS=6 (middle of the offered 4~8 range).

REAL DOCKER CRASH (2026-09-11, first real hardware-cost attempt at
PREFILL_SEQ_LEN=64) AND WHY IT IS NOT AN AREA-FRACTION PROBLEM: that run
derived a real wafer (mesh 64x256, real area used ~108.6M/6.2K/28.3K um^2
compute/memory/network) and then real-crashed 44 minutes in, inside layer
1's MLP ``silu(gate)`` call, with a ``ValueError`` from
``timeloop.py``'s ``_validate_cacti_safe_sram``: the real
``pe_scratchpad`` depth this op declared implied an 84,934,656-byte
(~81 MiB) CACTI ``cache_size``, 5.06x above this project's confirmed-safe
16 MiB ceiling (``SAFE_CACTI_CACHE_SIZE_BYTES`` in ``timeloop.py`` --
a real, Docker-confirmed CACTI integer-handling bug guard, not a
tunable). Reading ``wafersim.primitives.mul/exp/silu`` real source
confirmed the exact, arithmetic-exact cause: every elementwise primitive
in this project (mul/add/exp/silu/...) is modeled as ONE un-tiled
"representative PE" holding the op's ENTIRE element count at once
(``pe_scratchpad_depth_words = 3 * elements_per_pe`` in
``_timeloop_mul.py``) -- this is completely independent of
``WaferSpec``/mesh size/``AREA_COMPUTE_MEMORY_NETWORK_FRACTION``, so
re-running with a different area split would have hit the IDENTICAL
crash. The real numbers: gate tensor = PREFILL_SEQ_LEN(64) x
intermediate_size(12288) = 786,432 elements; ``simulate_silu`` calls
``simulate_exp`` which (see ``exp/_timeloop_exp.py``'s
``_MULS_PER_ELEMENT = 2 + _POLY_DEGREE + 1 = 9``, real range-reduction(2)
+ polynomial(6) + combine(1)) issues an internal ``MulSpec`` with
``elements_per_pe = 786,432 * 9 = 7,077,888``, giving
``3 * 7,077,888 = 21,233,664`` words -> exactly the 84,934,656-byte
cache_size the real crash reported. This is the first time this project
has fed an elementwise primitive a tensor this wide (prior real runs:
the tiny Qwen3 config's intermediate_size=768, and experiment1/2's pure
GEMM sweeps). The user was asked (real, not assumed) how to proceed and
chose to shrink PREFILL_SEQ_LEN rather than have the elementwise
primitives' real TimeLoop modeling rewritten to tile across the mesh
(the architecturally "correct" fix, but a much larger change touching
shared ``wafersim.primitives`` code every other experiment in this
project also depends on). PREFILL_SEQ_LEN=10 keeps real margin: at 10,
the same SiLU call's internal mul is
``10*12288*9 = 1,105,920`` elements_per_pe -> ``3,317,760`` words ->
a real 13,271,040-byte (~12.65 MiB) cache_size, about 79% of the 16 MiB
ceiling (PREFILL_SEQ_LEN=12 is the real arithmetic maximum that still
fits; 10 was chosen to leave headroom rather than sit right at the edge
for an untested real run). DECODE_STEPS was left at 6 -- every decode
step's own MLP call only ever sees a single new token
(1*12288*9 = 110,592 elements_per_pe, far under the ceiling), so the
PREFILL SiLU call analyzed above is the real peak element count anywhere
in this whole composition.
  - area budget: FIXED_AREA_BUDGET_UM2=800,000,000 um^2 -- the user chose
    to switch to the 800M baseline already used throughout
    experiment1_gemm_size/experiment2_area_sweep, instead of reusing the
    tiny run's 400M point, specifically so this result is horizontally
    comparable to that project's own already-collected real data at 800M.

A REAL, ARITHMETIC-CHECKED MEMORY PROBLEM AND ITS RESOLUTION (read this
before changing NUM_HIDDEN_LAYERS or HIDDEN_SIZE/INTERMEDIATE_SIZE/
VOCAB_SIZE below -- this is not a hypothetical concern, the naive
approach genuinely does not fit in memory):

  Naively building ``num_hidden_layers`` INDEPENDENT sets of real random
  weights at this config's real sizes costs, per layer (wq+wk+wv+wo+
  gate+up+down, float32): ~193M elements -> ~772 MB. Times 36 real
  layers: ~27.8 GB, on top of a real embed_tokens table
  (vocab_size x hidden_size = 151,936 x 4,096 = ~622M elements -> ~2.49
  GB) and a real lm_head weight of the same size (~2.49 GB) -- roughly
  33 GB total, just for weight tensors, before a single forward pass
  runs. That is not a safe assumption to build against in this sandboxed
  Mac terminal session.

  This project's own OWN foundational, already-established discipline
  supplies the exact fix, not a new hack: TimeLoop/Accelergy's real cost
  model is a pure function of an operation's SHAPE/COUNT, never of the
  tensor's actual data VALUES (see ``primitives/tensor``'s own module
  docstring, and this project's memoization design already relied on
  this exact fact for the tiny run). Two consequences, both applied here:

  (a) SHARED layer weights: since every one of the 36 real decoder layers
      has the IDENTICAL real architecture (same shapes throughout -- only
      the specific real random VALUES would differ per layer, which
      TimeLoop cannot see and does not affect its real cost), this script
      builds ONE real ``LayerWeights`` object and reuses the SAME Python
      object reference for all 36 layers, instead of building 36
      independently-random copies. This cuts weight memory from ~27.8 GB
      to ~772 MB, and changes NOTHING about the real hardware-cost
      numbers reported (they were already shape-only and identical
      across layers even in the naive version -- the memoization cache
      would have collapsed 36 layers' calls to the same real Docker
      results regardless).

  (b) SMALL stand-in vocab for the NUMERIC-only correctness check: the
      real ``gather`` primitive's cost (``GatherSpec.from_result``) is
      derived from the GATHERED OUTPUT tensor's real ``.size`` alone --
      confirmed by reading ``slice_gather/_timeloop_gather.py`` this
      round -- it does NOT depend on the source embedding table's row
      count at all (``np.take(table, index, axis=0)``'s output shape is
      fixed by ``len(index)`` and the table's OTHER axis, never by the
      table's row count). So a small stand-in embed_tokens table of shape
      (NUMERIC_STANDIN_VOCAB, hidden_size) with NUMERIC_STANDIN_VOCAB=1024
      (not the real 151,936) produces an IDENTICAL real gather cost to
      what the full real table would -- there is no approximation here at
      all, the real cost is exactly correct either way. For the LM head
      (a real ``matmul``, whose cost DOES depend on the output's vocab
      dimension), this script uses a small stand-in weight matrix for the
      NUMERIC pass only (to keep memory small while still exercising a
      real matmul and checking the output is finite), and separately
      issues the REAL hardware-cost call with the TRUE vocab_size=151,936
      via a direct ``cost.op_gemm(m, VOCAB_SIZE, k)`` call (this
      project's own ``CostTracker.op_gemm`` already takes explicit
      integers, not tensors, so this is not a new code path -- see
      ``op_matmul``'s tensor-shape-inferring wrapper vs. calling
      ``cost.op_gemm`` directly). The real LM head hardware cost reported
      by this script is therefore exactly correct for the true 151,936
      vocab dimension, even though the NUMERIC check ran against a
      smaller stand-in matrix.

  Neither (a) nor (b) touches the real hardware-cost numbers reported by
  STEP 2 below -- both are purely about what the STEP 1 numeric-only
  sanity check needs to allocate in real RAM to prove the composition's
  plumbing is correct (shapes align, output is finite). STEP 2's real
  Docker/TimeLoop calls always declare the TRUE architecture dimensions.

MISSING PRIMITIVE (unchanged from the tiny run): no dedicated ``sub``
primitive -- every subtraction (RoPE's ``-x2``, softmax's ``x - row_max``)
is composed via negate-by-mul(-1) + add, same as before.

DECODE DESIGN (mirrors the reference file's ``LayerKVCache``/
``Qwen3Attention.decode`` exactly):
  - Real per-layer cache: after PREFILL, each layer's real post-RoPE,
    pre-GQA-expand (S=PREFILL_SEQ_LEN, KVH, D) k/v tensors ARE the
    initial real cache (assigned directly, no concat needed for the very
    first fill -- exactly matching the reference's
    ``if self.k is None: self.k = k_new`` branch).
  - Each decode step computes a real new (1, KVH, D) k/v for the single
    new token, then REAL-concatenates it onto that layer's existing
    cache via the real ``concat`` primitive (mirrors ``torch.cat(...,
    dim=2)`` in the reference, here along this project's sequence axis).
  - GQA expand (real ``repeat``) is applied to the CURRENT cache (whatever
    length it is) every step, exactly like the reference's
    ``repeat_kv(k_all, n_rep)`` called fresh every decode step.
  - No causal mask during decode (matches the reference's own comment:
    "No causal mask is needed here: the cache contains only positions <=
    current token.") -- the single new query token legitimately attends
    to every real cached position.
  - RoPE position offset: PREFILL uses real positions 0..PREFILL_SEQ_LEN-1
    (unchanged from the tiny run); each decode step t (0-indexed) uses
    the real single position PREFILL_SEQ_LEN + t, mirroring the
    reference's ``position = cache.length``.
  - "Next token" selection between decode steps: this project has no
    sampling/argmax primitive (out of scope -- not part of the real
    16-primitive set this whole composition is built from), and since
    every weight/embedding here is already real-but-meaningless random
    data (not a pretrained model -- see the tiny run's own module
    docstring for why that is fine for this question), this script picks
    each decode step's "next token id" via a real ``numpy.argmax`` on the
    real (small-vocab, numeric-pass-only) logits -- a real, deterministic,
    reproducible choice, not a primitive-costed operation (matches the
    tiny run's treatment of ``cast`` as the one "illustrative, not
    otherwise required" real op left outside the strict cost trace).

Docker/TimeLoop failures RAISE here, same as every other script in this
project -- no fallback to a locally-estimated or fabricated cost number.
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
    CosSpec,
    ExpSpec,
    GatherSpec,
    GemmSpec,
    RepeatSpec,
    ReshapeSpec,
    RsqrtSpec,
    SiluSpec,
    SinSpec,
    SubSpec,
    Tensor,
    WaferSpec,
    WhereSpec,
    best_by_sim_time,
    cast as p_cast,
    concat as p_concat,
    cos as p_cos,
    derive_wafer_spec_from_area_budget,
    exp as p_exp,
    gather as p_gather,
    matmul as p_matmul,
    reshape as p_reshape,
    rsqrt as p_rsqrt,
    search_tile_k,
    silu as p_silu,
    simulate_cast,
    simulate_concat_stack,
    simulate_cos,
    simulate_exp,
    simulate_gather,
    simulate_repeat,
    simulate_reshape,
    simulate_rsqrt,
    simulate_silu,
    simulate_sin,
    simulate_sub,
    simulate_where,
    sin as p_sin,
    slice_ as p_slice,
    sub as p_sub,
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
# Configuration -- see module docstring for what is real/given vs. a
# deliberate stand-in for memory reasons only.
# --------------------------------------------------------------------------

DTYPE = np.float32
BYTES_PER_ELEMENT = 4
SEED = 0

PREFILL_SEQ_LEN = 10  # REVISED from 64 after a real Docker crash (elementwise
# single-PE SRAM ceiling) -- see module docstring's "REAL DOCKER CRASH" note
# for the exact, source-grounded arithmetic behind this number.
DECODE_STEPS = 6  # user-chosen input scale (middle of the offered 4~8)

NUMERIC_STANDIN_VOCAB = 1024  # NOT the real vocab_size -- see module
# docstring part (b). Only used to keep the NUMERIC-only pass's embed_tokens
# / lm_head weight allocations small; the real hardware-cost pass always
# declares the true vocab_size=151,936 explicitly (see op_gemm call for the
# LM head in main()).

RUN_HARDWARE_COST = True  # flipped ON: user explicitly confirmed the numeric-only
# pass (STEP 1) already verified finite output across prefill+decode, and asked
# for the real hardware-cost pass now. This run is expected to be substantially
# more expensive than the tiny run: real GEMM shapes up to k=12,288/n=151,936
# far exceed anything previously area-budget-derived in this project at
# 800M um^2 -- see module docstring; failures here are real and expected
# to need at least one real-data-driven area-fraction fix, same pattern as
# every other area-budget script in this project so far.

FIXED_AREA_BUDGET_UM2 = 800_000_000.0  # user's explicit choice this round
# -- matches experiment1_gemm_size/experiment2_area_sweep's baseline point,
# NOT the tiny run's 400M point (see module docstring).
AREA_COMPUTE_FRACTION = 0.15
AREA_MEMORY_FRACTION = 0.17
AREA_NETWORK_FRACTION = 0.68
# ^ STARTING POINT ONLY, carried over from the tiny run's real, working
# split at 400M -- explicitly NOT verified yet at 800M against THIS
# script's real (much larger, much more skewed -- e.g. the LM head's
# m=64 n=151,936 k=4,096 GEMM) op shapes. This is very likely to need a
# real-error-driven revision once STEP 2 actually runs, exactly like the
# tiny run needed one revision of its own -- left as the best real
# starting guess, not a claim that it will already work.

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "qwen3_8b_prefill_decode"


@dataclass(frozen=True)
class Qwen3Config8B:
    """Real values, confirmed straight from the user's now-complete
    ``qwen3_compute_sim.py`` upload's own ``Qwen3Config`` dataclass --
    nothing here is assumed or truncated any more."""

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0


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


def build_layer_weights(rng: np.random.Generator, cfg: Qwen3Config8B) -> LayerWeights:
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
# Real, Docker-backed hardware-cost tracker -- identical to the tiny run's,
# unchanged (see that script for the memoization/lane-accounting rationale).
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

    def op_sub(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        # NEW this round -- real elementwise subtract, aliased to add's own
        # already-Docker-verified AddSpec/simulate_add (SubSpec IS AddSpec,
        # simulate_sub IS simulate_add -- see wafersim.primitives.sub's own
        # module docstring). Replaces this script's old negate-by-mul(-1)
        # + add workaround (RoPE's rotate_half, softmax's x - row_max) with
        # one real dedicated subtract firing.
        self._elementwise("sub", SubSpec, simulate_sub, elements, bpe)

    def op_sin(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        # NEW this round -- real range-reduction + degree-17 Taylor-
        # polynomial decomposition (mul/add/sub/bitwise), see
        # wafersim.primitives.sin's own module docstring. Used for RoPE's
        # rotation-angle sin table, previously precomputed via plain,
        # UNCOSTED numpy (see build_rope_tables' old docstring) -- this is
        # the first time that real hardware cost is actually counted.
        self._elementwise("sin", SinSpec, simulate_sin, elements, bpe)

    def op_cos(self, elements: int, bpe: int = BYTES_PER_ELEMENT) -> None:
        # NEW this round -- reuses `sin`'s real pipeline via the real
        # identity cos(x) = sin(x + pi/2), plus one real add firing for the
        # phase shift (see wafersim.primitives.cos's own module docstring).
        # Same "previously uncosted numpy" note as op_sin above.
        self._elementwise("cos", CosSpec, simulate_cos, elements, bpe)

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
# Cost-tracked primitive wrappers -- identical to the tiny run's, plus one
# NEW wrapper (op_gather) the tiny run did not need (no embedding lookup
# there). Zero-real-cost ops (view/permute/transpose/slice_) are called
# directly, no wrapper needed.
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


def op_sub(a: Tensor, b: Tensor, cost: CostTracker | None) -> Tensor:
    """NEW this round -- real elementwise ``a - b``, replacing this
    script's old negate-by-mul(-1) + add workaround. See
    wafersim.primitives.sub's own module docstring for why this is a
    deliberate alias of `add`'s already-Docker-verified cost model."""
    out = p_sub(a, b)
    if cost is not None:
        cost.op_sub(out.size, out.bytes_per_element)
    return out


def op_sin(x: Tensor, cost: CostTracker | None) -> Tensor:
    """NEW this round -- real elementwise ``sin(x)``, real range-reduction
    + degree-17 Taylor-polynomial hardware cost (see
    wafersim.primitives.sin's own module docstring)."""
    out = p_sin(x)
    if cost is not None:
        cost.op_sin(out.size, out.bytes_per_element)
    return out


def op_cos(x: Tensor, cost: CostTracker | None) -> Tensor:
    """NEW this round -- real elementwise ``cos(x)``, reusing `sin`'s real
    pipeline via ``cos(x) = sin(x + pi/2)`` (see
    wafersim.primitives.cos's own module docstring)."""
    out = p_cos(x)
    if cost is not None:
        cost.op_cos(out.size, out.bytes_per_element)
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


def op_gather(table: Tensor, dim: int, index, cost: CostTracker | None) -> Tensor:
    """NEW vs. the tiny run -- real token-embedding lookup. See module
    docstring part (b) for why ``table`` may be a small stand-in without
    affecting the real hardware-cost number this records."""
    out = p_gather(table, dim, index)
    if cost is not None:
        cost.op_gather(out, out.bytes_per_element)
    return out


def op_matmul(a: Tensor, b: Tensor, cost: CostTracker | None) -> Tensor:
    out = p_matmul(a, b)
    if cost is not None:
        m, k = a.shape[-2], a.shape[-1]
        n = b.shape[-1]
        cost.op_gemm(m, n, k, a.bytes_per_element)
    return out


def zeros_like(x: Tensor) -> Tensor:
    """Real zero constant matching ``x``'s shape/dtype -- used with the
    real `sub` primitive (``op_sub(zeros_like(x), x, cost)`` = real
    ``-x``) below. A constant creation, same as this script's other
    ``Tensor.from_values(np.full(...))`` idioms elsewhere -- not itself a
    costed primitive call."""
    return Tensor.from_values(np.zeros(x.shape, dtype=x.dtype))


# --------------------------------------------------------------------------
# Model building blocks
# --------------------------------------------------------------------------


def rms_norm(x: Tensor, weight: Tensor, eps: float, cost: CostTracker | None) -> Tensor:
    xx = op_mul(x, x, cost)
    ss = op_sum(xx, dim=-1, cost=cost)
    n = x.shape[-1]
    recip_n = Tensor.from_values(np.full(ss.shape, 1.0 / n, dtype=x.dtype))
    mean = op_mul(ss, recip_n, cost)
    eps_t = Tensor.from_values(np.full(mean.shape, eps, dtype=x.dtype))
    var_eps = op_add(mean, eps_t, cost)
    inv = op_rsqrt(var_eps, cost)
    normed = op_mul(x, inv, cost)
    return op_mul(normed, weight, cost)


def build_rope_tables(
    num_positions: int, head_dim: int, theta: float, cost: CostTracker | None, start_pos: int = 0
) -> tuple[Tensor, Tensor]:
    """Real, precomputed RoPE cos/sin tables for real positions
    [start_pos, start_pos+num_positions). Shape (num_positions, 1, D).

    UPDATED this round -- cos/sin are now real, COST-TRACKED primitive
    calls (``op_cos``/``op_sin``, see wafersim.primitives.cos/sin's own
    module docstrings for the real Docker-backed range-reduction +
    degree-17 Taylor-polynomial decomposition), not plain uncosted numpy.
    Previously this table's real hardware cost was never counted anywhere
    in this script's STEP 2 total; it now is (pass ``cost=None`` for the
    numeric-only STEP 1 pass, same convention every other op_* wrapper in
    this script already uses). The angle table itself (inv_freq/positions/
    freqs/emb -- an outer product and a power, neither of which this
    project's primitive set models as a costed op) is still built via
    plain numpy: real, but data-independent scalar arithmetic, not an
    elementwise tensor primitive."""
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    positions = np.arange(start_pos, start_pos + num_positions, dtype=np.float64)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1).astype(DTYPE)
    emb_t = Tensor.from_values(emb)
    cos_t = op_cos(emb_t, cost)
    sin_t = op_sin(emb_t, cost)
    d = emb.shape[-1]
    cos = view(cos_t, (num_positions, 1, d))
    sin = view(sin_t, (num_positions, 1, d))
    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, cost: CostTracker | None) -> Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = p_slice(x, dim=-1, start=0, end=half)
    x2 = p_slice(x, dim=-1, start=half, end=d)
    # UPDATED this round -- real dedicated `sub` firing (0 - x2) instead of
    # the old negate-by-mul(-1) + (implicit) add workaround. See
    # wafersim.primitives.sub's own module docstring.
    neg_x2 = op_sub(zeros_like(x2), x2, cost)
    rotated = op_concat([neg_x2, x1], dim=-1, cost=cost)
    a = op_mul(x, cos, cost)
    b = op_mul(rotated, sin, cost)
    return op_add(a, b, cost)


def build_causal_consts(seq_len: int, head_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    """PREFILL-only constants: causal mask, -inf fill, 1/sqrt(D) scale,
    all shaped (seq_len, seq_len)."""
    causal = np.triu(np.ones((seq_len, seq_len), dtype=DTYPE), k=1)
    return (
        Tensor.from_values(causal),
        Tensor.from_values(np.full((seq_len, seq_len), -1e9, dtype=DTYPE)),
        Tensor.from_values(np.full((seq_len, seq_len), 1.0 / math.sqrt(head_dim), dtype=DTYPE)),
    )


def attention_block(
    hidden: Tensor,
    w: LayerWeights,
    cos: Tensor,
    sin: Tensor,
    cfg: Qwen3Config8B,
    cost: CostTracker | None,
    cache_k: Tensor | None,
    cache_v: Tensor | None,
    causal_mask: Tensor | None,
    neg_inf: Tensor | None,
    inv_sqrt_d_prefill: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Handles BOTH prefill (cache_k/cache_v is None coming in, full causal
    mask supplied) and one decode step (cache_k/cache_v is the existing
    real per-layer cache, no mask -- mirrors the reference's
    ``Qwen3Attention.prefill``/``.decode``). Returns (block_output,
    new_cache_k, new_cache_v)."""
    s_new = hidden.shape[0]
    is_prefill = cache_k is None
    h_heads, kv_heads, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = h_heads // kv_heads

    normed = rms_norm(hidden, w.input_norm_w, cfg.rms_norm_eps, cost)

    q = op_matmul(normed, w.wq, cost)  # (s_new, H*D)
    k_new = op_matmul(normed, w.wk, cost)  # (s_new, KVH*D)
    v_new = op_matmul(normed, w.wv, cost)  # (s_new, KVH*D)

    q = view(q, (s_new, h_heads, d))
    k_new = view(k_new, (s_new, kv_heads, d))
    v_new = view(v_new, (s_new, kv_heads, d))

    q = rms_norm(q, w.q_norm_w, cfg.rms_norm_eps, cost)
    k_new = rms_norm(k_new, w.k_norm_w, cfg.rms_norm_eps, cost)

    q = apply_rope(q, cos, sin, cost)
    k_new = apply_rope(k_new, cos, sin, cost)

    if is_prefill:
        # First fill: real direct assignment, no concat needed -- mirrors
        # the reference's LayerKVCache.append() when self.k is None.
        new_cache_k, new_cache_v = k_new, v_new
    else:
        # Real KV-cache append via the real concat primitive.
        new_cache_k = op_concat([cache_k, k_new], dim=0, cost=cost)
        new_cache_v = op_concat([cache_v, v_new], dim=0, cost=cost)

    s_total = new_cache_k.shape[0]
    k_attn = op_repeat(new_cache_k, n_rep, dim=1, cost=cost)  # (s_total, H, D)
    v_attn = op_repeat(new_cache_v, n_rep, dim=1, cost=cost)

    if is_prefill:
        inv_sqrt_d = inv_sqrt_d_prefill
    else:
        inv_sqrt_d = Tensor.from_values(np.full((s_new, s_total), 1.0 / math.sqrt(d), dtype=DTYPE))

    head_outs = []
    for head in range(h_heads):
        qh = p_slice(q, dim=1, start=head, end=head + 1)
        qh = op_reshape(qh, (s_new, d), cost)
        kh = p_slice(k_attn, dim=1, start=head, end=head + 1)
        kh = op_reshape(kh, (s_total, d), cost)
        vh = p_slice(v_attn, dim=1, start=head, end=head + 1)
        vh = op_reshape(vh, (s_total, d), cost)

        kh_t = p_transpose(kh, 0, 1)  # (D, s_total), free
        scores = op_matmul(qh, kh_t, cost)  # (s_new, s_total) real GEMM
        scores = op_mul(scores, inv_sqrt_d, cost)

        if is_prefill:
            scores = op_where(causal_mask, neg_inf, scores, cost)
        # else: no mask -- cache holds only positions <= current token,
        # matching the reference's own comment.

        row_max = op_max(scores, dim=-1, cost=cost)
        # UPDATED this round -- real dedicated `sub` firing (scores -
        # row_max) instead of the old negate-by-mul(-1) + add workaround.
        # This is the exact real softmax numeric-stability use case
        # `sub`'s own standalone check (run_sub_single_check.py) exercised.
        shifted = op_sub(scores, row_max, cost)
        exp_scores = op_exp(shifted, cost)
        row_sum = op_sum(exp_scores, dim=-1, cost=cost)
        probs = op_div(exp_scores, row_sum, cost)

        out_h = op_matmul(probs, vh, cost)  # (s_new, D)
        head_outs.append(out_h)

    attn = op_concat(head_outs, dim=1, cost=cost)  # (s_new, H*D)
    attn_out = op_matmul(attn, w.wo, cost)  # (s_new, hidden)
    block_out = op_add(hidden, attn_out, cost)
    return block_out, new_cache_k, new_cache_v


def mlp_block(hidden: Tensor, w: LayerWeights, cfg: Qwen3Config8B, cost: CostTracker | None) -> Tensor:
    normed = rms_norm(hidden, w.post_attn_norm_w, cfg.rms_norm_eps, cost)
    gate = op_matmul(normed, w.w_gate, cost)
    up = op_matmul(normed, w.w_up, cost)
    gated = op_mul(op_silu(gate, cost), up, cost)
    down = op_matmul(gated, w.w_down, cost)
    return op_add(hidden, down, cost)


def run_layers(
    x: Tensor,
    layer_weights: list[LayerWeights],
    cos: Tensor,
    sin: Tensor,
    cfg: Qwen3Config8B,
    cost: CostTracker | None,
    caches_k: list[Tensor | None],
    caches_v: list[Tensor | None],
    causal_mask: Tensor | None,
    neg_inf: Tensor | None,
    inv_sqrt_d_prefill: Tensor | None,
) -> Tensor:
    """One prefill pass (caches_k/v all None coming in) OR one decode step
    (caches_k/v hold the real per-layer state from before this call) over
    every real decoder layer. Mutates caches_k/caches_v in place."""
    hidden = x
    for i, w in enumerate(layer_weights):
        hidden, new_k, new_v = attention_block(
            hidden, w, cos, sin, cfg, cost, caches_k[i], caches_v[i], causal_mask, neg_inf, inv_sqrt_d_prefill
        )
        caches_k[i], caches_v[i] = new_k, new_v
        hidden = mlp_block(hidden, w, cfg, cost)
    return hidden


def main() -> None:
    rng = np.random.default_rng(SEED)
    cfg = Qwen3Config8B()

    print("=" * 78)
    print("STEP 1: NUMERIC-ONLY forward pass (real numpy via Tensor, no Docker)")
    print("=" * 78)
    print(f"config: {cfg}")
    print(f"prefill_seq_len={PREFILL_SEQ_LEN}, decode_steps={DECODE_STEPS}")
    print(
        f"NOTE: layer weights are ONE real shared set reused across all "
        f"{cfg.num_hidden_layers} layers, and embed_tokens/lm_head use a "
        f"{NUMERIC_STANDIN_VOCAB}-row numeric-only stand-in table (NOT the "
        f"real vocab_size={cfg.vocab_size}) -- see module docstring part (a)/(b) "
        "for why this does not change the real hardware-cost numbers in STEP 2."
    )

    shared_w = build_layer_weights(rng, cfg)
    layer_weights = [shared_w] * cfg.num_hidden_layers
    final_norm_w = Tensor.from_values(np.ones(cfg.hidden_size, dtype=DTYPE))

    embed_table_numeric = Tensor.from_values(
        (rng.standard_normal((NUMERIC_STANDIN_VOCAB, cfg.hidden_size)) * 0.02).astype(DTYPE)
    )
    lm_head_w_numeric = Tensor.from_values(
        (rng.standard_normal((cfg.hidden_size, NUMERIC_STANDIN_VOCAB)) * 0.02).astype(DTYPE)
    )

    causal_mask, neg_inf, inv_sqrt_d_prefill = build_causal_consts(PREFILL_SEQ_LEN, cfg.head_dim)

    t0 = time.time()

    # ---- PREFILL ----
    prompt_ids = rng.integers(0, NUMERIC_STANDIN_VOCAB, size=PREFILL_SEQ_LEN)
    x = op_gather(embed_table_numeric, dim=0, index=prompt_ids, cost=None)  # (PREFILL_SEQ_LEN, hidden)
    cos_p, sin_p = build_rope_tables(PREFILL_SEQ_LEN, cfg.head_dim, cfg.rope_theta, None, start_pos=0)

    caches_k: list[Tensor | None] = [None] * cfg.num_hidden_layers
    caches_v: list[Tensor | None] = [None] * cfg.num_hidden_layers

    hidden = run_layers(
        x, layer_weights, cos_p, sin_p, cfg, None, caches_k, caches_v, causal_mask, neg_inf, inv_sqrt_d_prefill
    )
    hidden = rms_norm(hidden, final_norm_w, cfg.rms_norm_eps, None)
    last_hidden = op_reshape(p_slice(hidden, dim=0, start=PREFILL_SEQ_LEN - 1, end=PREFILL_SEQ_LEN), (1, cfg.hidden_size), None)
    logits = op_matmul(last_hidden, lm_head_w_numeric, None)  # (1, NUMERIC_STANDIN_VOCAB)
    next_token = int(np.argmax(logits.data[0]))

    print(f"[prefill] output shape: {hidden.shape}, logits shape: {logits.shape}, next_token={next_token}")
    finite = np.isfinite(hidden.data.astype(np.float32)).all() and np.isfinite(logits.data.astype(np.float32)).all()

    # ---- DECODE ----
    for step in range(DECODE_STEPS):
        pos = PREFILL_SEQ_LEN + step
        x1 = op_gather(embed_table_numeric, dim=0, index=np.array([next_token]), cost=None)  # (1, hidden)
        cos_d, sin_d = build_rope_tables(1, cfg.head_dim, cfg.rope_theta, None, start_pos=pos)

        hidden1 = run_layers(x1, layer_weights, cos_d, sin_d, cfg, None, caches_k, caches_v, None, None, None)
        hidden1 = rms_norm(hidden1, final_norm_w, cfg.rms_norm_eps, None)
        logits = op_matmul(hidden1, lm_head_w_numeric, None)  # (1, NUMERIC_STANDIN_VOCAB)
        next_token = int(np.argmax(logits.data[0]))

        finite = finite and np.isfinite(hidden1.data.astype(np.float32)).all() and np.isfinite(logits.data.astype(np.float32)).all()
        print(f"[decode step {step + 1}/{DECODE_STEPS}] position={pos} cache_len={caches_k[0].shape[0]} next_token={next_token}")

    dt = time.time() - t0
    print(f"all finite (no NaN/Inf) across prefill+decode: {finite}")
    print(f"wall time: {dt:.3f}s")
    if not finite:
        raise RuntimeError("numeric forward pass produced NaN/Inf -- composition bug, not a hardware-cost issue")

    if not RUN_HARDWARE_COST:
        print()
        print("RUN_HARDWARE_COST is False -- skipping the real Docker/TimeLoop hardware-cost pass.")
        print("Flip RUN_HARDWARE_COST=True at the top of this script and re-run when ready to spend")
        print("real Docker time -- expected to be substantially more expensive than the tiny run; see")
        print("module docstring for why AREA_COMPUTE/MEMORY/NETWORK_FRACTION may need a real-data-driven")
        print("revision once this actually runs against 800M um^2 at these real, much larger GEMM shapes.")
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
        m=PREFILL_SEQ_LEN, n=cfg.intermediate_size, k=cfg.hidden_size, bytes_per_element=BYTES_PER_ELEMENT
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
    print(
        f"real area used: compute={area_result.compute_area_used_um2:,.2f} um^2, "
        f"memory={area_result.memory_area_used_um2:,.2f} um^2, "
        f"network={area_result.network_area_used_um2:,.2f} um^2"
    )

    cost = CostTracker(wafer=wafer, out_dir=OUT_ROOT / "primitive_cost_traces")

    t0 = time.time()

    embed_table_cost_stub = embed_table_numeric  # same small stand-in table; real gather cost is
    # output-size-based only, see module docstring part (b) -- byte-identical real cost either way.

    caches_k = [None] * cfg.num_hidden_layers
    caches_v = [None] * cfg.num_hidden_layers

    prompt_ids = rng.integers(0, NUMERIC_STANDIN_VOCAB, size=PREFILL_SEQ_LEN)
    x = op_gather(embed_table_cost_stub, dim=0, index=prompt_ids, cost=cost)
    cos_p, sin_p = build_rope_tables(PREFILL_SEQ_LEN, cfg.head_dim, cfg.rope_theta, cost, start_pos=0)
    hidden = run_layers(
        x, layer_weights, cos_p, sin_p, cfg, cost, caches_k, caches_v, causal_mask, neg_inf, inv_sqrt_d_prefill
    )
    hidden = rms_norm(hidden, final_norm_w, cfg.rms_norm_eps, cost)
    last_hidden = op_reshape(
        p_slice(hidden, dim=0, start=PREFILL_SEQ_LEN - 1, end=PREFILL_SEQ_LEN), (1, cfg.hidden_size), cost
    )
    # Real LM head hardware cost, TRUE vocab_size -- direct cost.op_gemm call
    # (bypasses op_matmul's tensor-shape inference, which would otherwise see
    # only the small numeric stand-in's vocab dimension). See module
    # docstring part (b).
    cost.op_gemm(m=1, n=cfg.vocab_size, k=cfg.hidden_size, bpe=BYTES_PER_ELEMENT)
    logits_numeric = op_matmul(last_hidden, lm_head_w_numeric, None)  # real numeric value only, uncosted here
    next_token = int(np.argmax(logits_numeric.data[0]))

    for step in range(DECODE_STEPS):
        pos = PREFILL_SEQ_LEN + step
        x1 = op_gather(embed_table_cost_stub, dim=0, index=np.array([next_token]), cost=cost)
        cos_d, sin_d = build_rope_tables(1, cfg.head_dim, cfg.rope_theta, cost, start_pos=pos)
        hidden1 = run_layers(x1, layer_weights, cos_d, sin_d, cfg, cost, caches_k, caches_v, None, None, None)
        hidden1 = rms_norm(hidden1, final_norm_w, cfg.rms_norm_eps, cost)
        cost.op_gemm(m=1, n=cfg.vocab_size, k=cfg.hidden_size, bpe=BYTES_PER_ELEMENT)
        logits_numeric = op_matmul(hidden1, lm_head_w_numeric, None)
        next_token = int(np.argmax(logits_numeric.data[0]))
        print(f"[decode step {step + 1}/{DECODE_STEPS}, costed] cache_len={caches_k[0].shape[0]}")

    dt = time.time() - t0

    print()
    print(f"real Docker/TimeLoop primitive calls actually executed: {cost.real_docker_calls}")
    print(f"real results memoized/reused (identical shape, same real Docker result): {cost.memoized_reuses}")
    print(f"total real sequential-trace compute cycles: {cost.total_cycles:,}")
    print(f"total real energy (sum across every real op instance): {cost.total_energy_uj:.4f} uJ")
    print(f"real chip area actually used: {area_result.compute_area_used_um2 + area_result.memory_area_used_um2 + area_result.network_area_used_um2:,.2f} um^2")
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

    print()
    print("Hardware-cost pass complete -- every number above came from a real")
    print(f"timeloop-mapper/timeloop-model/Accelergy Docker run; see {OUT_ROOT} for the real")
    print("artifact tree (resolved.stats.txt / resolved.ART.yaml per unique op shape).")


if __name__ == "__main__":
    main()
