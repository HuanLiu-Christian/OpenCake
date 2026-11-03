"""wafersim.primitives -- one folder per primitive op, unified interface here.

This package is the "最基础、最简单、但完整" (minimal-but-complete) primitive
set worked out earlier in this conversation, needed to implement a
decoder-only LLM forward pass + autoregressive generation (LLaMA/Qwen-style:
RMSNorm, RoPE, SwiGLU, grouped-query attention, causal masking, KV cache).

Layout: one folder per primitive under this package, each with its own
`__init__.py` exposing that primitive's public function(s)/class. This
top-level `__init__.py` re-exports all of them so callers can do a single
`from wafersim.primitives import matmul_related_name, add, mul, ...` instead
of reaching into each folder individually.

Status per primitive:
  - Tensor: REAL, implemented (wafersim.primitives.tensor) -- a real,
    numpy-backed data structure. No Docker/TimeLoop involved (see its own
    module docstring for why building/operating on real tensor VALUES
    never needs TimeLoop, which only ever estimates hardware COST for an
    operation count).
  - reshape/view, transpose/permute, concat/stack, repeat/expand,
    slice/gather, add, mul, div, exp, rsqrt, silu, sum, max, matmul,
    where, cast: numeric compute is REAL, built on Tensor (real numpy,
    still no Docker/TimeLoop). add/mul/div/sum/max/matmul additionally
    have a REAL, separate, Docker-verified hardware cost-model layer
    (simulate_add/simulate_mul/simulate_div/simulate_sum/simulate_max/
    simulate_gemm) -- see each of those folders' own `__init__.py` for how
    the two independent layers (numeric result vs. hardware cost) are
    meant to be bridged explicitly by the caller when both are wanted.
  - transpose/permute: cost-model layer is CONFIRMED permanently zero
    real hardware cost (no Spec/Result/simulate_* exists -- see its own
    package docstring for why no Docker run could ever be meaningful
    here).
  - view/expand/slice_: cost-model layer is CONFIRMED permanently zero
    real hardware cost too, but DO expose a Spec/Result/simulate_* trio
    (reshape_view.ReshapeSpec/repeat_expand.RepeatSpec/
    slice_gather.GatherSpec) because their SIBLING function in the same
    module (reshape/repeat/gather) is real-cost-bearing -- see each of
    those three folders' own package docstrings.
  - reshape (the copying branch), repeat, gather, concat/stack, where:
    REAL, Docker-backed hardware cost-model layer added this round --
    ReshapeSpec/simulate_reshape, RepeatSpec/simulate_repeat,
    GatherSpec/simulate_gather, ConcatStackSpec/simulate_concat_stack
    (all four composed from the shared real
    ``wafersim.primitives._memory_move`` pipeline -- a real identity-
    bitwise-pass-through per moved element, see that module's own
    docstring for the real Accelergy-source citation ruling out a
    dedicated DMA/copy-engine component in this project's pinned
    toolchain), and WhereSpec/simulate_where (composed from the same
    shared real ``wafersim.primitives._bitwise`` component via an EXACT
    3-gate XOR/AND/XOR mux decomposition, see
    ``where_select/_timeloop_where.py``). The underlying ``_bitwise``/
    ``bitwise_gate`` Accelergy component pipeline these all share is REAL,
    Docker-VERIFIED (all 5 real callers ran successfully via
    ``run_bridge_check2.py`` on its very first real Docker round -- see
    ``_bitwise.py``'s module docstring's "ROUND-1 REAL DOCKER RESULT"
    section for the full real-output citation and cross-checks).
  - rsqrt/exp/silu/cast: REAL, Docker-backed hardware cost-model layer
    added this round -- RsqrtSpec/simulate_rsqrt (real magic-number-seeded,
    3-iteration Newton-Raphson decomposition via mul/add/bitwise/intadder,
    see ``rsqrt/_timeloop_rsqrt.py``), ExpSpec/simulate_exp (real
    range-reduction + degree-6-Taylor-polynomial decomposition via
    mul/add/bitwise, see ``exp/_timeloop_exp.py``), SiluSpec/simulate_silu
    (a pure composition of the already-real bitwise/exp/add/div
    pipelines, needing no new numerical research of its own, see
    ``silu/_timeloop_silu.py``), and CastSpec/simulate_cast (real
    fp32->fp16 IEEE-754 unpack/rebias/round/repack bit-manipulation
    decomposition via bitwise/intadder, see ``cast/_timeloop_cast.py``).
    All four follow the same "no dedicated component exists, decompose
    into a real, cited, standard algorithm built from already-Docker-
    verified components" discipline `div`'s own Newton-Raphson reciprocal
    established, plus this round's new real ``intadder`` Accelergy
    component (distinct from `add`'s ``fpadder`` -- see ``_intadder.py``'s
    own module docstring) for the genuinely-integer steps (rsqrt's/cast's
    real bit-pattern integer arithmetic) that a floating-point adder would
    misrepresent. Each decomposition's chosen iteration count/polynomial
    degree was independently, empirically verified in this session against
    ``numpy``/``math`` reference values (not just cited) -- see
    ``tests/test_cost_model_numeric_approximation.py`` for the reproducible
    checks. ALL FOUR (and the new ``intadder`` component underneath
    rsqrt/cast) are now REAL, Docker-CONFIRMED WORKING -- all 16 real
    ``timeloop-model`` sub-invocations these 4 primitives trigger
    succeeded on the FIRST real Docker round via ``run_bridge_check3.py``,
    independently verified by reading the real
    ``resolved.stats.txt``/``resolved.ART.yaml`` output files (not just a
    "ran successfully" report) -- see each of
    ``_intadder.py``/``rsqrt/_timeloop_rsqrt.py``/``exp/_timeloop_exp.py``/
    ``silu/_timeloop_silu.py``/``cast/_timeloop_cast.py``'s own "REAL
    DOCKER RESULT" docstring section for the real per-element
    cycle/energy numbers this round measured.
  - sub: REAL, added this round -- a deliberate ALIAS of `add`'s own
    already-Docker-verified ``AddSpec``/``AddResult``/``simulate_add``,
    not a new Docker/TimeLoop architecture (see ``sub/__init__.py``'s own
    module docstring for the real Cerebras CSL citation -- a genuine,
    separate ``@fsubh``/``@fsubs`` hardware instruction exists on a real
    chip -- and for why this project's own pinned Accelergy toolchain has
    no separately-characterized subtractor component to measure
    independently, matching how real FPUs typically reuse the same adder
    datapath with a sign-flipped operand for subtraction). Closes this
    project's long-standing "no dedicated `sub`" gap -- RoPE's
    ``rotate_half``/softmax's ``x - row_max`` no longer need to be
    modeled as negate-by-mul(-1) + add.
  - sin/cos: REAL, added this round -- SinSpec/simulate_sin (real range-
    reduction + degree-17 odd-power Taylor-polynomial decomposition via
    mul/add/sub/bitwise, see ``sin/_timeloop_sin.py``) and
    CosSpec/simulate_cos (reuses `sin`'s exact real pipeline via the real
    identity ``cos(x) = sin(x + pi/2)``, plus one real `add` firing for
    the phase shift itself, see ``cos/_timeloop_cos.py``). Real Cerebras
    CSL docs confirm a real chip exposes ``sin``/``cos`` as built-in
    functions, but do not disclose the internal implementation -- so
    unlike `sub`, this decomposition is this project's own reasoned (not
    Cerebras-confirmed) engineering choice, independently, empirically
    verified against ``numpy.sin``/``numpy.cos`` across 20,000+ real
    points spanning x in [-100, 100] (max abs error ~2.24e-8, tighter
    than real fp32 epsilon) -- see each module's own docstring for the
    full derivation. Used for RoPE's rotary embedding tables, previously
    precomputed via plain, uncosted numpy in this project's Qwen3
    composition scripts.
  - clamp/topk: REAL, added for gpt-oss (SwiGLU clamp, MoE router topk).
    Numeric compute is real numpy on Tensor (``clamp`` matches
    ``torch.clamp``; ``topk`` returns descending ``(values, int64
    indices)``). Cost-model layer: ClampSpec/simulate_clamp (one real
    ``comparator`` firing + one exact 3-gate ``bitwise`` mux per element per
    bound) and TopKSpec/simulate_topk (k rounds of max-with-index plus
    winner masking, from the same two real components) -- see each folder's
    own package/``_timeloop_*`` docstring. Their Docker cost layer composes
    the already-Docker-verified comparator/bitwise pipelines but has NOT
    itself been run end to end yet; run ``run_clamp_single_check.py`` and
    ``run_topk_single_check.py`` with Docker open to confirm.
"""

from __future__ import annotations

from .add import add
from .cast import CastResult, CastSpec, cast, simulate_cast
from .clamp import ClampResult, ClampSpec, clamp, simulate_clamp
from .concat_stack import ConcatStackSpec, concat, simulate_concat_stack, stack
from .cos import CosResult, CosSpec, cos, simulate_cos
from .div import div
from .exp import ExpResult, ExpSpec, exp, simulate_exp
from .matmul import (
    AreaBudget,
    AreaBudgetResult,
    GemmMappingContract,
    GemmSpec,
    TileKCandidate,
    WaferGemmResult,
    WaferSpec,
    best_by_sim_time,
    derive_wafer_spec_from_area_budget,
    matmul,
    search_tile_k,
    simulate_gemm,
)
from .max import max
from .mul import mul
from .repeat_expand import RepeatSpec, expand, repeat, simulate_repeat
from .reshape_view import ReshapeSpec, reshape, simulate_reshape, view
from .rsqrt import RsqrtResult, RsqrtSpec, rsqrt, simulate_rsqrt
from .silu import SiluResult, SiluSpec, silu, simulate_silu
from .sin import SinResult, SinSpec, simulate_sin, sin
from .slice_gather import GatherSpec, gather, simulate_gather, slice_
from .sub import SubResult, SubSpec, simulate_sub, sub
from .sum import sum
from .tensor import Tensor
from .topk import TopKResult, TopKSpec, simulate_topk, topk
from .transpose_permute import permute, transpose
from .where_select import WhereResult, WhereSpec, simulate_where, where

__all__ = [
    "Tensor",
    "reshape",
    "view",
    "permute",
    "transpose",
    "concat",
    "stack",
    "repeat",
    "expand",
    "gather",
    "slice_",
    "add",
    "mul",
    "div",
    "exp",
    "rsqrt",
    "silu",
    "sum",
    "max",
    "matmul",
    "where",
    "cast",
    "sub",
    "sin",
    "cos",
    "clamp",
    "topk",
    # matmul's own real, Docker-backed GEMM cost-model pipeline (see
    # matmul/__init__.py) -- independent of the numeric `matmul` above.
    "GemmSpec",
    "WaferSpec",
    "GemmMappingContract",
    "WaferGemmResult",
    "simulate_gemm",
    "TileKCandidate",
    "search_tile_k",
    "best_by_sim_time",
    "AreaBudget",
    "AreaBudgetResult",
    "derive_wafer_spec_from_area_budget",
    # New real, Docker-backed memory-move/select cost-model pipelines
    # added this round (see this module's own docstring above).
    "ReshapeSpec",
    "simulate_reshape",
    "RepeatSpec",
    "simulate_repeat",
    "GatherSpec",
    "simulate_gather",
    "ConcatStackSpec",
    "simulate_concat_stack",
    "WhereSpec",
    "WhereResult",
    "simulate_where",
    # New real, Docker-backed rsqrt/exp/silu/cast cost-model pipelines
    # added this round (see this module's own docstring above).
    "RsqrtSpec",
    "RsqrtResult",
    "simulate_rsqrt",
    "ExpSpec",
    "ExpResult",
    "simulate_exp",
    "SiluSpec",
    "SiluResult",
    "simulate_silu",
    "CastSpec",
    "CastResult",
    "simulate_cast",
    # New real, Docker-backed sub/sin/cos cost-model pipelines added this
    # round (see this module's own docstring above).
    "SubSpec",
    "SubResult",
    "simulate_sub",
    "SinSpec",
    "SinResult",
    "simulate_sin",
    "CosSpec",
    "CosResult",
    "simulate_cos",
    # New real, Docker-backed clamp/topk cost-model pipelines added for
    # gpt-oss (MoE router topk, SwiGLU clamp) -- see each folder's own
    # package docstring.
    "ClampSpec",
    "ClampResult",
    "simulate_clamp",
    "TopKSpec",
    "TopKResult",
    "simulate_topk",
]
