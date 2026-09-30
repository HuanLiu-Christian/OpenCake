"""sub primitive.

STATUS: REAL on both sides, but deliberately implemented as a thin ALIAS
of `add`'s already-real, already-Docker-verified cost model, not a new
Docker/TimeLoop architecture -- see "WHY AN ALIAS, NOT A NEW ARCHITECTURE"
below for the real, cited justification. This closes this project's
long-standing "no dedicated `sub` primitive" gap (previously every
subtraction -- RoPE's ``-x2``, softmax's ``x - row_max`` -- was composed
as negate-by-mul(-1) + add, i.e. modeled as costing TWO real ops where a
real chip spends ONE).

REAL CEREBRAS HARDWARE FACT THAT MOTIVATED THIS (confirmed against
Cerebras' own official SDK docs this session, not assumed): CSL exposes
subtraction as its OWN, genuinely separate hardware instruction --
``@fsubh`` (f16) / ``@fsubs`` (f32) -- distinct from ``@faddh``/``@fadds``,
alongside distinct ``@fmulh``/``@fmuls`` and fused-multiply-accumulate
``@fmach``/``@fmacs``/``@fmachs``. Source:
https://sdk.cerebras.net/csl/language/libraries (CSL `dsd_ops` library
documentation, Cerebras' own official SDK docs domain). Subtraction is
real, first-class silicon on a real Cerebras-style chip -- not something
synthesized from negate+add at the instruction level.

WHY AN ALIAS, NOT A NEW ARCHITECTURE: this project's own pinned Docker
Accelergy toolchain (the same real 7-plugin search `add`'s own module
docstring already performed -- see
``wafersim.primitives.add._timeloop_add``) has exactly ONE real,
separately-characterized floating-point adder-class component
(``fpadder``, real ``fp_sp_adder.csv`` data) and NO separately-
characterized ``fpsubtractor`` anywhere. This is not a gap unique to this
project's toolchain -- it matches how virtually every real FPU is
actually built: floating-point subtraction is standard-textbook
implemented by feeding the SAME adder datapath with one operand's sign
bit flipped (`a - b` = `a + (-b)`, done at the bit level before the
shared adder, not via a second arithmetic circuit). TimeLoop's own cost
model has no visibility into which arithmetic identity a declared
compute-level component's firings implement either (see `add`'s own
module docstring's "WHY THE SAME MAC-SHAPED PROBLEM/MAPPING SCHEMA"
section: Accelergy's real ``fpadder_estimate_energy`` never branches on
the specific action name). So the real, honest answer to "what does one
real `sub` firing cost on this project's real Docker toolchain" is:
exactly what one real `add` firing already costs -- reusing `add`'s
already-Docker-verified ``AddSpec``/``AddResult``/``simulate_add`` here
verbatim is the faithful choice, not a shortcut around real measurement.

Numeric side (``sub(a, b)``) is real numpy ``a.data - b.data`` -- the
numeric and cost layers stay independent, same as every other primitive
in this package (see ``add/__init__.py``'s own module docstring for the
general pattern).

Used for: RoPE's ``rotate_half`` (``-x2``), softmax's ``x - row_max``, and
any other elementwise subtraction a composition script wants to declare
as a real, dedicated `sub` firing instead of the older negate+add
workaround.
"""

from __future__ import annotations

from ..add import AddResult, AddSpec, simulate_add
from ..tensor import Tensor

# Real, deliberate aliases -- see module docstring's "WHY AN ALIAS, NOT A
# NEW ARCHITECTURE" section. `sub`'s real per-firing cost model IS `add`'s;
# giving it its own name here (rather than making callers import AddSpec
# for a subtraction) keeps CostTracker's own per-op accounting/logging
# correctly labeled "sub" while the underlying real Docker numbers stay
# byte-for-byte what `add` already proved.
SubSpec = AddSpec
SubResult = AddResult
simulate_sub = simulate_add


def _require_same_dtype(a: Tensor, b: Tensor, op_name: str) -> None:
    if a.dtype != b.dtype:
        raise TypeError(
            f"{op_name}(a, b): dtype mismatch a.dtype={a.dtype} vs "
            f"b.dtype={b.dtype} -- cast one operand explicitly via "
            f"wafersim.primitives.cast.cast(...) first; this project models "
            f"mixed precision explicitly, not via numpy's implicit type "
            f"promotion."
        )


def sub(a: Tensor, b: Tensor) -> Tensor:
    """Real elementwise (broadcasting) ``a - b`` on real Tensor data."""
    _require_same_dtype(a, b, "sub")
    return Tensor(a.data - b.data)


__all__ = [
    "sub",
    "SubSpec",
    "SubResult",
    "simulate_sub",
]
