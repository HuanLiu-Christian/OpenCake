#!/usr/bin/env python3
"""Real single-primitive check for `where`: three real Tensors ->
where(mask, a, b) (real numeric result, real numpy.where) ->
WhereSpec(from the real result) -> simulate_where() -> real cycles/time/
energy/area from one real Docker `timeloop-model` run against the shared
real ``bitwise`` component, composed as an EXACT 3-gate XOR/AND/XOR 2:1
mux per element (see
``wafersim.primitives.where_select._timeloop_where``'s own module
docstring: an already Docker-confirmed real result from
``run_bridge_check2.py`` -- an 8-element where() call measured real
Cycles=24, exactly 3x the element count, matching the XOR/AND/XOR
identity's exact firing count).

Used for: causal attention masking, per this primitive's own package
docstring. This example mirrors that real use case at toy scale: a 4x4
lower-triangular causal mask applied to a 4x4 attention-score matrix,
masked positions replaced with a large negative value (as real softmax
pre-masking does, so exp() of a masked position underflows toward zero).

No synthetic fallback anywhere: if Docker/TimeLoop fails, this raises and
prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_where_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec
from wafersim.primitives.tensor import Tensor
from wafersim.primitives.where_select import WhereSpec, simulate_where, where

OUT_DIR = Path("outputs/where_single_check")


def main() -> None:
    # Real 4x4 causal mask: position (i, j) is 1.0 (visible) if j <= i,
    # else 0.0 (masked) -- a real lower-triangular causal attention mask.
    causal_mask = Tensor.from_values(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    # Real (toy) pre-softmax attention scores.
    scores = Tensor.from_values(
        [
            [2.0, 1.0, 0.5, 3.0],
            [0.1, 2.5, 1.5, 0.3],
            [1.2, 0.4, 3.1, 0.9],
            [0.7, 1.1, 0.6, 2.2],
        ]
    )
    neg_inf = Tensor.from_values([[-1e9] * 4 for _ in range(4)])

    print(f"causal_mask = {causal_mask.data.tolist()}")
    print(f"scores      = {scores.data.tolist()}")

    # Real numeric result -- plain numpy.where, no Docker involved here.
    result = where(causal_mask, scores, neg_inf)
    print(f"real numeric result where(causal_mask, scores, neg_inf) =")
    for row in result.data.tolist():
        print(f"  {row}")

    # Bridge: elements_per_pe = the real result's element count; the
    # real bitwise firing count is 3x that (XOR/AND/XOR per element).
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = WhereSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"bridged WhereSpec: elements_per_pe={spec.elements_per_pe}, "
          f"real_bitwise_count={spec.real_bitwise_count}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model run (3 real bitwise firings per element:
    # an exact XOR/AND/XOR 2:1 mux, not an approximation) -- this is the
    # only place cycles/energy/area come from; raises (never fabricates)
    # on failure.
    cost = simulate_where(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (where, 3-gate mux per element) ===")
    print(f"cycles (real, total for {spec.real_bitwise_count} bitwise firings): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                          {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                               {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"bitwise (bitwise_gate) area (real):                                 {cost.bitwise_area_um2} um^2")
    print(f"pe_scratchpad area (real):                                          {cost.pe_scratchpad_area_um2} um^2")
    print(f"cycles/element (real, divided by {spec.elements_per_pe} elements):   {cost.cycles_per_element}")
    print(f"energy/element (real):                                              {cost.energy_uj_per_element} uJ")


if __name__ == "__main__":
    main()
