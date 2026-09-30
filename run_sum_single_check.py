#!/usr/bin/env python3
"""Real single-primitive check for `sum`: one real Tensor -> sum(x, dim)
(real numeric reduction) -> SumSpec(from the real input's reduced axis)
-> simulate_sum() -> real cycles/time/energy/area from one real Docker
`timeloop-model` run against the SAME real, already-verified ``fpadder``
Accelergy component `add` uses (see
``wafersim.primitives.sum._timeloop_sum``'s own module docstring: summing
K elements is EXACTLY K-1 real sequential adder firings -- an exact
operation count, not an estimate -- so `sum` needed no new component
research, only `add`'s already-Docker-verified component reused K-1
times).

Note the real bridge formula this primitive uses is different from
add/mul/div: ``SumSpec(elements_per_pe=x.shape[dim], ...)`` -- the real
size of the axis being REDUCED, not ``result.size`` (the reduction keeps
its other axes, so ``result.size`` would undercount by counting only the
post-reduction shape).

Also note the real K=1 degenerate case (reducing a single element takes
zero real adds and skips Docker entirely, returning a true zero-cost
result -- see that module's own docstring) does NOT apply here: this
example reduces K=4 elements per row, so it's a real, non-degenerate
Docker-triggering case.

No synthetic fallback anywhere: if Docker/TimeLoop fails, this raises and
prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_sum_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec
from wafersim.primitives import sum as t_sum
from wafersim.primitives.sum import SumSpec, simulate_sum
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/sum_single_check")


def main() -> None:
    a = Tensor.from_values([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # shape (2, 4)
    print(f"a = {a.data.tolist()}  shape={a.shape}")

    # Real numeric result -- plain numpy sum(axis=1, keepdims=True), no
    # Docker involved here.
    result = t_sum(a, dim=1)
    print(f"real numeric result sum(a, dim=1) = {result.data.tolist()}  shape={result.shape}")

    # Bridge: K is the real size of the axis being reduced
    # (a.shape[dim]), NOT result.size -- per this primitive's own package
    # docstring's stated bridge formula.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = SumSpec(elements_per_pe=a.shape[1], bytes_per_element=a.bytes_per_element)
    print(f"bridged SumSpec: elements_per_pe (K)={spec.elements_per_pe}, "
          f"real_add_count (K-1)={spec.real_add_count}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model run (K-1=3 real sequential fpadder
    # firings) -- this is the only place cycles/energy/area come from;
    # raises (never fabricates) on failure.
    cost = simulate_sum(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (sum, K-1 real adds) ===")
    print(f"cycles (real, total for K-1={spec.real_add_count} adds): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                     {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"adder (fpadder_gate) area (real):                         {cost.adder_area_um2} um^2")
    print(f"pe_scratchpad area (real):                                {cost.pe_scratchpad_area_um2} um^2")
    print(f"cycles/element (real, divided by K={spec.elements_per_pe}): {cost.cycles_per_element}")
    print(f"energy/element (real):                                    {cost.energy_uj_per_element} uJ")


if __name__ == "__main__":
    main()
