#!/usr/bin/env python3
"""Real single-primitive check for `reshape`: one real Tensor -> permute()
(make it non-contiguous, so a later reshape is forced to copy) ->
reshape(x, shape) (real numeric result) -> ReshapeSpec.from_call() (checks
the REAL copy-vs-view outcome via numpy.shares_memory, never assumed) ->
simulate_reshape() -> real cycles/time/energy/area from one real Docker
`timeloop-model` run against the shared real ``bitwise`` component (see
``wafersim.primitives._memory_move``'s own module docstring: a real
identity-bitwise-pass-through per copied element -- there is no dedicated
DMA/copy-engine component in this project's pinned Docker toolchain).

WHY THE PERMUTE STEP: `reshape` is the ONE memory-move primitive whose
real cost is CONDITIONAL -- see
``wafersim.primitives.reshape_view._timeloop_reshape``'s own module
docstring. A reshape of an already-contiguous tensor is usually just a
real view (zero cost, no Docker run at all -- same as `view()`). To
actually exercise the real Docker-triggering copying branch, this script
first makes the tensor non-contiguous via a real `permute()` (numpy
transpose is always a real view, never a copy -- see
``transpose_permute``'s own module docstring), so the following
`reshape()` to a flat shape is NOT viewable in-place and must really
copy.

No synthetic fallback anywhere: if Docker/TimeLoop fails, this raises and
prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_reshape_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec
from wafersim.primitives.reshape_view import ReshapeSpec, reshape, simulate_reshape
from wafersim.primitives.tensor import Tensor
from wafersim.primitives.transpose_permute import permute

OUT_DIR = Path("outputs/reshape_single_check")


def main() -> None:
    a = Tensor.from_values([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # shape (2, 4)
    print(f"a = {a.data.tolist()}  shape={a.shape}")

    # Real transpose -- always a view, zero cost, never copies.
    non_contiguous = permute(a, (1, 0))  # shape (4, 2), non-contiguous
    print(f"permute(a, (1,0)) = {non_contiguous.data.tolist()}  shape={non_contiguous.shape}  "
          f"(real view, zero cost)")

    target_shape = (8,)
    # Real numeric result -- plain numpy reshape, no Docker involved here.
    result = reshape(non_contiguous, target_shape)
    print(f"real numeric result reshape(non_contiguous, {target_shape}) = {result.data.tolist()}")

    # Bridge: ReshapeSpec.from_call() checks the REAL outcome itself
    # (numpy.shares_memory), it does not trust the caller to say whether
    # a copy happened.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = ReshapeSpec.from_call(non_contiguous, target_shape)
    print(f"bridged ReshapeSpec: elements_per_pe={spec.elements_per_pe}, "
          f"bytes_per_element={spec.bytes_per_element}, copied={spec.copied}")
    assert spec.copied, "expected a real copy for this non-contiguous reshape"

    # Real Docker timeloop-model run (real identity-bitwise-pass-through
    # per copied element) -- this is the only place cycles/energy/area
    # come from; raises (never fabricates) on failure.
    cost = simulate_reshape(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (reshape, real copy) ===")
    print(f"cycles (real, total for {spec.elements_per_pe} elements): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                     {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"bitwise (bitwise_gate) area (real):                       {cost.bitwise_area_um2} um^2")
    print(f"pe_scratchpad area (real):                                {cost.pe_scratchpad_area_um2} um^2")
    print(f"cycles/element (real):                                    {cost.cycles_per_element}")
    print(f"energy/element (real):                                    {cost.energy_uj_per_element} uJ")


if __name__ == "__main__":
    main()
