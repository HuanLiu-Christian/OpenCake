#!/usr/bin/env python3
"""Real end-to-end check for the 5 NEW real hardware cost models added this
round: reshape (copying branch), repeat, gather, concat/stack, and
where/select. All 5 share ONE new Accelergy component pipeline
(``wafersim.primitives._bitwise`` -- a real ``bitwise`` Aladdin-plugin
gate) that has NOT yet been run against real Docker (see that module's own
"THIS IS THE FIRST REAL DOCKER ROUND" docstring section) -- so unlike
``run_bridge_check.py`` (which re-confirmed 6 ALREADY Docker-verified cost
models), THIS script is the first real test of whether this round's new
architecture.yaml/mapping.yaml design actually works.

If any of the 5 below crashes, that is real, expected-possible information
(the `add` primitive's own cost model needed 9 real Docker rounds to reach
its current, working design -- see ``add/_timeloop_add.py``'s module
docstring) -- paste the real Docker error back so the architecture/mapping
files can be fixed at the actual root cause, the same way `add`'s crash
investigation was done. No synthetic fallback anywhere: if Docker/TimeLoop
fails, this raises and prints nothing fabricated for that step.

Usage (run from the project root, with your usual venv active, Docker
Desktop open):
  source .venv/bin/activate
  python3 run_bridge_check2.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import (
    ConcatStackSpec,
    GatherSpec,
    RepeatSpec,
    ReshapeSpec,
    WhereSpec,
    WaferSpec,
    concat,
    gather,
    permute,
    repeat,
    reshape,
    simulate_concat_stack,
    simulate_gather,
    simulate_repeat,
    simulate_reshape,
    simulate_where,
    where,
)
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/bridge_check2")


def check_reshape_copy() -> None:
    print("\n--- reshape (copying branch): Tensor -> permute (non-contiguous) "
          "-> reshape(a, shape) -> ReshapeSpec.from_call (real copy check) "
          "-> simulate_reshape ---")
    a = Tensor.from_values([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    non_contiguous = permute(a, (1, 0))  # real transpose -> non-contiguous view
    spec = ReshapeSpec.from_call(non_contiguous, (8,))
    print(f"  real copy check: copied={spec.copied}, elements_per_pe={spec.elements_per_pe}")
    assert spec.copied, "expected a real copy for this non-contiguous reshape"
    result = reshape(non_contiguous, (8,))
    print(f"  numeric result: {result} values={result.data.tolist()}")
    cost = simulate_reshape(spec, WaferSpec(), OUT_DIR / "reshape")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}, "
          f"bitwise_area_um2={cost.bitwise_area_um2}")


def check_repeat() -> None:
    print("\n--- repeat: Tensor -> repeat(a, repeats, dim) -> RepeatSpec.from_result "
          "-> simulate_repeat ---")
    a = Tensor.from_values([[1.0, 2.0], [3.0, 4.0]])
    result = repeat(a, repeats=3, dim=0)
    print(f"  numeric result: {result} shape={result.shape}")
    spec = RepeatSpec.from_result(result)
    print(f"  bridged RepeatSpec: elements_per_pe={spec.elements_per_pe}")
    cost = simulate_repeat(spec, WaferSpec(), OUT_DIR / "repeat")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}")


def check_gather() -> None:
    print("\n--- gather: Tensor -> gather(a, dim, index) -> GatherSpec.from_result "
          "-> simulate_gather ---")
    a = Tensor.from_values([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = gather(a, dim=0, index=[2, 0, 0, 1])
    print(f"  numeric result: {result} shape={result.shape}")
    spec = GatherSpec.from_result(result)
    print(f"  bridged GatherSpec: elements_per_pe={spec.elements_per_pe}")
    cost = simulate_gather(spec, WaferSpec(), OUT_DIR / "gather")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}")


def check_concat_stack() -> None:
    print("\n--- concat: Tensor -> concat([a, b], dim) -> ConcatStackSpec.from_result "
          "-> simulate_concat_stack ---")
    a = Tensor.from_values([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor.from_values([[5.0, 6.0]])
    result = concat([a, b], dim=0)
    print(f"  numeric result: {result} shape={result.shape}")
    spec = ConcatStackSpec.from_result(result)
    print(f"  bridged ConcatStackSpec: elements_per_pe={spec.elements_per_pe}")
    cost = simulate_concat_stack(spec, WaferSpec(), OUT_DIR / "concat")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}")


def check_where() -> None:
    print("\n--- where: Tensor -> where(mask, a, b) -> WhereSpec(from result) "
          "-> simulate_where (3 real bitwise firings/element) ---")
    mask = Tensor.from_values([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    a = Tensor.from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = Tensor.from_values([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    result = where(mask, a, b)
    print(f"  numeric result: {result} values={result.data.tolist()}")
    spec = WhereSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"  bridged WhereSpec: elements_per_pe={spec.elements_per_pe}, "
          f"real_bitwise_count={spec.real_bitwise_count}")
    cost = simulate_where(spec, WaferSpec(), OUT_DIR / "where")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}, "
          f"bitwise_area_um2={cost.bitwise_area_um2}")


def main() -> None:
    print("Real cost-model check for the 5 NEW primitives added this round "
          "(reshape-copy/repeat/gather/concat-stack/where)")
    print("This is the FIRST real Docker test of the shared `bitwise` component "
          "pipeline -- if any step below crashes, that is real, expected-possible "
          "information (see this file's own module docstring), not a bug in your setup.")
    check_reshape_copy()
    check_repeat()
    check_gather()
    check_concat_stack()
    check_where()
    print("\nIf every step above printed a REAL cost line with no exception, "
          "all 5 new cost models ran a real Docker sub-run successfully.")


if __name__ == "__main__":
    main()
