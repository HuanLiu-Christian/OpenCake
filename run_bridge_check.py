#!/usr/bin/env python3
"""Real end-to-end check: Tensor -> numeric primitive -> real Docker cost
model, for all 6 primitives that currently have BOTH layers implemented
(add, mul, div, sum, max, matmul).

This is a genuinely new check, not a re-run of anything already verified.
run_add_950_1050.py/run_mul_950_1050.py/.../run_max_4elements.py already
verified each primitive's real Docker cost model in isolation, by hand-
building an AddSpec/MulSpec/.../MaxSpec directly. What was NOT verified
yet is the specific BRIDGE pattern documented in each primitive's package
docstring: build a real Tensor, run the real numeric op on it
(add(a, b)/mul(a, b)/.../matmul(a, b)), then derive the cost-model Spec's
fields (elements_per_pe/bytes_per_element, or m/n/k for matmul) FROM that
real Tensor result's real .size/.shape/.bytes_per_element -- instead of
typing those numbers in by hand. This script is that missing check.

No synthetic fallback anywhere -- every cost number below comes from a
real `timeloop-model`/Accelergy Docker sub-run. If Docker/TimeLoop fails
for any of the 6, this raises and prints nothing fabricated for that step.

Usage (run from the project root, with your usual venv active, Docker
Desktop open -- this launches 6 separate real Docker sub-runs, the matmul
one evaluates a handful of tile_k candidates so it's the slowest of the
six, but the GEMM here is tiny (32x32x32) so it should still be quick):
  source .venv/bin/activate
  python3 run_bridge_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives.add import AddSpec, simulate_add
from wafersim.primitives.div import DivSpec, simulate_div
from wafersim.primitives.matmul import GemmSpec, WaferSpec, best_by_sim_time, search_tile_k
from wafersim.primitives.max import MaxSpec, simulate_max
from wafersim.primitives.mul import MulSpec, simulate_mul
from wafersim.primitives.sum import SumSpec, simulate_sum
from wafersim.primitives.tensor import Tensor
from wafersim.primitives import add, div, matmul, max as t_max, mul, sum as t_sum

OUT_DIR = Path("outputs/bridge_check")


def check_add() -> None:
    print("\n--- add: Tensor -> add(a, b) -> AddSpec(from result) -> simulate_add ---")
    a = Tensor.from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = Tensor.from_values([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    result = add(a, b)
    print(f"  numeric result: {result} values={result.data.tolist()}")
    spec = AddSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    cost = simulate_add(spec, WaferSpec(), OUT_DIR / "add")
    print(f"  bridged AddSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}, adder_area_um2={cost.adder_area_um2}")


def check_mul() -> None:
    print("\n--- mul: Tensor -> mul(a, b) -> MulSpec(from result) -> simulate_mul ---")
    a = Tensor.from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = Tensor.from_values([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    result = mul(a, b)
    print(f"  numeric result: {result} values={result.data.tolist()}")
    spec = MulSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    cost = simulate_mul(spec, WaferSpec(), OUT_DIR / "mul")
    print(f"  bridged MulSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}, multiplier_area_um2={cost.multiplier_area_um2}")


def check_div() -> None:
    print("\n--- div: Tensor -> div(a, b) -> DivSpec(from result) -> simulate_div ---")
    a = Tensor.from_values([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    b = Tensor.from_values([2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    result = div(a, b)
    print(f"  numeric result: {result} values={result.data.tolist()}")
    spec = DivSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    cost = simulate_div(spec, WaferSpec(), OUT_DIR / "div")
    print(f"  bridged DivSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}")


def check_sum() -> None:
    print("\n--- sum: Tensor -> sum(x, dim) -> SumSpec(from x.shape[dim]) -> simulate_sum ---")
    x = Tensor.from_values([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    result = t_sum(x, dim=-1)
    print(f"  numeric result: {result} values={result.data.tolist()}")
    # Bridge uses x.shape[dim] (the K elements reduced in ONE lane), not
    # result.size or x.size -- see sum's own package docstring for why.
    spec = SumSpec(elements_per_pe=x.shape[-1], bytes_per_element=x.bytes_per_element)
    cost = simulate_sum(spec, WaferSpec(), OUT_DIR / "sum")
    print(f"  bridged SumSpec: elements_per_pe={spec.elements_per_pe} (=x.shape[-1]), bytes_per_element={spec.bytes_per_element}")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}, real_add_count={spec.real_add_count}")


def check_max() -> None:
    print("\n--- max: Tensor -> max(x, dim) -> MaxSpec(from x.shape[dim]) -> simulate_max ---")
    x = Tensor.from_values([[9.0, 2.0, 30.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    result = t_max(x, dim=-1)
    print(f"  numeric result: {result} values={result.data.tolist()}")
    spec = MaxSpec(elements_per_pe=x.shape[-1], bytes_per_element=x.bytes_per_element)
    cost = simulate_max(spec, WaferSpec(), OUT_DIR / "max")
    print(f"  bridged MaxSpec: elements_per_pe={spec.elements_per_pe} (=x.shape[-1]), bytes_per_element={spec.bytes_per_element}")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}, real_compare_count={spec.real_compare_count}")


def check_matmul() -> None:
    print("\n--- matmul: Tensor -> matmul(a, b) -> GemmSpec(from a/b.shape) -> search_tile_k/simulate_gemm ---")
    a = Tensor.random((32, 32), seed=1)
    b = Tensor.random((32, 32), seed=2)
    result = matmul(a, b)
    print(f"  numeric result: {result}")
    gemm = GemmSpec(m=a.shape[-2], n=b.shape[-1], k=a.shape[-1], bytes_per_element=a.bytes_per_element)
    print(f"  bridged GemmSpec: m={gemm.m}, n={gemm.n}, k={gemm.k}, bytes_per_element={gemm.bytes_per_element}")
    candidates = search_tile_k(gemm, WaferSpec(), OUT_DIR / "matmul")
    best = best_by_sim_time(candidates)
    print(f"  REAL cost (best of {len(candidates)} real tile_k candidates): "
          f"tile_k={best.tile_k}, sim_time_cycles={best.result.sim_time_cycles}, "
          f"total_energy_uj={best.result.energy.total_uj}")


def main() -> None:
    print("Real Tensor -> numeric primitive -> real Docker cost-model bridge check")
    print("(6 real Docker sub-runs total -- add/mul/div/sum/max are quick, matmul "
          "evaluates several tile_k candidates so it's the slowest)")
    check_add()
    check_mul()
    check_div()
    check_sum()
    check_max()
    check_matmul()
    print("\nAll 6 bridges ran a real Docker sub-run successfully -- no step above "
          "was skipped or fell back to a synthetic number.")


if __name__ == "__main__":
    main()
