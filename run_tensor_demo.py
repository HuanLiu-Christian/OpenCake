#!/usr/bin/env python3
"""Real numpy demo of the now-implemented Tensor data structure.

Unlike run_add_950_1050.py/run_mul_950_1050.py/run_div_950_1050.py/
run_sum_4elements.py/run_max_4elements.py, this script does NOT launch
Docker/timeloop-mapper/timeloop-model at all -- Tensor is a pure, real
numpy-backed data structure (see wafersim/primitives/tensor/__init__.py's
module docstring for why building/operating on real tensor VALUES never
needs TimeLoop, which only ever estimates hardware cost for an operation
COUNT, not real data). This script runs instantly, no Docker Desktop
required.

Usage:
  source .venv/bin/activate
  python3 run_tensor_demo.py
"""

from __future__ import annotations

import numpy as np

from wafersim.primitives.tensor import Tensor


def main() -> None:
    # Real data via from_values -- e.g. a tiny attention score row.
    scores = Tensor.from_values([950.0, 1050.0, 200.0, 300.0])
    print("scores:", scores)
    print("  shape              =", scores.shape)
    print("  dtype               =", scores.dtype)
    print("  bytes_per_element   =", scores.bytes_per_element)
    print("  size (elements)     =", scores.size)

    # Real reproducible random data -- same np.random.default_rng(seed)
    # convention as wafer/functional.py's own real A/B correctness matrices.
    weight = Tensor.random((4, 4), seed=0)
    print("\nweight:", weight)
    print("  shape              =", weight.shape)

    # zeros() constructor.
    bias = Tensor.zeros((4,))
    print("\nbias:", bias)

    # Real numeric comparison (never `==` -- see Tensor's class docstring).
    same_scores = Tensor.from_values([950.0, 1050.0, 200.0, 300.0])
    different_scores = Tensor.from_values([1.0, 2.0, 3.0, 4.0])
    print("\nscores.allclose(same_scores)      =", scores.allclose(same_scores))
    print("scores.allclose(different_scores) =", scores.allclose(different_scores))

    # The bridge to this project's real cost model, shown but not called:
    # once a numeric op like add(a, b) is implemented on top of Tensor, its
    # real output Tensor's .size/.bytes_per_element plug straight into the
    # already-real, Docker-verified AddSpec/simulate_add from
    # wafersim.primitives.add -- e.g.:
    #   from wafersim.primitives.add import AddSpec, simulate_add
    #   spec = AddSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    #   cost = simulate_add(spec, wafer, out_dir)  # <- THIS is the real Docker call
    # Tensor construction/introspection itself above never launches Docker.
    print(
        "\n(no Docker/timeloop launched by this script -- Tensor itself is "
        "pure numpy; only simulate_add()/simulate_mul()/.../simulate_gemm() "
        "launch the real Docker cost-model pipeline, see this script's "
        "module docstring)"
    )

    # dtype kind validation is real, not decorative.
    try:
        Tensor(np.array(["a", "b"]))
    except TypeError as exc:
        print("\nrejecting a non-numeric dtype, as expected:", exc)


if __name__ == "__main__":
    main()
