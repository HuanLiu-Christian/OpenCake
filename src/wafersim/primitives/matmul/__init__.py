"""matmul primitive.

This module has two independent parts:

1. The real GEMM COST MODEL -- a thin re-export layer, no new logic. This
is NOT a fresh implementation. `wafersim`'s top-level GEMM pipeline
(specs.py/contract.py/search.py/area_budget.py/timeloop.py/wafer/simulator.py)
already IS the real, Docker-verified `matmul` cost model: every number it
produces comes from an actual `timeloop-model`/accelergy run, never a local
estimate (see timeloop.py's module-level discipline). Moving that code here
would risk breaking the working CLI (`python -m wafersim`), the 41 existing
tests, the sweep script, and the real `outputs/...` paths cited by name in
README.md/PIPELINE_SUMMARY.md/TIMELOOP_STAGE_PIPELINE.md/
OPENCAKE_CODE_WALKTHROUGH.md as verified reference data -- so none of that
moves. This module only re-exports the existing real symbols under the
`wafersim.primitives.matmul` namespace so callers can do
`from wafersim.primitives import matmul` alongside the other 16 primitives.

Real, ready-to-use entry points re-exported below:
  - GemmSpec / WaferSpec: describe the GEMM problem and the target hardware.
  - GemmMappingContract: a parsed, real Docker-run mapping (tile sizes, real
    per-instance area/cycles/energy from Accelergy/TimeLoop).
  - simulate_gemm / WaferGemmResult: run the full wafer-level (compute +
    network) simulation on top of one real TimeLoop contract.
  - search_tile_k / best_by_sim_time / TileKCandidate: evaluate every
    SRAM-feasible tile_k with a real timeloop-model run each, and pick the
    full-system optimum (see search.py).
  - AreaBudget / AreaBudgetResult / derive_wafer_spec_from_area_budget:
    derive mesh/SRAM/FIFO sizing from a physical die-area budget via a real,
    Docker-backed binary search (see area_budget.py).

Low-level Docker plumbing (generate_timeloop_inputs, run_timeloop_model,
parse_mapping_contract, ...) stays internal to `wafersim.timeloop` -- it is
implementation detail, not part of the matmul primitive's public interface.

2. Numeric compute: ``matmul(a, b)`` below is REAL, implemented on top of
``wafersim.primitives.tensor.Tensor`` (real numpy ``@``, no Docker/TimeLoop
involved -- see ``Tensor``'s own module docstring, and
``wafer/functional.py``'s own precedent of checking this project's real
SUMMA tiling schedule against a plain ``A @ B`` reference). This is a
NEW addition, independent of part 1 above -- it answers "what are the real
output values", never "what would this cost on real wafer-scale hardware"
(that question is answered by ``simulate_gemm``/``search_tile_k``, which
never touch ``a``/``b``'s real values either -- see part 1's own module
history). Bridge them explicitly when both are wanted at once, e.g. via
``GemmSpec(m=a.shape[-2], n=b.shape[-1], k=a.shape[-1],
bytes_per_element=a.bytes_per_element)`` fed to ``search_tile_k``/
``simulate_gemm``.
"""

from __future__ import annotations

from ...area_budget import AreaBudget, AreaBudgetResult, derive_wafer_spec_from_area_budget
from ...contract import GemmMappingContract
from ...search import TileKCandidate, best_by_sim_time, search_tile_k
from ...specs import GemmSpec, WaferSpec
from ...wafer.simulator import WaferGemmResult, simulate_gemm
from ..tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Real matrix multiply (``a.data @ b.data``, numpy's own real
    ``matmul``, batched the same way numpy batches leading dimensions) on
    real Tensor data. Requires both operands to already share one real
    dtype (see e.g. ``wafersim.primitives.add``'s package docstring for why
    this project never silently up-casts via numpy's implicit type
    promotion)."""
    if a.dtype != b.dtype:
        raise TypeError(
            f"matmul(a, b): dtype mismatch a.dtype={a.dtype} vs "
            f"b.dtype={b.dtype} -- cast one operand explicitly via "
            f"wafersim.primitives.cast.cast(...) first."
        )
    return Tensor(a.data @ b.data)


__all__ = [
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
    "matmul",
]
