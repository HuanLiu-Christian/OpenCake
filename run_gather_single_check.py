#!/usr/bin/env python3
"""Real single-primitive check for `gather`: one real Tensor (a toy
embedding table) -> gather(x, dim, index) (real numeric result, real
numpy.take) -> GatherSpec.from_result() (bridged from the real OUTPUT
tensor's size) -> simulate_gather() -> real cycles/time/energy/area from
one real Docker `timeloop-model` run against the shared real ``bitwise``
component (see ``wafersim.primitives._memory_move``'s own module
docstring: a real identity-bitwise-pass-through per element -- there is
no dedicated DMA/copy-engine component in this project's pinned Docker
toolchain).

Like `repeat`, `gather` is UNCONDITIONAL -- every real call materializes
a genuinely new array (real ``numpy.take``), every time (see
``wafersim.primitives.slice_gather._timeloop_gather``'s own module
docstring), so a plain gather() call already triggers a real Docker run.

Used for: token embedding lookup (gather rows of the embedding table by
token id), per this primitive's own package docstring. This example
mirrors that real use case at toy scale: a 5-row x 4-dim embedding table,
gathering rows for 3 token ids (with a repeated id, to show a real
non-contiguous, non-injective gather -- not just a permutation).

No synthetic fallback anywhere: if Docker/TimeLoop fails, this raises and
prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_gather_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec
from wafersim.primitives.slice_gather import GatherSpec, gather, simulate_gather
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/gather_single_check")


def main() -> None:
    # Toy embedding table: 5 tokens (rows), 4-dim embeddings (cols).
    embedding_table = Tensor.from_values(
        [
            [0.1, 0.2, 0.3, 0.4],   # token id 0
            [1.1, 1.2, 1.3, 1.4],   # token id 1
            [2.1, 2.2, 2.3, 2.4],   # token id 2
            [3.1, 3.2, 3.3, 3.4],   # token id 3
            [4.1, 4.2, 4.3, 4.4],   # token id 4
        ]
    )
    print(f"embedding_table = {embedding_table.data.tolist()}  shape={embedding_table.shape}")

    token_ids = [2, 0, 4, 2]  # real sequence of token ids, id 2 repeated
    print(f"token_ids = {token_ids}")

    # Real numeric result -- plain numpy.take, no Docker involved here.
    result = gather(embedding_table, dim=0, index=token_ids)
    print(f"real numeric result gather(embedding_table, dim=0, index=token_ids) = "
          f"{result.data.tolist()}  shape={result.shape}")

    # Bridge: derived from the real OUTPUT tensor's size (every gathered
    # row is a real physical write) -- not from the input table.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = GatherSpec.from_result(result)
    print(f"bridged GatherSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model run -- this is the only place cycles/
    # energy/area come from; raises (never fabricates) on failure.
    cost = simulate_gather(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (gather, always-copying) ===")
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
