#!/usr/bin/env python3
"""Real single-primitive check for `concat`: two real Tensors -> concat([a,
b], dim) (real numeric result, real numpy.concatenate) ->
ConcatStackSpec.from_result() (bridged from the real OUTPUT tensor's
size) -> simulate_concat_stack() -> real cycles/time/energy/area from one
real Docker `timeloop-model` run against the shared real ``bitwise``
component (see ``wafersim.primitives._memory_move``'s own module
docstring: a real identity-bitwise-pass-through per element -- there is
no dedicated DMA/copy-engine component in this project's pinned Docker
toolchain).

Like `repeat`/`gather`, `concat`/`stack` are UNCONDITIONAL -- every real
call physically copies (there is no way to join two separately-allocated
real arrays without moving at least one of them -- see
``wafersim.primitives.concat_stack``'s own package docstring), so a plain
concat() call already triggers a real Docker run.

Used for: most importantly, appending each step's new K/V into the
growing KV cache during real autoregressive generation -- per this
primitive's own package docstring, this is NOT free (a real memory
write, not a metadata-only operation), confirmed as a first-class
`GGML_OP_CONCAT` op in llama.cpp's real production op table. This example
mirrors that real use case at toy scale: a 2-token KV cache (each a
4-dim K vector) with 1 new token's K vector appended.

No synthetic fallback anywhere: if Docker/TimeLoop fails, this raises and
prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_concat_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec
from wafersim.primitives.concat_stack import ConcatStackSpec, concat, simulate_concat_stack
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/concat_single_check")


def main() -> None:
    # Existing KV cache: 2 cached tokens' K vectors (4-dim each).
    kv_cache = Tensor.from_values([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # shape (2, 4)
    # New token's K vector, just computed this decode step.
    new_kv = Tensor.from_values([[9.0, 10.0, 11.0, 12.0]])  # shape (1, 4)
    print(f"kv_cache = {kv_cache.data.tolist()}  shape={kv_cache.shape}")
    print(f"new_kv   = {new_kv.data.tolist()}  shape={new_kv.shape}")

    # Real numeric result -- plain numpy.concatenate, no Docker involved
    # here.
    result = concat([kv_cache, new_kv], dim=0)
    print(f"real numeric result concat([kv_cache, new_kv], dim=0) = "
          f"{result.data.tolist()}  shape={result.shape}")

    # Bridge: derived from the real OUTPUT tensor's size (every element
    # of the updated cache is a real physical write, including the
    # elements copied over from the old cache) -- not from either input
    # alone.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = ConcatStackSpec.from_result(result)
    print(f"bridged ConcatStackSpec: elements_per_pe={spec.elements_per_pe}, "
          f"bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model run -- this is the only place cycles/
    # energy/area come from; raises (never fabricates) on failure.
    cost = simulate_concat_stack(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (concat, always-copying) ===")
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
