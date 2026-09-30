"""concat/stack primitive.

STATUS: fully split and REAL on both sides now --
  - Numeric compute: ``concat()``/``stack()`` are REAL, implemented on
    top of ``wafersim.primitives.tensor.Tensor`` (no Docker/TimeLoop
    involved for the numeric values themselves -- see ``Tensor``'s own
    module docstring).
  - Real cycle/energy/area COST MODEL for the LOCAL case (every input
    tensor already lives on one PE): both are unconditionally
    real-copying every call (see below) -- ``ConcatStackSpec``/
    ``simulate_concat_stack`` re-exported below from
    ``_timeloop_concat.py``, built on the shared real
    ``wafersim.primitives._memory_move`` pipeline (a real identity-
    bitwise-pass-through per written output element -- see that module's
    own docstring for the real Accelergy-source citation and modeling
    choice).
  - Real cycle/energy/area COST MODEL for the CROSS-PE case (the real
    input shards being joined live on DIFFERENT physical PEs -- e.g.
    recombining multi-head attention output when heads are sharded
    across the mesh, or appending a new KV-cache step whose existing
    entries are sharded across PEs): REAL Docker (one representative
    per-hop append firing, reusing the same real ``_memory_move``
    pipeline the local case uses) + REAL network simulation (the same
    ``PacketNetwork._send_hop``/``_packet_sizes`` real link/FIFO/packet
    timing GEMM's own cross-PE broadcast uses) + OpenCake's own
    analytical network/static energy and interconnect-area coefficients
    -- implemented in ``_cross_pe_concat_stack.py`` --
    ``ConcatStackCrossPESpec``/``ConcatStackCrossPEResult``/
    ``simulate_concat_stack_cross_pe`` re-exported below. Modeled as a
    real, named "shift-based redistribution" (WaferLLM, arXiv
    2502.04563's own phrase for real KV-cache-concat-style cross-PE data
    movement) sequential relay chain -- see that module's own module
    docstring for the full citation trail (including Cerebras's official
    ``collectives_2d.gather()`` primitive, the closest documented
    hardware mechanism for this "many real per-PE buffers -> one
    combined buffer" shape) and the explicit real/simulated/analytical
    breakdown of what each part of the cost is.

Structurally necessary (cannot be built from reshape/transpose/slice alone,
since those only ever operate on a single input tensor): RoPE's rotate_half
step, recombining multi-head attention output before the output projection,
and -- most importantly for real autoregressive generation -- appending each
step's new K/V into the growing KV cache. Confirmed as a first-class,
independently-implemented op (`GGML_OP_CONCAT`) in llama.cpp's real
production GGML op table, not something assumed.

Both real numpy ``concatenate``/``stack`` always copy (there is no way to
join two separately-allocated real arrays without moving at least one of
them) -- this matters for the KV-cache-append use case in particular:
appending is NOT free here, matching how a real KV cache append is a real
memory write, not a metadata-only operation. In the CROSS-PE case this is
even more literal: every shard not already on the destination PE must
physically travel over the real fabric before it can be written in.

For a LOCAL join (every input tensor already on one PE), bridge via
``ConcatStackSpec.from_result(result)``/``simulate_concat_stack``. For a
CROSS-PE join (each input shard lives on a different one of ``num_pes``
physical PEs), bridge via ``ConcatStackCrossPESpec(total_elements=
result.size, num_pes=num_pes, bytes_per_element=result.bytes_per_element)``
with ``simulate_concat_stack_cross_pe`` instead.
"""

from __future__ import annotations

import numpy as np

from ._cross_pe_concat_stack import (
    ConcatStackCrossPEResult,
    ConcatStackCrossPESpec,
    simulate_concat_stack_cross_pe,
)
from ._timeloop_concat import ConcatStackSpec, simulate_concat_stack
from ..tensor import Tensor


def _require_same_dtype(tensors: list[Tensor], op_name: str) -> None:
    dtypes = {t.dtype for t in tensors}
    if len(dtypes) > 1:
        raise TypeError(
            f"{op_name}(): all input tensors must share one real dtype, got "
            f"{sorted(str(d) for d in dtypes)} -- cast explicitly via "
            f"wafersim.primitives.cast.cast(...) first; this project models "
            f"mixed precision explicitly, not via numpy's implicit type "
            f"promotion."
        )


def concat(tensors: list[Tensor], dim: int) -> Tensor:
    if not tensors:
        raise ValueError("concat(): tensors must be a non-empty list of Tensor")
    _require_same_dtype(tensors, "concat")
    return Tensor(np.concatenate([t.data for t in tensors], axis=dim))


def stack(tensors: list[Tensor], dim: int) -> Tensor:
    if not tensors:
        raise ValueError("stack(): tensors must be a non-empty list of Tensor")
    _require_same_dtype(tensors, "stack")
    return Tensor(np.stack([t.data for t in tensors], axis=dim))


__all__ = [
    "concat",
    "stack",
    "ConcatStackSpec",
    "simulate_concat_stack",
    "ConcatStackCrossPESpec",
    "ConcatStackCrossPEResult",
    "simulate_concat_stack_cross_pe",
]
