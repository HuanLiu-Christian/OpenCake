"""PE tiling policy for LOCAL, embarrassingly-parallel primitives.

WHY THIS EXISTS. Every elementwise primitive here (add/mul/silu/clamp/...)
has a real Docker-backed cost model for ONE PE processing
``elements_per_pe`` elements one after another. Until this module, the
experiment scripts fed those models the WHOLE tensor -- i.e. they assumed
all N elements sit on a single PE and are processed serially -- while GEMM
is mapped across the whole mesh. That makes elementwise cycles look
enormous next to GEMM (silu alone was ~49% of a gpt-oss MoE block's cycles
and ~60% of Qwen3's), and it makes large tensors (e.g. a long-context KV
cache) crash against the single-PE SRAM ceiling.

WHAT IT DOES. Real tensors are distributed across the mesh and an
elementwise op touches only each PE's own slice, so the real latency is that
of the BUSIEST PE's slice, while the total energy is still the sum over all
PEs. ``Tiling`` computes that split:

    shard = tiling.elements(n, total_pes)
    cycles          = cycles_of(one PE doing shard.per_pe elements)
    total energy    = energy_of(one PE doing shard.per_pe) * shard.energy_multiplier

``energy_multiplier = n / per_pe`` (exactly the number of PE-equivalents), so
a one-PE ("serial") shard reproduces the old accounting bit for bit and a
tiled shard conserves total work/energy instead of inflating it by ceil
rounding.

TWO MODES (an explicit bracket, because the true mapping is not known):
  - ``Tiling.serial()``  -- everything on one PE. The OLD behavior; a
    pessimistic upper bound on cycles, kept so earlier results stay
    reproducible.
  - ``Tiling()`` ("mesh") -- spread over up to ``total_pes`` PEs with at
    least ``min_elements_per_pe`` elements each. With the default
    ``min_elements_per_pe=1`` this is an OPTIMISTIC bound on parallelism: it
    assumes the data is already laid out over as many PEs as there are
    elements. The real answer depends on the producing GEMM's output layout
    and lies between the two modes.

SCOPE. Only ops whose work is purely local and independent are tiled
(elementwise arithmetic, clamp, the independent lanes of a reduction, the
independent rows of a top-k). Data-movement ops that cross PEs (gather,
concat, repeat, reshape-copy) and single long reductions are deliberately
NOT tiled here -- they need the network/cross-PE models, and treating them
as local would hide real cost.
"""

from __future__ import annotations

from dataclasses import dataclass

_MODES = ("mesh", "serial")


@dataclass(frozen=True, slots=True)
class Shard:
    """How ``total`` independent work items are split over PEs."""

    total: int  # logical items (elements, reduction lanes or top-k rows)
    per_pe: int  # items the BUSIEST PE handles -> determines latency
    pes: int  # PEs actually used

    @property
    def energy_multiplier(self) -> float:
        """Total energy = single-PE energy at ``per_pe`` items times this."""
        return self.total / self.per_pe


def _split(total: int, max_pes: int) -> Shard:
    pes = max(1, min(max_pes, total))
    per_pe = -(-total // pes)  # ceil
    return Shard(total=total, per_pe=per_pe, pes=-(-total // per_pe))


@dataclass(frozen=True, slots=True)
class Tiling:
    mode: str = "mesh"
    min_elements_per_pe: int = 1

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if self.min_elements_per_pe < 1:
            raise ValueError("min_elements_per_pe must be >= 1")

    @classmethod
    def serial(cls) -> "Tiling":
        return cls(mode="serial")

    def elements(self, n: int, total_pes: int) -> Shard:
        """Split ``n`` elementwise-independent elements."""
        if n < 1 or total_pes < 1:
            raise ValueError("n and total_pes must be positive")
        if self.mode == "serial":
            return Shard(total=n, per_pe=n, pes=1)
        # floor, so every PE used holds at least min_elements_per_pe elements
        return _split(n, min(total_pes, max(1, n // self.min_elements_per_pe)))

    def lanes(self, n: int, total_pes: int) -> Shard:
        """Split ``n`` independent whole units (reduction lanes, top-k rows):
        a unit is never divided across PEs, so the minimum is 1 per PE."""
        if n < 1 or total_pes < 1:
            raise ValueError("n and total_pes must be positive")
        if self.mode == "serial":
            return Shard(total=n, per_pe=n, pes=1)
        return _split(n, total_pes)

    def describe(self, total_pes: int) -> str:
        if self.mode == "serial":
            return "serial (all elements on ONE PE; pessimistic upper bound on cycles, the old behavior)"
        return (f"mesh (up to {total_pes:,} PEs, >= {self.min_elements_per_pe} element(s)/PE; "
                "optimistic bound on parallelism)")
