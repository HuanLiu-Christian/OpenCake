from __future__ import annotations

from dataclasses import dataclass

from wafersim.opencake.ir import AllocOp, ComputeOp, DefineColorOp, OpenCakeProgram, SendOp

SPATIAL_IR_SCHEMA_VERSION = "opencake.spatial_ir.v1"


@dataclass(frozen=True, slots=True)
class SpatialIrValidation:
    schema: str
    ops: int
    ranks_with_allocations: int
    colors: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "ops": self.ops,
            "ranks_with_allocations": self.ranks_with_allocations,
            "colors": list(self.colors),
        }


def validate_spatial_program(program: OpenCakeProgram) -> SpatialIrValidation:
    colors: dict[str, str] = {}
    allocated: dict[tuple[int, int], list[tuple[int, int, str]]] = {}

    for op in program.ops:
        if isinstance(op, DefineColorOp):
            if op.color in colors and colors[op.color] != op.direction:
                raise ValueError(f"color {op.color} is defined with conflicting directions")
            colors[op.color] = op.direction
        elif isinstance(op, AllocOp):
            if op.addr < 0 or op.bytes <= 0:
                raise ValueError(f"invalid allocation {op.name} on rank {op.rank}")
            end = op.addr + op.bytes
            for start, existing_end, existing_name in allocated.get(op.rank, []):
                if op.addr < existing_end and end > start:
                    raise ValueError(
                        f"allocation {op.name} overlaps {existing_name} on rank {op.rank}"
                    )
            allocated.setdefault(op.rank, []).append((op.addr, end, op.name))
        elif isinstance(op, SendOp):
            if op.color not in colors:
                raise ValueError(f"send uses undefined color {op.color}")
            if op.bytes <= 0:
                raise ValueError("send bytes must be positive")
            if len(op.route) < 2:
                raise ValueError("send route must contain at least src and dst rank")
            if op.route[0] != op.src_rank or op.route[-1] != op.dst_rank:
                raise ValueError("send route endpoints must match src_rank and dst_rank")
        elif isinstance(op, ComputeOp):
            if op.ops <= 0 or op.memory_bytes <= 0:
                raise ValueError(f"invalid compute op on rank {op.rank}")
            if op.m <= 0 or op.n <= 0 or op.k <= 0:
                raise ValueError(f"invalid GEMM tile shape on rank {op.rank}")

    return SpatialIrValidation(
        schema=SPATIAL_IR_SCHEMA_VERSION,
        ops=len(program.ops),
        ranks_with_allocations=len(allocated),
        colors=tuple(sorted(colors)),
    )
