from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Literal

Coord = tuple[int, int]
Direction = Literal["east", "west", "north", "south"]


@dataclass(frozen=True, slots=True)
class DefineColorOp:
    op: Literal["define_color"]
    color: str
    direction: Direction


@dataclass(frozen=True, slots=True)
class AllocOp:
    op: Literal["alloc"]
    rank: Coord
    name: str
    addr: int
    bytes: int


@dataclass(frozen=True, slots=True)
class SendOp:
    op: Literal["send"]
    src_rank: Coord
    dst_rank: Coord
    color: str
    bytes: int
    route: tuple[Coord, ...]
    tensor: str = ""
    k_start: int = 0

    @property
    def hops(self) -> int:
        return max(0, len(self.route) - 1)


@dataclass(frozen=True, slots=True)
class ComputeOp:
    op: Literal["compute"]
    rank: Coord
    kernel: str
    m: int
    n: int
    k: int
    ops: int
    memory_bytes: int
    k_start: int = 0


SpatialOp = DefineColorOp | AllocOp | SendOp | ComputeOp


@dataclass(frozen=True, slots=True)
class OpenCakeProgram:
    """PE/rank spatial execution plan emitted by the OpenCake mapper."""

    workload: dict[str, object]
    hardware: dict[str, object]
    ops: tuple[SpatialOp, ...] = field(default_factory=tuple)

    def records(self) -> list[dict[str, object]]:
        return [_record(op) for op in self.ops]

    def write_jsonl(self, path: str | Path) -> None:
        output = Path(path)
        with output.open("w", encoding="utf-8") as handle:
            for record in self.records():
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")


def _record(op: SpatialOp) -> dict[str, object]:
    record = asdict(op)
    if isinstance(op, SendOp):
        record["route"] = [list(coord) for coord in op.route]
        record["src_rank"] = list(op.src_rank)
        record["dst_rank"] = list(op.dst_rank)
    elif isinstance(op, AllocOp | ComputeOp):
        record["rank"] = list(op.rank)
    return record
