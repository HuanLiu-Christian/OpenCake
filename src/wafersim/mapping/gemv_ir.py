from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from wafersim.config import Coord

GemvOp = Literal[
    "load_a_tile",
    "load_x_chunk",
    "load_bias",
    "forward_x_south",
    "compute_ax",
    "reduce_east",
    "expect_y",
]


@dataclass(frozen=True, slots=True)
class GemvCommand:
    op: GemvOp
    pe: Coord = (0, 0)
    src_pe: Coord = (0, 0)
    dst_pe: Coord | None = None
    color: str | None = None
    rows: int = 0
    cols: int = 0
    values: tuple[int, ...] = ()
    expected: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.op not in {
            "load_a_tile",
            "load_x_chunk",
            "load_bias",
            "forward_x_south",
            "compute_ax",
            "reduce_east",
            "expect_y",
        }:
            raise ValueError(f"unknown GEMV op: {self.op}")
        if self.rows < 0 or self.cols < 0:
            raise ValueError("rows and cols must be non-negative")
        if self.op == "load_a_tile":
            if self.rows <= 0 or self.cols <= 0:
                raise ValueError("load_a_tile needs positive rows and cols")
            if len(self.values) != self.rows * self.cols:
                raise ValueError("load_a_tile values must match rows * cols")
        if self.op == "load_x_chunk" and not self.values:
            raise ValueError("load_x_chunk needs values")
        if self.op == "load_bias" and not self.values:
            raise ValueError("load_bias needs values")
        if self.op in {"forward_x_south", "reduce_east"}:
            if self.dst_pe is None and self.color is None:
                raise ValueError(f"{self.op} needs either dst_pe or color")
            if self.dst_pe is not None and self.src_pe == self.dst_pe:
                raise ValueError(f"{self.op} needs distinct source and destination PEs")
        if self.op == "compute_ax" and (self.rows <= 0 or self.cols <= 0):
            raise ValueError("compute_ax needs positive rows and cols")
        if self.op == "expect_y" and not self.expected:
            raise ValueError("expect_y needs expected values")


def load_a_tile(
    pe: Coord,
    values: Iterable[int],
    *,
    rows: int,
    cols: int,
) -> GemvCommand:
    command = GemvCommand(
        op="load_a_tile",
        pe=pe,
        values=tuple(values),
        rows=rows,
        cols=cols,
    )
    command.validate()
    return command


def load_x_chunk(pe: Coord, values: Iterable[int]) -> GemvCommand:
    command = GemvCommand(op="load_x_chunk", pe=pe, values=tuple(values))
    command.validate()
    return command


def load_bias(pe: Coord, values: Iterable[int]) -> GemvCommand:
    command = GemvCommand(op="load_bias", pe=pe, values=tuple(values))
    command.validate()
    return command


def forward_x_south(
    *,
    src_pe: Coord,
    dst_pe: Coord | None = None,
    color: str | None = "x_color",
) -> GemvCommand:
    command = GemvCommand(
        op="forward_x_south",
        src_pe=src_pe,
        dst_pe=dst_pe,
        color=color,
    )
    command.validate()
    return command


def compute_ax(pe: Coord, *, rows: int, cols: int) -> GemvCommand:
    command = GemvCommand(op="compute_ax", pe=pe, rows=rows, cols=cols)
    command.validate()
    return command


def reduce_east(
    *,
    src_pe: Coord,
    dst_pe: Coord | None = None,
    color: str | None = "ax_color_1",
) -> GemvCommand:
    command = GemvCommand(
        op="reduce_east",
        src_pe=src_pe,
        dst_pe=dst_pe,
        color=color,
    )
    command.validate()
    return command


def expect_y(pe: Coord, expected: Iterable[int]) -> GemvCommand:
    command = GemvCommand(op="expect_y", pe=pe, expected=tuple(expected))
    command.validate()
    return command


@dataclass(slots=True)
class GemvProgram:
    commands: list[GemvCommand] = field(default_factory=list)

    def load_a_tile(
        self,
        pe: Coord,
        values: Iterable[int],
        *,
        rows: int,
        cols: int,
    ) -> GemvCommand:
        command = load_a_tile(pe, values, rows=rows, cols=cols)
        self.commands.append(command)
        return command

    def load_x_chunk(self, pe: Coord, values: Iterable[int]) -> GemvCommand:
        command = load_x_chunk(pe, values)
        self.commands.append(command)
        return command

    def load_bias(self, pe: Coord, values: Iterable[int]) -> GemvCommand:
        command = load_bias(pe, values)
        self.commands.append(command)
        return command

    def forward_x_south(
        self,
        *,
        src_pe: Coord,
        dst_pe: Coord | None = None,
        color: str | None = "x_color",
    ) -> GemvCommand:
        command = forward_x_south(src_pe=src_pe, dst_pe=dst_pe, color=color)
        self.commands.append(command)
        return command

    def compute_ax(self, pe: Coord, *, rows: int, cols: int) -> GemvCommand:
        command = compute_ax(pe, rows=rows, cols=cols)
        self.commands.append(command)
        return command

    def reduce_east(
        self,
        *,
        src_pe: Coord,
        dst_pe: Coord | None = None,
        color: str | None = "ax_color_1",
    ) -> GemvCommand:
        command = reduce_east(src_pe=src_pe, dst_pe=dst_pe, color=color)
        self.commands.append(command)
        return command

    def expect_y(self, pe: Coord, expected: Iterable[int]) -> GemvCommand:
        command = expect_y(pe, expected)
        self.commands.append(command)
        return command
