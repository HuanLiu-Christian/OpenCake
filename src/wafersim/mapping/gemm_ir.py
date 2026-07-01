from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from wafersim.config import Coord

GemmOp = Literal[
    "load_a_panel",
    "load_b_panel",
    "forward_a_east",
    "forward_b_south",
    "compute_c",
    "expect_c",
]


@dataclass(frozen=True, slots=True)
class GemmCommand:
    op: GemmOp
    pe: Coord = (0, 0)
    src_pe: Coord = (0, 0)
    dst_pe: Coord | None = None
    color: str | None = None
    rows: int = 0
    cols: int = 0
    k: int = 0
    values: tuple[int, ...] = ()
    expected: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.op not in {
            "load_a_panel",
            "load_b_panel",
            "forward_a_east",
            "forward_b_south",
            "compute_c",
            "expect_c",
        }:
            raise ValueError(f"unknown GEMM op: {self.op}")
        if self.rows < 0 or self.cols < 0 or self.k < 0:
            raise ValueError("rows, cols, and k must be non-negative")
        if self.op == "load_a_panel":
            if self.rows <= 0 or self.k <= 0:
                raise ValueError("load_a_panel needs positive rows and k")
            if len(self.values) != self.rows * self.k:
                raise ValueError("load_a_panel values must match rows * k")
        if self.op == "load_b_panel":
            if self.cols <= 0 or self.k <= 0:
                raise ValueError("load_b_panel needs positive cols and k")
            if len(self.values) != self.k * self.cols:
                raise ValueError("load_b_panel values must match k * cols")
        if self.op in {"forward_a_east", "forward_b_south"}:
            if self.dst_pe is None and self.color is None:
                raise ValueError(f"{self.op} needs either dst_pe or color")
            if self.dst_pe is not None and self.src_pe == self.dst_pe:
                raise ValueError(f"{self.op} needs distinct source and destination PEs")
        if self.op == "compute_c" and (
            self.rows <= 0 or self.cols <= 0 or self.k <= 0
        ):
            raise ValueError("compute_c needs positive rows, cols, and k")
        if self.op == "expect_c":
            if self.rows <= 0 or self.cols <= 0:
                raise ValueError("expect_c needs positive rows and cols")
            if len(self.expected) != self.rows * self.cols:
                raise ValueError("expect_c expected values must match rows * cols")


def load_a_panel(
    pe: Coord,
    values: Iterable[int],
    *,
    rows: int,
    k: int,
) -> GemmCommand:
    command = GemmCommand(
        op="load_a_panel",
        pe=pe,
        values=tuple(values),
        rows=rows,
        k=k,
    )
    command.validate()
    return command


def load_b_panel(
    pe: Coord,
    values: Iterable[int],
    *,
    cols: int,
    k: int,
) -> GemmCommand:
    command = GemmCommand(
        op="load_b_panel",
        pe=pe,
        values=tuple(values),
        cols=cols,
        k=k,
    )
    command.validate()
    return command


def forward_a_east(
    *,
    src_pe: Coord,
    dst_pe: Coord | None = None,
    color: str | None = "a_color",
) -> GemmCommand:
    command = GemmCommand(
        op="forward_a_east",
        src_pe=src_pe,
        dst_pe=dst_pe,
        color=color,
    )
    command.validate()
    return command


def forward_b_south(
    *,
    src_pe: Coord,
    dst_pe: Coord | None = None,
    color: str | None = "b_color",
) -> GemmCommand:
    command = GemmCommand(
        op="forward_b_south",
        src_pe=src_pe,
        dst_pe=dst_pe,
        color=color,
    )
    command.validate()
    return command


def compute_c(pe: Coord, *, rows: int, cols: int, k: int) -> GemmCommand:
    command = GemmCommand(op="compute_c", pe=pe, rows=rows, cols=cols, k=k)
    command.validate()
    return command


def expect_c(
    pe: Coord,
    expected: Iterable[int],
    *,
    rows: int,
    cols: int,
) -> GemmCommand:
    command = GemmCommand(
        op="expect_c",
        pe=pe,
        expected=tuple(expected),
        rows=rows,
        cols=cols,
    )
    command.validate()
    return command


@dataclass(slots=True)
class GemmProgram:
    commands: list[GemmCommand] = field(default_factory=list)

    def load_a_panel(
        self,
        pe: Coord,
        values: Iterable[int],
        *,
        rows: int,
        k: int,
    ) -> GemmCommand:
        command = load_a_panel(pe, values, rows=rows, k=k)
        self.commands.append(command)
        return command

    def load_b_panel(
        self,
        pe: Coord,
        values: Iterable[int],
        *,
        cols: int,
        k: int,
    ) -> GemmCommand:
        command = load_b_panel(pe, values, cols=cols, k=k)
        self.commands.append(command)
        return command

    def forward_a_east(
        self,
        *,
        src_pe: Coord,
        dst_pe: Coord | None = None,
        color: str | None = "a_color",
    ) -> GemmCommand:
        command = forward_a_east(src_pe=src_pe, dst_pe=dst_pe, color=color)
        self.commands.append(command)
        return command

    def forward_b_south(
        self,
        *,
        src_pe: Coord,
        dst_pe: Coord | None = None,
        color: str | None = "b_color",
    ) -> GemmCommand:
        command = forward_b_south(src_pe=src_pe, dst_pe=dst_pe, color=color)
        self.commands.append(command)
        return command

    def compute_c(self, pe: Coord, *, rows: int, cols: int, k: int) -> GemmCommand:
        command = compute_c(pe, rows=rows, cols=cols, k=k)
        self.commands.append(command)
        return command

    def expect_c(
        self,
        pe: Coord,
        expected: Iterable[int],
        *,
        rows: int,
        cols: int,
    ) -> GemmCommand:
        command = expect_c(pe, expected, rows=rows, cols=cols)
        self.commands.append(command)
        return command
