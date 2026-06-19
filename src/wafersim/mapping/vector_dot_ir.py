from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from wafersim.config import Coord

VectorDotOp = Literal["load_left", "load_right", "compute_dot", "reduce_east", "expect"]


@dataclass(frozen=True, slots=True)
class VectorDotCommand:
    op: VectorDotOp
    words: tuple[int, ...] = ()
    word_count: int = 0
    expected: int | None = None
    pe: Coord = (0, 0)
    src_pe: Coord = (0, 0)
    dst_pe: Coord | None = None
    color: str | None = None

    def validate(self) -> None:
        if self.op not in {"load_left", "load_right", "compute_dot", "reduce_east", "expect"}:
            raise ValueError(f"unknown vector dot op: {self.op}")
        if self.word_count < 0:
            raise ValueError("word_count must be non-negative")
        if self.op in {"load_left", "load_right"} and not self.words:
            raise ValueError(f"{self.op} needs at least one word")
        if self.op == "compute_dot" and self.word_count <= 0:
            raise ValueError("compute_dot needs a positive word_count")
        if self.op == "reduce_east":
            if self.dst_pe is None and self.color is None:
                raise ValueError("reduce_east needs either dst_pe or color")
            if self.dst_pe is not None and self.src_pe == self.dst_pe:
                raise ValueError("reduce_east needs distinct source and destination PEs")
        if self.op == "expect" and self.expected is None:
            raise ValueError("expect needs an expected result")


def load_left(words: Iterable[int], *, pe: Coord = (0, 0)) -> VectorDotCommand:
    command = VectorDotCommand(op="load_left", words=tuple(words), pe=pe)
    command.validate()
    return command


def load_right(words: Iterable[int], *, pe: Coord = (0, 0)) -> VectorDotCommand:
    command = VectorDotCommand(op="load_right", words=tuple(words), pe=pe)
    command.validate()
    return command


def compute_dot(word_count: int, *, pe: Coord = (0, 0)) -> VectorDotCommand:
    command = VectorDotCommand(op="compute_dot", word_count=word_count, pe=pe)
    command.validate()
    return command


def reduce_east(
    *,
    src_pe: Coord,
    dst_pe: Coord | None = None,
    color: str | None = "partial_sum_color",
) -> VectorDotCommand:
    command = VectorDotCommand(
        op="reduce_east",
        src_pe=src_pe,
        dst_pe=dst_pe,
        color=color,
    )
    command.validate()
    return command


def expect_dot(expected: int, *, pe: Coord = (0, 0)) -> VectorDotCommand:
    command = VectorDotCommand(op="expect", expected=expected, pe=pe)
    command.validate()
    return command


@dataclass(slots=True)
class VectorDotProgram:
    commands: list[VectorDotCommand] = field(default_factory=list)

    def load_left(self, words: Iterable[int], *, pe: Coord = (0, 0)) -> VectorDotCommand:
        command = load_left(words, pe=pe)
        self.commands.append(command)
        return command

    def load_right(self, words: Iterable[int], *, pe: Coord = (0, 0)) -> VectorDotCommand:
        command = load_right(words, pe=pe)
        self.commands.append(command)
        return command

    def compute_dot(self, word_count: int, *, pe: Coord = (0, 0)) -> VectorDotCommand:
        command = compute_dot(word_count, pe=pe)
        self.commands.append(command)
        return command

    def reduce_east(
        self,
        *,
        src_pe: Coord,
        dst_pe: Coord | None = None,
        color: str | None = "partial_sum_color",
    ) -> VectorDotCommand:
        command = reduce_east(src_pe=src_pe, dst_pe=dst_pe, color=color)
        self.commands.append(command)
        return command

    def expect_dot(self, expected: int, *, pe: Coord = (0, 0)) -> VectorDotCommand:
        command = expect_dot(expected, pe=pe)
        self.commands.append(command)
        return command
