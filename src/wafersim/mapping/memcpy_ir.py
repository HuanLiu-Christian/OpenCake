from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

Coord = tuple[int, int]

MemcpyOp = Literal["h2d_write", "local_copy", "d2h_read", "expect", "p2p_copy"]


@dataclass(frozen=True, slots=True)
class MemcpyCommand:
    """Portable PE-local memcpy command used by SimPy and RTL backends."""

    op: MemcpyOp
    src_addr: int = 0
    dst_addr: int = 0
    words: tuple[int, ...] = ()
    word_count: int = 0
    pe: Coord = (0, 0)
    src_pe: Coord = (0, 0)
    dst_pe: Coord = (0, 0)

    @property
    def effective_word_count(self) -> int:
        if self.word_count:
            return self.word_count
        return len(self.words)

    def validate(self) -> None:
        if self.op not in {
            "h2d_write",
            "local_copy",
            "d2h_read",
            "expect",
            "p2p_copy",
        }:
            raise ValueError(f"unknown memcpy op: {self.op}")
        for coord in (self.pe, self.src_pe, self.dst_pe):
            if len(coord) != 2 or coord[0] < 0 or coord[1] < 0:
                raise ValueError("PE coordinates must be non-negative row/col pairs")
        if self.src_addr < 0 or self.dst_addr < 0:
            raise ValueError("addresses must be non-negative")
        if self.word_count < 0:
            raise ValueError("word_count must be non-negative")
        if self.op == "h2d_write" and not self.words:
            raise ValueError("h2d_write needs at least one word")
        if self.op == "local_copy" and self.word_count <= 0:
            raise ValueError("local_copy needs a positive word_count")
        if self.op == "p2p_copy":
            if self.word_count <= 0:
                raise ValueError("p2p_copy needs a positive word_count")
            if self.src_pe == self.dst_pe:
                raise ValueError("p2p_copy needs distinct source and destination PEs")
        if self.op == "d2h_read" and self.effective_word_count <= 0:
            raise ValueError("d2h_read needs expected words or a word_count")
        if self.op == "expect" and len(self.words) != 1:
            raise ValueError("expect needs exactly one expected word")


def h2d_write(
    dst_addr: int,
    words: Iterable[int],
    *,
    pe: Coord = (0, 0),
) -> MemcpyCommand:
    command = MemcpyCommand(
        op="h2d_write",
        dst_addr=dst_addr,
        words=tuple(words),
        pe=pe,
    )
    command.validate()
    return command


def local_copy(
    src_addr: int,
    dst_addr: int,
    word_count: int,
    *,
    pe: Coord = (0, 0),
) -> MemcpyCommand:
    command = MemcpyCommand(
        op="local_copy",
        src_addr=src_addr,
        dst_addr=dst_addr,
        word_count=word_count,
        pe=pe,
    )
    command.validate()
    return command


def p2p_copy(
    src_pe: Coord,
    dst_pe: Coord,
    src_addr: int,
    dst_addr: int,
    word_count: int,
) -> MemcpyCommand:
    command = MemcpyCommand(
        op="p2p_copy",
        src_addr=src_addr,
        dst_addr=dst_addr,
        word_count=word_count,
        src_pe=src_pe,
        dst_pe=dst_pe,
    )
    command.validate()
    return command


def d2h_read(
    src_addr: int,
    expected_words: Iterable[int] = (),
    *,
    word_count: int = 0,
    pe: Coord = (0, 0),
) -> MemcpyCommand:
    command = MemcpyCommand(
        op="d2h_read",
        src_addr=src_addr,
        words=tuple(expected_words),
        word_count=word_count,
        pe=pe,
    )
    command.validate()
    return command


def expect_word(addr: int, expected: int, *, pe: Coord = (0, 0)) -> MemcpyCommand:
    command = MemcpyCommand(
        op="expect",
        src_addr=addr,
        words=(expected,),
        pe=pe,
    )
    command.validate()
    return command


@dataclass(slots=True)
class MemcpyProgram:
    """User-facing PE-local memcpy program that emits mapping IR automatically."""

    commands: list[MemcpyCommand] = field(default_factory=list)

    def h2d_write(
        self,
        dst_addr: int,
        words: Iterable[int],
        *,
        pe: Coord = (0, 0),
    ) -> MemcpyCommand:
        command = h2d_write(dst_addr, words, pe=pe)
        self.commands.append(command)
        return command

    def local_copy(
        self,
        src_addr: int,
        dst_addr: int,
        word_count: int,
        *,
        pe: Coord = (0, 0),
    ) -> MemcpyCommand:
        command = local_copy(src_addr, dst_addr, word_count, pe=pe)
        self.commands.append(command)
        return command

    def p2p_copy(
        self,
        src_pe: Coord,
        dst_pe: Coord,
        src_addr: int,
        dst_addr: int,
        word_count: int,
    ) -> MemcpyCommand:
        command = p2p_copy(src_pe, dst_pe, src_addr, dst_addr, word_count)
        self.commands.append(command)
        return command

    def d2h_read(
        self,
        src_addr: int,
        expected_words: Iterable[int] = (),
        *,
        word_count: int = 0,
        pe: Coord = (0, 0),
    ) -> MemcpyCommand:
        command = d2h_read(
            src_addr,
            expected_words,
            word_count=word_count,
            pe=pe,
        )
        self.commands.append(command)
        return command

    def expect_word(
        self,
        addr: int,
        expected: int,
        *,
        pe: Coord = (0, 0),
    ) -> MemcpyCommand:
        command = expect_word(addr, expected, pe=pe)
        self.commands.append(command)
        return command
