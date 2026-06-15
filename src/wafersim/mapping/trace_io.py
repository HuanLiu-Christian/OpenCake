from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from wafersim.mapping.memcpy_ir import MemcpyCommand


def _coord_to_record(coord: tuple[int, int]) -> list[int]:
    return [coord[0], coord[1]]


def _coord_from_record(value: object, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return (int(value[0]), int(value[1]))
    raise ValueError("PE coordinate must be a JSON array [row, col]")


def command_to_record(command: MemcpyCommand) -> dict[str, object]:
    command.validate()
    record: dict[str, object] = {"op": command.op}
    if command.src_addr:
        record["src_addr"] = command.src_addr
    if command.dst_addr:
        record["dst_addr"] = command.dst_addr
    if command.words:
        record["words"] = list(command.words)
    if command.word_count:
        record["word_count"] = command.word_count
    if command.pe != (0, 0):
        record["pe"] = _coord_to_record(command.pe)
    if command.op == "p2p_copy":
        record["src_pe"] = _coord_to_record(command.src_pe)
        record["dst_pe"] = _coord_to_record(command.dst_pe)
    return record


def command_from_record(record: dict[str, object]) -> MemcpyCommand:
    words_value = record.get("words", [])
    if not isinstance(words_value, list):
        raise ValueError("words must be a JSON array")

    command = MemcpyCommand(
        op=str(record["op"]),  # type: ignore[arg-type]
        src_addr=int(record.get("src_addr", 0)),
        dst_addr=int(record.get("dst_addr", 0)),
        words=tuple(int(word) for word in words_value),
        word_count=int(record.get("word_count", 0)),
        pe=_coord_from_record(record.get("pe"), (0, 0)),
        src_pe=_coord_from_record(record.get("src_pe"), (0, 0)),
        dst_pe=_coord_from_record(record.get("dst_pe"), (0, 0)),
    )
    command.validate()
    return command


def read_memcpy_commands(path: str | Path) -> list[MemcpyCommand]:
    commands: list[MemcpyCommand] = []
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("command record must be a JSON object")
            commands.append(command_from_record(record))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return commands


def write_memcpy_commands(
    path: str | Path,
    commands: Iterable[MemcpyCommand],
) -> None:
    lines = [
        json.dumps(command_to_record(command), sort_keys=True)
        for command in commands
    ]
    Path(path).write_text("\n".join(lines) + "\n")
