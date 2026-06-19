from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from wafersim.mapping.vector_dot_ir import VectorDotCommand


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


def vector_dot_command_to_record(command: VectorDotCommand) -> dict[str, object]:
    command.validate()
    record: dict[str, object] = {"op": command.op}
    if command.words:
        record["words"] = list(command.words)
    if command.word_count:
        record["word_count"] = command.word_count
    if command.expected is not None:
        record["expected"] = command.expected
    if command.op in {"load_left", "load_right", "compute_dot", "expect"}:
        record["pe"] = _coord_to_record(command.pe)
    if command.op == "reduce_east":
        record["src_pe"] = _coord_to_record(command.src_pe)
        if command.dst_pe is not None:
            record["dst_pe"] = _coord_to_record(command.dst_pe)
        if command.color is not None:
            record["color"] = command.color
    return record


def vector_dot_command_from_record(record: dict[str, object]) -> VectorDotCommand:
    words_value = record.get("words", [])
    if not isinstance(words_value, list):
        raise ValueError("words must be a JSON array")

    command = VectorDotCommand(
        op=str(record["op"]),  # type: ignore[arg-type]
        words=tuple(int(word) for word in words_value),
        word_count=int(record.get("word_count", 0)),
        expected=(
            int(record["expected"])
            if "expected" in record and record["expected"] is not None
            else None
        ),
        pe=_coord_from_record(record.get("pe"), (0, 0)),
        src_pe=_coord_from_record(record.get("src_pe"), (0, 0)),
        dst_pe=(
            _coord_from_record(record.get("dst_pe"), (0, 0))
            if "dst_pe" in record
            else None
        ),
        color=(str(record["color"]) if "color" in record else None),
    )
    command.validate()
    return command


def read_vector_dot_commands(path: str | Path) -> list[VectorDotCommand]:
    commands: list[VectorDotCommand] = []
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("command record must be a JSON object")
            commands.append(vector_dot_command_from_record(record))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return commands


def write_vector_dot_commands(
    path: str | Path,
    commands: Iterable[VectorDotCommand],
) -> None:
    lines = [
        json.dumps(vector_dot_command_to_record(command), sort_keys=True)
        for command in commands
    ]
    Path(path).write_text("\n".join(lines) + "\n")
