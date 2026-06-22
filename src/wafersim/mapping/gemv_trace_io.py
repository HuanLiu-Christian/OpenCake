from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from wafersim.mapping.gemv_ir import GemvCommand


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


def gemv_command_to_record(command: GemvCommand) -> dict[str, object]:
    command.validate()
    record: dict[str, object] = {"op": command.op}
    if command.op in {"load_a_tile", "load_x_chunk", "load_bias", "compute_ax", "expect_y"}:
        record["pe"] = _coord_to_record(command.pe)
    if command.op in {"forward_x_south", "reduce_east"}:
        record["src_pe"] = _coord_to_record(command.src_pe)
        if command.dst_pe is not None:
            record["dst_pe"] = _coord_to_record(command.dst_pe)
        if command.color is not None:
            record["color"] = command.color
    if command.rows:
        record["rows"] = command.rows
    if command.cols:
        record["cols"] = command.cols
    if command.values:
        record["values"] = list(command.values)
    if command.expected:
        record["expected"] = list(command.expected)
    return record


def gemv_command_from_record(record: dict[str, object]) -> GemvCommand:
    values = record.get("values", [])
    expected = record.get("expected", [])
    if not isinstance(values, list):
        raise ValueError("values must be a JSON array")
    if not isinstance(expected, list):
        raise ValueError("expected must be a JSON array")
    command = GemvCommand(
        op=str(record["op"]),  # type: ignore[arg-type]
        pe=_coord_from_record(record.get("pe"), (0, 0)),
        src_pe=_coord_from_record(record.get("src_pe"), (0, 0)),
        dst_pe=(
            _coord_from_record(record.get("dst_pe"), (0, 0))
            if "dst_pe" in record
            else None
        ),
        color=(str(record["color"]) if "color" in record else None),
        rows=int(record.get("rows", 0)),
        cols=int(record.get("cols", 0)),
        values=tuple(int(value) for value in values),
        expected=tuple(int(value) for value in expected),
    )
    command.validate()
    return command


def read_gemv_commands(path: str | Path) -> list[GemvCommand]:
    commands: list[GemvCommand] = []
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("command record must be a JSON object")
            commands.append(gemv_command_from_record(record))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return commands


def write_gemv_commands(path: str | Path, commands: Iterable[GemvCommand]) -> None:
    lines = [
        json.dumps(gemv_command_to_record(command), sort_keys=True)
        for command in commands
    ]
    Path(path).write_text("\n".join(lines) + "\n")
