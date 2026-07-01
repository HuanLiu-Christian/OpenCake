from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from wafersim.mapping.gemm_ir import GemmCommand


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


def gemm_command_to_record(command: GemmCommand) -> dict[str, object]:
    command.validate()
    record: dict[str, object] = {"op": command.op}
    if command.op in {"load_a_panel", "load_b_panel", "compute_c", "expect_c"}:
        record["pe"] = _coord_to_record(command.pe)
    if command.op in {"forward_a_east", "forward_b_south"}:
        record["src_pe"] = _coord_to_record(command.src_pe)
        if command.dst_pe is not None:
            record["dst_pe"] = _coord_to_record(command.dst_pe)
        if command.color is not None:
            record["color"] = command.color
    if command.rows:
        record["rows"] = command.rows
    if command.cols:
        record["cols"] = command.cols
    if command.k:
        record["k"] = command.k
    if command.values:
        record["values"] = list(command.values)
    if command.expected:
        record["expected"] = list(command.expected)
    return record


def gemm_command_from_record(record: dict[str, object]) -> GemmCommand:
    values = record.get("values", [])
    expected = record.get("expected", [])
    if not isinstance(values, list):
        raise ValueError("values must be a JSON array")
    if not isinstance(expected, list):
        raise ValueError("expected must be a JSON array")
    command = GemmCommand(
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
        k=int(record.get("k", 0)),
        values=tuple(int(value) for value in values),
        expected=tuple(int(value) for value in expected),
    )
    command.validate()
    return command


def read_gemm_commands(path: str | Path) -> list[GemmCommand]:
    commands: list[GemmCommand] = []
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("command record must be a JSON object")
            commands.append(gemm_command_from_record(record))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return commands


def write_gemm_commands(path: str | Path, commands: Iterable[GemmCommand]) -> None:
    lines = [
        json.dumps(gemm_command_to_record(command), sort_keys=True)
        for command in commands
    ]
    Path(path).write_text("\n".join(lines) + "\n")
