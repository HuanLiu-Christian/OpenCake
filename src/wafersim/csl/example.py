from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wafersim.csl.compat import Direction


_CONST_RE = re.compile(r"^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*;\s*$")
_RECT_RE = re.compile(r"@set_rectangle\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
_WAFERSIM_RE = re.compile(r"^\s*//\s*wafersim:\s*(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class CslExampleProgram:
    """Parsed CSL-style example metadata used by the SimPy adapter.

    This is intentionally a small structural parser, not a CSL compiler. The
    source files stay close to the Cerebras SDK example shape while `run.py`
    converts the declarations into the simulator's Python configuration.
    """

    layout_path: Path
    pe_program_path: Path
    constants: dict[str, int | float | str | bool] = field(default_factory=dict)
    colors: dict[str, Direction | None] = field(default_factory=dict)
    kernel: str = ""
    tasks: list[str] = field(default_factory=list)
    rectangle: tuple[str, str] | None = None

    def int_constant(self, name: str) -> int:
        value = self.constants[name]
        if not isinstance(value, int):
            raise TypeError(f"CSL constant is not an int: {name}")
        return value


def load_csl_example(example_dir: Path) -> CslExampleProgram:
    layout_path = example_dir / "layout.csl"
    pe_program_path = example_dir / "pe_program.csl"
    if not layout_path.exists():
        raise FileNotFoundError(f"missing CSL layout file: {layout_path}")
    if not pe_program_path.exists():
        raise FileNotFoundError(f"missing CSL PE program file: {pe_program_path}")

    constants: dict[str, int | float | str | bool] = {}
    colors: dict[str, Direction | None] = {}
    tasks: list[str] = []
    kernel = ""
    rectangle: tuple[str, str] | None = None

    for path in (layout_path, pe_program_path):
        for line in path.read_text(encoding="utf-8").splitlines():
            const_match = _CONST_RE.match(line)
            if const_match:
                name, value = const_match.groups()
                constants[name] = _parse_value(value)
                continue

            rect_match = _RECT_RE.search(line)
            if rect_match:
                rectangle = (rect_match.group(1), rect_match.group(2))

            directive_match = _WAFERSIM_RE.match(line)
            if not directive_match:
                continue
            _apply_directive(
                directive_match.group(1),
                colors=colors,
                tasks=tasks,
                set_kernel=lambda value: _set_kernel(value),
            )
            if _apply_directive.last_kernel is not None:
                kernel = _apply_directive.last_kernel
                _apply_directive.last_kernel = None

    return CslExampleProgram(
        layout_path=layout_path,
        pe_program_path=pe_program_path,
        constants=constants,
        colors=colors,
        kernel=kernel,
        tasks=tasks,
        rectangle=rectangle,
    )


def _parse_value(value: str) -> int | float | str | bool:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _apply_directive(
    directive: str,
    *,
    colors: dict[str, Direction | None],
    tasks: list[str],
    set_kernel,
) -> None:
    parts = directive.split()
    if not parts:
        return
    if parts[0] == "kernel" and len(parts) == 2:
        set_kernel(parts[1])
        return
    if parts[0] == "route" and len(parts) == 3:
        colors[parts[1]] = None if parts[2] == "ramp" else Direction(parts[2])
        return
    if parts[0] == "task" and len(parts) == 2:
        tasks.append(parts[1])


def _set_kernel(value: str) -> None:
    _apply_directive.last_kernel = value


_apply_directive.last_kernel = None  # type: ignore[attr-defined]
