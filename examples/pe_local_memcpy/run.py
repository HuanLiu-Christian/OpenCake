from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wafersim.hardware.local_memcpy import run_memcpy_commands
from wafersim.mapping import MemcpyProgram, write_memcpy_commands


def build_pe_local_memcpy_program() -> MemcpyProgram:
    """Build a PE-local memcpy program; each call emits MemcpyCommand IR."""

    program = MemcpyProgram()
    program.h2d_write(4, [11, 22, 33, 44])
    program.local_copy(4, 20, 4)
    program.d2h_read(20, [11, 22, 33, 44])
    program.expect_word(20, 11)
    program.expect_word(21, 22)
    program.expect_word(22, 33)
    program.expect_word(23, 44)
    program.h2d_write(12, [101, 202, 303, 404])
    program.d2h_read(12, [101, 202, 303, 404])
    return program


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ir-out",
        type=Path,
        default=None,
        help="Optional path for the generated MemcpyCommand JSONL IR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    program = build_pe_local_memcpy_program()
    commands = program.commands
    if args.ir_out is not None:
        write_memcpy_commands(args.ir_out, commands)
    result = run_memcpy_commands(commands)
    print(
        json.dumps(
            {
                "command_ir": (
                    str(args.ir_out)
                    if args.ir_out is not None
                    else "generated in memory"
                ),
                "commands": len(commands),
                "sim_time_cycles": result.sim_time_cycles,
                "memory_tasks_completed": result.memory_tasks_completed,
                "d2h_outputs": result.d2h_outputs,
                "sram_20_24": result.final_sram[20:24],
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
