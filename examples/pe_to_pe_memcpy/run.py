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
from wafersim.program import Direction


COLOR_ROUTES = {
    "east_memcpy": Direction.EAST,
}


def build_pe_to_pe_memcpy_program() -> MemcpyProgram:
    """Build a two-PE memcpy program; each call emits MemcpyCommand IR."""

    program = MemcpyProgram()
    program.h2d_write(4, [7, 8, 9, 10], pe=(0, 0))
    program.p2p_copy(
        src_pe=(0, 0),
        src_addr=4,
        dst_addr=12,
        word_count=4,
        color="east_memcpy",
    )
    program.d2h_read(12, [7, 8, 9, 10], pe=(0, 1))
    program.expect_word(12, 7, pe=(0, 1))
    program.expect_word(13, 8, pe=(0, 1))
    program.expect_word(14, 9, pe=(0, 1))
    program.expect_word(15, 10, pe=(0, 1))
    return program


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ir-out",
        type=Path,
        default=None,
        help="Optional path for the generated PE-to-PE MemcpyCommand JSONL IR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    program = build_pe_to_pe_memcpy_program()
    commands = program.commands
    if args.ir_out is not None:
        write_memcpy_commands(args.ir_out, commands)
    result = run_memcpy_commands(commands, color_routes=COLOR_ROUTES)
    print(
        json.dumps(
            {
                "command_ir": (
                    str(args.ir_out)
                    if args.ir_out is not None
                    else "generated in memory"
                ),
                "color_routes": {
                    color: direction.value for color, direction in COLOR_ROUTES.items()
                },
                "commands": len(commands),
                "d2h_outputs": result.d2h_outputs,
                "memory_tasks_completed": result.memory_tasks_completed,
                "pe_0_0_sram_4_8": result.final_srams[(0, 0)][4:8],
                "pe_0_1_sram_12_16": result.final_srams[(0, 1)][12:16],
                "sim_time_cycles": result.sim_time_cycles,
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
