from __future__ import annotations

from dataclasses import dataclass, field

import simpy

from wafersim.config import Coord, MeshConfig
from wafersim.hardware.pe import ProcessingElement
from wafersim.mapping.gemm_ir import GemmCommand


@dataclass(slots=True)
class GemmPeState:
    a_panel: list[list[int]] = field(default_factory=list)
    b_panel: list[list[int]] = field(default_factory=list)
    c_tile: list[list[int]] = field(default_factory=list)
    rows: int = 0
    cols: int = 0
    k: int = 0


@dataclass(frozen=True, slots=True)
class GemmResult:
    c_tiles: dict[Coord, tuple[int, ...]]
    result_c: tuple[int, ...]
    expected_c: tuple[int, ...]
    sim_time_cycles: float
    compute_tasks_completed: int
    memory_tasks_completed: int


class GemmPe:
    def __init__(
        self,
        env: simpy.Environment,
        *,
        coord: Coord,
        word_bytes: int = 4,
        ops_per_mac: int = 2,
        config: MeshConfig | None = None,
    ) -> None:
        self.env = env
        self.coord = coord
        self.word_bytes = word_bytes
        self.ops_per_mac = ops_per_mac
        self.state = GemmPeState()
        mesh_config = config or MeshConfig(rows=1, cols=1)
        self.pe = ProcessingElement(
            env=env,
            coord=coord,
            compute_throughput_ops_per_cycle=mesh_config.compute_throughput_ops_per_cycle,
            local_sram_bytes=64 * 1024,
            memory_bandwidth_bytes_per_cycle=mesh_config.memory_bandwidth_bytes_per_cycle,
        )

    def load_a_panel(self, values: tuple[int, ...], rows: int, k: int):
        yield from self.pe.access_memory(len(values) * self.word_bytes)
        self.state.rows = rows
        self.state.k = k
        self.state.a_panel = [
            list(values[row * k : (row + 1) * k])
            for row in range(rows)
        ]

    def load_b_panel(self, values: tuple[int, ...], cols: int, k: int):
        yield from self.pe.access_memory(len(values) * self.word_bytes)
        self.state.cols = cols
        self.state.k = k
        self.state.b_panel = [
            list(values[kk * cols : (kk + 1) * cols])
            for kk in range(k)
        ]

    def compute_c(self, rows: int, cols: int, k: int):
        if len(self.state.a_panel) != rows or len(self.state.b_panel) != k:
            raise ValueError(
                f"PE{self.coord} cannot compute {rows}x{cols}x{k}: "
                f"a_rows={len(self.state.a_panel)}, b_rows={len(self.state.b_panel)}"
            )
        if any(len(row) != k for row in self.state.a_panel):
            raise ValueError(f"PE{self.coord} has an invalid A panel")
        if any(len(row) != cols for row in self.state.b_panel):
            raise ValueError(f"PE{self.coord} has an invalid B panel")
        yield from self.pe.compute(rows * cols * k * self.ops_per_mac)
        self.state.c_tile = []
        for row in range(rows):
            c_row: list[int] = []
            for col in range(cols):
                c_row.append(
                    sum(
                        self.state.a_panel[row][kk] * self.state.b_panel[kk][col]
                        for kk in range(k)
                    )
                )
            self.state.c_tile.append(c_row)


def _move_by_color(command: GemmCommand) -> Coord:
    row, col = command.src_pe
    if command.op == "forward_a_east" and command.color == "a_color":
        return (row, col + 1)
    if command.op == "forward_b_south" and command.color == "b_color":
        return (row + 1, col)
    raise ValueError(f"unsupported GEMM color route: {command.color}")


def run_gemm_commands(
    commands: list[GemmCommand],
    *,
    word_bytes: int = 4,
    ops_per_mac: int = 2,
    kernel_cols: int | None = None,
) -> GemmResult:
    env = simpy.Environment()
    models: dict[Coord, GemmPe] = {}
    actual_by_coord: dict[Coord, tuple[int, ...]] = {}
    expected_by_coord: dict[Coord, tuple[int, ...]] = {}

    def _model(coord: Coord) -> GemmPe:
        if coord not in models:
            models[coord] = GemmPe(
                env,
                coord=coord,
                word_bytes=word_bytes,
                ops_per_mac=ops_per_mac,
            )
        return models[coord]

    def _dst(command: GemmCommand) -> Coord:
        if command.dst_pe is not None:
            return command.dst_pe
        return _move_by_color(command)

    def _run():
        for command in commands:
            if command.op == "load_a_panel":
                yield env.process(
                    _model(command.pe).load_a_panel(command.values, command.rows, command.k)
                )
            elif command.op == "load_b_panel":
                yield env.process(
                    _model(command.pe).load_b_panel(command.values, command.cols, command.k)
                )
            elif command.op == "forward_a_east":
                src = _model(command.src_pe)
                dst = _model(_dst(command))
                flat = tuple(value for row in src.state.a_panel for value in row)
                yield from src.pe.access_memory(len(flat) * word_bytes)
                yield from dst.pe.access_memory(len(flat) * word_bytes)
                dst.state.rows = src.state.rows
                dst.state.k = src.state.k
                dst.state.a_panel = [list(row) for row in src.state.a_panel]
            elif command.op == "forward_b_south":
                src = _model(command.src_pe)
                dst = _model(_dst(command))
                flat = tuple(value for row in src.state.b_panel for value in row)
                yield from src.pe.access_memory(len(flat) * word_bytes)
                yield from dst.pe.access_memory(len(flat) * word_bytes)
                dst.state.cols = src.state.cols
                dst.state.k = src.state.k
                dst.state.b_panel = [list(row) for row in src.state.b_panel]
            elif command.op == "compute_c":
                yield env.process(
                    _model(command.pe).compute_c(command.rows, command.cols, command.k)
                )
            elif command.op == "expect_c":
                actual = tuple(
                    value
                    for row in _model(command.pe).state.c_tile
                    for value in row
                )
                if actual != command.expected:
                    raise AssertionError(
                        f"PE{command.pe} C expected {command.expected}, got {actual}"
                    )
                actual_by_coord[command.pe] = actual
                expected_by_coord[command.pe] = command.expected
            else:
                raise ValueError(f"unknown GEMM op: {command.op}")

    env.run(until=env.process(_run()))
    if kernel_cols is None:
        kernel_cols = max((coord[1] for coord in actual_by_coord), default=0) + 1
    ordered_coords = sorted(actual_by_coord, key=lambda coord: coord[0] * kernel_cols + coord[1])
    result_c = tuple(value for coord in ordered_coords for value in actual_by_coord[coord])
    expected_c = tuple(value for coord in ordered_coords for value in expected_by_coord[coord])
    return GemmResult(
        c_tiles={
            coord: tuple(value for row in model.state.c_tile for value in row)
            for coord, model in models.items()
        },
        result_c=result_c,
        expected_c=expected_c,
        sim_time_cycles=env.now,
        compute_tasks_completed=sum(
            model.pe.compute_tasks_completed for model in models.values()
        ),
        memory_tasks_completed=sum(
            model.pe.memory_tasks_completed for model in models.values()
        ),
    )
