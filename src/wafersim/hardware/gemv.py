from __future__ import annotations

from dataclasses import dataclass, field

import simpy

from wafersim.config import Coord, MeshConfig
from wafersim.hardware.pe import ProcessingElement
from wafersim.mapping.gemv_ir import GemvCommand


@dataclass(slots=True)
class GemvPeState:
    a_tile: list[list[int]] = field(default_factory=list)
    x_chunk: list[int] = field(default_factory=list)
    y_chunk: list[int] = field(default_factory=list)
    rows: int = 0
    cols: int = 0


@dataclass(frozen=True, slots=True)
class GemvResult:
    y_chunks: dict[Coord, tuple[int, ...]]
    result_y: tuple[int, ...]
    expected_y: tuple[int, ...]
    sim_time_cycles: float
    compute_tasks_completed: int
    memory_tasks_completed: int


class GemvPe:
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
        self.state = GemvPeState()
        mesh_config = config or MeshConfig(rows=1, cols=1)
        self.pe = ProcessingElement(
            env=env,
            coord=coord,
            compute_throughput_ops_per_cycle=mesh_config.compute_throughput_ops_per_cycle,
            local_sram_bytes=64 * 1024,
            memory_bandwidth_bytes_per_cycle=mesh_config.memory_bandwidth_bytes_per_cycle,
        )

    def load_a_tile(self, values: tuple[int, ...], rows: int, cols: int):
        yield from self.pe.access_memory(len(values) * self.word_bytes)
        self.state.rows = rows
        self.state.cols = cols
        self.state.a_tile = [
            list(values[row * cols : (row + 1) * cols])
            for row in range(rows)
        ]
        if not self.state.y_chunk:
            self.state.y_chunk = [0 for _ in range(rows)]

    def load_x_chunk(self, values: tuple[int, ...]):
        yield from self.pe.access_memory(len(values) * self.word_bytes)
        self.state.x_chunk = list(values)

    def load_bias(self, values: tuple[int, ...]):
        yield from self.pe.access_memory(len(values) * self.word_bytes)
        self.state.y_chunk = list(values)

    def compute_ax(self, rows: int, cols: int):
        if len(self.state.a_tile) != rows or len(self.state.x_chunk) != cols:
            raise ValueError(
                f"PE{self.coord} cannot compute {rows}x{cols}: "
                f"a_rows={len(self.state.a_tile)}, x_cols={len(self.state.x_chunk)}"
            )
        if not self.state.y_chunk:
            self.state.y_chunk = [0 for _ in range(rows)]
        yield from self.pe.compute(rows * cols * self.ops_per_mac)
        for row in range(rows):
            self.state.y_chunk[row] += sum(
                a_value * x_value
                for a_value, x_value in zip(
                    self.state.a_tile[row],
                    self.state.x_chunk,
                    strict=True,
                )
            )


def _move_by_color(command: GemvCommand) -> Coord:
    row, col = command.src_pe
    if command.op == "forward_x_south" and command.color == "x_color":
        return (row + 1, col)
    if command.op == "reduce_east" and command.color in {"ax_color_1", "ax_color_2"}:
        return (row, col + 1)
    raise ValueError(f"unsupported GEMV color route: {command.color}")


def run_gemv_commands(
    commands: list[GemvCommand],
    *,
    word_bytes: int = 4,
    ops_per_mac: int = 2,
) -> GemvResult:
    env = simpy.Environment()
    models: dict[Coord, GemvPe] = {}
    actual_by_row: dict[int, tuple[int, ...]] = {}
    expected_by_row: dict[int, tuple[int, ...]] = {}

    def _model(coord: Coord) -> GemvPe:
        if coord not in models:
            models[coord] = GemvPe(
                env,
                coord=coord,
                word_bytes=word_bytes,
                ops_per_mac=ops_per_mac,
            )
        return models[coord]

    def _dst(command: GemvCommand) -> Coord:
        if command.dst_pe is not None:
            return command.dst_pe
        return _move_by_color(command)

    def _run():
        for command in commands:
            if command.op == "load_a_tile":
                yield env.process(
                    _model(command.pe).load_a_tile(command.values, command.rows, command.cols)
                )
            elif command.op == "load_x_chunk":
                yield env.process(_model(command.pe).load_x_chunk(command.values))
            elif command.op == "load_bias":
                yield env.process(_model(command.pe).load_bias(command.values))
            elif command.op == "forward_x_south":
                src = _model(command.src_pe)
                dst = _model(_dst(command))
                yield from src.pe.access_memory(len(src.state.x_chunk) * word_bytes)
                yield from dst.pe.access_memory(len(src.state.x_chunk) * word_bytes)
                dst.state.x_chunk = list(src.state.x_chunk)
            elif command.op == "compute_ax":
                yield env.process(_model(command.pe).compute_ax(command.rows, command.cols))
            elif command.op == "reduce_east":
                src = _model(command.src_pe)
                dst = _model(_dst(command))
                if not src.state.y_chunk:
                    raise ValueError(f"PE{command.src_pe} has no partial y to reduce")
                yield from src.pe.access_memory(len(src.state.y_chunk) * word_bytes)
                yield from dst.pe.access_memory(len(src.state.y_chunk) * word_bytes)
                if not dst.state.y_chunk:
                    dst.state.y_chunk = [0 for _ in src.state.y_chunk]
                dst.state.y_chunk = [
                    left + right
                    for left, right in zip(dst.state.y_chunk, src.state.y_chunk, strict=True)
                ]
            elif command.op == "expect_y":
                actual = tuple(_model(command.pe).state.y_chunk)
                if actual != command.expected:
                    raise AssertionError(
                        f"PE{command.pe} y expected {command.expected}, got {actual}"
                    )
                actual_by_row[command.pe[0]] = actual
                expected_by_row[command.pe[0]] = command.expected
            else:
                raise ValueError(f"unknown GEMV op: {command.op}")

    env.run(until=env.process(_run()))
    result_y = tuple(
        value
        for row in sorted(actual_by_row)
        for value in actual_by_row[row]
    )
    expected_y = tuple(
        value
        for row in sorted(expected_by_row)
        for value in expected_by_row[row]
    )
    return GemvResult(
        y_chunks={coord: tuple(model.state.y_chunk) for coord, model in models.items()},
        result_y=result_y,
        expected_y=expected_y,
        sim_time_cycles=env.now,
        compute_tasks_completed=sum(
            model.pe.compute_tasks_completed for model in models.values()
        ),
        memory_tasks_completed=sum(
            model.pe.memory_tasks_completed for model in models.values()
        ),
    )
