from __future__ import annotations

from dataclasses import dataclass

import simpy

from wafersim.config import Coord
from wafersim.csl import CslCompatibilityRuntime
from wafersim.hardware.network import WaferMesh
from wafersim.program import PeProgramSpec, PeTaskSpec


PE_PROGRAM = PeProgramSpec(
    kernel="gemv_09_streaming",
    tasks=(
        PeTaskSpec(
            name="recv_x",
            description="Receive an x chunk from host or the PE above.",
        ),
        PeTaskSpec(
            name="compute_local_ax",
            description="Compute the local A tile times the local x chunk.",
        ),
        PeTaskSpec(
            name="recv_b",
            description="Receive the row bias chunk on the west edge.",
        ),
        PeTaskSpec(
            name="reduce_ax_east",
            description="Accumulate partial y chunks from west to east.",
        ),
        PeTaskSpec(
            name="send_y_d2h",
            description="Return the final y chunk from the east edge to host.",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class GemvStreamingConfig:
    matrix_rows: int = 6
    matrix_cols: int = 8
    kernel_rows: int = 3
    kernel_cols: int = 4
    bytes_per_element: int = 4
    packet_payload_bytes: int = 64
    ops_per_mac: int = 2

    def validate(self) -> None:
        if self.matrix_rows <= 0 or self.matrix_cols <= 0:
            raise ValueError("GEMV rows and cols must be positive")
        if self.kernel_rows <= 0 or self.kernel_cols <= 0:
            raise ValueError("GEMV kernel dimensions must be positive")
        if self.matrix_rows % self.kernel_rows != 0:
            raise ValueError("matrix_rows must be divisible by kernel_rows")
        if self.matrix_cols % self.kernel_cols != 0:
            raise ValueError("matrix_cols must be divisible by kernel_cols")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes per element must be positive")
        if self.packet_payload_bytes <= 0:
            raise ValueError("packet payload bytes must be positive")
        if self.ops_per_mac <= 0:
            raise ValueError("ops per MAC must be positive")

    @property
    def rows_per_pe(self) -> int:
        return self.matrix_rows // self.kernel_rows

    @property
    def cols_per_pe(self) -> int:
        return self.matrix_cols // self.kernel_cols

    @property
    def active_pes(self) -> int:
        return self.kernel_rows * self.kernel_cols

    @property
    def a_tile_bytes_per_pe(self) -> int:
        return self.rows_per_pe * self.cols_per_pe * self.bytes_per_element

    @property
    def x_chunk_bytes_per_col(self) -> int:
        return self.cols_per_pe * self.bytes_per_element

    @property
    def b_chunk_bytes_per_row(self) -> int:
        return self.rows_per_pe * self.bytes_per_element

    @property
    def y_chunk_bytes_per_row(self) -> int:
        return self.rows_per_pe * self.bytes_per_element

    @property
    def compute_ops(self) -> int:
        return self.matrix_rows * self.matrix_cols * self.ops_per_mac

    @property
    def compute_ops_per_pe(self) -> int:
        return self.rows_per_pe * self.cols_per_pe * self.ops_per_mac

    @property
    def matrix_bytes(self) -> int:
        return self.matrix_rows * self.matrix_cols * self.bytes_per_element

    @property
    def x_bytes(self) -> int:
        return self.matrix_cols * self.bytes_per_element

    @property
    def b_bytes(self) -> int:
        return self.matrix_rows * self.bytes_per_element

    @property
    def y_bytes(self) -> int:
        return self.matrix_rows * self.bytes_per_element

    @property
    def memory_footprint_bytes_per_pe(self) -> int:
        return self.a_tile_bytes_per_pe + self.x_chunk_bytes_per_col + (
            2 * self.y_chunk_bytes_per_row
        )


class GemvStreamingPeProgram:
    """Executable GEMV9 PE-local program for this example.

    One instance is launched on each active PE. It owns local A/y state and
    SimPy queues; incoming x, b, or partial-y data wakes the corresponding
    PE-local task.
    """

    def __init__(
        self,
        *,
        env: simpy.Environment,
        mesh: WaferMesh,
        runtime: CslCompatibilityRuntime,
        config: GemvStreamingConfig,
        coord: Coord,
        matrix: list[float],
    ) -> None:
        self.env = env
        self.mesh = mesh
        self.runtime = runtime
        self.config = config
        self.coord = coord
        self.row, self.col = coord
        self.a_tile = self._local_a_tile(matrix)
        self.y_buffer = [0.0 for _ in range(config.rows_per_pe)]
        self.x_queue = simpy.Store(env, capacity=1)
        self.b_queue = simpy.Store(env, capacity=1)
        self.partial_queue = simpy.Store(env, capacity=1)
        self.x_done = env.event()
        self.b_done = env.event()
        self.done = env.event()
        self.pe_programs: dict[Coord, GemvStreamingPeProgram] = {}
        self.tasks_started = 0
        self.local_fmacs = 0

    def bind_programs(
        self,
        pe_programs: dict[Coord, "GemvStreamingPeProgram"],
    ) -> None:
        self.pe_programs = pe_programs

    def launch(self):
        self.tasks_started = 3
        return [
            self.env.process(self.recv_x_task()),
            self.env.process(self.recv_b_task()),
            self.env.process(self.reduce_task()),
        ]

    def recv_x_task(self):
        x_values = yield self.x_queue.get()
        events = [
            self.env.process(
                self.mesh.pes[self.coord].compute(self.config.compute_ops_per_pe)
            )
        ]
        if self.row < self.config.kernel_rows - 1:
            south = (self.row + 1, self.col)
            events.append(
                self.env.process(
                    self.runtime.colors.send(
                        "x_color",
                        self.coord,
                        south,
                        self.config.x_chunk_bytes_per_col,
                    )
                )
            )
        yield simpy.events.AllOf(self.env, events)
        self._accumulate_local_ax(x_values)
        if self.row < self.config.kernel_rows - 1:
            yield self.pe_programs[(self.row + 1, self.col)].x_queue.put(
                list(x_values)
            )
        self.x_done.succeed()

    def recv_b_task(self):
        if self.col != 0:
            self.b_done.succeed()
            return
        b_values = yield self.b_queue.get()
        for index, value in enumerate(b_values):
            self.y_buffer[index] += value
        self.b_done.succeed()

    def reduce_task(self):
        yield self.x_done
        if self.col == 0:
            yield self.b_done
        else:
            partial = yield self.partial_queue.get()
            for index, value in enumerate(partial):
                self.y_buffer[index] += value

        if self.col < self.config.kernel_cols - 1:
            east = (self.row, self.col + 1)
            color = "ax_color_1" if self.col % 2 == 0 else "ax_color_2"
            yield from self.runtime.colors.send(
                color,
                self.coord,
                east,
                self.config.y_chunk_bytes_per_row,
            )
            yield self.pe_programs[east].partial_queue.put(list(self.y_buffer))
        else:
            yield from self.runtime.colors.send(
                "MEMCPYD2H_DATA_1",
                self.coord,
                None,
                self.config.y_chunk_bytes_per_row,
            )
            self.done.succeed(list(self.y_buffer))

    def _local_a_tile(self, matrix: list[float]) -> list[list[float]]:
        row0 = self.row * self.config.rows_per_pe
        col0 = self.col * self.config.cols_per_pe
        tile = []
        for local_row in range(self.config.rows_per_pe):
            global_row = row0 + local_row
            start = global_row * self.config.matrix_cols + col0
            tile.append(matrix[start : start + self.config.cols_per_pe])
        return tile

    def _accumulate_local_ax(self, x_values: list[float]) -> None:
        for row_index, a_row in enumerate(self.a_tile):
            self.y_buffer[row_index] += sum(
                a_value * x_value
                for a_value, x_value in zip(a_row, x_values, strict=True)
            )
            self.local_fmacs += len(a_row)
