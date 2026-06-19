from __future__ import annotations

from dataclasses import dataclass

import simpy

from wafersim.config import Coord
from wafersim.csl import CslCompatibilityRuntime
from wafersim.hardware.network import WaferMesh
from wafersim.program import PeProgramSpec, PeTaskSpec


PE_PROGRAM = PeProgramSpec(
    kernel="vector_vector_dot",
    tasks=(
        PeTaskSpec(
            name="recv_a_b",
            description="Receive local chunks of both input vectors.",
        ),
        PeTaskSpec(
            name="compute_local_dot",
            description="Compute local sum(a_chunk * b_chunk).",
        ),
        PeTaskSpec(
            name="reduce_partial_sum_east",
            description="Accumulate partial dot products from west to east.",
        ),
        PeTaskSpec(
            name="send_scalar_d2h",
            description="Return the final scalar dot product from the east edge.",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class VectorDotConfig:
    vector_length: int = 16
    kernel_cols: int = 4
    bytes_per_element: int = 4
    packet_payload_bytes: int = 64
    ops_per_mac: int = 2

    def validate(self) -> None:
        if self.vector_length <= 0:
            raise ValueError("vector length must be positive")
        if self.kernel_cols <= 0:
            raise ValueError("kernel cols must be positive")
        if self.vector_length % self.kernel_cols != 0:
            raise ValueError("vector_length must be divisible by kernel_cols")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes per element must be positive")
        if self.packet_payload_bytes <= 0:
            raise ValueError("packet payload bytes must be positive")
        if self.ops_per_mac <= 0:
            raise ValueError("ops per MAC must be positive")

    @property
    def elements_per_pe(self) -> int:
        return self.vector_length // self.kernel_cols

    @property
    def chunk_bytes_per_pe(self) -> int:
        return self.elements_per_pe * self.bytes_per_element

    @property
    def scalar_bytes(self) -> int:
        return self.bytes_per_element

    @property
    def memory_footprint_bytes_per_pe(self) -> int:
        return 2 * self.chunk_bytes_per_pe + self.scalar_bytes

    @property
    def compute_ops(self) -> int:
        return self.vector_length * self.ops_per_mac

    @property
    def compute_ops_per_pe(self) -> int:
        return self.elements_per_pe * self.ops_per_mac


class VectorDotPeProgram:
    """PE-local executable program for a vector dot-product row pipeline."""

    def __init__(
        self,
        *,
        env: simpy.Environment,
        mesh: WaferMesh,
        runtime: CslCompatibilityRuntime,
        config: VectorDotConfig,
        coord: Coord,
    ) -> None:
        self.env = env
        self.mesh = mesh
        self.runtime = runtime
        self.config = config
        self.coord = coord
        self.row, self.col = coord
        self.a_queue = simpy.Store(env, capacity=1)
        self.b_queue = simpy.Store(env, capacity=1)
        self.partial_queue = simpy.Store(env, capacity=1)
        self.done = env.event()
        self.pe_programs: dict[Coord, VectorDotPeProgram] = {}
        self.tasks_started = 0
        self.local_fmacs = 0
        self.local_partial = 0.0

    def bind_programs(
        self,
        pe_programs: dict[Coord, "VectorDotPeProgram"],
    ) -> None:
        self.pe_programs = pe_programs

    def launch(self):
        self.tasks_started = 1
        return [self.env.process(self.run_task())]

    def run_task(self):
        a_values = yield self.a_queue.get()
        b_values = yield self.b_queue.get()
        yield self.env.process(
            self.mesh.pes[self.coord].compute(self.config.compute_ops_per_pe)
        )
        self.local_partial = sum(
            a_value * b_value
            for a_value, b_value in zip(a_values, b_values, strict=True)
        )
        self.local_fmacs = len(a_values)

        running_sum = self.local_partial
        if self.col > 0:
            running_sum += yield self.partial_queue.get()

        if self.col < self.config.kernel_cols - 1:
            east = (self.row, self.col + 1)
            yield from self.runtime.colors.send(
                "partial_sum_color",
                self.coord,
                east,
                self.config.scalar_bytes,
            )
            yield self.pe_programs[east].partial_queue.put(running_sum)
        else:
            yield from self.runtime.colors.send(
                "Y_D2H",
                self.coord,
                None,
                self.config.scalar_bytes,
            )
            self.done.succeed(running_sum)
