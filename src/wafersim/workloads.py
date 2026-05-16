from __future__ import annotations

import simpy

from wafersim.config import Coord, GemmWorkloadConfig, MemoryCopyWorkloadConfig
from wafersim.config import StencilWorkloadConfig
from wafersim.network import WaferMesh


class StencilWorkload:
    """Bulk-synchronous 2D stencil workload.

    Each PE computes one tile and exchanges halo packets with north/south/east/west
    neighbors at every time step.
    """

    def __init__(self, config: StencilWorkloadConfig) -> None:
        config.validate()
        self.config = config

    def summary(self, mesh: WaferMesh) -> dict[str, int | str]:
        return {
            "type": "stencil",
            "steps": self.config.steps,
            "tile_rows": self.config.tile_rows,
            "tile_cols": self.config.tile_cols,
            "ops_per_cell": self.config.ops_per_cell,
            "halo_bytes_per_edge": self.config.halo_bytes_per_edge,
            "bytes_per_cell": self.config.bytes_per_cell,
            "memory_reads_per_cell": self.config.memory_reads_per_cell,
            "memory_writes_per_cell": self.config.memory_writes_per_cell,
            "double_buffer_tiles": self.config.double_buffer_tiles,
            "tile_storage_bytes_per_pe": self.config.tile_storage_bytes,
            "halo_storage_bytes_per_pe": self.config.halo_storage_bytes,
            "memory_footprint_bytes_per_pe": (
                self.config.memory_footprint_bytes_per_pe
            ),
            "memory_access_bytes_per_step": self.config.memory_access_bytes_per_step,
        }

    def run(self, env: simpy.Environment, mesh: WaferMesh):
        coords = list(mesh.coords())
        for coord in coords:
            mesh.pes[coord].reserve_memory(self.config.memory_footprint_bytes_per_pe)

        for step in range(self.config.steps):
            memory_events = [
                env.process(
                    mesh.pes[coord].access_memory(
                        self.config.memory_access_bytes_per_step
                    )
                )
                for coord in coords
            ]
            if memory_events:
                yield simpy.events.AllOf(env, memory_events)

            compute_events = [
                env.process(mesh.pes[coord].compute(self.config.ops_per_tile))
                for coord in coords
            ]
            if compute_events:
                yield simpy.events.AllOf(env, compute_events)

            comm_events = []
            if self.config.halo_bytes_per_edge > 0:
                for coord in coords:
                    for neighbor in mesh.neighbors(coord):
                        comm_events.append(
                            env.process(
                                mesh.send(
                                    src=coord,
                                    dst=neighbor,
                                    size_bytes=self.config.halo_bytes_per_edge,
                                    kind=f"halo:{step}",
                                )
                            )
                        )
            if comm_events:
                yield simpy.events.AllOf(env, comm_events)


class MemoryCopyWorkload:
    """Copy an array from one PE's local memory to another PE's local memory."""

    def __init__(self, config: MemoryCopyWorkloadConfig) -> None:
        config.validate()
        self.config = config
        self.source: Coord | None = None
        self.destination: Coord | None = None
        self.started_at = 0.0
        self.completed_at = 0.0

    def run(self, env: simpy.Environment, mesh: WaferMesh):
        self.source, self.destination = self._endpoints(mesh)
        self.started_at = env.now
        mesh.pes[self.source].reserve_memory(self.config.array_bytes)
        mesh.pes[self.destination].reserve_memory(self.config.array_bytes)

        yield from mesh.pes[self.source].access_memory(self.config.array_bytes)

        remaining_bytes = self.config.array_bytes
        packet_index = 0
        send_events = []

        while remaining_bytes > 0:
            packet_bytes = min(self.config.packet_payload_bytes, remaining_bytes)
            send_events.append(
                env.process(
                    mesh.send(
                        src=self.source,
                        dst=self.destination,
                        size_bytes=packet_bytes,
                        kind=f"memory_copy:{packet_index}",
                    )
                )
            )
            remaining_bytes -= packet_bytes
            packet_index += 1
            if self.config.injection_gap_cycles > 0 and remaining_bytes > 0:
                yield env.timeout(self.config.injection_gap_cycles)

        if send_events:
            yield simpy.events.AllOf(env, send_events)

        yield from mesh.pes[self.destination].access_memory(self.config.array_bytes)
        self.completed_at = env.now

    def summary(self, mesh: WaferMesh) -> dict[str, float | int | list[int] | str]:
        source, destination = self._endpoints(mesh)
        route_hops = len(mesh.route_xy(source, destination))
        return {
            "type": "memory_copy",
            "elements": self.config.elements,
            "bytes_per_element": self.config.bytes_per_element,
            "array_bytes": self.config.array_bytes,
            "packet_payload_bytes": self.config.packet_payload_bytes,
            "expected_packets": self.config.packet_count,
            "source": list(source),
            "destination": list(destination),
            "route_hops": route_hops,
            "injection_gap_cycles": self.config.injection_gap_cycles,
            "source_read_bytes": self.config.array_bytes,
            "destination_write_bytes": self.config.array_bytes,
        }

    def result_summary(
        self, mesh: WaferMesh, elapsed_cycles: float
    ) -> dict[str, float | int]:
        throughput = (
            self.config.array_bytes / elapsed_cycles if elapsed_cycles > 0 else 0.0
        )
        path_bandwidth = mesh.config.link_bandwidth_bytes_per_cycle
        efficiency = throughput / path_bandwidth if path_bandwidth > 0 else 0.0
        return {
            "memory_copy_throughput_bytes_per_cycle": throughput,
            "path_bandwidth_efficiency": efficiency,
        }

    def _endpoints(self, mesh: WaferMesh) -> tuple[Coord, Coord]:
        source = (self.config.source_row, self.config.source_col)
        destination = (
            mesh.config.rows - 1 if self.config.dest_row is None else self.config.dest_row,
            mesh.config.cols - 1 if self.config.dest_col is None else self.config.dest_col,
        )
        if not mesh.contains(source):
            raise ValueError(f"source coordinate out of mesh: {source}")
        if not mesh.contains(destination):
            raise ValueError(f"destination coordinate out of mesh: {destination}")
        return source, destination


class GemmWorkload:
    """Output-stationary tiled GEMM on a 2D PE mesh.

    C is partitioned across the PE mesh. Each PE owns one C tile. For every K
    tile, A tiles move east along rows and B tiles move south along columns.
    """

    def __init__(self, config: GemmWorkloadConfig) -> None:
        config.validate()
        self.config = config

    def run(self, env: simpy.Environment, mesh: WaferMesh):
        self._validate_mesh_shape(mesh)
        active_coords = [
            (row, col)
            for row in range(self.config.pe_rows)
            for col in range(self.config.pe_cols)
        ]
        for coord in active_coords:
            mesh.pes[coord].reserve_memory(self.config.memory_footprint_bytes_per_pe)

        for k_tile in range(self.config.k_tiles):
            if self.config.pipeline_k_tiles:
                stream_events = self._start_stream_events(env, mesh, k_tile)
                memory_events = self._start_memory_events(env, mesh, active_coords)
                compute_events = self._start_compute_events(env, mesh, active_coords)
                phase_events = stream_events + memory_events + compute_events
                if phase_events:
                    yield simpy.events.AllOf(env, phase_events)
            else:
                stream_events = self._start_stream_events(env, mesh, k_tile)
                if stream_events:
                    yield simpy.events.AllOf(env, stream_events)
                memory_events = self._start_memory_events(env, mesh, active_coords)
                if memory_events:
                    yield simpy.events.AllOf(env, memory_events)
                compute_events = self._start_compute_events(env, mesh, active_coords)
                if compute_events:
                    yield simpy.events.AllOf(env, compute_events)

        writeback_events = [
            env.process(mesh.pes[coord].access_memory(self.config.c_tile_bytes))
            for coord in active_coords
        ]
        if writeback_events:
            yield simpy.events.AllOf(env, writeback_events)

    def summary(self, mesh: WaferMesh) -> dict[str, int | bool | str]:
        return {
            "type": "gemm",
            "m": self.config.m,
            "n": self.config.n,
            "k": self.config.k,
            "tile_m": self.config.tile_m,
            "tile_n": self.config.tile_n,
            "tile_k": self.config.tile_k,
            "pe_rows": self.config.pe_rows,
            "pe_cols": self.config.pe_cols,
            "active_pes": self.config.pe_rows * self.config.pe_cols,
            "k_tiles": self.config.k_tiles,
            "bytes_per_element": self.config.bytes_per_element,
            "ops_per_mac": self.config.ops_per_mac,
            "pipeline_k_tiles": self.config.pipeline_k_tiles,
            "a_tile_bytes": self.config.a_tile_bytes,
            "b_tile_bytes": self.config.b_tile_bytes,
            "c_tile_bytes": self.config.c_tile_bytes,
            "double_buffer_inputs": self.config.double_buffer_inputs,
            "memory_footprint_bytes_per_pe": (
                self.config.memory_footprint_bytes_per_pe
            ),
            "ops_per_pe_per_k_tile": self.config.ops_per_pe_per_k_tile,
            "ops_per_pe_total": self.config.ops_per_pe_total,
            "total_ops": self.config.total_ops,
        }

    def result_summary(
        self, mesh: WaferMesh, elapsed_cycles: float
    ) -> dict[str, float | int]:
        ops_per_cycle = (
            self.config.total_ops / elapsed_cycles if elapsed_cycles > 0 else 0.0
        )
        peak_ops_per_cycle = (
            self.config.pe_rows
            * self.config.pe_cols
            * mesh.config.compute_throughput_ops_per_cycle
        )
        compute_efficiency = (
            ops_per_cycle / peak_ops_per_cycle if peak_ops_per_cycle > 0 else 0.0
        )
        return {
            "gemm_ops_per_cycle": ops_per_cycle,
            "gemm_peak_ops_per_cycle": peak_ops_per_cycle,
            "gemm_compute_efficiency": compute_efficiency,
        }

    def _validate_mesh_shape(self, mesh: WaferMesh) -> None:
        if self.config.pe_rows > mesh.config.rows:
            raise ValueError("GEMM tile rows require more PE rows than the mesh has")
        if self.config.pe_cols > mesh.config.cols:
            raise ValueError("GEMM tile cols require more PE cols than the mesh has")

    def _start_stream_events(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        k_tile: int,
    ):
        stream_events = []
        for row in range(self.config.pe_rows):
            for col in range(self.config.pe_cols - 1):
                stream_events.append(
                    env.process(
                        mesh.send(
                            src=(row, col),
                            dst=(row, col + 1),
                            size_bytes=self.config.a_tile_bytes,
                            kind=f"gemm:a:{k_tile}",
                        )
                    )
                )
        for row in range(self.config.pe_rows - 1):
            for col in range(self.config.pe_cols):
                stream_events.append(
                    env.process(
                        mesh.send(
                            src=(row, col),
                            dst=(row + 1, col),
                            size_bytes=self.config.b_tile_bytes,
                            kind=f"gemm:b:{k_tile}",
                        )
                    )
                )
        return stream_events

    def _start_memory_events(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        active_coords: list[Coord],
    ):
        return [
            env.process(
                mesh.pes[coord].access_memory(
                    self.config.a_tile_bytes + self.config.b_tile_bytes
                )
            )
            for coord in active_coords
        ]

    def _start_compute_events(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        active_coords: list[Coord],
    ):
        return [
            env.process(mesh.pes[coord].compute(self.config.ops_per_pe_per_k_tile))
            for coord in active_coords
        ]
