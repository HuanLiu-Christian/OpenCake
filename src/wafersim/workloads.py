from __future__ import annotations

import math

import simpy

from wafersim.config import CslDemoWorkloadConfig, CslRuntimeWorkloadConfig, Coord
from wafersim.config import GemmWorkloadConfig
from wafersim.config import MemoryCopyWorkloadConfig
from wafersim.config import SdkExamplesWorkloadConfig
from wafersim.config import StencilWorkloadConfig
from wafersim.csl_compat import CslBuiltins, CslCompatibilityRuntime, Direction
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


class CslDemoWorkload:
    """Concept-level SimPy models for common Cerebras CSL tutorial topics.

    This is not a CSL compiler or WSE hardware model. It maps the programming
    ideas from the public CSL examples onto the simulator's PE, memory, FIFO,
    and packet abstractions so the same architecture counters stay visible.
    """

    def __init__(self, config: CslDemoWorkloadConfig) -> None:
        config.validate()
        self.config = config
        self.topic_results: dict[str, int | float | list[int]] = {}
        self.debug_log: list[str] = []
        self.simprint_log: list[str] = []

    def run(self, env: simpy.Environment, mesh: WaferMesh):
        self._validate_mesh_shape(mesh)
        coords = self._active_coords()
        for coord in coords:
            mesh.pes[coord].reserve_memory(self.config.memory_footprint_bytes_per_pe)

        yield from self._arrays_and_pointers(env, mesh, coords)
        yield from self._libraries(env, mesh, coords)
        yield from self._streaming_wavelet_data(env, mesh, coords)
        yield from self._sparse_wavelets(env, mesh, coords)
        yield from self._sentinels(env, mesh, coords)
        yield from self._switches(env, mesh, coords)
        yield from self._switches_and_control(env, mesh, coords)
        yield from self._filters(env, mesh, coords)
        yield from self._fifos(env, mesh, coords)
        yield from self._map_builtin(env, mesh, coords)
        yield from self._collectives(env, mesh, coords)
        yield from self._debug_and_simprint(env, mesh, coords)
        yield from self._color_swap(env, mesh, coords)
        yield from self._microthreads(env, mesh, coords)
        yield from self._reuse_output_queues(env, mesh, coords)

    def summary(
        self, mesh: WaferMesh
    ) -> dict[str, int | float | str | dict[str, int | float | list[int]] | list[str]]:
        return {
            "type": "csl_demo",
            "active_rows": self.config.active_rows,
            "active_cols": self.config.active_cols,
            "active_pes": self.config.active_pes,
            "elements_per_pe": self.config.elements_per_pe,
            "bytes_per_element": self.config.bytes_per_element,
            "packet_payload_bytes": self.config.packet_payload_bytes,
            "fifo_capacity": self.config.fifo_capacity,
            "wavelet_levels": self.config.wavelet_levels,
            "sparse_density": self.config.sparse_density,
            "collective_payload_bytes": self.config.collective_payload_bytes,
            "microthreads_per_pe": self.config.microthreads_per_pe,
            "output_queue_reuse_rounds": self.config.output_queue_reuse_rounds,
            "color_swap_payload_bytes": self.config.color_swap_payload_bytes,
            "sentinel_bytes": self.config.sentinel_bytes,
            "memory_footprint_bytes_per_pe": (
                self.config.memory_footprint_bytes_per_pe
            ),
            "topic_results": self.topic_results,
            "debug_log": self.debug_log,
            "simprint_log": self.simprint_log,
        }

    def _validate_mesh_shape(self, mesh: WaferMesh) -> None:
        if self.config.active_rows > mesh.config.rows:
            raise ValueError("CSL demo active rows exceed mesh rows")
        if self.config.active_cols > mesh.config.cols:
            raise ValueError("CSL demo active cols exceed mesh cols")

    def _active_coords(self) -> list[Coord]:
        return [
            (row, col)
            for row in range(self.config.active_rows)
            for col in range(self.config.active_cols)
        ]

    def _arrays_and_pointers(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        bytes_accessed = self.config.local_array_bytes_per_pe
        events = [env.process(mesh.pes[coord].access_memory(bytes_accessed)) for coord in coords]
        yield simpy.events.AllOf(env, events)
        self.topic_results["topic_1_arrays_and_pointers_bytes"] = (
            bytes_accessed * len(coords)
        )

    def _libraries(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        ops = self.config.library_ops_per_pe
        events = [env.process(mesh.pes[coord].compute(ops)) for coord in coords]
        yield simpy.events.AllOf(env, events)
        self.topic_results["topic_2_library_ops"] = ops * len(coords)

    def _streaming_wavelet_data(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        events = []
        for row in range(self.config.active_rows):
            for col in range(self.config.active_cols - 1):
                events.append(
                    env.process(
                        mesh.send(
                            src=(row, col),
                            dst=(row, col + 1),
                            size_bytes=self.config.packet_payload_bytes,
                            kind="csl:topic3:streaming_wavelet",
                        )
                    )
                )
        if events:
            yield simpy.events.AllOf(env, events)
        self.topic_results["topic_3_streaming_wavelet_packets"] = len(events)

    def _sparse_wavelets(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        ops = self.config.sparse_ops_per_pe
        events = [env.process(mesh.pes[coord].compute(ops)) for coord in coords]
        yield simpy.events.AllOf(env, events)
        self.topic_results["topic_4_sparse_wavelet_ops"] = ops * len(coords)

    def _sentinels(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        events = [
            env.process(
                mesh.send(
                    src=coords[-1],
                    dst=coords[0],
                    size_bytes=self.config.sentinel_bytes,
                    kind="csl:topic5:sentinel:end_of_stream",
                )
            )
        ]
        yield simpy.events.AllOf(env, events)
        self.topic_results["topic_5_sentinel_packets"] = len(events)

    def _switches(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        root = coords[0]
        destinations = [
            (0, self.config.active_cols - 1),
            (self.config.active_rows - 1, 0),
            (self.config.active_rows - 1, self.config.active_cols - 1),
        ]
        events = [
            env.process(
                mesh.send(
                    src=root,
                    dst=dst,
                    size_bytes=self.config.packet_payload_bytes,
                    kind="csl:topic6:switch_fanout",
                )
            )
            for dst in destinations
            if dst != root
        ]
        if events:
            yield simpy.events.AllOf(env, events)
        self.topic_results["topic_6_switch_fanout_packets"] = len(events)

    def _switches_and_control(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        control = (0, 0)
        events = [
            env.process(
                mesh.send(
                    src=control,
                    dst=coord,
                    size_bytes=self.config.sentinel_bytes,
                    kind="csl:topic7:control_entrypoint",
                )
            )
            for coord in coords
            if coord != control
        ]
        if events:
            yield simpy.events.AllOf(env, events)
        self.topic_results["topic_7_control_entrypoint_packets"] = len(events)

    def _filters(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        selected = [
            coord
            for coord in coords
            if (coord[0] + coord[1]) % 2 == 0 and coord[1] + 1 < self.config.active_cols
        ]
        events = [
            env.process(
                mesh.send(
                    src=coord,
                    dst=(coord[0], coord[1] + 1),
                    size_bytes=self.config.packet_payload_bytes,
                    kind="csl:topic8:filtered_stream",
                )
            )
            for coord in selected
        ]
        if events:
            yield simpy.events.AllOf(env, events)
        self.topic_results["topic_8_filter_pass_packets"] = len(events)

    def _fifos(self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]):
        fifo = simpy.Store(env, capacity=self.config.fifo_capacity)
        producer = env.process(self._fifo_producer(env, fifo))
        consumer = env.process(self._fifo_consumer(env, fifo, mesh, coords[0]))
        yield simpy.events.AllOf(env, [producer, consumer])
        self.topic_results["topic_9_fifo_items"] = self.config.fifo_capacity * 2

    def _fifo_producer(self, env: simpy.Environment, fifo: simpy.Store):
        for item in range(self.config.fifo_capacity * 2):
            yield fifo.put(item)
            yield env.timeout(1)

    def _fifo_consumer(
        self,
        env: simpy.Environment,
        fifo: simpy.Store,
        mesh: WaferMesh,
        coord: Coord,
    ):
        for _ in range(self.config.fifo_capacity * 2):
            yield fifo.get()
            yield from mesh.pes[coord].access_memory(self.config.packet_payload_bytes)

    def _map_builtin(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        ops = self.config.map_ops_per_pe
        events = [env.process(mesh.pes[coord].compute(ops)) for coord in coords]
        yield simpy.events.AllOf(env, events)
        mapped_ids = [
            row * self.config.active_cols + col
            for row, col in coords[: min(4, len(coords))]
        ]
        self.topic_results["topic_10_map_ops"] = ops * len(coords)
        self.topic_results["topic_10_first_mapped_ids"] = mapped_ids

    def _collectives(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        root = (0, 0)
        broadcast_events = [
            env.process(
                mesh.send(
                    src=root,
                    dst=coord,
                    size_bytes=self.config.collective_payload_bytes,
                    kind="csl:topic11:broadcast",
                )
            )
            for coord in coords
            if coord != root
        ]
        reduce_events = [
            env.process(
                mesh.send(
                    src=coord,
                    dst=root,
                    size_bytes=self.config.collective_payload_bytes,
                    kind="csl:topic11:reduce",
                )
            )
            for coord in coords
            if coord != root
        ]
        events = broadcast_events + reduce_events
        if events:
            yield simpy.events.AllOf(env, events)
        self.topic_results["topic_11_collective_packets"] = len(events)

    def _debug_and_simprint(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        self.debug_log.append(
            f"active_pes={len(coords)} fifo_capacity={self.config.fifo_capacity}"
        )
        self.simprint_log.append(
            f"PE(0,0) local_array_bytes={self.config.local_array_bytes_per_pe}"
        )
        yield env.timeout(0)
        self.topic_results["topic_12_13_debug_messages"] = (
            len(self.debug_log) + len(self.simprint_log)
        )

    def _color_swap(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        events = []
        for row in range(self.config.active_rows):
            for col in range(0, self.config.active_cols - 1, 2):
                events.append(
                    env.process(
                        mesh.send(
                            src=(row, col),
                            dst=(row, col + 1),
                            size_bytes=self.config.color_swap_payload_bytes,
                            kind="csl:topic14:color_swap",
                        )
                    )
                )
                events.append(
                    env.process(
                        mesh.send(
                            src=(row, col + 1),
                            dst=(row, col),
                            size_bytes=self.config.color_swap_payload_bytes,
                            kind="csl:topic14:color_swap",
                        )
                    )
                )
        if events:
            yield simpy.events.AllOf(env, events)
        self.topic_results["topic_14_color_swap_packets"] = len(events)

    def _microthreads(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        ops_per_thread = max(1, self.config.library_ops_per_pe // self.config.microthreads_per_pe)
        events = [
            env.process(mesh.pes[coord].compute(ops_per_thread))
            for coord in coords
            for _ in range(self.config.microthreads_per_pe)
        ]
        yield simpy.events.AllOf(env, events)
        self.topic_results["topic_15_microthread_tasks"] = len(events)

    def _reuse_output_queues(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        events = []
        for round_id in range(self.config.output_queue_reuse_rounds):
            for row in range(self.config.active_rows - 1):
                dst = (row + 1, 0)
                events.append(
                    env.process(
                        mesh.send(
                            src=(row, 0),
                            dst=dst,
                            size_bytes=self.config.packet_payload_bytes,
                            kind=f"csl:topic16:reuse_output_queue:{round_id}",
                        )
                    )
                )
        if events:
            yield simpy.events.AllOf(env, events)
        self.topic_results["topic_16_reused_output_queue_packets"] = len(events)


class SdkExamplesWorkload:
    """Concept-level models for later Cerebras SDK code examples.

    The official examples use CSL, SdkRuntime, fabric DSDs, colors, and SDK
    libraries. This workload keeps the architectural behavior: streams, FIFOs,
    ports, collectives, stencils, sparse matvecs, iterative solvers, FFT-style
    transposes, and cellular automata.
    """

    def __init__(self, config: SdkExamplesWorkloadConfig) -> None:
        config.validate()
        self.config = config
        self.example_results: dict[str, int | float | list[int]] = {}

    def run(self, env: simpy.Environment, mesh: WaferMesh):
        self._validate_mesh_shape(mesh)
        coords = self._active_coords()
        for coord in coords:
            mesh.pes[coord].reserve_memory(self.config.memory_footprint_bytes_per_pe)

        yield from self._pipeline_redirect(env, mesh, coords)
        yield from self._pipeline_h2d_fifo(env, mesh, coords)
        yield from self._pipeline_artificial_halo(env, mesh, coords)
        yield from self._sdklayout_intro(env, mesh, coords)
        yield from self._sdklayout_basic_routing(env, mesh, coords)
        yield from self._sdklayout_ports(env, mesh, coords)
        yield from self._sdklayout_h2d_d2h(env, mesh, coords)
        yield from self._gemv(env, mesh, coords)
        yield from self._gemv_checkerboard(env, mesh, coords)
        yield from self._gemv_collective(env, mesh, coords)
        yield from self._gemm_collective(env, mesh, coords)
        yield from self._residual(env, mesh, coords)
        yield from self._cholesky(env, mesh, coords)
        yield from self._stencil_25_point(env, mesh, coords)
        yield from self._bandwidth_test(env, mesh, coords)
        yield from self._stencil_3d_7_point_spmv(env, mesh, coords)
        yield from self._power_method(env, mesh, coords)
        yield from self._conjugate_gradient(env, mesh, coords)
        yield from self._preconditioned_conjugate_gradient(env, mesh, coords)
        yield from self._bicgstab(env, mesh, coords)
        yield from self._single_tile_matvec(env, mesh, coords)
        yield from self._host_to_device_broadcast(env, mesh, coords)
        yield from self._fft_3d(env, mesh, coords)
        yield from self._game_of_life(env, mesh, coords)

    def summary(
        self, mesh: WaferMesh
    ) -> dict[str, int | float | str | dict[str, int | float | list[int]]]:
        return {
            "type": "sdk_examples",
            "active_rows": self.config.active_rows,
            "active_cols": self.config.active_cols,
            "active_pes": self.config.active_pes,
            "bytes_per_element": self.config.bytes_per_element,
            "packet_payload_bytes": self.config.packet_payload_bytes,
            "fifo_capacity": self.config.fifo_capacity,
            "stream_bytes": self.config.stream_bytes,
            "tile_rows": self.config.tile_rows,
            "tile_cols": self.config.tile_cols,
            "vector_length": self.config.vector_length,
            "matrix_rows": self.config.matrix_rows,
            "matrix_cols": self.config.matrix_cols,
            "gemm_m": self.config.gemm_m,
            "gemm_n": self.config.gemm_n,
            "gemm_k": self.config.gemm_k,
            "iterations": self.config.iterations,
            "stencil_halo_bytes_per_direction": (
                self.config.stencil_halo_bytes_per_direction
            ),
            "spmv_nonzeros_per_row": self.config.spmv_nonzeros_per_row,
            "fft_points": self.config.fft_points,
            "memory_footprint_bytes_per_pe": (
                self.config.memory_footprint_bytes_per_pe
            ),
            "example_results": self.example_results,
        }

    def result_summary(
        self, mesh: WaferMesh, elapsed_cycles: float
    ) -> dict[str, float | int]:
        total_ops = sum(
            value
            for key, value in self.example_results.items()
            if key.endswith("_ops") and isinstance(value, int | float)
        )
        total_stream_bytes = sum(
            value
            for key, value in self.example_results.items()
            if key.endswith("_bytes") and isinstance(value, int | float)
        )
        peak = (
            self.config.active_pes * mesh.config.compute_throughput_ops_per_cycle
        )
        return {
            "sdk_examples_total_ops": total_ops,
            "sdk_examples_ops_per_cycle": (
                total_ops / elapsed_cycles if elapsed_cycles > 0 else 0.0
            ),
            "sdk_examples_compute_efficiency": (
                (total_ops / elapsed_cycles) / peak
                if elapsed_cycles > 0 and peak > 0
                else 0.0
            ),
            "sdk_examples_stream_bytes": total_stream_bytes,
            "sdk_examples_stream_bytes_per_cycle": (
                total_stream_bytes / elapsed_cycles if elapsed_cycles > 0 else 0.0
            ),
        }

    def _validate_mesh_shape(self, mesh: WaferMesh) -> None:
        if self.config.active_rows > mesh.config.rows:
            raise ValueError("SDK examples active rows exceed mesh rows")
        if self.config.active_cols > mesh.config.cols:
            raise ValueError("SDK examples active cols exceed mesh cols")

    def _active_coords(self) -> list[Coord]:
        return [
            (row, col)
            for row in range(self.config.active_rows)
            for col in range(self.config.active_cols)
        ]

    def _packet_chunks(self, total_bytes: int) -> list[int]:
        chunks = []
        remaining = total_bytes
        while remaining > 0:
            chunk = min(self.config.packet_payload_bytes, remaining)
            chunks.append(chunk)
            remaining -= chunk
        return chunks

    def _send_stream(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        src: Coord,
        dst: Coord,
        total_bytes: int,
        kind: str,
    ):
        events = [
            env.process(mesh.send(src=src, dst=dst, size_bytes=chunk, kind=kind))
            for chunk in self._packet_chunks(total_bytes)
        ]
        if events:
            yield simpy.events.AllOf(env, events)
        return len(events)

    def _compute_all(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        coords: list[Coord],
        ops_per_pe: int,
    ):
        events = [env.process(mesh.pes[coord].compute(ops_per_pe)) for coord in coords]
        if events:
            yield simpy.events.AllOf(env, events)

    def _memory_all(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        coords: list[Coord],
        bytes_per_pe: int,
    ):
        events = [
            env.process(mesh.pes[coord].access_memory(bytes_per_pe)) for coord in coords
        ]
        if events:
            yield simpy.events.AllOf(env, events)

    def _pipeline_redirect(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        packets = yield from self._send_stream(
            env, mesh, coords[0], coords[-1], self.config.stream_bytes, "sdk:pipeline1"
        )
        self.example_results["pipeline_1_redirect_packets"] = packets
        self.example_results["pipeline_1_redirect_bytes"] = self.config.stream_bytes

    def _pipeline_h2d_fifo(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        fifo = simpy.Store(env, capacity=self.config.fifo_capacity)
        chunks = self._packet_chunks(self.config.stream_bytes)
        producer = env.process(self._fifo_stream_producer(env, fifo, chunks))
        consumer = env.process(self._fifo_stream_consumer(env, fifo, mesh, coords[0]))
        yield simpy.events.AllOf(env, [producer, consumer])
        self.example_results["pipeline_2_h2d_fifo_items"] = len(chunks)
        self.example_results["pipeline_2_h2d_fifo_bytes"] = self.config.stream_bytes

    def _fifo_stream_producer(
        self, env: simpy.Environment, fifo: simpy.Store, chunks: list[int]
    ):
        for chunk in chunks:
            yield fifo.put(chunk)
            yield env.timeout(1)

    def _fifo_stream_consumer(
        self, env: simpy.Environment, fifo: simpy.Store, mesh: WaferMesh, coord: Coord
    ):
        for _ in range(len(self._packet_chunks(self.config.stream_bytes))):
            chunk = yield fifo.get()
            yield from mesh.pes[coord].access_memory(chunk)

    def _pipeline_artificial_halo(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        events = []
        for coord in coords:
            for neighbor in mesh.neighbors(coord):
                if neighbor in coords:
                    events.append(
                        env.process(
                            mesh.send(
                                src=coord,
                                dst=neighbor,
                                size_bytes=self.config.stencil_halo_bytes_per_direction,
                                kind="sdk:pipeline3:artificial_halo",
                            )
                        )
                    )
        if events:
            yield simpy.events.AllOf(env, events)
        self.example_results["pipeline_3_artificial_halo_packets"] = len(events)

    def _sdklayout_intro(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        yield env.timeout(0)
        self.example_results["sdklayout_1_active_pes"] = len(coords)

    def _sdklayout_basic_routing(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        route = mesh.route_xy(coords[0], coords[-1])
        packets = yield from self._send_stream(
            env,
            mesh,
            coords[0],
            coords[-1],
            self.config.packet_payload_bytes,
            "sdk:layout2:basic_routing",
        )
        self.example_results["sdklayout_2_route_hops"] = len(route)
        self.example_results["sdklayout_2_packets"] = packets

    def _sdklayout_ports(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        west_port, east_port = (0, 0), (0, self.config.active_cols - 1)
        packets = yield from self._send_stream(
            env,
            mesh,
            west_port,
            east_port,
            self.config.stream_bytes // 2,
            "sdk:layout3:ports",
        )
        self.example_results["sdklayout_3_port_packets"] = packets

    def _sdklayout_h2d_d2h(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        h2d = yield from self._send_stream(
            env, mesh, coords[0], coords[-1], self.config.stream_bytes, "sdk:layout4:h2d"
        )
        d2h = yield from self._send_stream(
            env, mesh, coords[-1], coords[0], self.config.stream_bytes, "sdk:layout4:d2h"
        )
        self.example_results["sdklayout_4_stream_packets"] = h2d + d2h
        self.example_results["sdklayout_4_stream_bytes"] = self.config.stream_bytes * 2

    def _gemv(self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]):
        ops_per_pe = max(1, self.config.gemv_ops // len(coords))
        yield from self._memory_all(env, mesh, coords, self.config.vector_bytes)
        yield from self._compute_all(env, mesh, coords, ops_per_pe)
        self.example_results["sdklayout_5_gemv_ops"] = self.config.gemv_ops

    def _gemv_checkerboard(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        active = [coord for coord in coords if (coord[0] + coord[1]) % 2 == 0]
        ops_per_pe = max(1, self.config.gemv_ops // max(1, len(active)))
        yield from self._compute_all(env, mesh, active, ops_per_pe)
        self.example_results["gemv_checkerboard_active_pes"] = len(active)
        self.example_results["gemv_checkerboard_ops"] = self.config.gemv_ops

    def _gemv_collective(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        packets = yield from self._broadcast_and_reduce(
            env, mesh, coords, self.config.vector_bytes, "sdk:gemv_collective"
        )
        ops_per_pe = max(1, self.config.gemv_ops // len(coords))
        yield from self._compute_all(env, mesh, coords, ops_per_pe)
        self.example_results["gemv_collective_packets"] = packets
        self.example_results["gemv_collective_ops"] = self.config.gemv_ops

    def _gemm_collective(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        tile_bytes = self.config.tile_bytes_per_pe
        packets = yield from self._broadcast_and_reduce(
            env, mesh, coords, tile_bytes, "sdk:gemm_collective"
        )
        ops_per_pe = max(1, self.config.gemm_ops // len(coords))
        yield from self._compute_all(env, mesh, coords, ops_per_pe)
        self.example_results["gemm_collective_packets"] = packets
        self.example_results["gemm_collective_ops"] = self.config.gemm_ops

    def _residual(self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]):
        ops = self.config.spmv_ops_per_iteration + self.config.matrix_rows
        yield from self._compute_all(env, mesh, coords, max(1, ops // len(coords)))
        self.example_results["residual_ops"] = ops

    def _cholesky(self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]):
        n = min(self.config.matrix_rows, self.config.matrix_cols)
        ops = max(1, n * n * n // 3)
        panel_steps = min(self.config.active_rows, self.config.active_cols)
        for step in range(panel_steps):
            panel = coords[step]
            trailing = coords[step + 1 :]
            packets = [
                env.process(
                    mesh.send(
                        src=panel,
                        dst=coord,
                        size_bytes=self.config.packet_payload_bytes,
                        kind=f"sdk:cholesky:panel:{step}",
                    )
                )
                for coord in trailing
            ]
            if packets:
                yield simpy.events.AllOf(env, packets)
        yield from self._compute_all(env, mesh, coords, max(1, ops // len(coords)))
        self.example_results["cholesky_ops"] = ops

    def _stencil_25_point(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        packets = yield from self._neighbor_exchange(
            env, mesh, coords, self.config.stencil_halo_bytes_per_direction, "sdk:25pt"
        )
        ops = self.config.stencil_ops_per_pe * len(coords)
        yield from self._compute_all(env, mesh, coords, self.config.stencil_ops_per_pe)
        self.example_results["stencil_25_point_packets"] = packets
        self.example_results["stencil_25_point_ops"] = ops

    def _bandwidth_test(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        packets = yield from self._send_stream(
            env, mesh, coords[0], coords[-1], self.config.stream_bytes * 2, "sdk:bandwidth"
        )
        self.example_results["bandwidth_test_packets"] = packets
        self.example_results["bandwidth_test_bytes"] = self.config.stream_bytes * 2

    def _stencil_3d_7_point_spmv(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        packets = yield from self._neighbor_exchange(
            env, mesh, coords, self.config.packet_payload_bytes, "sdk:3d7pt_spmv"
        )
        ops = self.config.spmv_ops_per_iteration
        yield from self._compute_all(env, mesh, coords, max(1, ops // len(coords)))
        self.example_results["stencil_3d_7_point_spmv_packets"] = packets
        self.example_results["stencil_3d_7_point_spmv_ops"] = ops

    def _power_method(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        ops = self.config.iterations * (
            self.config.spmv_ops_per_iteration + 2 * self.config.matrix_rows
        )
        packets = 0
        for iteration in range(self.config.iterations):
            packets += yield from self._broadcast_and_reduce(
                env,
                mesh,
                coords,
                self.config.packet_payload_bytes,
                f"sdk:power_method:{iteration}",
            )
        yield from self._compute_all(env, mesh, coords, max(1, ops // len(coords)))
        self.example_results["power_method_packets"] = packets
        self.example_results["power_method_ops"] = ops

    def _conjugate_gradient(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        ops = self.config.iterations * (
            self.config.spmv_ops_per_iteration + 6 * self.config.matrix_rows
        )
        packets = yield from self._iterative_solver_collectives(
            env, mesh, coords, "sdk:cg", reductions_per_iteration=2
        )
        yield from self._compute_all(env, mesh, coords, max(1, ops // len(coords)))
        self.example_results["conjugate_gradient_packets"] = packets
        self.example_results["conjugate_gradient_ops"] = ops

    def _preconditioned_conjugate_gradient(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        ops = self.config.iterations * (
            self.config.spmv_ops_per_iteration + 8 * self.config.matrix_rows
        )
        packets = yield from self._iterative_solver_collectives(
            env, mesh, coords, "sdk:pcg", reductions_per_iteration=2
        )
        yield from self._compute_all(env, mesh, coords, max(1, ops // len(coords)))
        self.example_results["preconditioned_conjugate_gradient_packets"] = packets
        self.example_results["preconditioned_conjugate_gradient_ops"] = ops

    def _bicgstab(self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]):
        ops = self.config.iterations * (
            2 * self.config.spmv_ops_per_iteration + 10 * self.config.matrix_rows
        )
        packets = yield from self._iterative_solver_collectives(
            env, mesh, coords, "sdk:bicgstab", reductions_per_iteration=4
        )
        yield from self._compute_all(env, mesh, coords, max(1, ops // len(coords)))
        self.example_results["bicgstab_packets"] = packets
        self.example_results["bicgstab_ops"] = ops

    def _single_tile_matvec(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        ops = 2 * self.config.tile_rows * self.config.tile_cols
        yield from mesh.pes[coords[0]].compute(ops)
        self.example_results["single_tile_matvec_ops"] = ops

    def _host_to_device_broadcast(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        packets = yield from self._broadcast(
            env, mesh, coords, self.config.packet_payload_bytes, "sdk:h2d_broadcast"
        )
        self.example_results["host_to_device_broadcast_packets"] = packets
        self.example_results["host_to_device_broadcast_bytes"] = (
            self.config.packet_payload_bytes * max(0, len(coords) - 1)
        )

    def _fft_3d(self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]):
        log_points = max(1, math.ceil(math.log2(self.config.fft_points)))
        ops = 5 * self.config.fft_points * log_points
        transpose_bytes = self.config.fft_points * self.config.bytes_per_element
        packets = yield from self._all_to_all(
            env, mesh, coords[: min(4, len(coords))], transpose_bytes, "sdk:3d_fft"
        )
        yield from self._compute_all(env, mesh, coords, max(1, ops // len(coords)))
        self.example_results["fft_3d_packets"] = packets
        self.example_results["fft_3d_ops"] = ops

    def _game_of_life(
        self, env: simpy.Environment, mesh: WaferMesh, coords: list[Coord]
    ):
        packets = 0
        ops = 0
        for step in range(self.config.iterations):
            packets += yield from self._neighbor_exchange(
                env,
                mesh,
                coords,
                self.config.packet_payload_bytes,
                f"sdk:game_of_life:{step}",
            )
            step_ops = self.config.game_of_life_ops_per_pe * len(coords)
            ops += step_ops
            yield from self._compute_all(
                env, mesh, coords, self.config.game_of_life_ops_per_pe
            )
        self.example_results["game_of_life_packets"] = packets
        self.example_results["game_of_life_ops"] = ops

    def _broadcast(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        coords: list[Coord],
        bytes_per_message: int,
        kind: str,
    ):
        root = coords[0]
        events = [
            env.process(mesh.send(src=root, dst=coord, size_bytes=bytes_per_message, kind=kind))
            for coord in coords
            if coord != root
        ]
        if events:
            yield simpy.events.AllOf(env, events)
        return len(events)

    def _broadcast_and_reduce(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        coords: list[Coord],
        bytes_per_message: int,
        kind: str,
    ):
        root = coords[0]
        events = [
            env.process(
                mesh.send(src=root, dst=coord, size_bytes=bytes_per_message, kind=f"{kind}:bcast")
            )
            for coord in coords
            if coord != root
        ]
        events.extend(
            env.process(
                mesh.send(src=coord, dst=root, size_bytes=bytes_per_message, kind=f"{kind}:reduce")
            )
            for coord in coords
            if coord != root
        )
        if events:
            yield simpy.events.AllOf(env, events)
        return len(events)

    def _neighbor_exchange(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        coords: list[Coord],
        bytes_per_message: int,
        kind: str,
    ):
        coord_set = set(coords)
        events = []
        for coord in coords:
            for neighbor in mesh.neighbors(coord):
                if neighbor in coord_set:
                    events.append(
                        env.process(
                            mesh.send(
                                src=coord,
                                dst=neighbor,
                                size_bytes=bytes_per_message,
                                kind=kind,
                            )
                        )
                    )
        if events:
            yield simpy.events.AllOf(env, events)
        return len(events)

    def _iterative_solver_collectives(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        coords: list[Coord],
        kind: str,
        reductions_per_iteration: int,
    ):
        packets = 0
        for iteration in range(self.config.iterations):
            for reduction in range(reductions_per_iteration):
                packets += yield from self._broadcast_and_reduce(
                    env,
                    mesh,
                    coords,
                    self.config.packet_payload_bytes,
                    f"{kind}:{iteration}:{reduction}",
                )
        return packets

    def _all_to_all(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        coords: list[Coord],
        total_bytes: int,
        kind: str,
    ):
        chunk = max(1, total_bytes // max(1, len(coords)))
        events = [
            env.process(mesh.send(src=src, dst=dst, size_bytes=chunk, kind=kind))
            for src in coords
            for dst in coords
            if src != dst
        ]
        if events:
            yield simpy.events.AllOf(env, events)
        return len(events)


class CslRuntimeWorkload:
    """Integrated smoke workload for CSL builtins/library compatibility APIs."""

    def __init__(self, config: CslRuntimeWorkloadConfig) -> None:
        config.validate()
        self.config = config
        self.runtime_summary: dict[str, int | float | list[str]] = {}
        self.compat_results: dict[str, int | float | list[int] | str] = {}

    def run(self, env: simpy.Environment, mesh: WaferMesh):
        self._validate_mesh_shape(mesh)
        region = CslBuiltins.set_rectangle(
            self.config.active_rows, self.config.active_cols
        )
        coords = region.coords()
        runtime = CslCompatibilityRuntime(
            env=env,
            mesh=mesh,
            region=region,
            packet_payload_bytes=self.config.packet_payload_bytes,
        )

        zeros = CslBuiltins.zeros(4)
        constants = CslBuiltins.constant(4, 7)
        pointer = CslBuiltins.ptr(constants, 0)
        pointer.store(pointer.load() + 1)
        module_name = CslBuiltins.import_module("<math>")
        mapped_ids = [CslBuiltins.map_builtin(coord, region) for coord in coords]
        bitcast_value = CslBuiltins.bitcast_int(-1, 8)
        selected_value = CslBuiltins.select(True, 11, 22)
        concat_length = len(CslBuiltins.concat([1, 2], [3, 4]))
        range_sum = sum(CslBuiltins.range_values(4))

        random_values = [
            runtime.random.random_f32(-1.0, 1.0)
            for _ in range(self.config.math_calls)
        ]
        math_ops = runtime.math.estimate_ops(self.config.math_calls)
        math_result = sum(
            runtime.math.sqrt_f32(value * value)
            + runtime.math.cos_f32(value)
            + runtime.math.abs_f32(value)
            for value in random_values
        )
        complex_magnitude = runtime.complex.magnitude(
            runtime.complex.mul(1 + 2j, 3 + 4j)
        )
        flattened = runtime.data_utils.flatten([[1, 2], [3, 4]])
        reshaped = runtime.data_utils.reshape(flattened, 2, 2)
        chunks = runtime.data_utils.chunk(self.config.memcpy_bytes, self.config.packet_payload_bytes)
        moved_coord = runtime.directions.move(coords[0], Direction.SOUTH)
        opposite = runtime.directions.opposite(Direction.SOUTH).value
        dsd_indices = runtime.dsd_ops.strided(0, 4, 2)
        dsd_buffer = [0, 10, 20, 30, 40, 50, 60, 70]
        gathered = runtime.dsd_ops.gather(dsd_buffer, dsd_indices)
        runtime.dsd_ops.scatter(dsd_buffer, dsd_indices, [1, 2, 3, 4])
        runtime.empty.noop()
        string_length = runtime.string.length(runtime.string.concat("cs", "l"))
        type_size = runtime.types.sizeof("f32")
        clamped = runtime.types.clamp_to_unsigned(-1, 16)
        saxpy = runtime.kernels.saxpy(2.0, [1.0, 2.0], [3.0, 4.0])
        dot = runtime.kernels.dot([1.0, 2.0], [3.0, 4.0])
        fft_ops = runtime.fft.estimate_ops(8)
        bit_reverse_sum = sum(runtime.fft.bit_reverse_indices(8))
        tally = runtime.tally.histogram([0, 1, 1, 2, 5], 4)
        runtime.control.block_cmd_stream()
        runtime.control.unblock_cmd_stream()
        runtime.timer.start("compat")
        yield env.timeout(0)
        timer_elapsed = runtime.timer.elapsed("compat")
        runtime.malloc.malloc(self.config.malloc_bytes)
        runtime.debug.trace("math_result", coords[0], math_result)
        runtime.simprint.print(coords[0], "compat runtime started")

        fifo = runtime.tile_config.fifo(self.config.fifo_capacity)
        for item in range(self.config.fifo_capacity):
            yield fifo.put(item)
        for _ in range(self.config.fifo_capacity):
            yield fifo.get()

        selected = runtime.tile_config.filter_coords(
            coords, lambda coord: (coord[0] + coord[1]) % 2 == 0
        )
        switched = runtime.tile_config.switch_destinations(coords[0], coords)

        yield from runtime.memcpy.h2d(coords[-1], self.config.memcpy_bytes)
        yield from runtime.memcpy.d2h(coords[-1], self.config.memcpy_bytes)
        yield from runtime.memcpy.p2p(coords[0], coords[-1], self.config.memcpy_bytes)
        yield from runtime.collectives.broadcast(coords[0], self.config.collective_bytes)
        yield from runtime.collectives.reduce(coords[0], self.config.collective_bytes)
        yield from runtime.collectives.allreduce(coords[0], self.config.collective_bytes)
        yield from runtime.collectives.scatter(coords[0], self.config.collective_bytes)
        yield from runtime.collectives.gather(coords[0], self.config.collective_bytes)

        self.runtime_summary = runtime.summary()
        self.compat_results = {
            "zeros_length": len(zeros),
            "constant_first_value_after_pointer_store": constants[0],
            "imported_module": module_name,
            "sizeof_bytes": CslBuiltins.sizeof(16, 4),
            "bitcast_int_8_bits": bitcast_value,
            "selected_value": selected_value,
            "concat_length": concat_length,
            "range_sum": range_sum,
            "mapped_ids": mapped_ids,
            "selected_coords": len(selected),
            "switch_destinations": len(switched),
            "fifo_items": self.config.fifo_capacity,
            "math_ops": math_ops,
            "math_result": math_result,
            "complex_magnitude": complex_magnitude,
            "data_utils_chunks": len(chunks),
            "data_utils_reshaped_rows": len(reshaped),
            "direction_moved_row": moved_coord[0],
            "direction_opposite": opposite,
            "dsd_gather_sum": sum(gathered),
            "dsd_scatter_first": dsd_buffer[0],
            "string_length": string_length,
            "type_size_f32": type_size,
            "clamped_u16": clamped,
            "saxpy_sum": sum(saxpy),
            "dot_result": dot,
            "fft_ops": fft_ops,
            "fft_bit_reverse_sum": bit_reverse_sum,
            "tally_total": sum(tally),
            "collective_params_size": runtime.collective_params.size,
            "collective_rank_1_1": runtime.collective_pe.rank(coords[-1]),
            "timer_elapsed": timer_elapsed,
        }

    def summary(
        self, mesh: WaferMesh
    ) -> dict[str, int | float | str | dict[str, int | float | list[int] | str | list[str]]]:
        return {
            "type": "csl_runtime",
            "active_rows": self.config.active_rows,
            "active_cols": self.config.active_cols,
            "active_pes": self.config.active_pes,
            "packet_payload_bytes": self.config.packet_payload_bytes,
            "memcpy_bytes": self.config.memcpy_bytes,
            "collective_bytes": self.config.collective_bytes,
            "fifo_capacity": self.config.fifo_capacity,
            "math_calls": self.config.math_calls,
            "malloc_bytes": self.config.malloc_bytes,
            "compat_results": self.compat_results,
            "runtime_summary": self.runtime_summary,
        }

    def result_summary(
        self, mesh: WaferMesh, elapsed_cycles: float
    ) -> dict[str, float | int]:
        math_ops = self.compat_results.get("math_ops", 0)
        memcpy_bytes = self.runtime_summary.get("memcpy_bytes", 0)
        collective_bytes = self.runtime_summary.get("collective_bytes", 0)
        return {
            "csl_runtime_ops": math_ops if isinstance(math_ops, int | float) else 0,
            "csl_runtime_ops_per_cycle": (
                math_ops / elapsed_cycles
                if elapsed_cycles > 0 and isinstance(math_ops, int | float)
                else 0.0
            ),
            "csl_runtime_bytes": (
                (memcpy_bytes if isinstance(memcpy_bytes, int | float) else 0)
                + (collective_bytes if isinstance(collective_bytes, int | float) else 0)
            ),
        }

    def _validate_mesh_shape(self, mesh: WaferMesh) -> None:
        if self.config.active_rows > mesh.config.rows:
            raise ValueError("CSL runtime active rows exceed mesh rows")
        if self.config.active_cols > mesh.config.cols:
            raise ValueError("CSL runtime active cols exceed mesh cols")
