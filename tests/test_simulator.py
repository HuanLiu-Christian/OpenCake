from __future__ import annotations

from pathlib import Path
import unittest

import simpy

from wafersim.config import CslDemoWorkloadConfig, CslGemmWorkloadConfig
from wafersim.config import CslGemvStreamingWorkloadConfig
from wafersim.config import CslRuntimeWorkloadConfig
from wafersim.config import MemoryCopyWorkloadConfig
from wafersim.config import MeshConfig, SimulationConfig
from wafersim.config import SdkExamplesWorkloadConfig
from wafersim.config import StencilWorkloadConfig
from wafersim.csl import CslBuiltins, CslCompatibilityRuntime, Direction
from wafersim.csl import csl_compat_coverage
from wafersim.csl import load_csl_example
from wafersim.hardware.network import WaferMesh, percentile
from wafersim.simulator import run_simulation, strip_internal_records


class NetworkTests(unittest.TestCase):
    def test_xy_route_uses_manhattan_hops(self) -> None:
        env = simpy.Environment()
        mesh = WaferMesh(env, MeshConfig(rows=4, cols=4))

        route = mesh.route_xy((0, 0), (2, 3))

        self.assertEqual(len(route), 5)
        self.assertEqual(route[0], ((0, 0), (0, 1)))
        self.assertEqual(route[-1], ((1, 3), (2, 3)))

    def test_single_packet_latency_matches_hops(self) -> None:
        config = MeshConfig(
            rows=2,
            cols=2,
            link_bandwidth_bytes_per_cycle=4.0,
            router_latency_cycles=1.0,
        )
        env = simpy.Environment()
        mesh = WaferMesh(env, config)
        env.run(until=env.process(mesh.send((0, 0), (1, 1), 8)))

        self.assertEqual(len(mesh.packet_records), 1)
        self.assertEqual(mesh.packet_records[0].hops, 2)
        self.assertEqual(mesh.packet_records[0].latency, 6.0)

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(percentile([0, 10], 95), 9.5)


class SimulatorTests(unittest.TestCase):
    def test_tiny_stencil_packet_count(self) -> None:
        summary = run_simulation(
            SimulationConfig(
                mesh=MeshConfig(rows=2, cols=2),
                workload=StencilWorkloadConfig(
                    steps=1,
                    tile_rows=2,
                    tile_cols=2,
                    ops_per_cell=1,
                    halo_bytes_per_edge=4,
                ),
            )
        )
        public_summary = strip_internal_records(summary)

        self.assertEqual(public_summary["pes"], 4)
        self.assertEqual(public_summary["compute_tasks_completed"], 4)
        self.assertEqual(public_summary["memory_tasks_completed"], 4)
        self.assertEqual(public_summary["packets"], 8)
        self.assertEqual(public_summary["bytes"], 32)
        self.assertEqual(public_summary["workload"]["memory_footprint_bytes_per_pe"], 48)

    def test_memory_copy_defaults_to_bottom_right(self) -> None:
        summary = run_simulation(
            SimulationConfig(
                mesh=MeshConfig(rows=4, cols=4),
                workload=MemoryCopyWorkloadConfig(
                    elements=16,
                    bytes_per_element=4,
                    packet_payload_bytes=16,
                ),
            )
        )
        public_summary = strip_internal_records(summary)

        self.assertEqual(public_summary["workload"]["source"], [0, 0])
        self.assertEqual(public_summary["workload"]["destination"], [3, 3])
        self.assertEqual(public_summary["workload"]["route_hops"], 6)
        self.assertEqual(public_summary["packets"], 4)
        self.assertEqual(public_summary["bytes"], 64)
        self.assertEqual(public_summary["memory_tasks_completed"], 2)
        self.assertGreater(public_summary["path_bandwidth_efficiency"], 0)

    def test_stencil_tile_must_fit_local_sram(self) -> None:
        with self.assertRaises(ValueError):
            run_simulation(
                SimulationConfig(
                    mesh=MeshConfig(rows=1, cols=1, local_sram_bytes_per_pe=16),
                    workload=StencilWorkloadConfig(
                        steps=1,
                        tile_rows=4,
                        tile_cols=4,
                        ops_per_cell=1,
                        halo_bytes_per_edge=0,
                    ),
                )
            )

    def test_csl_demo_runs_all_topic_models(self) -> None:
        summary = run_simulation(
            SimulationConfig(
                mesh=MeshConfig(rows=4, cols=4),
                workload=CslDemoWorkloadConfig(
                    active_rows=2,
                    active_cols=2,
                    elements_per_pe=16,
                    packet_payload_bytes=16,
                    collective_payload_bytes=16,
                    color_swap_payload_bytes=8,
                ),
            )
        )
        public_summary = strip_internal_records(summary)
        topic_results = public_summary["workload"]["topic_results"]

        self.assertEqual(public_summary["workload"]["type"], "csl_demo")
        self.assertEqual(public_summary["workload"]["active_pes"], 4)
        self.assertGreater(public_summary["packets"], 0)
        self.assertIn("topic_1_arrays_and_pointers_bytes", topic_results)
        self.assertIn("topic_9_fifo_items", topic_results)
        self.assertIn("topic_11_collective_packets", topic_results)
        self.assertIn("topic_15_microthread_tasks", topic_results)

    def test_sdk_examples_runs_later_example_models(self) -> None:
        summary = run_simulation(
            SimulationConfig(
                mesh=MeshConfig(rows=4, cols=4),
                workload=SdkExamplesWorkloadConfig(
                    active_rows=2,
                    active_cols=2,
                    stream_bytes=128,
                    packet_payload_bytes=32,
                    matrix_rows=32,
                    matrix_cols=32,
                    gemm_m=32,
                    gemm_n=32,
                    gemm_k=32,
                    vector_length=32,
                    tile_rows=4,
                    tile_cols=4,
                    iterations=2,
                    fft_points=32,
                ),
            )
        )
        public_summary = strip_internal_records(summary)
        example_results = public_summary["workload"]["example_results"]

        self.assertEqual(public_summary["workload"]["type"], "sdk_examples")
        self.assertGreater(public_summary["packets"], 0)
        self.assertIn("pipeline_1_redirect_packets", example_results)
        self.assertIn("gemv_collective_ops", example_results)
        self.assertIn("gemm_collective_ops", example_results)
        self.assertIn("game_of_life_ops", example_results)
        self.assertGreater(public_summary["sdk_examples_total_ops"], 0)

    def test_csl_gemm_uses_compat_runtime(self) -> None:
        summary = run_simulation(
            SimulationConfig(
                mesh=MeshConfig(rows=4, cols=4),
                workload=CslGemmWorkloadConfig(
                    m=64,
                    n=64,
                    k=64,
                    tile_m=16,
                    tile_n=16,
                    tile_k=16,
                    packet_payload_bytes=256,
                ),
            )
        )
        public_summary = strip_internal_records(summary)
        workload = public_summary["workload"]
        runtime_summary = workload["runtime_summary"]

        self.assertEqual(workload["type"], "csl_gemm")
        self.assertEqual(workload["active_pes"], 16)
        self.assertEqual(workload["total_ops"], 2 * 64 * 64 * 64)
        self.assertGreater(runtime_summary["memcpy_packets"], 0)
        self.assertGreater(public_summary["csl_gemm_compute_efficiency"], 0)

    def test_csl_gemv_streaming_color_pipeline(self) -> None:
        summary = run_simulation(
            SimulationConfig(
                mesh=MeshConfig(rows=3, cols=4),
                workload=CslGemvStreamingWorkloadConfig(
                    matrix_rows=6,
                    matrix_cols=8,
                    kernel_rows=3,
                    kernel_cols=4,
                    packet_payload_bytes=64,
                ),
            )
        )
        public_summary = strip_internal_records(summary)
        workload = public_summary["workload"]

        self.assertEqual(workload["type"], "csl_gemv_streaming")
        self.assertTrue(workload["verified"])
        self.assertEqual(workload["result_y"], [30.0, 94.0, 158.0, 222.0, 286.0, 350.0])
        self.assertEqual(workload["color_directions"]["x_color"], "south")
        self.assertEqual(workload["color_directions"]["ax_color_1"], "east")
        self.assertIn("x_color", workload["color_packets"])
        self.assertGreater(public_summary["csl_gemv_streaming_color_packets"], 0)

    def test_csl_style_gemv_example_files_drive_simulation(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/gemv_09_streaming"
        program = load_csl_example(example_dir)

        self.assertEqual(program.kernel, "gemv_09_streaming")
        self.assertEqual(program.rectangle, ("kernel_rows", "kernel_cols"))
        self.assertEqual(program.colors["x_color"], Direction.SOUTH)
        self.assertIn("compute_local_ax", program.tasks)

        summary = run_simulation(
            SimulationConfig(
                mesh=MeshConfig(
                    rows=program.int_constant("mesh_rows"),
                    cols=program.int_constant("mesh_cols"),
                ),
                workload=CslGemvStreamingWorkloadConfig(
                    matrix_rows=program.int_constant("matrix_rows"),
                    matrix_cols=program.int_constant("matrix_cols"),
                    kernel_rows=program.int_constant("kernel_rows"),
                    kernel_cols=program.int_constant("kernel_cols"),
                    bytes_per_element=program.int_constant("bytes_per_element"),
                    packet_payload_bytes=program.int_constant("packet_payload_bytes"),
                    ops_per_mac=program.int_constant("ops_per_mac"),
                ),
            )
        )
        public_summary = strip_internal_records(summary)

        self.assertTrue(public_summary["workload"]["verified"])
        self.assertGreater(public_summary["csl_gemv_streaming_color_packets"], 0)

    def test_csl_builtins_mapping(self) -> None:
        region = CslBuiltins.set_rectangle(2, 3)
        buffer = CslBuiltins.constant(2, 5)
        pointer = CslBuiltins.ptr(buffer, 1)

        pointer.store(9)

        self.assertEqual(CslBuiltins.import_module("<math>"), "math")
        self.assertEqual(CslBuiltins.map_builtin((1, 2), region), 5)
        self.assertEqual(CslBuiltins.zeros(3), [0, 0, 0])
        self.assertEqual(buffer, [5, 9])
        self.assertEqual(CslBuiltins.sizeof(8, 4), 32)
        self.assertEqual(CslBuiltins.bitcast_int(-1, 8), 255)

    def test_csl_runtime_libraries_mapping(self) -> None:
        env = simpy.Environment()
        mesh = WaferMesh(env, MeshConfig(rows=2, cols=2))
        region = CslBuiltins.set_rectangle(2, 2)
        runtime = CslCompatibilityRuntime(env, mesh, region, packet_payload_bytes=16)

        self.assertAlmostEqual(runtime.math.sqrt_f32(9.0), 3.0)
        self.assertTrue(-1.0 <= runtime.random.random_f32(-1.0, 1.0) <= 1.0)
        self.assertEqual(runtime.layout.get_linear_id((1, 1)), 3)
        self.assertEqual(runtime.layout.get_coord(2), (1, 0))
        self.assertEqual(runtime.malloc.malloc(32), 0)
        runtime.debug.trace("value", (0, 0), 7)
        runtime.simprint.print((0, 0), "hello")

        env.run(until=env.process(runtime.memcpy.p2p((0, 0), (1, 1), 32)))
        env.run(until=env.process(runtime.collectives.allreduce((0, 0), 8)))

        fifo = runtime.tile_config.fifo(2)
        selected = runtime.tile_config.filter_coords(region.coords(), lambda coord: coord[0] == 0)

        self.assertEqual(fifo.capacity, 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(runtime.memcpy.packets, 2)
        self.assertEqual(runtime.collectives.packets, 6)
        self.assertEqual(len(runtime.debug.events), 1)
        self.assertEqual(len(runtime.simprint.messages), 1)

    def test_extended_csl_library_mappings(self) -> None:
        env = simpy.Environment()
        mesh = WaferMesh(env, MeshConfig(rows=2, cols=2))
        region = CslBuiltins.set_rectangle(2, 2)
        runtime = CslCompatibilityRuntime(env, mesh, region, packet_payload_bytes=16)

        self.assertEqual(runtime.complex.mul(1 + 1j, 1 - 1j), 2 + 0j)
        runtime.control.block_cmd_stream()
        self.assertTrue(runtime.control.blocked)
        runtime.control.unblock_cmd_stream()
        self.assertFalse(runtime.control.blocked)
        self.assertEqual(runtime.data_utils.chunk(33, 16), [16, 16, 1])
        self.assertEqual(runtime.data_utils.reshape([1, 2, 3, 4], 2, 2), [[1, 2], [3, 4]])
        self.assertEqual(runtime.directions.move((1, 1), Direction.WEST), (1, 0))
        self.assertEqual(runtime.directions.opposite(Direction.NORTH), Direction.SOUTH)
        indices = runtime.dsd_ops.strided(0, 3, 2)
        buffer = [10, 20, 30, 40, 50]
        self.assertEqual(runtime.dsd_ops.gather(buffer, indices), [10, 30, 50])
        runtime.dsd_ops.scatter(buffer, indices, [1, 2, 3])
        self.assertEqual(buffer, [1, 20, 2, 40, 3])
        self.assertEqual(runtime.string.concat("wafer", "sim"), "wafersim")
        self.assertEqual(runtime.types.sizeof("f32"), 4)
        self.assertEqual(runtime.kernels.dot([1.0, 2.0], [3.0, 4.0]), 11.0)
        self.assertEqual(runtime.fft.estimate_ops(8), 120)
        self.assertEqual(sum(runtime.tally.histogram([0, 1, 1, 3], 4)), 4)
        self.assertEqual(runtime.collective_params.size, 4)
        self.assertEqual(runtime.collective_pe.rank((1, 1)), 3)

    def test_csl_compat_coverage_manifest_documents_scope(self) -> None:
        coverage = csl_compat_coverage()

        self.assertIn("<memcpy>", coverage["libraries"])
        self.assertIn("<collectives_2d>", coverage["libraries"])
        self.assertIn("<dsd_ops>", coverage["libraries"])
        self.assertIn("@set_tile_code / compiler code placement", coverage["builtins"])
        self.assertIn("omitted", coverage["builtins"]["@set_tile_code / compiler code placement"])

    def test_csl_runtime_workload_integrates_compat_layer(self) -> None:
        summary = run_simulation(
            SimulationConfig(
                mesh=MeshConfig(rows=2, cols=2),
                workload=CslRuntimeWorkloadConfig(
                    active_rows=2,
                    active_cols=2,
                    packet_payload_bytes=16,
                    memcpy_bytes=32,
                    collective_bytes=8,
                    fifo_capacity=2,
                    math_calls=4,
                    malloc_bytes=64,
                ),
            )
        )
        public_summary = strip_internal_records(summary)
        workload = public_summary["workload"]
        compat_results = workload["compat_results"]
        runtime_summary = workload["runtime_summary"]

        self.assertEqual(workload["type"], "csl_runtime")
        self.assertEqual(compat_results["imported_module"], "math")
        self.assertEqual(compat_results["constant_first_value_after_pointer_store"], 8)
        self.assertEqual(compat_results["direction_opposite"], "north")
        self.assertEqual(compat_results["type_size_f32"], 4)
        self.assertGreater(compat_results["fft_ops"], 0)
        self.assertEqual(runtime_summary["malloc_used_bytes"], 64)
        self.assertGreater(runtime_summary["memcpy_packets"], 0)
        self.assertGreater(runtime_summary["collective_packets"], 0)
        self.assertGreater(public_summary["csl_runtime_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
