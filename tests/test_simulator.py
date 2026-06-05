from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import simpy

from wafersim.config import MeshConfig
from wafersim.csl import CslBuiltins, CslCompatibilityRuntime, Direction
from wafersim.csl import csl_compat_coverage
from wafersim.hardware.network import WaferMesh, percentile


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


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


class CslCompatibilityTests(unittest.TestCase):
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

    def test_csl_compat_coverage_manifest_documents_scope(self) -> None:
        coverage = csl_compat_coverage()

        self.assertIn("<memcpy>", coverage["libraries"])
        self.assertIn("<collectives_2d>", coverage["libraries"])
        self.assertIn("<dsd_ops>", coverage["libraries"])
        self.assertIn("@set_tile_code / compiler code placement", coverage["builtins"])
        self.assertIn("omitted", coverage["builtins"]["@set_tile_code / compiler code placement"])


class GemvStreamingExampleTests(unittest.TestCase):
    def test_python_native_gemv_example_files_drive_simulation(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/gemv_09_streaming"
        layout_module = load_module(example_dir / "layout.py", "gemv_layout")
        pe_program_module = load_module(example_dir / "pe_program.py", "gemv_pe_program")
        run_module = load_module(example_dir / "run.py", "gemv_run")
        layout = layout_module.LAYOUT
        pe_program = pe_program_module.PE_PROGRAM

        self.assertEqual(pe_program.kernel, "gemv_09_streaming")
        self.assertEqual((layout.kernel_rows, layout.kernel_cols), (3, 4))
        self.assertEqual(layout.routes["x_color"], Direction.SOUTH)
        self.assertIn("compute_local_ax", [task.name for task in pe_program.tasks])
        self.assertTrue(hasattr(pe_program_module, "GemvStreamingPeProgram"))

        summary = run_module.run_gemv_streaming(
            layout=layout,
            problem=run_module.GemvProblemSpec(matrix_rows=6, matrix_cols=8),
        )
        workload = summary["workload"]

        self.assertEqual(workload["type"], "gemv_09_streaming_example")
        self.assertTrue(workload["verified"])
        self.assertEqual(workload["result_y"], [30.0, 94.0, 158.0, 222.0, 286.0, 350.0])
        self.assertEqual(workload["color_directions"]["x_color"], "south")
        self.assertEqual(workload["color_directions"]["ax_color_1"], "east")
        self.assertIn("x_color", workload["color_packets"])
        self.assertGreater(summary["gemv_09_streaming_color_packets"], 0)


if __name__ == "__main__":
    unittest.main()
