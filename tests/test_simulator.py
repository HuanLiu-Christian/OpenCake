from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import simpy

from wafersim.config import MeshConfig
from wafersim.csl import CslBuiltins, CslCompatibilityRuntime, Direction
from wafersim.csl import csl_compat_coverage
from wafersim.hardware.local_memcpy import (
    parse_local_memcpy_trace,
    run_memcpy_commands,
    run_local_memcpy_trace,
)
from wafersim.hardware.gemm import run_gemm_commands
from wafersim.hardware.gemv import run_gemv_commands
from wafersim.hardware.vector_dot import run_vector_dot_commands
from wafersim.hardware.network import WaferMesh, percentile
from wafersim.mapping import (
    read_gemm_commands,
    read_gemv_commands,
    read_memcpy_commands,
    read_vector_dot_commands,
)
from wafersim.mapping.gemm_trace_io import gemm_command_to_record
from wafersim.mapping.gemv_trace_io import gemv_command_to_record
from wafersim.mapping.trace_io import command_to_record
from wafersim.mapping.vector_dot_trace_io import vector_dot_command_to_record
from wafersim.opencake import (
    ChakraTrace,
    GemmWorkload,
    HardwareSpec,
    OpenCakeBackend,
    OpenCakeMapper,
    run_chakra_to_wafer_sim,
    run_formal_opencake_pipeline,
    validate_spatial_program,
    write_gemm_et_trace,
)


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


def protobuf_available() -> bool:
    try:
        import google.protobuf  # noqa: F401
    except ImportError:
        return False
    return True


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


class LocalMemcpyMappingTests(unittest.TestCase):
    def test_example_program_auto_generates_the_checked_in_memcpy_ir(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/pe_local_memcpy"
        run_module = load_module(example_dir / "run.py", "pe_local_memcpy_run")
        ir_path = Path(__file__).resolve().parent / "fixtures" / "pe_local_memcpy.jsonl"
        program = run_module.build_pe_local_memcpy_program()

        self.assertEqual(
            [command_to_record(command) for command in program.commands],
            [command_to_record(command) for command in read_memcpy_commands(ir_path)],
        )

    def test_pe_to_pe_program_auto_generates_and_runs_memcpy_ir(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/pe_to_pe_memcpy"
        run_module = load_module(example_dir / "run.py", "pe_to_pe_memcpy_run")
        ir_path = Path(__file__).resolve().parent / "fixtures" / "pe_to_pe_memcpy.jsonl"
        program = run_module.build_pe_to_pe_memcpy_program()
        commands = read_memcpy_commands(ir_path)
        result = run_memcpy_commands(
            program.commands,
            color_routes=run_module.COLOR_ROUTES,
        )

        self.assertEqual(
            [command_to_record(command) for command in program.commands],
            [command_to_record(command) for command in commands],
        )
        self.assertEqual([command.op for command in commands], [
            "h2d_write",
            "p2p_copy",
            "d2h_read",
            "expect",
            "expect",
            "expect",
            "expect",
        ])
        self.assertEqual(commands[1].color, "east_memcpy")
        self.assertIsNone(commands[1].dst_pe)
        self.assertEqual(result.final_srams[(0, 0)][4:8], (7, 8, 9, 10))
        self.assertEqual(result.final_srams[(0, 1)][12:16], (7, 8, 9, 10))
        self.assertEqual(result.d2h_outputs[0], (7, 8, 9, 10))

    def test_simpy_pe_local_memcpy_command_ir_is_the_mapping_source(self) -> None:
        ir_path = Path(__file__).resolve().parent / "fixtures" / "pe_local_memcpy.jsonl"
        commands = read_memcpy_commands(ir_path)
        result = run_memcpy_commands(commands)

        self.assertEqual(
            [command.op for command in commands],
            [
                "h2d_write",
                "local_copy",
                "d2h_read",
                "expect",
                "expect",
                "expect",
                "expect",
                "h2d_write",
                "d2h_read",
            ],
        )
        self.assertEqual(result.d2h_outputs[0], (11, 22, 33, 44))
        self.assertEqual(result.d2h_outputs[1], (101, 202, 303, 404))
        self.assertEqual(result.final_sram[20:24], (11, 22, 33, 44))
        self.assertEqual(result.memory_tasks_completed, 6)

    def test_legacy_text_trace_converts_to_memcpy_command_ir(self) -> None:
        trace_path = Path(__file__).resolve().parent / "fixtures" / "pe_local_memcpy.trace"
        ir_path = Path(__file__).resolve().parent / "fixtures" / "pe_local_memcpy.jsonl"
        steps = parse_local_memcpy_trace(trace_path)
        commands = read_memcpy_commands(ir_path)
        result = run_local_memcpy_trace(trace_path)

        self.assertEqual(steps, commands)
        self.assertEqual(result.d2h_outputs[0], (11, 22, 33, 44))
        self.assertEqual(result.d2h_outputs[1], (101, 202, 303, 404))
        self.assertEqual(result.final_sram[20:24], (11, 22, 33, 44))
        self.assertEqual(result.memory_tasks_completed, 6)


class GemvStreamingExampleTests(unittest.TestCase):
    def test_gemv_program_auto_generates_and_runs_cross_pe_ir(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/gemv_09_streaming"
        sys.modules.pop("layout", None)
        sys.modules.pop("pe_program", None)
        run_module = load_module(example_dir / "run.py", "gemv_ir_run")
        ir_path = Path(__file__).resolve().parent / "fixtures" / "gemv_09_streaming.jsonl"
        config = run_module.GemvStreamingConfig(
            matrix_rows=6,
            matrix_cols=8,
            kernel_rows=3,
            kernel_cols=4,
            bytes_per_element=4,
            packet_payload_bytes=64,
            ops_per_mac=2,
        )
        matrix = [index for index in range(config.matrix_rows * config.matrix_cols)]
        vector = [1 for _ in range(config.matrix_cols)]
        bias = [2 for _ in range(config.matrix_rows)]

        program = run_module.build_gemv_program(config, matrix, vector, bias)
        commands = read_gemv_commands(ir_path)
        result = run_gemv_commands(commands, ops_per_mac=config.ops_per_mac)

        self.assertEqual(
            [gemv_command_to_record(command) for command in program.commands],
            [gemv_command_to_record(command) for command in commands],
        )
        self.assertEqual(
            [command.op for command in commands],
            (["load_a_tile"] * 12)
            + (["load_x_chunk"] + ["forward_x_south"] * 2) * 4
            + (["load_bias"] * 3)
            + (["compute_ax"] * 12)
            + (["reduce_east"] * 9)
            + (["expect_y"] * 3),
        )
        self.assertEqual(result.result_y, (30, 94, 158, 222, 286, 350))
        self.assertEqual(result.expected_y, (30, 94, 158, 222, 286, 350))
        self.assertEqual(result.compute_tasks_completed, 12)
        self.assertEqual(result.memory_tasks_completed, 53)

    def test_python_native_gemv_example_files_drive_simulation(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/gemv_09_streaming"
        layout_module = load_module(example_dir / "layout.py", "gemv_layout")
        pe_program_module = load_module(example_dir / "pe_program.py", "gemv_pe_program")
        sys.modules.pop("layout", None)
        sys.modules.pop("pe_program", None)
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


class GemmStreamingExampleTests(unittest.TestCase):
    def test_gemm_program_auto_generates_and_runs_cross_pe_ir(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/gemm_streaming"
        sys.modules.pop("layout", None)
        sys.modules.pop("pe_program", None)
        run_module = load_module(example_dir / "run.py", "gemm_ir_run")
        ir_path = Path(__file__).resolve().parent / "fixtures" / "gemm_streaming.jsonl"
        config = run_module.GemmStreamingConfig(
            m=4,
            n=4,
            k=4,
            kernel_rows=2,
            kernel_cols=2,
            bytes_per_element=4,
            packet_payload_bytes=64,
            ops_per_mac=2,
        )
        left = [index for index in range(config.m * config.k)]
        right = [1 for _ in range(config.k * config.n)]

        program = run_module.build_gemm_program(config, left, right)
        commands = read_gemm_commands(ir_path)
        result = run_gemm_commands(
            commands,
            ops_per_mac=config.ops_per_mac,
            kernel_cols=config.kernel_cols,
        )

        self.assertEqual(
            [gemm_command_to_record(command) for command in program.commands],
            [gemm_command_to_record(command) for command in commands],
        )
        self.assertEqual([command.op for command in commands], [
            "load_a_panel",
            "forward_a_east",
            "load_a_panel",
            "forward_a_east",
            "load_b_panel",
            "forward_b_south",
            "load_b_panel",
            "forward_b_south",
            "compute_c",
            "compute_c",
            "compute_c",
            "compute_c",
            "expect_c",
            "expect_c",
            "expect_c",
            "expect_c",
        ])
        self.assertEqual(result.c_tiles[(0, 0)], (6, 6, 22, 22))
        self.assertEqual(result.c_tiles[(1, 1)], (38, 38, 54, 54))
        self.assertEqual(result.compute_tasks_completed, 4)
        self.assertEqual(result.memory_tasks_completed, 12)

    def test_python_native_gemm_example_files_drive_simulation(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/gemm_streaming"
        layout_module = load_module(example_dir / "layout.py", "gemm_layout")
        pe_program_module = load_module(example_dir / "pe_program.py", "gemm_pe_program")
        sys.modules.pop("layout", None)
        sys.modules.pop("pe_program", None)
        run_module = load_module(example_dir / "run.py", "gemm_run")

        summary = run_module.run_gemm_streaming(
            layout=layout_module.LAYOUT,
            problem=run_module.GemmProblemSpec(m=4, n=4, k=4),
        )

        self.assertEqual(pe_program_module.PE_PROGRAM.kernel, "gemm_streaming")
        self.assertEqual(summary["type"], "gemm_streaming_example")
        self.assertTrue(summary["verified"])
        self.assertEqual(summary["program"]["colors"]["a_color"], "east")
        self.assertEqual(summary["program"]["colors"]["b_color"], "south")
        self.assertEqual(summary["result_c_tiles"]["(0, 0)"], [6, 6, 22, 22])
        self.assertEqual(summary["result_c_tiles"]["(1, 1)"], [38, 38, 54, 54])


class VectorDotExampleTests(unittest.TestCase):
    def test_vector_dot_program_auto_generates_and_runs_compute_ir(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/vector_vector_dot"
        run_module = load_module(example_dir / "run.py", "vector_dot_ir_run")
        ir_path = Path(__file__).resolve().parent / "fixtures" / "vector_dot.jsonl"

        program = run_module.build_vector_dot_program(
            list(range(1, 17)),
            [1 for _ in range(16)],
        )
        commands = read_vector_dot_commands(ir_path)
        result = run_vector_dot_commands(commands)

        self.assertEqual(
            [vector_dot_command_to_record(command) for command in program.commands],
            [vector_dot_command_to_record(command) for command in commands],
        )
        self.assertEqual([command.op for command in commands], [
            "load_left",
            "load_right",
            "load_left",
            "load_right",
            "load_left",
            "load_right",
            "load_left",
            "load_right",
            "compute_dot",
            "compute_dot",
            "compute_dot",
            "compute_dot",
            "reduce_east",
            "reduce_east",
            "reduce_east",
            "expect",
        ])
        self.assertEqual(result.pe_results[(0, 0)], 10)
        self.assertEqual(result.pe_results[(0, 1)], 36)
        self.assertEqual(result.pe_results[(0, 2)], 78)
        self.assertEqual(result.pe_results[(0, 3)], 136)
        self.assertEqual(result.result, 136)
        self.assertEqual(result.expected, 136)
        self.assertEqual(result.compute_tasks_completed, 4)
        self.assertEqual(result.memory_tasks_completed, 14)

    def test_vector_dot_example_files_drive_simulation(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/vector_vector_dot"
        layout_module = load_module(example_dir / "layout.py", "vector_dot_layout")
        pe_program_module = load_module(example_dir / "pe_program.py", "vector_dot_pe_program")
        sys.modules.pop("layout", None)
        sys.modules.pop("pe_program", None)
        run_module = load_module(example_dir / "run.py", "vector_dot_run")
        layout = layout_module.LAYOUT
        pe_program = pe_program_module.PE_PROGRAM

        self.assertEqual(pe_program.kernel, "vector_vector_dot")
        self.assertEqual((layout.kernel_rows, layout.kernel_cols), (1, 4))
        self.assertEqual(layout.routes["partial_sum_color"], Direction.EAST)
        self.assertIn("compute_local_dot", [task.name for task in pe_program.tasks])
        self.assertTrue(hasattr(pe_program_module, "VectorDotPeProgram"))

        summary = run_module.run_vector_dot(
            layout=layout,
            config=pe_program_module.VectorDotConfig(vector_length=16, kernel_cols=4),
        )
        workload = summary["workload"]

        self.assertEqual(workload["type"], "vector_vector_dot_example")
        self.assertTrue(workload["verified"])
        self.assertEqual(workload["result_dot"], 136.0)
        self.assertEqual(workload["expected_dot"], 136.0)
        self.assertEqual(workload["local_partials"]["(0, 0)"], 10.0)
        self.assertEqual(workload["local_partials"]["(0, 3)"], 58.0)
        self.assertEqual(workload["color_directions"]["partial_sum_color"], "east")
        self.assertEqual(workload["color_packets"]["partial_sum_color"], 3)
        self.assertGreater(summary["vector_dot_ops_per_cycle"], 0.0)


class OpenCakePrototypeTests(unittest.TestCase):
    def test_chakra_trace_loads_gemm_workload(self) -> None:
        trace_path = Path(__file__).resolve().parents[1] / "examples/opencake_pipeline/chakra_gemm.jsonl"

        trace = ChakraTrace.read(trace_path)
        workload = trace.gemm_workload()

        self.assertEqual(len(trace.nodes), 4)
        self.assertEqual((workload.m, workload.n, workload.k), (1024, 1024, 1024))
        self.assertEqual(workload.total_ops, 2_147_483_648)

    @unittest.skipUnless(protobuf_available(), "protobuf is not installed in this Python environment")
    def test_real_chakra_et_trace_loads_gemm_workload(self) -> None:
        trace_path = Path("/tmp/opencake_chakra_gemm.et")

        write_gemm_et_trace(trace_path, GemmWorkload(m=64, n=32, k=16))
        trace = ChakraTrace.read(trace_path)
        workload = trace.gemm_workload()
        result = run_chakra_to_wafer_sim(
            chakra_trace=trace_path,
            hardware=HardwareSpec(mesh_rows=4, mesh_cols=4, pe_block_rows=2, pe_block_cols=2),
        )

        self.assertEqual(len(trace.nodes), 1)
        self.assertEqual((workload.m, workload.n, workload.k), (64, 32, 16))
        self.assertEqual(result.summary()["workload"]["total_ops"], 65_536)

    def test_mapper_lowers_gemm_to_pe_block_spatial_ir(self) -> None:
        workload = GemmWorkload(m=8, n=8, k=4)
        hardware = HardwareSpec(
            mesh_rows=4,
            mesh_cols=4,
            pe_block_rows=2,
            pe_block_cols=2,
            compute_ops_per_cycle_per_rank=64.0,
            sram_bandwidth_bytes_per_cycle_per_rank=32.0,
            link_bandwidth_bytes_per_cycle=16.0,
            router_latency_cycles=1.0,
            sram_bytes_per_rank=1024,
        )

        program = OpenCakeMapper().map_gemm(workload, hardware)
        records = program.records()
        result = OpenCakeBackend().run(program, hardware)

        self.assertEqual(program.workload["type"], "gemm")
        self.assertEqual(program.hardware["rank_rows"], 2)
        self.assertEqual(program.hardware["rank_cols"], 2)
        self.assertEqual(records[0], {"op": "define_color", "color": "a_east", "direction": "east"})
        self.assertEqual(records[1], {"op": "define_color", "color": "b_south", "direction": "south"})
        self.assertEqual(len([record for record in records if record["op"] == "compute"]), 4)
        self.assertEqual(len([record for record in records if record["op"] == "send"]), 4)
        self.assertEqual(result.ranks, 4)
        self.assertEqual(result.compute_ops, 512)
        self.assertEqual(result.compute_cycles, 2.0)
        self.assertEqual(result.memory_cycles, 6.0)
        self.assertEqual(result.network_cycles, 5.0)
        self.assertEqual(result.sim_time_cycles, 6.0)
        self.assertAlmostEqual(result.max_sram_utilization, 192 / 1024)
        self.assertGreater(result.compute_efficiency, 0.0)

    def test_formal_spatial_ir_schema_validation(self) -> None:
        program = OpenCakeMapper().map_gemm(
            GemmWorkload(m=8, n=8, k=4),
            HardwareSpec(mesh_rows=4, mesh_cols=4, pe_block_rows=2, pe_block_cols=2),
        )

        validation = validate_spatial_program(program)

        self.assertEqual(validation.schema, "opencake.spatial_ir.v1")
        self.assertEqual(validation.ops, 22)
        self.assertEqual(validation.ranks_with_allocations, 4)
        self.assertEqual(validation.colors, ("a_east", "b_south"))

    def test_opencake_gemm_example_writes_jsonl(self) -> None:
        example_dir = Path(__file__).resolve().parents[1] / "examples/opencake_gemm"
        run_module = load_module(example_dir / "run.py", "opencake_gemm_run")
        ir_path = Path("/tmp/opencake_gemm_test.jsonl")

        summary = run_module.run_opencake_gemm(
            m=8,
            n=8,
            k=4,
            mesh_rows=4,
            mesh_cols=4,
            pe_block_rows=2,
            pe_block_cols=2,
            ir_out=ir_path,
        )

        self.assertEqual(summary["workload"]["total_ops"], 512)
        self.assertEqual(summary["hardware"]["ranks"], 4)
        self.assertGreater(summary["spatial_ir_ops"], 0)
        self.assertTrue(ir_path.exists())
        self.assertEqual(ir_path.read_text(encoding="utf-8").splitlines()[0], '{"color": "a_east", "direction": "east", "op": "define_color"}')

    def test_full_chakra_to_wafer_sim_pipeline(self) -> None:
        trace_path = Path(__file__).resolve().parents[1] / "examples/opencake_pipeline/chakra_gemm.jsonl"
        spatial_ir_path = Path("/tmp/opencake_pipeline_spatial_ir.jsonl")

        result = run_chakra_to_wafer_sim(
            chakra_trace=trace_path,
            hardware=HardwareSpec(),
            spatial_ir_out=spatial_ir_path,
        )
        summary = result.summary()

        self.assertEqual(summary["chakra_nodes"], 4)
        self.assertEqual(summary["workload"]["type"], "gemm")
        self.assertEqual(summary["hardware"]["ranks"], 16)
        self.assertEqual(summary["spatial_ir_ops"], 90)
        self.assertEqual(summary["simulator"]["backend"], "opencake_astrasim_wafer_python")
        self.assertEqual(summary["simulator"]["sim_time_cycles"], 32768.0)
        self.assertTrue(spatial_ir_path.exists())
        self.assertEqual(len(spatial_ir_path.read_text(encoding="utf-8").splitlines()), 90)

    def test_formal_pipeline_exports_astrasim_package(self) -> None:
        trace_path = Path(__file__).resolve().parents[1] / "examples/opencake_pipeline/chakra_gemm.jsonl"
        out_dir = Path("/tmp/opencake_formal_package_test")

        result = run_formal_opencake_pipeline(
            chakra_trace=trace_path,
            out_dir=out_dir,
            hardware=HardwareSpec(),
        )
        summary = result.summary()
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        system = json.loads((out_dir / "system.json").read_text(encoding="utf-8"))
        reference = json.loads((out_dir / "reference_result.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["validation"]["schema"], "opencake.spatial_ir.v1")
        self.assertEqual(manifest["backend_contract"], "Astra-Sim C++ OpenCake wafer backend input package")
        self.assertEqual(system["rank_model"], "pe_block")
        self.assertEqual(system["ranks"], 16)
        self.assertIn("OpenCake2DMesh", (out_dir / "network.yml").read_text(encoding="utf-8"))
        self.assertEqual(reference["sim_time_cycles"], 32768.0)
        self.assertEqual(len((out_dir / "workload/opencake_spatial_ir.jsonl").read_text(encoding="utf-8").splitlines()), 90)

    def test_cpp_opencake_backend_matches_python_reference(self) -> None:
        trace_path = Path(__file__).resolve().parents[1] / "examples/opencake_pipeline/chakra_gemm.jsonl"
        backend_dir = Path(__file__).resolve().parents[1] / "cpp/opencake_wafer_backend"
        out_dir = Path("/tmp/opencake_cpp_backend_test_package")

        run_formal_opencake_pipeline(
            chakra_trace=trace_path,
            out_dir=out_dir,
            hardware=HardwareSpec(),
        )
        subprocess.run(["make"], cwd=backend_dir, check=True)
        completed = subprocess.run(
            [
                str(backend_dir / "build/opencake_wafer_backend"),
                "--package",
                str(out_dir),
                "--check-reference",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["backend"], "opencake_wafer_backend_cpp")
        self.assertEqual(result["sim_time_cycles"], 32768.0)
        self.assertEqual(result["compute_ops"], 2_147_483_648)


if __name__ == "__main__":
    unittest.main()
