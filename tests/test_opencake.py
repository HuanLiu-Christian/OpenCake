from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from wafersim.opencake.backend import OpenCakeBackend
from wafersim.opencake.chakra import ChakraTrace, write_gemm_et_trace
from wafersim.opencake.frontend.mlir import read_linalg_matmul
from wafersim.opencake.mapper import OpenCakeMapper, SramAwareTiler
from wafersim.opencake.schema import SPATIAL_IR_SCHEMA_VERSION, validate_spatial_program
from wafersim.opencake.workload import GemmWorkload, HardwareSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


class CurrentOpenCakePipelineTest(unittest.TestCase):
    def test_default_workload_and_hardware_contract(self) -> None:
        workload = read_linalg_matmul(REPO_ROOT / "examples/mlir_gemm/matmul.mlir").to_workload()
        hardware = HardwareSpec()

        self.assertEqual((workload.m, workload.n, workload.k), (1024, 1024, 1024))
        self.assertEqual(workload.bytes_per_element, 4)
        self.assertEqual((hardware.rank_rows, hardware.rank_cols), (32, 32))
        self.assertEqual(hardware.ranks, 1024)
        self.assertEqual(hardware.rank_model, "pe")
        self.assertEqual(hardware.sram_bytes_per_rank, 48 * 1024)

    def test_current_32_cubed_tile_fits_local_sram(self) -> None:
        workload = GemmWorkload(m=1024, n=1024, k=1024)
        hardware = HardwareSpec()
        plan = SramAwareTiler().plan_for_tile(
            workload,
            hardware,
            tile_m=32,
            tile_n=32,
            tile_k=32,
        )

        self.assertEqual(plan.a_buffer_bytes, 4096)
        self.assertEqual(plan.b_buffer_bytes, 4096)
        self.assertEqual(plan.c_tile_bytes, 4096)
        self.assertEqual(plan.sram_bytes_per_rank, 20_480)
        self.assertAlmostEqual(plan.sram_utilization, 20_480 / 49_152)

    def test_small_spatial_ir_is_valid_and_timed(self) -> None:
        workload = GemmWorkload(m=4, n=4, k=4)
        hardware = HardwareSpec(
            mesh_rows=2,
            mesh_cols=2,
            sram_bytes_per_rank=1024,
            gemm_k_tile=4,
        )
        program = OpenCakeMapper().map_gemm(workload, hardware)
        validation = validate_spatial_program(program)
        result = OpenCakeBackend().run(program, hardware)

        self.assertEqual(validation.schema, SPATIAL_IR_SCHEMA_VERSION)
        self.assertEqual(validation.ranks_with_allocations, 4)
        self.assertEqual(validation.colors, ("a_east", "b_south"))
        self.assertEqual(result.compute_ops, workload.total_ops)
        self.assertTrue(result.sram_capacity_feasible)
        self.assertGreater(result.sim_time_cycles, 0)

    def test_chakra_et_round_trip_preserves_gemm_semantics(self) -> None:
        workload = GemmWorkload(m=1024, n=1024, k=1024)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gemm.et"
            write_gemm_et_trace(path, workload)
            recovered = ChakraTrace.read(path).gemm_workload()

        self.assertEqual(recovered, workload)

    def test_checked_in_chisel_fixture_uses_only_current_contract(self) -> None:
        fixture = REPO_ROOT / "tests/fixtures/opencake_chisel_gemm"
        config = json.loads((fixture / "chisel_config.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (fixture / "opencake_spatial_ir.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(config["schema"], SPATIAL_IR_SCHEMA_VERSION)
        self.assertEqual(config["workload"]["type"], "gemm")
        self.assertEqual(config["hardware"]["ranks"], 4)
        self.assertEqual(
            {record["op"] for record in records},
            {"define_color", "alloc", "send", "compute"},
        )


if __name__ == "__main__":
    unittest.main()
