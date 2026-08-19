from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wafersim.contract import GemmMappingContract
from wafersim.specs import GemmSpec, WaferSpec
from wafersim.timeloop import generate_timeloop_inputs
from wafersim.wafer.simulator import simulate_gemm


class TimeLoopInputGenerationTest(unittest.TestCase):
    """Covers the deterministic tile-selection math in generate_timeloop_inputs.

    Deliberately does NOT call run_timeloop_mapper/parse_mapping_contract:
    those require a real `timeloop-mapper` running inside Docker (see
    wafersim/timeloop.py) -- there is no offline/synthetic fallback -- so
    this unit test suite doesn't assume Docker is available. generate_
    timeloop_inputs records the tile plan it decided on in mapping_plan.json,
    which is enough to verify the SRAM-fit logic on its own.
    """

    def test_respects_48_kib_sram(self) -> None:
        gemm = GemmSpec(m=1024, n=1024, k=1024)
        wafer = WaferSpec(mesh_rows=32, mesh_cols=32, sram_kib_per_pe=48)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = generate_timeloop_inputs(gemm, wafer, Path(directory))
            plan = json.loads(artifacts.mapping_plan_json.read_text(encoding="ascii"))

        # tile_k=128, not 64: single-buffered A/B (see contract.py's
        # data_sram_bytes_per_pe -- real Cerebras CSL task synchronization
        # does not prefetch a phase ahead, so this project no longer
        # reserves a second A/B slot) roughly doubles how much of the 48 KiB
        # budget is available per phase versus the old double-buffered
        # formula, so the largest SRAM-feasible divisor of K=1024 doubles too.
        self.assertEqual((plan["tile_m"], plan["tile_n"], plan["tile_k"]), (32, 32, 128))
        a_bytes = plan["tile_m"] * plan["tile_k"] * gemm.bytes_per_element
        b_bytes = plan["tile_k"] * plan["tile_n"] * gemm.bytes_per_element
        c_bytes = plan["tile_m"] * plan["tile_n"] * gemm.bytes_per_element
        data_sram_bytes_per_pe = c_bytes + a_bytes + b_bytes
        self.assertEqual(data_sram_bytes_per_pe, 36_864)
        self.assertLessEqual(data_sram_bytes_per_pe, wafer.sram_bytes_per_pe)


class WaferGemmPipelineTest(unittest.TestCase):
    def test_mesh_pipeline_models_broadcast_and_compute(self) -> None:
        gemm = GemmSpec(m=64, n=64, k=128)
        wafer = WaferSpec(
            mesh_rows=2,
            mesh_cols=2,
            sram_kib_per_pe=48,
            packet_payload_bytes=256,
            fifo_capacity_packets=8,
        )
        # local_compute_cycles=131072 is not arbitrary: it is tile_m *
        # tile_n * gemm.k (32 * 32 * 128), i.e. exactly what this
        # synthetic-but-real-shaped contract implies a real TimeLoop run
        # would report at a real 1-cycle-per-MAC rate (see
        # wafer/simulator.py's simulate_gemm -- it derives cycles_per_mac =
        # local_compute_cycles / (tile_m * tile_n * gemm.k), so this choice
        # makes that derived rate exactly 1.0, matching the real 1
        # cycle/MAC rate this project's architecture.yaml declares and a
        # real Docker run confirmed).
        contract = GemmMappingContract(
            gemm=gemm,
            wafer=wafer,
            active_rows=2,
            active_cols=2,
            tile_m=32,
            tile_n=32,
            tile_k=64,
            mapper_mode="test",
            map_path="test.map.yaml",
            stats_path="test.stats.txt",
            local_compute_cycles=131072,
            local_energy_uj=0.0,
            mac_area_um2=6628.0,
            pe_scratchpad_area_um2=82504.0,
        )
        result = simulate_gemm(contract)

        self.assertEqual(result.mac_operations, 64 * 64 * 128)
        # cycles_per_mac == 1.0 exactly by the local_compute_cycles choice
        # above, so the per-PE/per-phase compute timeline (which now comes
        # from that real-rate derivation, not an assumed throughput
        # constant) should sum to exactly the total MAC count, with no
        # rounding slack introduced anywhere.
        self.assertEqual(result.compute_busy_cycles, result.mac_operations)
        self.assertGreater(result.sim_time_cycles, 0)
        self.assertGreater(result.network["packet_hops"], 0)
        self.assertGreater(result.network["link_bytes"], 0)
        self.assertLessEqual(result.network["max_send_fifo_depth"], wafer.fifo_capacity_packets)
        self.assertGreater(result.energy.total_uj, 0)

        # Area: PE-internal comes straight from the (here, synthetic-but-
        # real-shaped) contract fields; interconnect defaults to 0 because
        # WaferSpec's router/link/fifo area coefficients default to 0.0 (see
        # specs.py) -- so active_pe_array_area_um2 should be exactly
        # pe_internal_area_um2 * active_pes, and die_pe_array_area_um2
        # exactly pe_internal_area_um2 * mesh_rows * mesh_cols, with no
        # interconnect contribution mixed in.
        self.assertEqual(contract.pe_internal_area_um2, 6628.0 + 82504.0)
        self.assertEqual(result.area.interconnect_area_um2_per_pe, 0.0)
        self.assertEqual(
            result.area.active_pe_array_area_um2,
            contract.pe_internal_area_um2 * contract.active_pes,
        )
        self.assertEqual(
            result.area.die_pe_array_area_um2,
            contract.pe_internal_area_um2 * wafer.mesh_rows * wafer.mesh_cols,
        )


if __name__ == "__main__":
    unittest.main()
