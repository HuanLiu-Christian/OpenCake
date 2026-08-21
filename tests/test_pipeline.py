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

        # tile_k=64, not 128: FIFO/router now really reserve
        # wafer.fifo_router_bytes_per_pe (8*8*256 + 5*4 = 16,404 bytes, with
        # this WaferSpec's defaults) out of the same 48 KiB budget, per the
        # project's decision that these are not a separate/additional
        # allocation (see specs.py's fifo_router_bytes_per_pe and
        # timeloop.py's feasible_tile_k_candidates). That shrinks the
        # GEMM-data-only budget back down to 49,152 - 16,404 = 32,748 bytes,
        # so the largest SRAM-feasible divisor of K=1024 is 64 again (128
        # would need 36,864 data bytes, which no longer fits).
        self.assertEqual((plan["tile_m"], plan["tile_n"], plan["tile_k"]), (32, 32, 64))
        a_bytes = plan["tile_m"] * plan["tile_k"] * gemm.bytes_per_element
        b_bytes = plan["tile_k"] * plan["tile_n"] * gemm.bytes_per_element
        c_bytes = plan["tile_m"] * plan["tile_n"] * gemm.bytes_per_element
        gemm_data_bytes_per_pe = c_bytes + a_bytes + b_bytes
        self.assertEqual(gemm_data_bytes_per_pe, 20_480)
        data_sram_budget = wafer.sram_bytes_per_pe - wafer.fifo_router_bytes_per_pe
        self.assertLessEqual(gemm_data_bytes_per_pe, data_sram_budget)


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
            # Synthetic-but-real-shaped, mirroring the real fifo_buffer/
            # router_buffer SRAM components now declared per PE (see
            # timeloop.py's generate_timeloop_inputs) -- nonzero so this
            # test actually exercises pe_internal_area_um2 summing all
            # four terms, not just the pre-refactor two.
            fifo_area_um2=1200.0,
            router_area_um2=400.0,
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
        # real-shaped) contract fields -- now mac + pe_scratchpad +
        # fifo_area_um2 + router_area_um2, all four folded into one real
        # per-PE number (see contract.py's pe_internal_area_um2).
        # interconnect_area_um2_per_pe is now ONLY the link term, which
        # defaults to 0.0 (see specs.py's link_area_um2_per_pe) since FIFO/
        # router area is already inside pe_internal_area_um2 -- so
        # active_pe_array_area_um2 should be exactly pe_internal_area_um2 *
        # active_pes, and die_pe_array_area_um2 exactly pe_internal_area_um2
        # * mesh_rows * mesh_cols, with no separate interconnect
        # contribution mixed in.
        self.assertEqual(contract.pe_internal_area_um2, 6628.0 + 82504.0 + 1200.0 + 400.0)
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
