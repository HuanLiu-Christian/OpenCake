from __future__ import annotations

import unittest

import simpy

from wafersim.config import CslDemoWorkloadConfig, MemoryCopyWorkloadConfig
from wafersim.config import MeshConfig, SimulationConfig
from wafersim.config import StencilWorkloadConfig
from wafersim.network import WaferMesh, percentile
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


if __name__ == "__main__":
    unittest.main()
