from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from wafersim.contract import GemmMappingContract
from wafersim.specs import GemmSpec, WaferSpec
from wafersim.timeloop import (
    SAFE_CACTI_CACHE_SIZE_BYTES,
    _cacti_cache_size_bytes,
    _latest_fresh_docker_output,
    generate_timeloop_inputs,
)
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


class CactiCacheSizeFormulaTest(unittest.TestCase):
    """_cacti_cache_size_bytes must match the real accelergy-cacti-plug-in
    formula EXACTLY -- verified against the two real, confirmed data points
    from an actual Docker run (see SAFE_CACTI_CACHE_SIZE_BYTES's docstring
    in timeloop.py): fifo_capacity_packets=8 (fifo_buffer depth=16384,
    word-bits=8, block-size=1) produced a real, correctly-scaled CACTI call
    with cache_size=65536; fifo_capacity_packets=524288 (depth=1_073_741_824)
    would imply cache_size=2**32 exactly, past which CACTI silently broke."""

    def test_matches_the_real_confirmed_working_fifo_buffer_call(self) -> None:
        self.assertEqual(_cacti_cache_size_bytes(word_bits=8, block_size_bytes=1, depth_words=16384), 65536)

    def test_matches_the_real_confirmed_broken_fifo_buffer_call(self) -> None:
        self.assertEqual(
            _cacti_cache_size_bytes(word_bits=8, block_size_bytes=1, depth_words=1_073_741_824),
            2**32,
        )

    def test_matches_the_real_pe_scratchpad_call_from_the_48kib_default_run(self) -> None:
        # From the real fifo_router_in_pe_check run: pe_scratchpad
        # depth=8955 words, word-bits=8, block-size=gemm.bytes_per_element=4
        # -> real "Calling CACTI with cache_size=35820" (8955*4).
        self.assertEqual(_cacti_cache_size_bytes(word_bits=8, block_size_bytes=4, depth_words=8955), 35820)

    def test_width_floors_at_32_bits_for_narrow_components(self) -> None:
        # router_buffer/fifo_buffer declare width_bits=8 (word-bits=8,
        # block-size=1), below CACTI's own 32-bit scaled-width floor --
        # confirmed against a real run's router_buffer call: cache_size=256
        # regardless of a small depth, i.e. scaled_width stays 32 not 8.
        self.assertEqual(_cacti_cache_size_bytes(word_bits=8, block_size_bytes=1, depth_words=64), 256)


class GenerateTimeloopInputsCactiSafetyTest(unittest.TestCase):
    """generate_timeloop_inputs() must reject (in plain Python, before any
    real Docker call) any SRAM-class declaration whose implied CACTI
    cache_size exceeds SAFE_CACTI_CACHE_SIZE_BYTES -- this is the general,
    root-cause-level guard against the real silent-near-zero-area failure
    mode (see timeloop.py's SAFE_CACTI_CACHE_SIZE_BYTES docstring), and
    covers BOTH area_budget.py's automatic search (which stays under this
    ceiling by construction -- see its own _FIFO_CAPACITY_PACKETS_MAX/
    _SRAM_KIB_PER_PE_MAX derivation) AND a plain, direct
    --sram-kib/--fifo-packets CLI run, which area_budget.py's own bounds
    cannot constrain at all."""

    def test_default_wafer_spec_is_safe(self) -> None:
        gemm = GemmSpec(m=1024, n=1024, k=1024)
        wafer = WaferSpec(mesh_rows=32, mesh_cols=32)
        with tempfile.TemporaryDirectory() as directory:
            generate_timeloop_inputs(gemm, wafer, Path(directory))  # must not raise

    def test_fifo_capacity_far_past_the_safe_ceiling_raises_before_docker(self) -> None:
        gemm = GemmSpec(m=1024, n=1024, k=1024)
        # sram_kib_per_pe set generously large so WaferSpec.validate()'s own
        # "FIFO+router reserved bytes leave no room for GEMM data" check
        # does not fire first -- this test isolates the CACTI-safety check
        # on fifo_buffer specifically, not that unrelated pre-existing check.
        wafer = WaferSpec(mesh_rows=32, mesh_cols=32, sram_kib_per_pe=8192, fifo_capacity_packets=2049)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as ctx:
                generate_timeloop_inputs(gemm, wafer, Path(directory))
        self.assertIn("fifo_buffer", str(ctx.exception))

    def test_sram_kib_far_past_the_safe_ceiling_raises_before_docker(self) -> None:
        gemm = GemmSpec(m=1024, n=1024, k=1024)
        wafer = WaferSpec(mesh_rows=32, mesh_cols=32, sram_kib_per_pe=20_000)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as ctx:
                generate_timeloop_inputs(gemm, wafer, Path(directory))
        self.assertIn("pe_scratchpad", str(ctx.exception))

    def test_area_budgets_own_derived_fifo_ceiling_stays_safe(self) -> None:
        # Cross-module integration check: area_budget.py's own
        # _FIFO_CAPACITY_PACKETS_MAX (derived FROM SAFE_CACTI_CACHE_SIZE_BYTES)
        # must never itself trip this check -- if it ever drifted out of
        # sync, this would fail loudly in CI rather than silently at the
        # search's own artificial ceiling. sram_kib_per_pe is set generously
        # large (well above _SRAM_KIB_PER_PE_MAX) purely so WaferSpec.
        # validate()'s unrelated "no room for GEMM data" check does not fire
        # first -- see test_sram_far_past below for that ceiling in isolation.
        from wafersim.area_budget import _FIFO_CAPACITY_PACKETS_MAX

        gemm = GemmSpec(m=1024, n=1024, k=1024)
        wafer = WaferSpec(mesh_rows=32, mesh_cols=32, sram_kib_per_pe=16_384, fifo_capacity_packets=_FIFO_CAPACITY_PACKETS_MAX)
        with tempfile.TemporaryDirectory() as directory:
            generate_timeloop_inputs(gemm, wafer, Path(directory))  # must not raise

    def test_area_budgets_own_derived_sram_ceiling_stays_safe(self) -> None:
        # Same cross-module check for _SRAM_KIB_PER_PE_MAX, in isolation
        # (fifo_capacity_packets left at WaferSpec's own small real default).
        from wafersim.area_budget import _SRAM_KIB_PER_PE_MAX

        gemm = GemmSpec(m=1024, n=1024, k=1024)
        wafer = WaferSpec(mesh_rows=32, mesh_cols=32, sram_kib_per_pe=_SRAM_KIB_PER_PE_MAX)
        with tempfile.TemporaryDirectory() as directory:
            generate_timeloop_inputs(gemm, wafer, Path(directory))  # must not raise


class LatestFreshDockerOutputTest(unittest.TestCase):
    """Regression test for a real, confirmed bug (2026-08-23): re-running
    into an already-used --out-dir silently picked a STALE canonicalized
    copy (e.g. resolved.ART.yaml, left over from an EARLIER run into that
    same tile_k_<N>/area_budget_*_<N> directory) back up instead of the
    genuinely fresh file a just-completed real Docker invocation produced
    right next to it -- because the old `sorted(root.glob(pattern))[0]`
    lookup did not exclude the canonical destination path, and
    "resolved.*" sorts alphabetically before "timeloop-mapper.*"/
    "timeloop-model.*". Confirmed against real project files: a re-run's
    resolved.ART.yaml still held fifo_buffer area=1.0 from an earlier,
    unrelated fifo_capacity_packets config, while that same directory's
    fresh timeloop-model.ART.yaml correctly reported fifo_buffer
    area=65197.7 for the actual re-run -- and the parsed contract silently
    used the stale one. This test reproduces that exact directory layout
    with local files, no Docker needed: _latest_fresh_docker_output must
    exclude the canonical path outright and pick by mtime among what's
    left, so a stale leftover can never be mistaken for this run's output
    again, regardless of filename alphabetical order.
    """

    def test_excludes_the_canonical_path_even_when_it_sorts_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "resolved.ART.yaml"  # sorts before "timeloop-model..."
            fresh = root / "timeloop-model.ART.yaml"
            canonical.write_text("STALE from an earlier run", encoding="utf-8")
            # Ensure the fresh file's mtime is strictly later, matching a
            # real sequence of events (canonical copy already existed from
            # a prior run; Docker just wrote a new file after that).
            os.utime(canonical, (time.time() - 10, time.time() - 10))
            fresh.write_text("FRESH from this run", encoding="utf-8")

            found = _latest_fresh_docker_output(root, "*.ART.yaml", exclude=canonical)
            self.assertEqual(found, fresh)
            self.assertEqual(found.read_text(encoding="utf-8"), "FRESH from this run")

    def test_returns_none_when_only_the_canonical_path_matches(self) -> None:
        # This is exactly run_timeloop_model's real map.yaml situation: no
        # genuine fresh Docker output exists (timeloop-model never emits
        # its own *.map.yaml), only a stale resolved.map.yaml sitting at
        # the canonical path from an earlier run -- must return None (so
        # the caller correctly falls back), never that stale file.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "resolved.map.yaml"
            canonical.write_text("STALE from an earlier, unrelated run", encoding="utf-8")

            found = _latest_fresh_docker_output(root, "*.map.yaml", exclude=canonical)
            self.assertIsNone(found)

    def test_picks_the_most_recently_modified_among_multiple_fresh_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "resolved.stats.txt"
            older = root / "aaa.stats.txt"
            newer = root / "zzz.stats.txt"  # alphabetically last, but freshest
            older.write_text("older", encoding="utf-8")
            os.utime(older, (time.time() - 10, time.time() - 10))
            newer.write_text("newer", encoding="utf-8")

            found = _latest_fresh_docker_output(root, "*.stats.txt", exclude=canonical)
            self.assertEqual(found, newer)


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
        # interconnect_area_um2_per_pe is now ONLY the link term
        # (wafer.link_area_um2_per_pe, this WaferSpec's default -- a
        # reasoned real-cited estimate, no longer 0.0; see specs.py) since
        # FIFO/router area is already inside pe_internal_area_um2 -- so
        # active_pe_array_area_um2/die_pe_array_area_um2 both roll up
        # (pe_internal_area_um2 + interconnect_area_um2_per_pe) per PE, not
        # pe_internal_area_um2 alone (see wafer/area.py's area_um2_per_pe).
        self.assertEqual(contract.pe_internal_area_um2, 6628.0 + 82504.0 + 1200.0 + 400.0)
        self.assertEqual(result.area.interconnect_area_um2_per_pe, wafer.link_area_um2_per_pe)
        area_um2_per_pe = contract.pe_internal_area_um2 + wafer.link_area_um2_per_pe
        self.assertEqual(
            result.area.active_pe_array_area_um2,
            area_um2_per_pe * contract.active_pes,
        )
        self.assertEqual(
            result.area.die_pe_array_area_um2,
            area_um2_per_pe * wafer.mesh_rows * wafer.mesh_cols,
        )

        # Non-overlapping compute/memory/network re-bucketing of the same
        # five terms (see contract.py's compute_area_um2/memory_area_um2/
        # network_area_um2): compute=mac only, memory=pe_scratchpad only,
        # network=fifo+router+link -- so compute+memory+network must equal
        # pe_internal_area_um2 + link (i.e. area_um2_per_pe above), with no
        # term double-counted and no term dropped.
        self.assertEqual(contract.compute_area_um2, 6628.0)
        self.assertEqual(contract.memory_area_um2, 82504.0)
        self.assertEqual(contract.network_area_um2, 1200.0 + 400.0 + wafer.link_area_um2_per_pe)
        self.assertEqual(
            contract.compute_area_um2 + contract.memory_area_um2 + contract.network_area_um2,
            area_um2_per_pe,
        )
        self.assertEqual(contract.total_area_um2, area_um2_per_pe)
        self.assertAlmostEqual(
            contract.compute_area_fraction
            + contract.memory_area_fraction
            + contract.network_area_fraction,
            1.0,
        )
        self.assertAlmostEqual(
            contract.compute_area_fraction, contract.compute_area_um2 / area_um2_per_pe
        )


if __name__ == "__main__":
    unittest.main()
