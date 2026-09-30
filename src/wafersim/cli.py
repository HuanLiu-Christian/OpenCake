"""Command-line entry point for the clean TimeLoop + wafer GEMM model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wafersim.area_budget import (
    MAX_SEARCH_ITERATIONS,
    AreaBudget,
    derive_wafer_spec_from_area_budget,
)
from wafersim.search import best_by_sim_time, search_tile_k
from wafersim.specs import GemmSpec, WaferSpec
from wafersim.timeloop import (
    generate_timeloop_inputs,
    parse_mapping_contract,
    run_timeloop_mapper,
)
from wafersim.wafer.simulator import simulate_gemm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TimeLoop-guided wafer-scale GEMM simulator")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--mesh-rows", type=int, default=32)
    parser.add_argument("--mesh-cols", type=int, default=32)
    parser.add_argument("--sram-kib", type=int, default=48)
    parser.add_argument("--link-bandwidth", type=float, default=32.0, help="bytes/cycle")
    parser.add_argument("--router-latency", type=float, default=1.0, help="cycles/hop")
    parser.add_argument("--packet-bytes", type=int, default=256)
    parser.add_argument("--fifo-packets", type=int, default=8)
    parser.add_argument(
        "--area-budget-um2",
        type=float,
        default=None,
        help=(
            "Physical die-area upper limit for the WHOLE mesh, in um^2 (e.g. "
            "200000000). When given (together with --compute-fraction/"
            "--memory-fraction/--network-fraction), --mesh-rows/--mesh-cols/"
            "--sram-kib/--fifo-packets are IGNORED: mesh size, per-PE SRAM "
            "capacity, and per-PE FIFO capacity are instead derived by a real, "
            "Docker-backed binary search for the largest hardware config that "
            "fits each category's percentage share of this budget (see "
            "wafersim/area_budget.py) -- this runs many real timeloop-model "
            "invocations (roughly 1 + 2*--area-search-iterations), so it is "
            "much slower than a direct --mesh-rows/--sram-kib run."
        ),
    )
    parser.add_argument(
        "--compute-fraction",
        type=float,
        default=None,
        help="Share of --area-budget-um2 given to compute (mac) area, 0..1.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=None,
        help="Share of --area-budget-um2 given to memory (pe_scratchpad) area, 0..1.",
    )
    parser.add_argument(
        "--network-fraction",
        type=float,
        default=None,
        help=(
            "Share of --area-budget-um2 given to network (fifo_buffer+"
            "router_buffer+link) area, 0..1. compute+memory+network fractions "
            "must not exceed 1.0."
        ),
    )
    parser.add_argument(
        "--area-search-iterations",
        type=int,
        default=MAX_SEARCH_ITERATIONS,
        help=(
            "Binary-search step cap per stage (memory, network) when "
            "--area-budget-um2 is given -- bounds real Docker call count to "
            "roughly 1 + 2*this value. The search range now needs roughly "
            "12 halvings in the worst case (sram_kib_per_pe/"
            "fifo_capacity_packets are bounded well below their old values -- "
            "see area_budget.py's _SAFE_CACHE_SIZE_BYTES derivation), so the "
            "default of 24 leaves headroom; don't lower this much below the "
            "default anyway, since fewer iterations can wrongly report 'no "
            "config fits the budget' when a real one exists but the search "
            "just ran out of steps before finding it (confirmed against a "
            "real run at the old, larger bounds)."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gemm"))
    parser.add_argument(
        "--timeloop-image", default="timeloopaccelergy/accelergy-timeloop-infrastructure:latest"
    )
    parser.add_argument(
        "--mode",
        choices=["search", "mapper"],
        default="search",
        help=(
            "'search' (default): OpenCake enumerates every SRAM-feasible tile_k "
            "and evaluates each with a real timeloop-model run, then picks the "
            "full-system (compute + network) optimum. 'mapper': the older single "
            "run of timeloop-mapper's own internal search (no network cost "
            "visibility, kept for comparison/debugging)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the complete result JSON")
    return parser


def _run_mapper_mode(gemm: GemmSpec, wafer: WaferSpec, args: argparse.Namespace) -> int:
    # mapspace_constraints.yaml (generated below) leaves the temporal K
    # split completely open: timeloop-mapper genuinely searches it
    # (bounded by the halved SRAM depth in architecture.yaml, see that
    # function's comments), and parse_mapping_contract() reads the real
    # chosen tile_k back out of the mapper's own .map.yaml -- not out of
    # mapping_plan.json. There is no fallback: if Docker/timeloop-mapper/
    # accelergy fails, both calls below raise instead of substituting a
    # locally-computed tiling. This mode has no visibility into network
    # cost -- it picks a tile_k purely from timeloop-mapper's own delay/
    # energy metric -- see 'search' mode for the network-cost-aware
    # OpenCake decision.
    artifacts = generate_timeloop_inputs(gemm, wafer, args.out_dir)
    run_timeloop_mapper(artifacts, image=args.timeloop_image)
    contract = parse_mapping_contract(artifacts, gemm, wafer)
    result = simulate_gemm(contract)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "mapping_contract.json").write_text(
        json.dumps(contract.to_dict(), indent=2) + "\n"
    )
    (args.out_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"mapper_mode: {contract.mapper_mode}")
    print(f"tile: {contract.tile_m} x {contract.tile_n} x {contract.tile_k}")
    print(f"K phases: {contract.k_phases}")
    print(
        "data SRAM/PE: "
        f"{contract.data_sram_bytes_per_pe} / {wafer.sram_bytes_per_pe} bytes"
    )
    print(f"sim_time_cycles: {result.sim_time_cycles:.0f}")
    print(f"mean_pe_utilization: {result.mean_pe_utilization:.4f}")
    print(f"packet_hops: {result.network['packet_hops']}")
    print(f"max_link_utilization: {result.network['max_link_utilization']:.4f}")
    print(f"total_energy_uj: {result.energy.total_uj:.4f}")
    _print_area(result)
    print(f"artifacts: {args.out_dir.resolve()}")
    return 0


def _print_area(result) -> None:
    # pe_internal_area_um2 (mac + pe_scratchpad + fifo_buffer + router_buffer,
    # all real Accelergy ART.yaml numbers -- see contract.py's
    # pe_internal_area_um2) is per ONE PE; interconnect_area_um2_per_pe is
    # now ONLY the inter-PE link/wire term, OpenCake's own analytical
    # coefficient (a reasoned estimate built from real-cited ITRS wire-pitch
    # + ORION 2.0 formula + Cerebras tile-area inputs, not an Accelergy
    # number -- see specs.py's link_area_um2_per_pe for the full derivation).
    # Both PE counts are shown side by side rather than blended into one
    # number -- active_pes is what THIS problem used, mesh_pes is the
    # physical chip.
    area = result.area
    print(
        "pe_internal_area_um2 (1 PE, real Accelergy -- mac+pe_scratchpad+"
        f"fifo_buffer+router_buffer): {area.pe_internal_area_um2:.2f}"
    )
    print(
        "interconnect_area_um2_per_pe (1 PE, OpenCake link-only "
        f"estimate): {area.interconnect_area_um2_per_pe:.2f}"
    )
    print(
        f"active_pe_array_area_um2 ({area.active_pes} PEs used by this run): "
        f"{area.active_pe_array_area_um2:.2f}"
    )
    print(
        f"die_pe_array_area_um2 ({area.mesh_pes} PEs on the physical wafer mesh): "
        f"{area.die_pe_array_area_um2:.2f}"
    )
    # Non-overlapping compute/memory/network re-bucketing of the same five
    # area terms (mac, pe_scratchpad, fifo_buffer, router_buffer, link) --
    # see contract.py's compute_area_um2/memory_area_um2/network_area_um2
    # property comments for exactly which term goes in which category.
    contract = result.contract
    print(
        "area by category (1 PE, no overlap, sums to 100%): "
        f"compute {contract.compute_area_um2:.2f} um^2 "
        f"({contract.compute_area_fraction * 100:.2f}%), "
        f"memory {contract.memory_area_um2:.2f} um^2 "
        f"({contract.memory_area_fraction * 100:.2f}%), "
        f"network {contract.network_area_um2:.2f} um^2 "
        f"({contract.network_area_fraction * 100:.2f}%)"
    )


def _run_search_mode(gemm: GemmSpec, wafer: WaferSpec, args: argparse.Namespace) -> int:
    # OpenCake owns the decision here: every SRAM-feasible tile_k gets one
    # real timeloop-model run (TimeLoop's real, non-fabricated per-
    # candidate Cycles/Energy), then simulate_gemm() adds the full-system
    # network cost TimeLoop cannot see, and best_by_sim_time() picks the
    # global optimum by that full-system number -- not by TimeLoop's
    # per-candidate metric alone.
    candidates = search_tile_k(gemm, wafer, args.out_dir, image=args.timeloop_image)
    winner = best_by_sim_time(candidates)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "candidates.json").write_text(
        json.dumps([c.to_dict() for c in candidates], indent=2) + "\n"
    )
    (args.out_dir / "mapping_contract.json").write_text(
        json.dumps(winner.result.contract.to_dict(), indent=2) + "\n"
    )
    (args.out_dir / "result.json").write_text(
        json.dumps(winner.result.to_dict(), indent=2) + "\n"
    )

    if args.json:
        print(json.dumps(winner.result.to_dict(), indent=2))
        return 0

    print(f"evaluated {len(candidates)} SRAM-feasible tile_k candidates via real timeloop-model:")
    header = f"{'tile_k':>8}  {'k_phases':>8}  {'sim_time_cycles':>16}  {'total_energy_uj':>16}  {'timeloop_cycles':>16}"
    print(header)
    for c in candidates:
        marker = " <-- best" if c is winner else ""
        d = c.to_dict()
        print(
            f"{d['tile_k']:>8}  {d['k_phases']:>8}  {d['sim_time_cycles']:>16.0f}  "
            f"{d['total_energy_uj']:>16.4f}  {d['local_compute_cycles']:>16}{marker}"
        )
    print()
    contract = winner.result.contract
    result = winner.result
    print(f"best tile_k: {contract.tile_k}")
    print(f"mapper_mode: {contract.mapper_mode}")
    print(f"tile: {contract.tile_m} x {contract.tile_n} x {contract.tile_k}")
    print(f"K phases: {contract.k_phases}")
    print(
        "data SRAM/PE: "
        f"{contract.data_sram_bytes_per_pe} / {wafer.sram_bytes_per_pe} bytes"
    )
    print(f"sim_time_cycles: {result.sim_time_cycles:.0f}")
    print(f"mean_pe_utilization: {result.mean_pe_utilization:.4f}")
    print(f"packet_hops: {result.network['packet_hops']}")
    print(f"max_link_utilization: {result.network['max_link_utilization']:.4f}")
    print(f"total_energy_uj: {result.energy.total_uj:.4f}")
    _print_area(result)
    print(f"artifacts: {args.out_dir.resolve()}")
    return 0


def _validate_mesh_fits_gemm(gemm: GemmSpec, wafer: WaferSpec) -> None:
    """Pre-flight check for the direct --mesh-rows/--mesh-cols CLI path,
    same discipline as timeloop.py's _validate_cacti_safe_sram: catch a
    real TimeLoop hard constraint in plain Python before spending a real
    Docker call on a config TimeLoop will reject anyway.

    timeloop.py's generate_timeloop_inputs() pins active_rows/active_cols =
    min(wafer.mesh_rows, gemm.m)/min(wafer.mesh_cols, gemm.n) as the DRAM
    spatial M/N fanout (architecture_constraints.yaml), and TimeLoop
    requires the product of every level's factor for a dimension to equal
    that dimension's total size exactly -- no boundary/remainder tiles.
    Two real, Docker-confirmed failure modes follow from that:
      - active_rows/active_cols must evenly divide gemm.m/gemm.n, or a real
        run fails with "product of all factors of dimension M is X, which
        is not equal to the dimension size of the workload Y".
      - active_rows/active_cols must be >= 2, or a real run fails with
        "cannot find spatial tiling level associated with storage level
        DRAM ... the number of instances of the next-inner level
        (pe_scratchpad) is the same as this level" (TimeLoop can't tell a
        spatial fanout of 1 apart from no spatial level at all).

    Only the direct --mesh-rows/--mesh-cols path needs this: --area-budget-
    um2's own mesh search (area_budget.py's _derive_mesh) only ever
    proposes mesh_rows/mesh_cols pairs that are already true divisors of
    gemm.m/gemm.n with both >= 2, by construction -- it cannot produce an
    invalid mesh here.
    """
    active_rows = min(wafer.mesh_rows, gemm.m)
    active_cols = min(wafer.mesh_cols, gemm.n)
    problems = []
    if active_rows < 2:
        problems.append(
            f"--mesh-rows={wafer.mesh_rows} gives an effective row count of "
            f"{active_rows} (= min(mesh_rows, --m={gemm.m})) -- TimeLoop "
            "requires at least 2; a real Docker run confirmed a 1-wide "
            "spatial dimension fails with 'cannot find spatial tiling level "
            "associated with storage level DRAM'"
        )
    if active_cols < 2:
        problems.append(
            f"--mesh-cols={wafer.mesh_cols} gives an effective column count "
            f"of {active_cols} (= min(mesh_cols, --n={gemm.n})) -- TimeLoop "
            "requires at least 2, same real failure mode as above"
        )
    if gemm.m % active_rows:
        problems.append(
            f"--mesh-rows={wafer.mesh_rows} (effective row count "
            f"{active_rows}) does not evenly divide --m={gemm.m} -- a real "
            "Docker run confirmed TimeLoop rejects this with \"product of "
            "all factors of dimension M is ..., which is not equal to the "
            "dimension size of the workload\"; pick a --mesh-rows that "
            "divides --m (or a --mesh-rows >= --m, which is capped down to "
            "--m and always divides it)"
        )
    if gemm.n % active_cols:
        problems.append(
            f"--mesh-cols={wafer.mesh_cols} (effective column count "
            f"{active_cols}) does not evenly divide --n={gemm.n} -- same "
            "real TimeLoop failure mode as above for dimension N; pick a "
            "--mesh-cols that divides --n (or a --mesh-cols >= --n)"
        )
    if problems:
        raise SystemExit(
            "invalid --mesh-rows/--mesh-cols for this GEMM shape -- this would "
            "otherwise fail deep inside a real Docker timeloop-mapper/-model "
            "call with an opaque TimeLoop error instead of failing fast here:\n"
            "  - " + "\n  - ".join(problems)
        )


def _derive_wafer_from_area_budget(gemm: GemmSpec, args: argparse.Namespace) -> WaferSpec:
    if None in (args.compute_fraction, args.memory_fraction, args.network_fraction):
        raise SystemExit(
            "--area-budget-um2 requires --compute-fraction, --memory-fraction, "
            "and --network-fraction to all be given"
        )
    budget = AreaBudget(
        wafer_area_budget_um2=args.area_budget_um2,
        compute_area_fraction=args.compute_fraction,
        memory_area_fraction=args.memory_fraction,
        network_area_fraction=args.network_fraction,
    )
    # Every OTHER field (technology_node, clock_ghz, router_ports_per_pe/
    # router_port_bytes, link_area_um2_per_pe, ...) comes from WaferSpec's
    # own real-cited defaults, matched by link_bandwidth/router_latency/
    # packet_bytes here so the CLI's other flags still apply -- only
    # mesh_rows/mesh_cols/sram_kib_per_pe/fifo_capacity_packets are
    # overwritten by the search below (see area_budget.py).
    base_wafer = WaferSpec(
        link_bandwidth_bytes_per_cycle=args.link_bandwidth,
        router_latency_cycles=args.router_latency,
        packet_payload_bytes=args.packet_bytes,
    )
    print(
        f"deriving hardware config from a {args.area_budget_um2:.2f} um^2 area budget "
        f"(compute {args.compute_fraction * 100:.2f}%, memory {args.memory_fraction * 100:.2f}%, "
        f"network {args.network_fraction * 100:.2f}%) -- this runs many real "
        f"timeloop-model Docker invocations, this may take a while..."
    )
    result = derive_wafer_spec_from_area_budget(
        budget,
        gemm,
        base_wafer,
        args.out_dir,
        max_iterations=args.area_search_iterations,
        image=args.timeloop_image,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "area_budget.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    print(
        f"derived mesh: {result.wafer.mesh_rows} x {result.wafer.mesh_cols} "
        f"({result.compute_pes} PEs, real mac_area_um2={result.mac_area_um2:.2f}, "
        f"compute area used {result.compute_area_used_um2:.2f} / "
        f"{budget.compute_area_budget_um2:.2f} um^2 budget)"
    )
    print(
        f"derived sram_kib_per_pe: {result.wafer.sram_kib_per_pe} "
        f"(real pe_scratchpad_area_um2={result.memory_area_used_um2:.2f} / "
        f"{budget.memory_area_budget_um2 / result.compute_pes:.2f} um^2 per-PE budget)"
    )
    print(
        f"derived fifo_capacity_packets: {result.wafer.fifo_capacity_packets} "
        f"(real fifo+router+link area_um2={result.network_area_used_um2:.2f} / "
        f"{budget.network_area_budget_um2 / result.compute_pes:.2f} um^2 per-PE budget)"
    )
    if result.saturation_warning is not None:
        print(f"WARNING: {result.saturation_warning}", file=sys.stderr)
    return result.wafer


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gemm = GemmSpec(m=args.m, n=args.n, k=args.k)
    gemm.validate()

    if args.area_budget_um2 is not None:
        wafer = _derive_wafer_from_area_budget(gemm, args)
    else:
        wafer = WaferSpec(
            mesh_rows=args.mesh_rows,
            mesh_cols=args.mesh_cols,
            sram_kib_per_pe=args.sram_kib,
            link_bandwidth_bytes_per_cycle=args.link_bandwidth,
            router_latency_cycles=args.router_latency,
            packet_payload_bytes=args.packet_bytes,
            fifo_capacity_packets=args.fifo_packets,
        )
        # Only the direct --mesh-rows/--mesh-cols path needs this check --
        # see _validate_mesh_fits_gemm's docstring for why the
        # --area-budget-um2 path (_derive_wafer_from_area_budget above) is
        # inherently safe from this without any extra guard.
        _validate_mesh_fits_gemm(gemm, wafer)
    wafer.validate()

    if args.mode == "mapper":
        return _run_mapper_mode(gemm, wafer, args)
    return _run_search_mode(gemm, wafer, args)


if __name__ == "__main__":
    raise SystemExit(main())
