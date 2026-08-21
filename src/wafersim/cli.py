"""Command-line entry point for the clean TimeLoop + wafer GEMM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    # now ONLY the inter-PE link/wire term, OpenCake's own placeholder
    # coefficient (0 by default -- see specs.py's link_area_um2_per_pe).
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
        f"placeholder): {area.interconnect_area_um2_per_pe:.2f}"
    )
    print(
        f"active_pe_array_area_um2 ({area.active_pes} PEs used by this run): "
        f"{area.active_pe_array_area_um2:.2f}"
    )
    print(
        f"die_pe_array_area_um2 ({area.mesh_pes} PEs on the physical wafer mesh): "
        f"{area.die_pe_array_area_um2:.2f}"
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gemm = GemmSpec(m=args.m, n=args.n, k=args.k)
    wafer = WaferSpec(
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        sram_kib_per_pe=args.sram_kib,
        link_bandwidth_bytes_per_cycle=args.link_bandwidth,
        router_latency_cycles=args.router_latency,
        packet_payload_bytes=args.packet_bytes,
        fifo_capacity_packets=args.fifo_packets,
    )
    gemm.validate()
    wafer.validate()

    if args.mode == "mapper":
        return _run_mapper_mode(gemm, wafer, args)
    return _run_search_mode(gemm, wafer, args)


if __name__ == "__main__":
    raise SystemExit(main())
