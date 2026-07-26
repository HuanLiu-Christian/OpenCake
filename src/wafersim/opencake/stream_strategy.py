from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import ceil, isqrt
import os
from pathlib import Path
import re
import subprocess
from time import monotonic
from typing import Any

from wafersim.opencake.frontend.tiled_mlir import TiledGemmLoopNest
from wafersim.opencake.mapper.tiler import GemmTilePlan
from wafersim.opencake.workload import GemmWorkload, HardwareSpec


_LINK_RE = re.compile(r"Link\s+(\d+)\N{RIGHTWARDS ARROW}(\d+)")
_TENSOR_RE = re.compile(r"Transfer\(([^)]+)\)")


@dataclass(frozen=True, slots=True)
class StreamRouteHop:
    tensor: str
    src_core: int
    dst_core: int
    start_cycle: float
    duration_cycles: float
    direction: str
    color: str


@dataclass(frozen=True, slots=True)
class StreamStrategyResult:
    """Normalized result produced by the real Stream/ZigZag optimizer."""

    schema: str
    tool: str
    solver: str
    total_latency_cycles: float
    compute_cores: tuple[int, ...]
    core_coordinates: dict[int, tuple[int, int]]
    route_hops: tuple[StreamRouteHop, ...]
    allocation: dict[str, Any]
    wafer_constraint: str
    constraint_stats: dict[str, dict[str, int]]
    ideal_a_east_b_south: bool
    limitation: str
    target_mesh: tuple[int, int] = (0, 0)
    tile_sizes: tuple[int, int, int] = (0, 0, 0)
    route_family: str = "unspecified"
    solve_time_seconds: float = 0.0
    trace_window_cycles: float = 0.0

    @classmethod
    def read(cls, path: str | Path) -> StreamStrategyResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            schema=str(data["schema"]),
            tool=str(data["tool"]),
            solver=str(data["solver"]),
            total_latency_cycles=float(data["total_latency_cycles"]),
            compute_cores=tuple(int(core) for core in data["compute_cores"]),
            core_coordinates={
                int(core): (int(coord[0]), int(coord[1]))
                for core, coord in data["core_coordinates"].items()
            },
            route_hops=tuple(StreamRouteHop(**hop) for hop in data["route_hops"]),
            allocation=dict(data["allocation"]),
            wafer_constraint=str(data["wafer_constraint"]),
            constraint_stats={
                str(tensor): {str(key): int(value) for key, value in stats.items()}
                for tensor, stats in data["constraint_stats"].items()
            },
            ideal_a_east_b_south=bool(data["ideal_a_east_b_south"]),
            limitation=str(data["limitation"]),
            target_mesh=tuple(int(value) for value in data.get("target_mesh", (0, 0))),
            tile_sizes=tuple(int(value) for value in data.get("tile_sizes", (0, 0, 0))),
            route_family=str(data.get("route_family", "unspecified")),
            solve_time_seconds=float(data.get("solve_time_seconds", 0.0)),
            trace_window_cycles=float(data.get("trace_window_cycles", 0.0)),
        )

    def summary(self) -> dict[str, object]:
        directions: dict[str, set[str]] = {}
        for hop in self.route_hops:
            if hop.direction == "offchip":
                continue
            directions.setdefault(hop.tensor, set()).add(hop.direction)
        return {
            "schema": self.schema,
            "tool": self.tool,
            "solver": self.solver,
            "total_latency_cycles": self.total_latency_cycles,
            "latency_metric": "tetra_steady_state_cycles_per_iteration",
            "trace_window_cycles": self.trace_window_cycles,
            "compute_cores": list(self.compute_cores),
            "tensor_directions": {
                tensor: sorted(values) for tensor, values in sorted(directions.items())
            },
            "route_hops": [asdict(hop) for hop in self.route_hops],
            "wafer_constraint": self.wafer_constraint,
            "constraint_stats": self.constraint_stats,
            "ideal_a_east_b_south": self.ideal_a_east_b_south,
            "limitation": self.limitation,
            "target_mesh": list(self.target_mesh),
            "tile_sizes": list(self.tile_sizes),
            "route_family": self.route_family,
            "solve_time_seconds": self.solve_time_seconds,
        }


@dataclass(frozen=True, slots=True)
class StreamWaferPackage:
    """Files consumed by the official Stream/TETRA pipeline."""

    root: Path
    contract: Path
    hardware: Path
    mapping: Path
    compute_core: Path
    offchip_core: Path

    def summary(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "tiled_mlir_contract": str(self.contract),
            "hardware": str(self.hardware),
            "mapping": str(self.mapping),
            "compute_core": str(self.compute_core),
            "offchip_core": str(self.offchip_core),
        }


class StreamWaferInputBuilder:
    """Translate a materialized tiled-MLIR loop nest into Stream inputs.

    The builder describes resources and legal placement domains only. It does
    not score streaming directions or synthesize a schedule; those decisions
    remain inside Stream's TETRA/GSCIP pipeline.
    """

    def build(
        self,
        loop_nest: TiledGemmLoopNest,
        hardware: HardwareSpec,
        output_dir: str | Path,
    ) -> StreamWaferPackage:
        workload = GemmWorkload(
            m=loop_nest.output_shape[0],
            n=loop_nest.output_shape[1],
            k=loop_nest.k_upper,
            bytes_per_element=loop_nest.bytes_per_element,
        )
        loop_nest.validate_against(workload, hardware)
        loop_nest.to_tile_plan(workload, hardware)
        if (loop_nest.grid_rows, loop_nest.grid_cols) != (
            hardware.rank_rows,
            hardware.rank_cols,
        ):
            raise ValueError(
                "formal Stream mapping requires the transformed tiled grid to equal "
                f"the active rank grid; got {(loop_nest.grid_rows, loop_nest.grid_cols)} "
                f"and {(hardware.rank_rows, hardware.rank_cols)}"
            )

        root = Path(output_dir)
        cores = root / "cores"
        cores.mkdir(parents=True, exist_ok=True)
        contract = root / "tiled_gemm_contract.json"
        hardware_path = root / "wafer_mesh.yaml"
        mapping_path = root / "gemm_mapping.yaml"
        compute_path = cores / "wafer_pe.yaml"
        offchip_path = cores / "offchip.yaml"

        contract.write_text(
            json.dumps(
                {
                    "schema": "opencake.stream_tiled_gemm_input.v1",
                    "workload": {
                        "m": workload.m,
                        "n": workload.n,
                        "k": workload.k,
                        "bytes_per_element": workload.bytes_per_element,
                    },
                    "tiled_loop_nest": loop_nest.summary(),
                    "target_mesh": [loop_nest.grid_rows, loop_nest.grid_cols],
                    "rank_model": hardware.rank_model,
                    "rank_to_pe": "1:1" if hardware.pes_per_rank == 1 else "1:PE-block",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self._write_yaml(
            compute_path,
            self._compute_core(hardware, workload.bytes_per_element),
        )
        self._write_yaml(
            offchip_path,
            self._offchip_core(workload.bytes_per_element),
        )
        self._write_yaml(hardware_path, self._hardware(hardware))
        self._write_yaml(mapping_path, self._mapping(loop_nest))
        return StreamWaferPackage(
            root=root,
            contract=contract,
            hardware=hardware_path,
            mapping=mapping_path,
            compute_core=compute_path,
            offchip_core=offchip_path,
        )

    @staticmethod
    def _hardware(hardware: HardwareSpec) -> dict[str, object]:
        rows, cols = hardware.rank_rows, hardware.rank_cols
        core_id = lambda row, col: row * cols + col
        offchip = rows * cols
        cores = {index: "./cores/wafer_pe.yaml" for index in range(offchip)}
        cores[offchip] = "./cores/offchip.yaml"
        coordinates = {
            core_id(row, col): [col, row]
            for row in range(rows)
            for col in range(cols)
        }
        coordinates[offchip] = [-1, -1]
        links: list[dict[str, object]] = []
        for row in range(rows):
            for col in range(cols):
                here = core_id(row, col)
                if col + 1 < cols:
                    links.append(
                        {
                            "type": "link",
                            "cores": [here, core_id(row, col + 1)],
                            "bandwidth": hardware.link_bandwidth_bytes_per_cycle,
                        }
                    )
                if row + 1 < rows:
                    links.append(
                        {
                            "type": "link",
                            "cores": [here, core_id(row + 1, col)],
                            "bandwidth": hardware.link_bandwidth_bytes_per_cycle,
                        }
                    )
                if row in {0, rows - 1} or col in {0, cols - 1}:
                    links.append(
                        {
                            "type": "link",
                            "cores": [offchip, here],
                            "bandwidth": hardware.link_bandwidth_bytes_per_cycle,
                        }
                    )
        return {
            "name": f"opencake_wafer_{rows}x{cols}",
            "cores": cores,
            "core_coordinates": coordinates,
            "offchip_core_id": offchip,
            "unit_energy_cost": 0,
            "core_connectivity": links,
        }

    @staticmethod
    def _write_yaml(path: Path, document: dict[str, object]) -> None:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - optional integration dependency
            raise RuntimeError(
                "Stream input generation requires PyYAML; install wafersim[stream]"
            ) from error
        path.write_text(
            yaml.safe_dump(document, sort_keys=False),
            encoding="utf-8",
        )

    @staticmethod
    def _mapping(loop_nest: TiledGemmLoopNest) -> dict[str, object]:
        cores = list(range(loop_nest.grid_rows * loop_nest.grid_cols))
        return {
            "layers": [
                {
                    "name": "TiledGemm",
                    "core_allocation": [cores],
                    "inter_core_tiling": [
                        [
                            {"dim": "D0", "split": loop_nest.grid_rows},
                            {"dim": "D2", "split": loop_nest.grid_cols},
                        ]
                    ],
                }
            ],
            "fused_groups": [
                {
                    "name": "Fused_Group_1",
                    "layers": ["TiledGemm"],
                    "intra_core_tiling": [
                        {"dim": "TiledGemm.D1", "tile": loop_nest.tile_k}
                    ],
                }
            ],
        }

    @staticmethod
    def _compute_core(hardware: HardwareSpec, bytes_per_element: int) -> dict[str, object]:
        precision = bytes_per_element * 8
        macs = max(1, int(hardware.compute_ops_per_cycle_per_rank // 2))
        array_rows = max(1, isqrt(macs))
        while macs % array_rows:
            array_rows -= 1
        array_cols = macs // array_rows
        return {
            "name": "opencake_wafer_pe",
            "type": "zigzag.compute",
            "memories": {
                "operand_rf": {
                    "size": max(precision, macs * precision),
                    "r_cost": 0,
                    "w_cost": 0,
                    "area": 0,
                    "latency": 1,
                    "auto_cost_extraction": False,
                    "operands": ["I1", "I2", "O"],
                    "ports": [
                        {
                            "name": "rw_port_1",
                            "type": "read_write",
                            "bandwidth_min": precision,
                            "bandwidth_max": max(precision, macs * precision),
                            "allocation": [
                                f"{operand}, {transfer}"
                                for operand in ("I1", "I2", "O")
                                for transfer in ("fh", "tl", "fl", "th")
                            ],
                        }
                    ],
                    "served_dimensions": [],
                },
                "local_sram_48kib": {
                    "size": hardware.sram_bytes_per_rank * 8,
                    "r_cost": 1,
                    "w_cost": 1,
                    "area": 0,
                    "latency": 1,
                    "auto_cost_extraction": False,
                    "operands": ["I1", "I2", "O"],
                    "ports": [
                        {
                            "name": "rw_port_1",
                            "type": "read_write",
                            "bandwidth_min": precision,
                            "bandwidth_max": int(
                                hardware.sram_bandwidth_bytes_per_cycle_per_rank * 8
                            ),
                            "allocation": [
                                f"{operand}, {transfer}"
                                for operand in ("I1", "I2", "O")
                                for transfer in ("fh", "tl", "fl", "th")
                            ],
                        }
                    ],
                    "served_dimensions": ["D1", "D2"],
                },
            },
            "operational_array": {
                "unit_energy": 0,
                "unit_area": 1,
                "dimensions": ["D1", "D2"],
                "sizes": [array_rows, array_cols],
            },
        }

    @staticmethod
    def _offchip_core(bytes_per_element: int) -> dict[str, object]:
        precision = bytes_per_element * 8
        return {
            "name": "opencake_offchip",
            "type": "zigzag.offchip",
            "memories": {
                "dram": {
                    "size": 1 << 50,
                    "r_cost": 0,
                    "w_cost": 0,
                    "area": 0,
                    "latency": 1,
                    "operands": ["I1", "I2", "O"],
                    "ports": [
                        {
                            "name": "rw_port_1",
                            "type": "read_write",
                            "bandwidth_min": precision,
                            "bandwidth_max": 1024,
                            "allocation": [
                                f"{operand}, {transfer}"
                                for operand in ("I1", "I2", "O")
                                for transfer in ("fh", "tl", "fl", "th")
                            ],
                        }
                    ],
                    "served_dimensions": ["D1", "D2"],
                }
            },
            "operational_array": {
                "unit_energy": 0,
                "unit_area": 0,
                "dimensions": ["D1", "D2"],
                "sizes": [0, 0],
            },
        }


@dataclass(frozen=True, slots=True)
class GemmStreamAllocationContract:
    """Exact full-mesh strategy selected by Stream/TETRA AllocationIR."""

    schema: str
    source: str
    solver: str
    status: str
    target_mesh: tuple[int, int]
    tile_m: int
    tile_n: int
    tile_k: int
    ab_buffer_count: int
    a_color: str
    b_color: str
    a_direction: str
    b_direction: str
    a_injection_boundary: str
    b_injection_boundary: str
    objective_cycles: float
    trace_window_cycles: float
    optimality_gap: float
    solve_time_seconds: float
    route_family: str
    allocation: dict[str, Any]
    route_hops: tuple[StreamRouteHop, ...]

    @classmethod
    def from_stream_result(
        cls,
        result: StreamStrategyResult,
        *,
        buffer_count: int,
    ) -> GemmStreamAllocationContract:
        if result.target_mesh == (0, 0) or result.tile_sizes == (0, 0, 0):
            raise ValueError("Stream result lacks exact tiled-mesh metadata")
        a_direction, b_direction = result.route_family.split("-")[1::2]
        if a_direction not in {"east", "west"} or b_direction not in {"north", "south"}:
            raise ValueError(f"invalid Stream GEMM route family: {result.route_family}")
        return cls(
            schema="opencake.gemm_stream_allocation_contract.v1",
            source="KULeuven-MICAS Stream/TETRA AllocationIR",
            solver=result.solver,
            status="feasible",
            target_mesh=result.target_mesh,
            tile_m=result.tile_sizes[0],
            tile_n=result.tile_sizes[1],
            tile_k=result.tile_sizes[2],
            ab_buffer_count=buffer_count,
            a_color=f"a_{a_direction}",
            b_color=f"b_{b_direction}",
            a_direction=a_direction,
            b_direction=b_direction,
            a_injection_boundary="west" if a_direction == "east" else "east",
            b_injection_boundary="north" if b_direction == "south" else "south",
            objective_cycles=result.total_latency_cycles,
            trace_window_cycles=result.trace_window_cycles,
            optimality_gap=-1.0,
            solve_time_seconds=result.solve_time_seconds,
            route_family=result.route_family,
            allocation=result.allocation,
            route_hops=result.route_hops,
        )

    @property
    def tile_sizes(self) -> tuple[int, int, int]:
        return self.tile_m, self.tile_n, self.tile_k

    def tile_plan(self, workload: GemmWorkload, hardware: HardwareSpec) -> GemmTilePlan:
        plan = GemmTilePlan(
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            ab_buffer_count=self.ab_buffer_count,
            a_buffer_bytes=self.tile_m * self.tile_k * workload.bytes_per_element,
            b_buffer_bytes=self.tile_k * self.tile_n * workload.bytes_per_element,
            c_tile_bytes=self.tile_m * self.tile_n * workload.bytes_per_element,
            sram_bytes_per_rank=(
                self.tile_m * self.tile_n
                + self.ab_buffer_count
                * (self.tile_m * self.tile_k + self.tile_k * self.tile_n)
            )
            * workload.bytes_per_element,
            sram_utilization=0.0,
        )
        plan = GemmTilePlan(
            **{
                **asdict(plan),
                "sram_utilization": plan.sram_bytes_per_rank / hardware.sram_bytes_per_rank,
            }
        )
        if plan.sram_bytes_per_rank > hardware.sram_bytes_per_rank:
            raise ValueError("Stream-selected tiled contract exceeds PE-local SRAM")
        expected = (ceil(workload.m / self.tile_m), ceil(workload.n / self.tile_n))
        if expected != self.target_mesh:
            raise ValueError(f"Stream target mesh {self.target_mesh} does not match {expected}")
        return plan

    def validate_target_mesh(self, rows: int, cols: int) -> None:
        if (rows, cols) != self.target_mesh:
            raise ValueError(
                f"materialized tiled mesh {(rows, cols)} differs from Stream AllocationIR "
                f"mesh {self.target_mesh}; proof-mesh scaling is forbidden"
            )

    def summary(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source": self.source,
            "solver": self.solver,
            "status": self.status,
            "target_mesh": list(self.target_mesh),
            "tile_sizes": list(self.tile_sizes),
            "ab_buffer_count": self.ab_buffer_count,
            "a": {
                "direction": self.a_direction,
                "color": self.a_color,
                "injection_boundary": self.a_injection_boundary,
            },
            "b": {
                "direction": self.b_direction,
                "color": self.b_color,
                "injection_boundary": self.b_injection_boundary,
            },
            "objective_cycles": self.objective_cycles,
            "objective_metric": "tetra_steady_state_cycles_per_iteration",
            "trace_window_cycles": self.trace_window_cycles,
            "optimality_gap": self.optimality_gap,
            "solve_time_seconds": self.solve_time_seconds,
            "route_family": self.route_family,
            "allocation_ir_consumed": True,
            "route_hops": len(self.route_hops),
        }


class StreamStrategyRunner:
    """Launch an external official Stream installation and normalize its result.

    Stream is intentionally optional. The main wafersim environment remains
    lightweight; callers provide the Python executable from a Stream virtual
    environment and the checkout root used as its working directory.
    """

    def __init__(self, *, python: str | Path, stream_root: str | Path) -> None:
        self.python = Path(python)
        self.stream_root = Path(stream_root)

    def run(
        self,
        *,
        hardware: str | Path,
        mapping: str | Path,
        output_dir: str | Path,
        matrix_size: int = 4,
        wafer_constraint: str = "a-east-b-south",
    ) -> StreamStrategyResult:
        return self._run_worker(
            hardware=hardware,
            mapping=mapping,
            output_dir=output_dir,
            matrix_size=matrix_size,
            wafer_constraint=wafer_constraint,
            contract=None,
        )

    def run_tiled_gemm(
        self,
        *,
        loop_nest: TiledGemmLoopNest,
        hardware: HardwareSpec,
        output_dir: str | Path,
        route_families: tuple[str, ...] = (
            "a-east-b-south",
            "a-east-b-north",
            "a-west-b-south",
            "a-west-b-north",
        ),
    ) -> tuple[GemmStreamAllocationContract, StreamWaferPackage, tuple[StreamStrategyResult, ...]]:
        """Run full Stream/TETRA for each legal wafer route family.

        OpenCake defines only the legal target route families. Stream performs
        allocation, transfer-plan selection, and steady-state scheduling for
        each family. The selected contract is the result with minimum latency
        reported by Stream itself; no OpenCake analytical score participates.
        """

        root = Path(output_dir)
        package = StreamWaferInputBuilder().build(loop_nest, hardware, root / "input")
        results: list[StreamStrategyResult] = []
        for route_family in route_families:
            family_output = root / route_family
            results.append(
                self._run_worker(
                    hardware=package.hardware,
                    mapping=package.mapping,
                    output_dir=family_output,
                    matrix_size=loop_nest.output_shape[0],
                    wafer_constraint=route_family,
                    contract=package.contract,
                )
            )
        selected = min(results, key=lambda result: result.total_latency_cycles)
        contract = GemmStreamAllocationContract.from_stream_result(
            selected,
            buffer_count=2 if hardware.double_buffer_ab else 1,
        )
        (root / "selected_stream_allocation.json").write_text(
            json.dumps(contract.summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return contract, package, tuple(results)

    def _run_worker(
        self,
        *,
        hardware: str | Path,
        mapping: str | Path,
        output_dir: str | Path,
        matrix_size: int,
        wafer_constraint: str,
        contract: str | Path | None,
    ) -> StreamStrategyResult:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "stream_strategy.json"
        command = [
            str(self.python),
            "-m",
            "wafersim.opencake.stream_strategy",
            "--worker",
            "--hardware",
            str(Path(hardware).resolve()),
            "--mapping",
            str(Path(mapping).resolve()),
            "--output-dir",
            str(output.resolve()),
            "--result",
            str(result_path.resolve()),
            "--matrix-size",
            str(matrix_size),
            "--wafer-constraint",
            wafer_constraint,
        ]
        if contract is not None:
            command.extend(("--contract", str(Path(contract).resolve())))
        worker_env = os.environ.copy()
        worker_env["MPLCONFIGDIR"] = str(output / ".matplotlib")
        worker_env["XDG_CACHE_HOME"] = str(output / ".cache")
        project_src = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = worker_env.get("PYTHONPATH")
        worker_env["PYTHONPATH"] = (
            project_src
            if not existing_pythonpath
            else os.pathsep.join((project_src, existing_pythonpath))
        )
        completed = subprocess.run(
            tuple(command),
            cwd=self.stream_root,
            check=False,
            capture_output=True,
            text=True,
            env=worker_env,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Stream/ZigZag strategy worker failed:\n"
                + completed.stdout
                + completed.stderr
            )
        if not result_path.exists():
            raise RuntimeError(
                "Stream completed without writing a strategy result:\n"
                + completed.stdout
                + completed.stderr
            )
        return StreamStrategyResult.read(result_path)


def _direction(
    src_core: int,
    dst_core: int,
    coordinates: dict[int, tuple[int, int]],
) -> str:
    if src_core not in coordinates or dst_core not in coordinates:
        return "offchip"
    src_col, src_row = coordinates[src_core]
    dst_col, dst_row = coordinates[dst_core]
    delta = (dst_col - src_col, dst_row - src_row)
    directions = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}
    try:
        return directions[delta]
    except KeyError as error:
        raise ValueError(f"non-neighbor Stream route hop {src_core}->{dst_core}: {delta}") from error


def wafer_path_satisfies_constraint(
    tensor: str,
    links: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    coordinates: dict[int, tuple[int, int]],
    offchip_core: int,
    constraint: str = "a-east-b-south",
) -> bool:
    """Check a physical multicast tree against wafer directional constraints."""

    if constraint == "none" or tensor not in {"A", "B"}:
        return True
    match = re.fullmatch(r"a-(east|west)-b-(north|south)", constraint)
    if match is None:
        raise ValueError(f"unknown wafer constraint: {constraint}")
    a_direction, b_direction = match.groups()
    a_source_col = min(col for col, unused_row in coordinates.values()) if a_direction == "east" else max(
        col for col, unused_row in coordinates.values()
    )
    b_source_row = min(row for unused_col, row in coordinates.values()) if b_direction == "south" else max(
        row for unused_col, row in coordinates.values()
    )

    for src_core, dst_core in links:
        if src_core == offchip_core:
            if dst_core not in coordinates:
                return False
            col, row = coordinates[dst_core]
            if tensor == "A" and col != a_source_col:
                return False
            if tensor == "B" and row != b_source_row:
                return False
            continue
        if dst_core == offchip_core:
            return False
        direction = _direction(src_core, dst_core, coordinates)
        if tensor == "A" and direction != a_direction:
            return False
        if tensor == "B" and direction != b_direction:
            return False
    return True


def parse_stream_route_hops(
    trace: dict[str, Any],
    coordinates: dict[int, tuple[int, int]],
) -> tuple[StreamRouteHop, ...]:
    hops: dict[tuple[str, int, int, float, float], StreamRouteHop] = {}
    for event in trace.get("traceEvents", []):
        args = event.get("args", {})
        if args.get("iteration") != "i" or not str(event.get("cat", "")).startswith("transfer"):
            continue
        tensor_match = _TENSOR_RE.search(str(event.get("name", "")))
        link_match = _LINK_RE.search(str(args.get("this_link", "")))
        if tensor_match is None or link_match is None:
            continue
        tensor = tensor_match.group(1).split("_")[0]
        src_core, dst_core = (int(value) for value in link_match.groups())
        start = float(event["ts"])
        duration = float(event["dur"])
        direction = _direction(src_core, dst_core, coordinates)
        color = f"{tensor.lower()}_{direction}"
        key = (tensor, src_core, dst_core, start, duration)
        hops[key] = StreamRouteHop(
            tensor=tensor,
            src_core=src_core,
            dst_core=dst_core,
            start_cycle=start,
            duration_cycles=duration,
            direction=direction,
            color=color,
        )
    return tuple(sorted(hops.values(), key=lambda hop: (hop.start_cycle, hop.tensor, hop.src_core, hop.dst_core)))


def _worker(args: argparse.Namespace) -> None:
    # Imports live in the worker so wafersim does not require Stream at runtime.
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, shape_inference
    import yaml

    from stream.api import optimize_allocation_co_with_mapping
    from stream.cost_model.communication_manager import CommunicationManager
    from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
    from stream.ir import AllocationIR
    from stream.workload.workload import Workload

    started = monotonic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.contract is not None:
        contract_data = json.loads(Path(args.contract).read_text(encoding="utf-8"))
        workload_data = contract_data["workload"]
        m, n, k = (int(workload_data[name]) for name in ("m", "n", "k"))
        target_mesh = tuple(int(value) for value in contract_data["target_mesh"])
        tile_data = contract_data["tiled_loop_nest"]
        tile_sizes = (
            int(tile_data["tile_m"]),
            int(tile_data["tile_n"]),
            int(tile_data["tile_k"]),
        )
    else:
        m = n = k = args.matrix_size
        target_mesh = (0, 0)
        tile_sizes = (0, 0, 0)
    workload_path = output_dir / f"gemm_{m}x{n}x{k}.onnx"

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [m, n])
    b = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [k, n],
        np.zeros((k, n), dtype=np.float32),
    )
    node = helper.make_node("Gemm", ["A", "B"], ["C"], name="TiledGemm")
    graph = helper.make_graph([node], "OpenCakeTiledGemm", [a], [c], [b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(shape_inference.infer_shapes(model), workload_path)

    hardware_data = yaml.safe_load(Path(args.hardware).read_text(encoding="utf-8"))
    offchip = int(hardware_data["offchip_core_id"])
    coordinates = {
        int(core): (int(coord[0]), int(coord[1]))
        for core, coord in hardware_data["core_coordinates"].items()
        if int(core) != offchip
    }

    constraint_stats: dict[str, dict[str, int]] = {}
    original_get_transfer_plans = CommunicationManager.get_possible_transfer_plan
    original_update_mapping = SteadyStateScheduler.update_mapping_for_transfer

    def expanded_transfer_plans(manager, src_allocs, dst_allocs):
        if args.wafer_constraint == "none":
            return original_get_transfer_plans(manager, src_allocs, dst_allocs)
        return tuple(
            manager._get_simple_no_meeting_node_plans(
                tuple(src_allocs),
                tuple(dst_allocs),
                k_paths=8,
                beam_width=512,
                max_allocations=512,
                offchip_mem_penalty=1,
            )
        )

    def constrained_update_mapping(scheduler, transfer, src, destinations):
        original_update_mapping(scheduler, transfer, src, destinations)
        match = _TENSOR_RE.fullmatch(str(transfer.name))
        tensor = match.group(1).split("_")[0] if match else ""
        if tensor not in {"A", "B"} or args.wafer_constraint == "none":
            return

        node_mapping = scheduler.mapping.get(transfer)
        candidates = tuple(node_mapping.resource_allocation)
        legal = tuple(
            plan
            for plan in candidates
            if wafer_path_satisfies_constraint(
                tensor,
                tuple((link.sender.id, link.receiver.id) for link in plan.links_used),
                coordinates,
                offchip,
                args.wafer_constraint,
            )
        )
        constraint_stats[tensor] = {
            "candidate_paths": len(candidates),
            "legal_paths": len(legal),
            "rejected_paths": len(candidates) - len(legal),
        }
        if not legal:
            raise ValueError(
                f"wafer constraint {args.wafer_constraint!r} rejected every {tensor} multicast plan"
            )
        scheduler.mapping.set_for_node(
            transfer,
            resource_allocation=legal,
            inter_core_tiling=node_mapping.inter_core_tiling,
            memory_allocation=node_mapping.memory_allocation,
            kernel=node_mapping.kernel,
        )

    # Graphviz output is not part of the strategy contract.
    Workload.visualize = lambda self, *unused_args, **unused_kwargs: None
    CommunicationManager.get_possible_transfer_plan = expanded_transfer_plans
    SteadyStateScheduler.update_mapping_for_transfer = constrained_update_mapping
    try:
        ctx = optimize_allocation_co_with_mapping(
            hardware=args.hardware,
            workload=str(workload_path),
            mapping=args.mapping,
            experiment_id="stream-wafer-strategy",
            output_path=str(output_dir),
            skip_if_exists=False,
            backend="ortools_gscip",
        )
    finally:
        SteadyStateScheduler.update_mapping_for_transfer = original_update_mapping
        CommunicationManager.get_possible_transfer_plan = original_get_transfer_plans
    scheduler = ctx.get("scheduler")
    allocation = AllocationIR.from_internal(scheduler)
    allocation_data = allocation.compiler_view().model_dump()

    trace_path = output_dir / "stream-wafer-strategy" / "tetra" / "steady_state_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    route_hops = parse_stream_route_hops(trace, coordinates)
    timed_events = [
        event
        for event in trace.get("traceEvents", ())
        if "ts" in event and event.get("ph") != "M"
    ]
    trace_window_cycles = (
        max(float(event["ts"]) + float(event.get("dur", 0.0)) for event in timed_events)
        - min(float(event["ts"]) for event in timed_events)
        if timed_events
        else 0.0
    )
    onchip_directions: dict[str, set[str]] = {}
    for hop in route_hops:
        if hop.direction != "offchip":
            onchip_directions.setdefault(hop.tensor, set()).add(hop.direction)
    family_match = re.fullmatch(r"a-(east|west)-b-(north|south)", args.wafer_constraint)
    if family_match is None:
        ideal = False
    else:
        expected_a, expected_b = family_match.groups()
        ideal = onchip_directions.get("A", set()) <= {expected_a}
        ideal = ideal and onchip_directions.get("B", set()) <= {expected_b}
        ideal = ideal and bool(onchip_directions.get("A")) and bool(onchip_directions.get("B"))

    compute_node = allocation_data["mapping_nodes"]["TiledGemm"]
    compute_cores = tuple(
        sorted(
            resource["id"]
            for slot in compute_node["resource_allocation"]
            for resource in slot
            if resource.get("type") == "core"
        )
    )
    result = {
        "schema": "opencake.stream_strategy.v1",
        "tool": "KULeuven-MICAS Stream + ZigZag",
        "solver": allocation.backend,
        "total_latency_cycles": scheduler.latency_total,
        "trace_window_cycles": trace_window_cycles,
        "compute_cores": list(compute_cores),
        "core_coordinates": {str(core): list(coord) for core, coord in coordinates.items()},
        "route_hops": [asdict(hop) for hop in route_hops],
        "allocation": allocation_data,
        "wafer_constraint": args.wafer_constraint,
        "constraint_stats": constraint_stats,
        "ideal_a_east_b_south": ideal,
        "target_mesh": list(target_mesh),
        "tile_sizes": list(tile_sizes),
        "route_family": args.wafer_constraint,
        "solve_time_seconds": monotonic() - started,
        "limitation": (
            "Stream/TETRA selected allocation, transfer plans, and schedule on the exact target mesh. "
            "OpenCake supplied only wafer resource constraints and the legal route family, then mapped "
            "solver-selected directional hops to target color names because Stream has no Cerebras-color dialect."
        ),
    }
    Path(args.result).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def _parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--matrix-size", type=int, default=4)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument(
        "--wafer-constraint",
        choices=(
            "none",
            "a-east-b-south",
            "a-east-b-north",
            "a-west-b-south",
            "a-west-b-north",
        ),
        default="a-east-b-south",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _worker(_parse_worker_args())
