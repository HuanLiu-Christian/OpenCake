from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

from wafersim.opencake.frontend.air import (
    AirDependencyContract,
    AirDialectRunner,
    AirTransformResult,
)
from wafersim.opencake.frontend.mlir import MlirMatmulOp
from wafersim.opencake.frontend.tiled_mlir import (
    TiledGemmLoopNest,
    parse_tiled_gemm_mlir,
)
from wafersim.opencake.frontend.transform import MlirTransformResult
from wafersim.opencake.strategy import GemmStreamingStrategyContract
from wafersim.opencake.ir import AllocOp, ComputeOp, DefineColorOp, OpenCakeProgram, SendOp
from wafersim.opencake.lowering import MlirToSpatialLowering, MlirToSpatialLoweringResult
from wafersim.opencake.mapper.streamer import GemmStreamingPlan
from wafersim.opencake.workload import GemmWorkload, HardwareSpec

Coord = tuple[int, int]
TaskKind = Literal["control", "memory", "communication", "compute"]


@dataclass(frozen=True, slots=True)
class IreeWorkloadIr:
    """Stage 1: semantic GEMM recovered from MLIR/IREE workload IR."""

    schema: str
    source_op: str
    m: int
    n: int
    k: int
    dtype: str
    bytes_per_element: int
    total_ops: int

    def summary(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TiledOperatorIr:
    """Stage 2: explicit M/N placement tiles and K streaming chunks."""

    schema: str
    op: str
    tile_m: int
    tile_n: int
    tile_k: int
    grid_rows: int
    grid_cols: int
    k_starts: tuple[int, ...]
    double_buffer_ab: bool
    a_buffer_bytes: int
    b_buffer_bytes: int
    c_tile_bytes: int
    sram_bytes_per_rank: int
    sram_utilization: float

    @property
    def k_chunks(self) -> int:
        return len(self.k_starts)

    def summary(self) -> dict[str, object]:
        result = asdict(self)
        result["k_chunks"] = self.k_chunks
        return result


@dataclass(frozen=True, slots=True)
class TensorLayout:
    tensor: str
    partitioned_dimensions: tuple[str, ...]
    mesh_axes: tuple[str, ...]
    replicated_mesh_axes: tuple[str, ...]
    movement: str
    color: str | None


@dataclass(frozen=True, slots=True)
class RankPlacement:
    rank: Coord
    m_range: tuple[int, int]
    n_range: tuple[int, int]
    c_owner: bool
    sram_buffers: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PlacementDataLayoutIr:
    """Stage 3: physical rank placement and operand data layouts."""

    schema: str
    rank_model: str
    mesh: tuple[int, int]
    active_grid: tuple[int, int]
    layouts: tuple[TensorLayout, ...]
    placements: tuple[RankPlacement, ...]

    def summary(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rank_model": self.rank_model,
            "mesh": list(self.mesh),
            "active_grid": list(self.active_grid),
            "layouts": [asdict(layout) for layout in self.layouts],
            "placement_count": len(self.placements),
            "placements": [_jsonable(asdict(placement)) for placement in self.placements],
        }


@dataclass(frozen=True, slots=True)
class DagTask:
    id: int
    kind: TaskKind
    op: str
    rank: Coord | None
    deps: tuple[int, ...]
    attrs: dict[str, Any]

    def record(self) -> dict[str, object]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class ComputeMemoryCommunicationDag:
    """Stage 4: dependency graph spanning all three resource classes."""

    schema: str
    dependency_source: str
    tasks: tuple[DagTask, ...]

    def summary(self) -> dict[str, object]:
        counts = {kind: 0 for kind in ("control", "memory", "communication", "compute")}
        for task in self.tasks:
            counts[task.kind] += 1
        return {
            "schema": self.schema,
            "dependency_source": self.dependency_source,
            "tasks": len(self.tasks),
            "task_counts": counts,
        }

    def write_jsonl(self, path: str | Path) -> None:
        _write_jsonl(path, (task.record() for task in self.tasks))


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    id: int
    kind: TaskKind
    op: str
    rank: Coord | None
    deps: tuple[int, ...]
    start_cycle: float
    end_cycle: float
    attrs: dict[str, Any]

    @property
    def duration_cycles(self) -> float:
        return self.end_cycle - self.start_cycle

    def chakra_record(self) -> dict[str, object]:
        record = {
            "id": self.id,
            "type": self.kind,
            "op": self.op,
            "deps": list(self.deps),
            "start_cycle": self.start_cycle,
            "end_cycle": self.end_cycle,
            "duration_cycles": self.duration_cycles,
            **self.attrs,
        }
        if self.rank is not None:
            record["rank"] = list(self.rank)
        return _jsonable(record)


@dataclass(frozen=True, slots=True)
class ChakraExecutionTrace:
    """Stage 5: dependency- and boundary-scheduled Chakra-like trace."""

    schema: str
    tasks: tuple[ScheduledTask, ...]

    @property
    def end_cycle(self) -> float:
        return max((task.end_cycle for task in self.tasks), default=0.0)

    def summary(self) -> dict[str, object]:
        return {"schema": self.schema, "nodes": len(self.tasks), "end_cycle": self.end_cycle}

    def write_jsonl(self, path: str | Path) -> None:
        _write_jsonl(path, (task.chakra_record() for task in self.tasks))

    def write_et(self, path: str | Path) -> Path:
        from wafersim.opencake.chakra import write_execution_trace_et

        return write_execution_trace_et(
            path,
            (task.chakra_record() for task in self.tasks),
        )

    def write_ranked_et(
        self,
        prefix: str | Path,
        *,
        rank_rows: int,
        rank_cols: int,
    ):
        from wafersim.opencake.chakra import write_ranked_execution_trace_et

        return write_ranked_execution_trace_et(
            prefix,
            (task.chakra_record() for task in self.tasks),
            rank_rows=rank_rows,
            rank_cols=rank_cols,
        )


@dataclass(frozen=True, slots=True)
class StageArtifactPaths:
    root: Path
    workload_ir: Path
    tiled_operator_ir: Path
    transform_schedule: Path
    scheduled_payload: Path
    tiled_payload: Path
    tiled_mlir_lowering: Path
    air_transform_schedule: Path
    air_scheduled_payload: Path
    air_herd: Path
    air_streaming_input: Path
    air_streaming_async: Path
    air_to_opencake: Path
    placement_data_layout: Path
    task_dag: Path
    chakra_trace: Path
    chakra_et: Path
    spatial_ir: Path
    manifest: Path

    def summary(self) -> dict[str, str]:
        return {name: str(value) for name, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class StagedGemmCompilation:
    workload_ir: IreeWorkloadIr
    tiled_operator_ir: TiledOperatorIr
    mlir_transform: MlirTransformResult
    air_transform: AirTransformResult
    tiled_mlir_lowering: MlirToSpatialLoweringResult
    placement_data_layout: PlacementDataLayoutIr
    task_dag: ComputeMemoryCommunicationDag
    chakra_trace: ChakraExecutionTrace
    program: OpenCakeProgram

    def summary(self) -> dict[str, object]:
        return {
            "semantic_recovery": self.workload_ir.summary(),
            "mlir_transform_tiling": self.mlir_transform.summary(),
            "mlir_air_hierarchy": self.air_transform.summary(),
            "air_to_opencake_lowering": self.tiled_mlir_lowering.summary(),
            "operator_decomposition": self.tiled_operator_ir.summary(),
            "spatial_mapping": {
                "schema": self.placement_data_layout.schema,
                "rank_model": self.placement_data_layout.rank_model,
                "mesh": list(self.placement_data_layout.mesh),
                "active_grid": list(self.placement_data_layout.active_grid),
                "placements": len(self.placement_data_layout.placements),
                "layouts": [asdict(layout) for layout in self.placement_data_layout.layouts],
            },
            "communication_synthesis": self.task_dag.summary(),
            "schedule_generation": self.chakra_trace.summary(),
        }

    def export(self, out_dir: str | Path) -> StageArtifactPaths:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        paths = StageArtifactPaths(
            root=root,
            workload_ir=root / "01_iree_workload_ir.json",
            tiled_operator_ir=root / "02_tiled_operator_ir.json",
            transform_schedule=root / "02a_transform_schedule.mlir",
            scheduled_payload=root / "02b_scheduled_payload.mlir",
            tiled_payload=root / "02c_tiled_payload.mlir",
            tiled_mlir_lowering=root / "02d_tiled_mlir_lowering.json",
            air_transform_schedule=root / "02e_air_transform_schedule.mlir",
            air_scheduled_payload=root / "02f_air_scheduled_payload.mlir",
            air_herd=root / "02g_air_herd.mlir",
            air_streaming_input=root / "02h_air_streaming_input.mlir",
            air_streaming_async=root / "02i_air_streaming_async.mlir",
            air_to_opencake=root / "02j_air_to_opencake.json",
            placement_data_layout=root / "03_placement_data_layout.json",
            task_dag=root / "04_compute_memory_communication_dag.jsonl",
            chakra_trace=root / "05_chakra_execution_trace.jsonl",
            chakra_et=root / "05_chakra_execution_trace.et",
            spatial_ir=root / "opencake_spatial_ir.jsonl",
            manifest=root / "pipeline_manifest.json",
        )
        _write_json(paths.workload_ir, self.workload_ir.summary())
        _write_json(paths.tiled_operator_ir, self.tiled_operator_ir.summary())
        paths.transform_schedule.write_text(self.mlir_transform.schedule_mlir, encoding="utf-8")
        paths.scheduled_payload.write_text(self.mlir_transform.scheduled_payload_mlir, encoding="utf-8")
        paths.tiled_payload.write_text(self.mlir_transform.tiled_payload_mlir, encoding="utf-8")
        _write_json(paths.tiled_mlir_lowering, self.tiled_mlir_lowering.summary())
        paths.air_transform_schedule.write_text(
            self.air_transform.schedule_mlir,
            encoding="utf-8",
        )
        paths.air_scheduled_payload.write_text(
            self.air_transform.scheduled_payload_mlir,
            encoding="utf-8",
        )
        paths.air_herd.write_text(self.air_transform.herd_mlir, encoding="utf-8")
        paths.air_streaming_input.write_text(
            self.air_transform.streaming_input_mlir,
            encoding="utf-8",
        )
        paths.air_streaming_async.write_text(self.air_transform.air_mlir, encoding="utf-8")
        _write_json(paths.air_to_opencake, self.air_transform.summary())
        _write_json(paths.placement_data_layout, self.placement_data_layout.summary())
        self.task_dag.write_jsonl(paths.task_dag)
        self.chakra_trace.write_jsonl(paths.chakra_trace)
        self.chakra_trace.write_et(paths.chakra_et)
        self.program.write_jsonl(paths.spatial_ir)
        _write_json(
            paths.manifest,
            {
                "pipeline": [
                    "semantic_recovery",
                    "mlir_transform_tiling",
                    "mlir_air_hierarchy",
                    "air_to_opencake_lowering",
                    "operator_decomposition",
                    "spatial_mapping",
                    "communication_synthesis",
                    "schedule_generation",
                ],
                "schemas": {
                    "workload_ir": self.workload_ir.schema,
                    "tiled_operator_ir": self.tiled_operator_ir.schema,
                    "mlir_transform": self.mlir_transform.schema,
                    "mlir_air": self.air_transform.schema,
                    "tiled_mlir_lowering": self.tiled_mlir_lowering.schema,
                    "placement_data_layout": self.placement_data_layout.schema,
                    "task_dag": self.task_dag.schema,
                    "chakra_trace": self.chakra_trace.schema,
                },
                "artifacts": {
                    "workload_ir": paths.workload_ir.name,
                    "tiled_operator_ir": paths.tiled_operator_ir.name,
                    "transform_schedule": paths.transform_schedule.name,
                    "scheduled_payload": paths.scheduled_payload.name,
                    "tiled_payload": paths.tiled_payload.name,
                    "tiled_mlir_lowering": paths.tiled_mlir_lowering.name,
                    "air_transform_schedule": paths.air_transform_schedule.name,
                    "air_scheduled_payload": paths.air_scheduled_payload.name,
                    "air_herd": paths.air_herd.name,
                    "air_streaming_input": paths.air_streaming_input.name,
                    "air_streaming_async": paths.air_streaming_async.name,
                    "air_to_opencake": paths.air_to_opencake.name,
                    "placement_data_layout": paths.placement_data_layout.name,
                    "task_dag": paths.task_dag.name,
                    "chakra_trace": paths.chakra_trace.name,
                    "chakra_et": paths.chakra_et.name,
                    "spatial_ir": paths.spatial_ir.name,
                },
            },
        )
        return paths


class OpenCakeStageCompiler:
    """Materialize the explicit mapping stages for one GEMM."""

    def __init__(self, *, air_runner: AirDialectRunner | None = None) -> None:
        self._air = air_runner or AirDialectRunner()
        self._lowering = MlirToSpatialLowering()

    @staticmethod
    def schedule_spatial_program(
        program: OpenCakeProgram,
        hardware: HardwareSpec,
    ) -> tuple[ComputeMemoryCommunicationDag, ChakraExecutionTrace]:
        """Build the Chakra execution DAG for an already-mapped Spatial IR."""

        dag = OpenCakeStageCompiler._synthesize_communication(program)
        trace = OpenCakeStageCompiler._schedule(dag, hardware)
        return dag, trace

    def compile_tiled_gemm(
        self,
        tiled_mlir_file: str | Path,
        hardware: HardwareSpec,
        *,
        streaming_strategy: GemmStreamingStrategyContract | None = None,
    ) -> StagedGemmCompilation:
        """Compile an already tiled GEMM without rerunning tile search or MLIR Transform.

        The input operation tree is the source of truth for the active PE grid,
        M/N tile sizes, and K streaming step. Tile search and Transform Dialect
        execution happen once in the host launcher before this stage compiler.
        """

        tiled_payload = Path(tiled_mlir_file).read_text(encoding="utf-8")
        loop_nest = parse_tiled_gemm_mlir(tiled_payload)
        frontend = MlirMatmulOp(
            m=loop_nest.output_shape[0],
            n=loop_nest.output_shape[1],
            k=loop_nest.k_upper,
            dtype=loop_nest.dtype,
            bytes_per_element=loop_nest.bytes_per_element,
        )
        workload = frontend.to_workload()
        loop_nest.validate_against(workload, hardware)
        loop_nest.to_tile_plan(workload, hardware)
        workload_ir = self._recover_semantics(
            frontend,
            workload,
            source_op="pre_tiled_mlir.linalg.matmul",
        )
        pre_tiled_input = MlirTransformResult(
            schema="opencake.pre_tiled_mlir_input.v1",
            tile_m=loop_nest.tile_m,
            tile_n=loop_nest.tile_n,
            tile_k=loop_nest.tile_k,
            schedule_mlir="",
            scheduled_payload_mlir=tiled_payload,
            tiled_payload_mlir=tiled_payload,
            command=(),
            stderr="",
            interpreter="pre-tiled MLIR input; Transform Dialect not rerun",
            applied=True,
        )
        return self._compile_materialized_gemm(
            frontend,
            hardware,
            workload_ir=workload_ir,
            mlir_transform=pre_tiled_input,
            loop_nest=loop_nest,
            streaming_strategy=streaming_strategy,
        )

    def _compile_materialized_gemm(
        self,
        frontend: MlirMatmulOp,
        hardware: HardwareSpec,
        *,
        workload_ir: IreeWorkloadIr,
        mlir_transform: MlirTransformResult,
        loop_nest: TiledGemmLoopNest,
        streaming_strategy: GemmStreamingStrategyContract | None,
    ) -> StagedGemmCompilation:
        workload = frontend.to_workload()
        air_transform = self._lower_to_air(
            mlir_transform,
            loop_nest,
            workload,
            hardware,
            streaming_strategy,
        )
        if air_transform.hierarchy is not None:
            loop_nest = air_transform.hierarchy.to_loop_nest(loop_nest)
        tiled_mlir_lowering = self._lowering.lower_gemm(
            loop_nest,
            workload,
            hardware,
            air_hierarchy=air_transform.hierarchy if air_transform.applied else None,
            streaming_strategy=streaming_strategy,
        )
        tiled_ir = self._decompose_operator(hardware, tiled_mlir_lowering)
        placement_ir = self._map_spatially(
            workload,
            hardware,
            tiled_ir,
            tiled_mlir_lowering.streaming_plan,
        )
        program = tiled_mlir_lowering.program
        air_dependencies = (
            air_transform.hierarchy.dependencies
            if air_transform.applied and air_transform.hierarchy is not None
            else None
        )
        dag = self._synthesize_communication(program, air_dependencies)
        trace = self._schedule(dag, hardware)
        return StagedGemmCompilation(
            workload_ir=workload_ir,
            tiled_operator_ir=tiled_ir,
            mlir_transform=mlir_transform,
            air_transform=air_transform,
            tiled_mlir_lowering=tiled_mlir_lowering,
            placement_data_layout=placement_ir,
            task_dag=dag,
            chakra_trace=trace,
            program=program,
        )

    def _lower_to_air(
        self,
        transform: MlirTransformResult,
        loop_nest: TiledGemmLoopNest,
        workload: GemmWorkload,
        hardware: HardwareSpec,
        streaming_strategy: GemmStreamingStrategyContract | None,
    ) -> AirTransformResult:
        if not transform.applied:
            return AirTransformResult(
                schema="opencake.air_transform_result.v1",
                schedule_mlir="",
                scheduled_payload_mlir="",
                herd_mlir="",
                streaming_input_mlir="",
                air_mlir="",
                herd_command=(),
                command=(),
                stderr="MLIR tiling was not materialized, so AIR lowering was skipped",
                runner="unavailable",
                applied=False,
                hierarchy=None,
            )
        return self._air.lower_gemm(
            transform.tiled_payload_mlir,
            loop_nest,
            bytes_per_element=workload.bytes_per_element,
            buffer_count=2 if hardware.double_buffer_ab else 1,
            streaming_strategy=streaming_strategy,
        )

    @staticmethod
    def _recover_semantics(
        frontend: MlirMatmulOp,
        workload: GemmWorkload,
        *,
        source_op: str = "mlir.linalg.matmul",
    ) -> IreeWorkloadIr:
        return IreeWorkloadIr(
            schema="opencake.iree_workload_ir.v1",
            source_op=source_op,
            m=workload.m,
            n=workload.n,
            k=workload.k,
            dtype=frontend.dtype,
            bytes_per_element=workload.bytes_per_element,
            total_ops=workload.total_ops,
        )

    def _decompose_operator(
        self,
        hardware: HardwareSpec,
        lowering: MlirToSpatialLoweringResult,
    ) -> TiledOperatorIr:
        tile = lowering.tile_plan
        stream = lowering.streaming_plan
        loop_nest = lowering.loop_nest
        return TiledOperatorIr(
            schema="opencake.tiled_operator_ir.v1",
            op="gemm",
            tile_m=tile.tile_m,
            tile_n=tile.tile_n,
            tile_k=tile.tile_k,
            grid_rows=loop_nest.grid_rows,
            grid_cols=loop_nest.grid_cols,
            k_starts=stream.k_starts,
            double_buffer_ab=hardware.double_buffer_ab,
            a_buffer_bytes=tile.a_buffer_bytes,
            b_buffer_bytes=tile.b_buffer_bytes,
            c_tile_bytes=tile.c_tile_bytes,
            sram_bytes_per_rank=tile.sram_bytes_per_rank,
            sram_utilization=tile.sram_utilization,
        )

    @staticmethod
    def _map_spatially(
        workload: GemmWorkload,
        hardware: HardwareSpec,
        tiled: TiledOperatorIr,
        streaming: GemmStreamingPlan,
    ) -> PlacementDataLayoutIr:
        layouts = (
            TensorLayout(
                "A",
                ("M",),
                ("Y",),
                ("X",),
                f"stream_{streaming.a_direction}",
                streaming.a_color,
            ),
            TensorLayout(
                "B",
                ("N",),
                ("X",),
                ("Y",),
                f"stream_{streaming.b_direction}",
                streaming.b_color,
            ),
            TensorLayout("C", ("M", "N"), ("Y", "X"), (), "stationary", None),
        )
        placements = []
        for row in range(tiled.grid_rows):
            m_start = row * tiled.tile_m
            m_end = min(workload.m, m_start + tiled.tile_m)
            for col in range(tiled.grid_cols):
                n_start = col * tiled.tile_n
                n_end = min(workload.n, n_start + tiled.tile_n)
                rows = m_end - m_start
                cols = n_end - n_start
                a_bytes = rows * tiled.tile_k * workload.bytes_per_element
                b_bytes = tiled.tile_k * cols * workload.bytes_per_element
                c_bytes = rows * cols * workload.bytes_per_element
                buffers = [("c_tile", c_bytes)]
                for index in range(2 if tiled.double_buffer_ab else 1):
                    buffers.extend(((f"a_buffer{index}", a_bytes), (f"b_buffer{index}", b_bytes)))
                placements.append(
                    RankPlacement(
                        rank=(row, col),
                        m_range=(m_start, m_end),
                        n_range=(n_start, n_end),
                        c_owner=True,
                        sram_buffers=tuple(buffers),
                    )
                )
        return PlacementDataLayoutIr(
            schema="opencake.placement_data_layout.v1",
            rank_model=hardware.rank_model,
            mesh=(hardware.rank_rows, hardware.rank_cols),
            active_grid=(tiled.grid_rows, tiled.grid_cols),
            layouts=layouts,
            placements=tuple(placements),
        )

    @staticmethod
    def _synthesize_communication(
        program: OpenCakeProgram,
        air_dependencies: AirDependencyContract | None = None,
    ) -> ComputeMemoryCommunicationDag:
        if air_dependencies is not None and (
            air_dependencies.async_channel_gets == 0
            or not air_dependencies.compute_waits_for_channels
        ):
            raise ValueError("official AIR dependencies do not describe a valid streaming compute chain")
        dependency_source = (
            "official_air_async_tokens"
            if air_dependencies is not None
            else "opencake_program_order_fallback"
        )
        tasks: list[DagTask] = []
        next_id = 0
        allocation_ids: dict[Coord, list[int]] = {}
        send_ids: dict[tuple[Coord, str, int], int] = {}
        previous_compute: dict[Coord, int] = {}

        for op in program.ops:
            if isinstance(op, DefineColorOp):
                tasks.append(
                    DagTask(next_id, "control", "define_color", None, (), {"color": op.color, "direction": op.direction})
                )
                next_id += 1
            elif isinstance(op, AllocOp):
                tasks.append(
                    DagTask(
                        next_id,
                        "memory",
                        "alloc",
                        op.rank,
                        (),
                        {"name": op.name, "addr": op.addr, "bytes": op.bytes},
                    )
                )
                allocation_ids.setdefault(op.rank, []).append(next_id)
                next_id += 1

        for op in program.ops:
            if isinstance(op, SendOp):
                allocation_deps = tuple(allocation_ids.get(op.src_rank, ())) + tuple(
                    allocation_ids.get(op.dst_rank, ())
                )
                deps = set(allocation_deps)
                # A non-boundary sender can forward a chunk only after the
                # same color/tensor chunk arrived from its upstream neighbor.
                upstream = send_ids.get((op.src_rank, op.tensor, op.k_start))
                if upstream is not None:
                    deps.add(upstream)
                tasks.append(
                    DagTask(
                        next_id,
                        "communication",
                        "send",
                        op.src_rank,
                        tuple(sorted(deps)),
                        {
                            "src_rank": op.src_rank,
                            "dst_rank": op.dst_rank,
                            "color": op.color,
                            "bytes": op.bytes,
                            "route": op.route,
                            "tensor": op.tensor,
                            "k_start": op.k_start,
                            "hops": op.hops,
                        },
                    )
                )
                send_ids[(op.dst_rank, op.tensor, op.k_start)] = next_id
                next_id += 1
            elif isinstance(op, ComputeOp):
                deps = set(allocation_ids.get(op.rank, ()))
                for tensor in ("A", "B"):
                    send_id = send_ids.get((op.rank, tensor, op.k_start))
                    if send_id is not None:
                        deps.add(send_id)
                memory_id = next_id
                tasks.append(
                    DagTask(
                        memory_id,
                        "memory",
                        "read_write_tile",
                        op.rank,
                        tuple(sorted(deps)),
                        {
                            "bytes": op.memory_bytes,
                            "k_start": op.k_start,
                            "dependency_source": dependency_source,
                        },
                    )
                )
                next_id += 1
                compute_id = next_id
                compute_deps = {memory_id}
                if op.rank in previous_compute:
                    compute_deps.add(previous_compute[op.rank])
                tasks.append(
                    DagTask(
                        compute_id,
                        "compute",
                        op.kernel,
                        op.rank,
                        tuple(sorted(compute_deps)),
                        {
                            "m": op.m,
                            "n": op.n,
                            "k": op.k,
                            "ops": op.ops,
                            "k_start": op.k_start,
                            "dependency_source": dependency_source,
                        },
                    )
                )
                previous_compute[op.rank] = compute_id
                next_id += 1

        return ComputeMemoryCommunicationDag(
            schema="opencake.compute_memory_communication_dag.v2",
            dependency_source=dependency_source,
            tasks=tuple(tasks),
        )

    @staticmethod
    def _schedule(
        dag: ComputeMemoryCommunicationDag,
        hardware: HardwareSpec,
    ) -> ChakraExecutionTrace:
        end_by_task: dict[int, float] = {}
        compute_ready: dict[Coord, float] = {}
        memory_ready: dict[Coord, float] = {}
        link_ready: dict[tuple[Coord, Coord], float] = {}
        scheduled: list[ScheduledTask] = []

        for task in dag.tasks:
            dependency_ready = max((end_by_task[dep] for dep in task.deps), default=0.0)
            start = dependency_ready
            end = start
            if task.kind == "memory" and task.op == "alloc":
                # Allocation declares PE-local address ownership; it does not
                # move bytes through the SRAM port in this timing model.
                end = start
            elif task.kind == "memory" and task.rank is not None:
                start = max(start, memory_ready.get(task.rank, 0.0))
                duration = (
                    float(task.attrs.get("bytes", 0))
                    / hardware.sram_bandwidth_bytes_per_cycle_per_rank
                )
                end = start + duration
                memory_ready[task.rank] = end
            elif task.kind == "compute" and task.rank is not None:
                start = max(start, compute_ready.get(task.rank, 0.0))
                duration = float(task.attrs["ops"]) / hardware.compute_ops_per_cycle_per_rank
                end = start + duration
                compute_ready[task.rank] = end
            elif task.kind == "communication":
                route = tuple(tuple(coord) for coord in task.attrs["route"])
                per_link = (
                    hardware.router_latency_cycles
                    + float(task.attrs["bytes"]) / hardware.link_bandwidth_bytes_per_cycle
                )
                arrival = start
                first_start = start
                for index, (src, dst) in enumerate(zip(route, route[1:])):
                    link_start = max(arrival, link_ready.get((src, dst), 0.0))
                    if index == 0:
                        first_start = link_start
                    arrival = link_start + per_link
                    link_ready[(src, dst)] = arrival
                start = first_start
                end = arrival
            end_by_task[task.id] = end
            scheduled.append(
                ScheduledTask(
                    id=task.id,
                    kind=task.kind,
                    op=task.op,
                    rank=task.rank,
                    deps=task.deps,
                    start_cycle=start,
                    end_cycle=end,
                    attrs=task.attrs,
                )
            )
        return ChakraExecutionTrace(
            schema="opencake.chakra_execution_trace.v1",
            tasks=tuple(scheduled),
        )


def _write_json(path: str | Path, data: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: str | Path, records) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), sort_keys=True))
            handle.write("\n")


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value
