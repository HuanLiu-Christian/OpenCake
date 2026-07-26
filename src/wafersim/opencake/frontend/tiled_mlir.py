from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterator

from wafersim.opencake.mapper.tiler import GemmTilePlan
from wafersim.opencake.workload import GemmWorkload, HardwareSpec


@dataclass(frozen=True, slots=True)
class TiledGemmLoopNest:
    """GEMM structure recovered from transformed MLIR operations."""

    schema: str
    source: str
    parallel_lower_bounds: tuple[int, int]
    parallel_upper_bounds: tuple[int, int]
    parallel_steps: tuple[int, int]
    grid_rows: int
    grid_cols: int
    k_lower: int
    k_upper: int
    k_step: int
    tile_m: int
    tile_n: int
    tile_k: int
    a_tile_shape: tuple[int, int]
    b_tile_shape: tuple[int, int]
    c_tile_shape: tuple[int, int]
    output_shape: tuple[int, int]
    dtype: str

    @property
    def k_starts(self) -> tuple[int, ...]:
        return tuple(range(self.k_lower, self.k_upper, self.k_step))

    @property
    def bytes_per_element(self) -> int:
        return _dtype_bytes(self.dtype)

    def summary(self) -> dict[str, object]:
        result = asdict(self)
        result["parallel_lower_bounds"] = list(self.parallel_lower_bounds)
        result["parallel_upper_bounds"] = list(self.parallel_upper_bounds)
        result["parallel_steps"] = list(self.parallel_steps)
        result["a_tile_shape"] = list(self.a_tile_shape)
        result["b_tile_shape"] = list(self.b_tile_shape)
        result["c_tile_shape"] = list(self.c_tile_shape)
        result["output_shape"] = list(self.output_shape)
        result["k_starts"] = list(self.k_starts)
        result["k_chunks"] = len(self.k_starts)
        return result

    def validate_against(self, workload: GemmWorkload, hardware: HardwareSpec) -> None:
        if self.parallel_lower_bounds != (0, 0) or self.parallel_steps != (1, 1):
            raise ValueError(
                "OpenCake rank placement currently requires zero-based unit-step scf.forall loops"
            )
        if self.output_shape != (workload.m, workload.n):
            raise ValueError(
                f"tiled MLIR output shape {self.output_shape} does not match "
                f"workload C shape {(workload.m, workload.n)}"
            )
        if self.k_lower != 0 or self.k_upper != workload.k:
            raise ValueError(
                f"tiled MLIR K loop [{self.k_lower}, {self.k_upper}) does not cover "
                f"workload K=[0, {workload.k})"
            )
        if self.grid_rows > hardware.rank_rows or self.grid_cols > hardware.rank_cols:
            raise ValueError(
                f"tiled MLIR grid {(self.grid_rows, self.grid_cols)} exceeds physical "
                f"rank grid {(hardware.rank_rows, hardware.rank_cols)}"
            )
        if self.grid_rows * self.tile_m < workload.m or self.grid_cols * self.tile_n < workload.n:
            raise ValueError("tiled MLIR parallel grid does not cover the complete output tensor")
        if _dtype_bytes(self.dtype) != workload.bytes_per_element:
            raise ValueError(
                f"tiled MLIR dtype {self.dtype} does not match the workload's "
                f"{workload.bytes_per_element}-byte elements"
            )

    def to_tile_plan(self, workload: GemmWorkload, hardware: HardwareSpec) -> GemmTilePlan:
        self.validate_against(workload, hardware)
        ab_buffer_count = 2 if hardware.double_buffer_ab else 1
        a_buffer_bytes = self.tile_m * self.tile_k * workload.bytes_per_element
        b_buffer_bytes = self.tile_k * self.tile_n * workload.bytes_per_element
        c_tile_bytes = self.tile_m * self.tile_n * workload.bytes_per_element
        sram_bytes = c_tile_bytes + ab_buffer_count * (a_buffer_bytes + b_buffer_bytes)
        if sram_bytes > hardware.sram_bytes_per_rank:
            raise ValueError(
                f"transformed MLIR tile requires {sram_bytes} bytes per rank, exceeding "
                f"the {hardware.sram_bytes_per_rank}-byte SRAM capacity"
            )
        return GemmTilePlan(
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            ab_buffer_count=ab_buffer_count,
            a_buffer_bytes=a_buffer_bytes,
            b_buffer_bytes=b_buffer_bytes,
            c_tile_bytes=c_tile_bytes,
            sram_bytes_per_rank=sram_bytes,
            sram_utilization=sram_bytes / hardware.sram_bytes_per_rank,
        )


def read_tiled_gemm_mlir(path: str | Path) -> TiledGemmLoopNest:
    return parse_tiled_gemm_mlir(Path(path).read_text(encoding="utf-8"))


def parse_tiled_gemm_mlir(text: str) -> TiledGemmLoopNest:
    """Traverse transformed MLIR and recover its concrete GEMM loop nest.

    The IREE compiler package includes MLIR Python bindings. Keeping the import
    local preserves OpenCake's non-IREE fallback for environments that only run
    the lightweight simulator.
    """

    try:
        from iree.compiler import ir as mlir_ir
    except ImportError as error:
        raise RuntimeError(
            "IREE MLIR Python bindings are required to lower transformed MLIR; "
            "install wafersim[iree]"
        ) from error

    with mlir_ir.Context() as context:
        context.allow_unregistered_dialects = True
        module = mlir_ir.Module.parse(text)
        forall = _single_operation(module.operation, "scf.forall")
        reduction = _single_operation(forall, "scf.for")
        matmul = _single_operation(reduction, "linalg.matmul")

        lower = tuple(int(value) for value in forall.attributes["staticLowerBound"])
        upper = tuple(int(value) for value in forall.attributes["staticUpperBound"])
        steps = tuple(int(value) for value in forall.attributes["staticStep"])
        if len(lower) != 2 or len(upper) != 2 or len(steps) != 2:
            raise ValueError("OpenCake currently expects a two-dimensional scf.forall GEMM grid")
        if any(step <= 0 for step in steps):
            raise ValueError("scf.forall steps must be positive")

        k_lower, k_upper, k_step = (
            _constant_index(operand, mlir_ir) for operand in reduction.operands[:3]
        )
        if k_step <= 0:
            raise ValueError("GEMM reduction step must be positive")

        operand_shapes = tuple(_ranked_tensor_shape(value.type, mlir_ir) for value in matmul.operands[:3])
        if len(operand_shapes) != 3:
            raise ValueError("tiled linalg.matmul must have A, B, and C operands")
        a_shape, b_shape, c_shape = operand_shapes
        if len(a_shape) != 2 or len(b_shape) != 2 or len(c_shape) != 2:
            raise ValueError("OpenCake currently supports rank-2 tiled GEMM operands")
        tile_m, tile_k = a_shape
        b_k, tile_n = b_shape
        if b_k != tile_k or c_shape != (tile_m, tile_n):
            raise ValueError(
                f"inconsistent tiled matmul shapes: A={a_shape}, B={b_shape}, C={c_shape}"
            )
        if tile_k != k_step:
            raise ValueError(
                f"K loop step {k_step} does not match linalg.matmul K tile {tile_k}"
            )

        if len(forall.results) != 1:
            raise ValueError("OpenCake expects scf.forall to produce one tiled C tensor")
        output_shape = _ranked_tensor_shape(forall.results[0].type, mlir_ir)
        if len(output_shape) != 2:
            raise ValueError("OpenCake expects scf.forall to return one rank-2 C tensor")
        output_type = mlir_ir.RankedTensorType(forall.results[0].type)
        dtype = str(output_type.element_type)

        return TiledGemmLoopNest(
            schema="opencake.tiled_mlir_loop_nest.v1",
            source="mlir_operation_tree",
            parallel_lower_bounds=(lower[0], lower[1]),
            parallel_upper_bounds=(upper[0], upper[1]),
            parallel_steps=(steps[0], steps[1]),
            grid_rows=ceil((upper[0] - lower[0]) / steps[0]),
            grid_cols=ceil((upper[1] - lower[1]) / steps[1]),
            k_lower=k_lower,
            k_upper=k_upper,
            k_step=k_step,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            a_tile_shape=(tile_m, tile_k),
            b_tile_shape=(tile_k, tile_n),
            c_tile_shape=(tile_m, tile_n),
            output_shape=(output_shape[0], output_shape[1]),
            dtype=dtype,
        )


def _walk_operations(operation: Any) -> Iterator[Any]:
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child
                yield from _walk_operations(child)


def _single_operation(root: Any, name: str) -> Any:
    matches = [operation for operation in _walk_operations(root) if operation.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} operation, found {len(matches)}")
    return matches[0]


def _constant_index(value: Any, mlir_ir: Any) -> int:
    owner = value.owner
    if owner.name != "arith.constant" or "value" not in owner.attributes:
        raise ValueError("expected static arith.constant bounds for tiled GEMM reduction")
    return int(mlir_ir.IntegerAttr(owner.attributes["value"]).value)


def _ranked_tensor_shape(value_type: Any, mlir_ir: Any) -> tuple[int, ...]:
    tensor_type = mlir_ir.RankedTensorType(value_type)
    return tuple(int(dimension) for dimension in tensor_type.shape)


def _dtype_bytes(dtype: str) -> int:
    normalized = dtype.lower()
    if normalized in {"f16", "bf16", "i16", "ui16"}:
        return 2
    if normalized in {"f32", "i32", "ui32"}:
        return 4
    if normalized in {"f64", "i64", "ui64"}:
        return 8
    if normalized in {"i8", "ui8"}:
        return 1
    raise ValueError(f"unsupported tiled MLIR element type: {dtype}")
