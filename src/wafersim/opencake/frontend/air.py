from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from wafersim.opencake.frontend.tiled_mlir import TiledGemmLoopNest

if TYPE_CHECKING:
    from wafersim.opencake.strategy import GemmStreamingStrategyContract


@dataclass(frozen=True, slots=True)
class AirChannelContract:
    """Explicit streaming channel recovered from official AIR IR."""

    symbol: str
    tensor: str
    color: str
    direction: str
    fanout_axis: str
    source_shape: tuple[int, int]
    broadcast_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    bytes_per_transfer: int
    buffer_count: int
    channel_type: str = "npu_dma_stream"
    put_sites: int = 1
    get_sites: int = 1

    def summary(self) -> dict[str, object]:
        result = asdict(self)
        result["source_shape"] = list(self.source_shape)
        result["broadcast_shape"] = list(self.broadcast_shape)
        result["tile_shape"] = list(self.tile_shape)
        return result


@dataclass(frozen=True, slots=True)
class AirDependencyContract:
    """Dependency properties proven by the official ``air-dependency`` pass."""

    async_channel_puts: int
    async_channel_gets: int
    async_dma_copies: int
    async_compute_regions: int
    loop_carried_token: bool
    compute_waits_for_channels: bool
    output_dma_waits_for_loop: bool

    def summary(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AirGemmHierarchy:
    """GEMM hierarchy recovered from the real MLIR-AIR operation tree."""

    schema: str
    herd_rows: int
    herd_cols: int
    k_lower: int
    k_upper: int
    k_step: int
    a_tile_shape: tuple[int, int]
    b_tile_shape: tuple[int, int]
    c_tile_shape: tuple[int, int]
    dtype: str
    channels: tuple[AirChannelContract, ...]
    dependencies: AirDependencyContract | None = None

    def summary(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source": "official_mlir_air_operation_tree",
            "herd": [self.herd_rows, self.herd_cols],
            "k_loop": [self.k_lower, self.k_upper, self.k_step],
            "a_tile_shape": list(self.a_tile_shape),
            "b_tile_shape": list(self.b_tile_shape),
            "c_tile_shape": list(self.c_tile_shape),
            "dtype": self.dtype,
            "channels": [channel.summary() for channel in self.channels],
            "channel_source": "explicit_official_air_channel_operations",
            "dependencies": self.dependencies.summary() if self.dependencies is not None else None,
        }

    def to_loop_nest(self, tiled: TiledGemmLoopNest) -> TiledGemmLoopNest:
        return TiledGemmLoopNest(
            schema="opencake.air_gemm_loop_nest.v1",
            source="air_operation_tree",
            parallel_lower_bounds=(0, 0),
            parallel_upper_bounds=(self.herd_rows, self.herd_cols),
            parallel_steps=(1, 1),
            grid_rows=self.herd_rows,
            grid_cols=self.herd_cols,
            k_lower=self.k_lower,
            k_upper=self.k_upper,
            k_step=self.k_step,
            tile_m=self.a_tile_shape[0],
            tile_n=self.b_tile_shape[1],
            tile_k=self.a_tile_shape[1],
            a_tile_shape=self.a_tile_shape,
            b_tile_shape=self.b_tile_shape,
            c_tile_shape=self.c_tile_shape,
            output_shape=tiled.output_shape,
            dtype=self.dtype,
        )


@dataclass(frozen=True, slots=True)
class AirTransformResult:
    """Result of official AIR hierarchy, streaming, and dependency lowering."""

    schema: str
    schedule_mlir: str
    scheduled_payload_mlir: str
    herd_mlir: str
    streaming_input_mlir: str
    air_mlir: str
    herd_command: tuple[str, ...]
    command: tuple[str, ...]
    stderr: str
    runner: str
    applied: bool
    hierarchy: AirGemmHierarchy | None

    def summary(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "runner": self.runner,
            "applied": self.applied,
            "command": list(self.command),
            "herd_command": list(self.herd_command),
            "stderr": self.stderr.strip(),
            "air_ir_ops": {
                "air.herd": _count_op(self.air_mlir, "air.herd"),
                "air.channel": _count_op(self.air_mlir, "air.channel"),
                "air.channel.put": _count_op(self.air_mlir, "air.channel.put"),
                "air.channel.get": _count_op(self.air_mlir, "air.channel.get"),
                "air.dma_memcpy_nd": _count_op(self.air_mlir, "air.dma_memcpy_nd"),
                "air.execute": _count_op(self.air_mlir, "air.execute"),
                "air.wait_all": _count_op(self.air_mlir, "air.wait_all"),
                "scf.for": _count_op(self.air_mlir, "scf.for"),
                "linalg.matmul": _count_op(self.air_mlir, "linalg.matmul"),
            },
            "hierarchy": self.hierarchy.summary() if self.hierarchy is not None else None,
        }


@dataclass(frozen=True, slots=True)
class OfficialAirPipelineResult:
    """AIR IR produced only by official ``air-opt`` transformation passes."""

    schema: str
    source_mlir: str
    air_mlir: str
    command: tuple[str, ...]
    passes: tuple[str, ...]
    stderr: str
    runner: str
    applied: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "runner": self.runner,
            "applied": self.applied,
            "command": list(self.command),
            "passes": list(self.passes),
            "custom_air_generation": False,
            "air_ir_ops": {
                "air.herd": _count_op(self.air_mlir, "air.herd"),
                "air.channel": _count_op(self.air_mlir, "air.channel"),
                "air.channel.put": _count_op(self.air_mlir, "air.channel.put"),
                "air.channel.get": _count_op(self.air_mlir, "air.channel.get"),
                "air.dma_memcpy_nd": _count_op(self.air_mlir, "air.dma_memcpy_nd"),
                "air.execute": _count_op(self.air_mlir, "air.execute"),
                "air.wait_all": _count_op(self.air_mlir, "air.wait_all"),
                "linalg.matmul": _count_op(self.air_mlir, "linalg.matmul"),
            },
        }


class AirDialectRunner:
    """Run the official MLIR-AIR toolchain on OpenCake tiled MLIR.

    MLIR-AIR currently publishes Linux wheels. On Linux, pass an ``air-opt``
    path or put it on ``PATH``. On macOS, a prebuilt Linux container can be
    selected with ``docker_container``; the payload is streamed over stdin, so
    the container does not need access to the host workspace.
    """

    def __init__(
        self,
        binary: str | Path | None = None,
        *,
        docker_container: str | None = None,
    ) -> None:
        self.binary = Path(binary) if binary is not None else self._find_binary()
        self.docker_container = docker_container

    @classmethod
    def available(cls) -> bool:
        return cls._find_binary() is not None

    def lower_gemm(
        self,
        tiled_payload_mlir: str,
        tiled_loop_nest: TiledGemmLoopNest,
        *,
        bytes_per_element: int,
        buffer_count: int,
        streaming_strategy: GemmStreamingStrategyContract | None = None,
    ) -> AirTransformResult:
        schedule = generate_air_transform_schedule()
        scheduled = embed_air_transform_schedule(tiled_payload_mlir)
        if self.binary is None and self.docker_container is None:
            return AirTransformResult(
                schema="opencake.air_transform_result.v1",
                schedule_mlir=schedule,
                scheduled_payload_mlir=scheduled,
                herd_mlir="",
                streaming_input_mlir="",
                air_mlir="",
                herd_command=(),
                command=(),
                stderr=(
                    "air-opt was not found; use --air-opt on Linux or "
                    "--air-docker-container on macOS"
                ),
                runner="unavailable",
                applied=False,
                hierarchy=None,
            )

        herd_arguments = (
            "-",
            "--transform-interpreter=entry-point=__air_transform_main",
            "--air-par-to-herd=depth=-1",
            "--mlir-print-op-generic",
        )
        herd_command, runner = self._command(herd_arguments)
        herd_completed = self._run(herd_command, scheduled)
        herd_mlir = herd_completed.stdout
        base_hierarchy = parse_air_gemm_mlir(
            herd_mlir,
            tiled_loop_nest,
            bytes_per_element=bytes_per_element,
            buffer_count=buffer_count,
        )
        streaming_input = generate_air_streaming_module(
            tiled_loop_nest,
            base_hierarchy,
            buffer_count=buffer_count,
            streaming_strategy=streaming_strategy,
        )
        dependency_arguments = (
            "-",
            "--air-dependency",
            "--air-dependency-canonicalize",
            "--canonicalize",
        )
        command, _ = self._command(dependency_arguments)
        dependency_completed = self._run(command, streaming_input)
        air_mlir = dependency_completed.stdout
        hierarchy = parse_air_streaming_mlir(
            air_mlir,
            base_hierarchy,
            bytes_per_element=bytes_per_element,
            buffer_count=buffer_count,
        )
        return AirTransformResult(
            schema="opencake.air_streaming_transform_result.v2",
            schedule_mlir=schedule,
            scheduled_payload_mlir=scheduled,
            herd_mlir=herd_mlir,
            streaming_input_mlir=streaming_input,
            air_mlir=air_mlir,
            herd_command=herd_command,
            command=command,
            stderr="\n".join(
                part for part in (herd_completed.stderr.strip(), dependency_completed.stderr.strip()) if part
            ),
            runner=runner,
            applied=True,
            hierarchy=hierarchy,
        )

    def lower_bufferized_tiled_gemm(self, source_mlir: str) -> OfficialAirPipelineResult:
        """Lower standard bufferized tiled MLIR using only official AIR passes.

        The source must express parallel tiles, local memory allocations, and
        ``memref.copy`` operations. AIR is responsible for converting those
        operations to herds, DMA channels, and asynchronous dependencies. This
        entry point deliberately does not call ``generate_air_streaming_module``.
        """

        passes = (
            "--air-copy-to-dma",
            "--air-par-to-herd=depth=-1",
            "--air-specialize-dma-broadcast",
            "--air-dma-to-channel",
            "--air-dependency",
            "--air-dependency-canonicalize",
            "--canonicalize",
        )
        if self.binary is None and self.docker_container is None:
            return OfficialAirPipelineResult(
                schema="opencake.official_air_pipeline.v1",
                source_mlir=source_mlir,
                air_mlir="",
                command=(),
                passes=passes,
                stderr=(
                    "air-opt was not found; pass an air-opt binary or select "
                    "an MLIR-AIR Docker container"
                ),
                runner="unavailable",
                applied=False,
            )

        command, runner = self._command(("-", *passes))
        completed = self._run(command, source_mlir)
        air_mlir = completed.stdout
        required = ("air.herd", "air.channel", "air.channel.put", "air.channel.get")
        missing = [operation for operation in required if _count_op(air_mlir, operation) == 0]
        if missing:
            raise ValueError(
                "official AIR pipeline did not materialize required operations: "
                + ", ".join(missing)
            )
        return OfficialAirPipelineResult(
            schema="opencake.official_air_pipeline.v1",
            source_mlir=source_mlir,
            air_mlir=air_mlir,
            command=command,
            passes=passes,
            stderr=completed.stderr,
            runner=runner,
            applied=True,
        )

    def _command(self, arguments: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
        if self.docker_container is not None:
            return (
                (
                    "docker",
                    "exec",
                    "-i",
                    self.docker_container,
                    "sh",
                    "-lc",
                    "air-opt " + " ".join(arguments),
                ),
                f"official air-opt in Docker container {self.docker_container}",
            )
        assert self.binary is not None
        return (str(self.binary), *arguments), f"official air-opt at {self.binary}"

    @staticmethod
    def _run(command: tuple[str, ...], payload: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            input=payload,
            check=True,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _find_binary() -> Path | None:
        found = shutil.which("air-opt")
        return Path(found) if found is not None else None


def generate_air_transform_schedule() -> str:
    """Return the Transform schedule interpreted by official ``air-opt``."""

    return """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__air_transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{[\"func.func\"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %bufferized = transform.bufferization.one_shot_bufferize %func
        : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{[\"scf.forall\"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %forall
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""


def embed_air_transform_schedule(tiled_payload: str) -> str:
    if "@__air_transform_main" in tiled_payload:
        raise ValueError("payload already contains the OpenCake AIR transform schedule")
    closing = tiled_payload.rfind("}")
    if closing < 0:
        raise ValueError("tiled MLIR module is missing its closing brace")
    sequence = """  transform.named_sequence @__air_transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{[\"func.func\"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %bufferized = transform.bufferization.one_shot_bufferize %func
        : (!transform.any_op) -> !transform.any_op
    %forall = transform.structured.match ops{[\"scf.forall\"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %forall
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
"""
    return tiled_payload[:closing].rstrip() + "\n\n" + sequence + tiled_payload[closing:]


def parse_air_gemm_mlir(
    text: str,
    tiled: TiledGemmLoopNest,
    *,
    bytes_per_element: int,
    buffer_count: int,
) -> AirGemmHierarchy:
    """Recover one GEMM herd from generic AIR assembly emitted by air-opt.

    Generic assembly keeps this adapter independent of a particular version of
    MLIR Python bindings while still requiring the official AIR parser and
    transformations to have accepted and produced the input.
    """

    constants = {
        name: int(value)
        for name, value in re.findall(
            r'(%[A-Za-z0-9_]+)\s*=\s*"arith\.constant"\(\)\s*<\{value\s*=\s*(-?\d+)\s*:\s*index',
            text,
        )
    }
    herd_match = re.search(r'"air\.herd"\((%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)', text)
    if herd_match is None:
        raise ValueError("official AIR output does not contain exactly one parseable air.herd")
    herd_dims = tuple(_constant_ref(constants, reference) for reference in herd_match.groups())

    for_match = re.search(
        r'"scf\.for"\((%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)',
        text,
    )
    if for_match is None:
        raise ValueError("AIR herd does not contain a static scf.for K loop")
    k_lower, k_upper, k_step = (
        _constant_ref(constants, reference) for reference in for_match.groups()
    )

    matmul_start = text.find('"linalg.matmul"')
    if matmul_start < 0:
        raise ValueError("AIR herd does not contain linalg.matmul")
    # Generic MLIR assembly prints the scalar linalg region before the
    # operation's memref signature. Keep enough trailing text to include both.
    matmul_assembly = text[matmul_start : matmul_start + 4096]
    shapes = [
        (int(rows), int(cols), dtype)
        for rows, cols, dtype in re.findall(
            r"memref<(\d+)x(\d+)x([A-Za-z0-9]+)",
            matmul_assembly,
        )
    ]
    if len(shapes) < 3:
        raise ValueError("could not recover A, B, and C memref shapes from AIR linalg.matmul")
    (a_rows, a_cols, dtype), (b_rows, b_cols, b_dtype), (c_rows, c_cols, c_dtype) = shapes[:3]
    if dtype != b_dtype or dtype != c_dtype:
        raise ValueError("AIR linalg.matmul operands have inconsistent element types")
    if a_cols != b_rows or (a_rows, b_cols) != (c_rows, c_cols):
        raise ValueError("AIR linalg.matmul operands have inconsistent tile shapes")

    actual = (herd_dims, (k_lower, k_upper, k_step), (a_rows, a_cols), (b_rows, b_cols), (c_rows, c_cols))
    expected = (
        (tiled.grid_rows, tiled.grid_cols),
        (tiled.k_lower, tiled.k_upper, tiled.k_step),
        tiled.a_tile_shape,
        tiled.b_tile_shape,
        tiled.c_tile_shape,
    )
    if actual != expected:
        raise ValueError(f"AIR hierarchy {actual} does not preserve tiled MLIR structure {expected}")

    return AirGemmHierarchy(
        schema="opencake.air_gemm_herd.v2",
        herd_rows=herd_dims[0],
        herd_cols=herd_dims[1],
        k_lower=k_lower,
        k_upper=k_upper,
        k_step=k_step,
        a_tile_shape=(a_rows, a_cols),
        b_tile_shape=(b_rows, b_cols),
        c_tile_shape=(c_rows, c_cols),
        dtype=dtype,
        channels=(),
    )


def generate_air_streaming_module(
    tiled: TiledGemmLoopNest,
    hierarchy: AirGemmHierarchy,
    *,
    buffer_count: int,
    streaming_strategy: GemmStreamingStrategyContract | None = None,
) -> str:
    """Materialize a GEMM stream with official AIR channel and DMA operations."""

    if buffer_count not in (1, 2):
        raise ValueError("AIR GEMM streaming currently supports one or two A/B buffers")
    a_direction = streaming_strategy.a_direction if streaming_strategy is not None else "east"
    b_direction = streaming_strategy.b_direction if streaming_strategy is not None else "south"
    a_color = streaming_strategy.a_color if streaming_strategy is not None else "a_east"
    b_color = streaming_strategy.b_color if streaming_strategy is not None else "b_south"
    if a_direction not in {"east", "west"} or b_direction not in {"south", "north"}:
        raise ValueError("AIR GEMM requires horizontal A and vertical B streaming")
    a_source_col = 0 if a_direction == "east" else hierarchy.herd_cols - 1
    b_source_row = 0 if b_direction == "south" else hierarchy.herd_rows - 1
    m, n = tiled.output_shape
    k = tiled.k_upper
    tm, tn, tk = tiled.tile_m, tiled.tile_n, tiled.tile_k
    dtype = tiled.dtype
    a_type = f"memref<{tm}x{tk}x{dtype}, 2>"
    b_type = f"memref<{tk}x{tn}x{dtype}, 2>"
    c_type = f"memref<{tm}x{tn}x{dtype}, 2>"
    a_global = f"memref<{m}x{k}x{dtype}>"
    b_global = f"memref<{k}x{n}x{dtype}>"
    c_global = f"memref<{m}x{n}x{dtype}>"
    a_view = f"memref<{tm}x{tk}x{dtype}, strided<[{k}, 1], offset: ?>>"
    b_view = f"memref<{tk}x{tn}x{dtype}, strided<[{n}, 1], offset: ?>>"
    a_buffers = "\n".join(f"      %a{index} = memref.alloc() : {a_type}" for index in range(buffer_count))
    b_buffers = "\n".join(f"      %b{index} = memref.alloc() : {b_type}" for index in range(buffer_count))
    dealloc_a = "\n".join(f"      memref.dealloc %a{index} : {a_type}" for index in range(buffer_count))
    dealloc_b = "\n".join(f"      memref.dealloc %b{index} : {b_type}" for index in range(buffer_count))
    if buffer_count == 1:
        select_buffers = "      %a_buffer = memref.cast %a0 : {a_type} to {a_type}\n      %b_buffer = memref.cast %b0 : {b_type} to {b_type}".format(
            a_type=a_type,
            b_type=b_type,
        )
    else:
        select_buffers = f"""      %phase = arith.divui %k0, %kstep : index
      %slot = arith.remui %phase, %buffers : index
      %use_buffer0 = arith.cmpi eq, %slot, %z : index
      %a_buffer, %b_buffer = scf.if %use_buffer0 -> ({a_type}, {b_type}) {{
        scf.yield %a0, %b0 : {a_type}, {b_type}
      }} else {{
        scf.yield %a1, %b1 : {a_type}, {b_type}
      }}"""

    return f"""#map_m = affine_map<(d0) -> (d0 * {tm})>
#map_n = affine_map<(d0) -> (d0 * {tn})>
module {{
  air.channel @{a_color} [{hierarchy.herd_rows}, 1] {{broadcast_shape = [{hierarchy.herd_rows}, {hierarchy.herd_cols}], channel_type = "npu_dma_stream", opencake.color = "{a_color}", opencake.direction = "{a_direction}", opencake.tensor = "A"}}
  air.channel @{b_color} [1, {hierarchy.herd_cols}] {{broadcast_shape = [{hierarchy.herd_rows}, {hierarchy.herd_cols}], channel_type = "npu_dma_stream", opencake.color = "{b_color}", opencake.direction = "{b_direction}", opencake.tensor = "B"}}
  func.func @opencake_stream_gemm(%A: {a_global}, %B: {b_global}, %C: {c_global}) {{
    %herd_rows = arith.constant {hierarchy.herd_rows} : index
    %herd_cols = arith.constant {hierarchy.herd_cols} : index
    air.herd @opencake_gemm_herd tile (%row, %col) in (%rows=%herd_rows, %cols=%herd_cols) args(%ha=%A, %hb=%B, %hc=%C) : {a_global}, {b_global}, {c_global} {{
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %tm = arith.constant {tm} : index
      %tn = arith.constant {tn} : index
      %kstep = arith.constant {tk} : index
      %kupper = arith.constant {k} : index
      %nstride = arith.constant {n} : index
      %buffers = arith.constant {buffer_count} : index
      %a_source_coord = arith.constant {a_source_col} : index
      %b_source_coord = arith.constant {b_source_row} : index
      %m0 = affine.apply #map_m(%row)
      %n0 = affine.apply #map_n(%col)
{a_buffers}
{b_buffers}
      %c_buffer = memref.alloc() : {c_type}
      air.dma_memcpy_nd (%c_buffer[] [] [], %hc[%m0, %n0] [%tm, %tn] [%nstride, %one]) : ({c_type}, {c_global})
      scf.for %k0 = %z to %kupper step %kstep {{
        %a_source = memref.subview %ha[%m0, %k0] [{tm}, {tk}] [1, 1] : {a_global} to {a_view}
        %b_source = memref.subview %hb[%k0, %n0] [{tk}, {tn}] [1, 1] : {b_global} to {b_view}
{select_buffers}
        %is_a_source = arith.cmpi eq, %col, %a_source_coord : index
        scf.if %is_a_source {{
          air.channel.put @{a_color}[%row, %z] (%a_source[%z, %z] [%tm, %kstep] [%one, %one]) : ({a_view})
        }}
        %is_b_source = arith.cmpi eq, %row, %b_source_coord : index
        scf.if %is_b_source {{
          air.channel.put @{b_color}[%z, %col] (%b_source[%z, %z] [%kstep, %tn] [%one, %one]) : ({b_view})
        }}
        air.channel.get @{a_color}[%row, %col] (%a_buffer[%z, %z] [%tm, %kstep] [%one, %one]) : ({a_type})
        air.channel.get @{b_color}[%row, %col] (%b_buffer[%z, %z] [%kstep, %tn] [%one, %one]) : ({b_type})
        linalg.matmul ins(%a_buffer, %b_buffer : {a_type}, {b_type}) outs(%c_buffer : {c_type})
      }}
      air.dma_memcpy_nd (%hc[%m0, %n0] [%tm, %tn] [%nstride, %one], %c_buffer[] [] []) : ({c_global}, {c_type})
{dealloc_a}
{dealloc_b}
      memref.dealloc %c_buffer : {c_type}
    }}
    return
  }}
}}
"""


def parse_air_streaming_mlir(
    text: str,
    base: AirGemmHierarchy,
    *,
    bytes_per_element: int,
    buffer_count: int,
) -> AirGemmHierarchy:
    """Recover explicit channels and dependencies from ``air-dependency`` output."""

    channels: list[AirChannelContract] = []
    pattern = re.compile(
        r"air\.channel\s+@(?P<symbol>[A-Za-z0-9_]+)\s+"
        r"\[(?P<s0>\d+),\s*(?P<s1>\d+)\]\s+\{(?P<attrs>[^}]*)\}"
    )
    tile_shapes = {"A": base.a_tile_shape, "B": base.b_tile_shape}
    for match in pattern.finditer(text):
        attrs = match.group("attrs")
        tensor = _string_attr(attrs, "opencake.tensor")
        color = _string_attr(attrs, "opencake.color")
        direction = _string_attr(attrs, "opencake.direction")
        broadcast = re.search(r"broadcast_shape\s*=\s*\[(\d+),\s*(\d+)\]", attrs)
        if broadcast is None:
            raise ValueError(f"AIR channel @{match.group('symbol')} lacks broadcast_shape")
        source_shape = (int(match.group("s0")), int(match.group("s1")))
        broadcast_shape = (int(broadcast.group(1)), int(broadcast.group(2)))
        expected_broadcast = (base.herd_rows, base.herd_cols)
        if broadcast_shape != expected_broadcast:
            raise ValueError(
                f"AIR channel @{match.group('symbol')} broadcasts to {broadcast_shape}, "
                f"expected herd {expected_broadcast}"
            )
        tile_shape = tile_shapes[tensor]
        channels.append(
            AirChannelContract(
                symbol=match.group("symbol"),
                tensor=tensor,
                color=color,
                direction=direction,
                fanout_axis=(
                    "herd_cols" if direction in {"east", "west"} else "herd_rows"
                ),
                source_shape=source_shape,
                broadcast_shape=broadcast_shape,
                tile_shape=tile_shape,
                bytes_per_transfer=tile_shape[0] * tile_shape[1] * bytes_per_element,
                buffer_count=buffer_count,
                put_sites=len(re.findall(rf"air\.channel\.put[^\n]*@{match.group('symbol')}\b", text)),
                get_sites=len(re.findall(rf"air\.channel\.get[^\n]*@{match.group('symbol')}\b", text)),
            )
        )
    if {channel.tensor for channel in channels} != {"A", "B"}:
        raise ValueError("official AIR streaming IR must contain explicit A and B channels")

    loop_result = re.search(
        r"(?m)^\s*(%[A-Za-z0-9_]+)\s*=\s*scf\.for\b[^\n]*"
        r"->\s*\(!air\.async\.token\)",
        text,
    )
    output_dma_waits_for_loop = False
    if loop_result is not None:
        output_dma_waits_for_loop = bool(
            re.search(
                rf"air\.dma_memcpy_nd\s+async\s+\[[^\]]*"
                rf"{re.escape(loop_result.group(1))}[^\]]*\]",
                text,
            )
        )

    dependencies = AirDependencyContract(
        async_channel_puts=len(re.findall(r"air\.channel\.put\s+async", text)),
        async_channel_gets=len(re.findall(r"air\.channel\.get\s+async", text)),
        async_dma_copies=len(re.findall(r"air\.dma_memcpy_nd\s+async", text)),
        async_compute_regions=len(re.findall(r"air\.execute\s+\[[^\]]+\]\s*\{\s*\n?\s*linalg\.matmul", text)),
        loop_carried_token=bool(
            re.search(
                r"scf\.for\b(?:(?!\{).)*?iter_args\([^)]*\)"
                r"(?:(?!\{).)*?!air\.async\.token",
                text,
                flags=re.DOTALL,
            )
        ),
        compute_waits_for_channels=bool(
            re.search(r"air\.execute\s+\[[^\]]*%[^,\]]+\s*,\s*%[^\]]+\]\s*\{\s*\n?\s*linalg\.matmul", text)
        ),
        output_dma_waits_for_loop=output_dma_waits_for_loop,
    )
    k_chunks = (base.k_upper - base.k_lower + base.k_step - 1) // base.k_step
    if (k_chunks > 1 and not dependencies.loop_carried_token) or not dependencies.compute_waits_for_channels:
        raise ValueError(
            "air-dependency did not materialize the required streaming dependency chain: "
            f"{dependencies.summary()}"
        )
    return AirGemmHierarchy(
        schema="opencake.air_streaming_hierarchy.v2",
        herd_rows=base.herd_rows,
        herd_cols=base.herd_cols,
        k_lower=base.k_lower,
        k_upper=base.k_upper,
        k_step=base.k_step,
        a_tile_shape=base.a_tile_shape,
        b_tile_shape=base.b_tile_shape,
        c_tile_shape=base.c_tile_shape,
        dtype=base.dtype,
        channels=tuple(sorted(channels, key=lambda channel: channel.tensor)),
        dependencies=dependencies,
    )


def _string_attr(attrs: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}\s*=\s*\"([^\"]+)\"", attrs)
    if match is None:
        raise ValueError(f"AIR channel is missing required {name} attribute")
    return match.group(1)


def _count_op(text: str, name: str) -> int:
    custom_syntax = re.findall(
        rf"(?m)^\s*(?:%[^=\n]+\s*=\s*)?{re.escape(name)}(?:\s|$)",
        text,
    )
    return text.count(f'"{name}"') + len(custom_syntax)


def _constant_ref(constants: dict[str, int], reference: str) -> int:
    try:
        return constants[reference]
    except KeyError as error:
        raise ValueError(f"AIR dimension {reference} is not a static index constant") from error
