from __future__ import annotations

from dataclasses import asdict

from wafersim.opencake.ir import AllocOp, ComputeOp, DefineColorOp, OpenCakeProgram, SendOp
from wafersim.opencake.mapper.streamer import GemmStreamingPlan, StreamingPlanner
from wafersim.opencake.mapper.tiler import GemmTilePlan, SramAwareTiler
from wafersim.opencake.workload import GemmWorkload, HardwareSpec

Coord = tuple[int, int]


class OpenCakeMapper:
    """Lower high-level workloads into wafer-scale Spatial IR."""

    def __init__(
        self,
        *,
        tiler: SramAwareTiler | None = None,
        streamer: StreamingPlanner | None = None,
    ) -> None:
        self._tiler = tiler or SramAwareTiler()
        self._streamer = streamer or StreamingPlanner()

    def map_gemm(
        self,
        workload: GemmWorkload,
        hardware: HardwareSpec,
        *,
        tile_plan: GemmTilePlan | None = None,
        stream_plan: GemmStreamingPlan | None = None,
        rank_grid: tuple[int, int] | None = None,
        mapping_source: str | None = None,
    ) -> OpenCakeProgram:
        return self._map_gemm_like(
            workload=workload,
            hardware=hardware,
            workload_metadata={
                "type": "gemm",
                "m": workload.m,
                "n": workload.n,
                "k": workload.k,
            },
            kernel="gemm_k_chunk",
            tile_plan=tile_plan,
            stream_plan=stream_plan,
            rank_grid=rank_grid,
            mapping_source=mapping_source,
        )

    def _map_gemm_like(
        self,
        *,
        workload: GemmWorkload,
        hardware: HardwareSpec,
        workload_metadata: dict[str, object],
        kernel: str,
        tile_plan: GemmTilePlan | None,
        stream_plan: GemmStreamingPlan | None,
        rank_grid: tuple[int, int] | None,
        mapping_source: str | None,
    ) -> OpenCakeProgram:
        workload.validate()
        hardware.validate()

        tile_plan = tile_plan or self._tiler.choose_gemm_tile(workload, hardware)
        stream_plan = stream_plan or self._streamer.plan_gemm_streaming(workload, tile_plan)
        expected_k_starts = tuple(range(0, workload.k, tile_plan.tile_k))
        if stream_plan.k_tile != tile_plan.tile_k or stream_plan.k_starts != expected_k_starts:
            raise ValueError("streaming plan does not match the tiled GEMM reduction loop")
        if stream_plan.a_direction not in {"east", "west"}:
            raise ValueError("A streaming direction must be east or west")
        if stream_plan.b_direction not in {"south", "north"}:
            raise ValueError("B streaming direction must be south or north")

        rank_rows, rank_cols = rank_grid or (hardware.rank_rows, hardware.rank_cols)
        if rank_rows <= 0 or rank_cols <= 0:
            raise ValueError("active rank grid dimensions must be positive")
        if rank_rows > hardware.rank_rows or rank_cols > hardware.rank_cols:
            raise ValueError("active rank grid exceeds the physical wafer rank grid")

        ops = [
            DefineColorOp(
                op="define_color", color=stream_plan.a_color, direction=stream_plan.a_direction
            ),
            DefineColorOp(
                op="define_color", color=stream_plan.b_color, direction=stream_plan.b_direction
            ),
        ]

        for rank_r in range(rank_rows):
            for rank_c in range(rank_cols):
                rows = self._tile_extent(workload.m, tile_plan.tile_m, rank_r)
                cols = self._tile_extent(workload.n, tile_plan.tile_n, rank_c)
                a_buffer_bytes = rows * tile_plan.tile_k * workload.bytes_per_element
                b_buffer_bytes = tile_plan.tile_k * cols * workload.bytes_per_element
                c_bytes = rows * cols * workload.bytes_per_element
                next_addr = 0

                ops.append(
                    AllocOp(op="alloc", rank=(rank_r, rank_c), name="c_tile", addr=next_addr, bytes=c_bytes)
                )
                next_addr += c_bytes
                for buffer_index in range(stream_plan.ab_buffer_count):
                    ops.append(
                        AllocOp(
                            op="alloc",
                            rank=(rank_r, rank_c),
                            name=f"a_buffer{buffer_index}",
                            addr=next_addr,
                            bytes=a_buffer_bytes,
                        )
                    )
                    next_addr += a_buffer_bytes
                    ops.append(
                        AllocOp(
                            op="alloc",
                            rank=(rank_r, rank_c),
                            name=f"b_buffer{buffer_index}",
                            addr=next_addr,
                            bytes=b_buffer_bytes,
                        )
                    )
                    next_addr += b_buffer_bytes

        # Each logical multicast is materialized as adjacent PE-to-PE hops.
        # This preserves the color direction and exposes every contested link
        # to the performance backend instead of hiding the path in one send.
        for k_start in stream_plan.k_starts:
            current_k = min(tile_plan.tile_k, workload.k - k_start)
            horizontal_edges = (
                tuple((col, col + 1) for col in range(rank_cols - 1))
                if stream_plan.a_direction == "east"
                else tuple((col, col - 1) for col in range(rank_cols - 1, 0, -1))
            )
            vertical_edges = (
                tuple((row, row + 1) for row in range(rank_rows - 1))
                if stream_plan.b_direction == "south"
                else tuple((row, row - 1) for row in range(rank_rows - 1, 0, -1))
            )
            for rank_r in range(rank_rows):
                rows = self._tile_extent(workload.m, tile_plan.tile_m, rank_r)
                a_chunk_bytes = rows * current_k * workload.bytes_per_element
                for src_col, dst_col in horizontal_edges:
                    ops.append(
                        SendOp(
                            op="send",
                            src_rank=(rank_r, src_col),
                            dst_rank=(rank_r, dst_col),
                            color=stream_plan.a_color,
                            bytes=a_chunk_bytes,
                            route=((rank_r, src_col), (rank_r, dst_col)),
                            tensor="A",
                            k_start=k_start,
                        )
                    )
            for rank_c in range(rank_cols):
                cols = self._tile_extent(workload.n, tile_plan.tile_n, rank_c)
                b_chunk_bytes = current_k * cols * workload.bytes_per_element
                for src_row, dst_row in vertical_edges:
                    ops.append(
                        SendOp(
                            op="send",
                            src_rank=(src_row, rank_c),
                            dst_rank=(dst_row, rank_c),
                            color=stream_plan.b_color,
                            bytes=b_chunk_bytes,
                            route=((src_row, rank_c), (dst_row, rank_c)),
                            tensor="B",
                            k_start=k_start,
                        )
                    )

            for rank_r in range(rank_rows):
                rows = self._tile_extent(workload.m, tile_plan.tile_m, rank_r)
                for rank_c in range(rank_cols):
                    cols = self._tile_extent(workload.n, tile_plan.tile_n, rank_c)
                    a_chunk_bytes = rows * current_k * workload.bytes_per_element
                    b_chunk_bytes = current_k * cols * workload.bytes_per_element
                    c_bytes = rows * cols * workload.bytes_per_element
                    ops.append(
                        ComputeOp(
                            op="compute",
                            rank=(rank_r, rank_c),
                            kernel=kernel,
                            m=rows,
                            n=cols,
                            k=current_k,
                            ops=workload.ops_per_mac * rows * cols * current_k,
                            memory_bytes=a_chunk_bytes + b_chunk_bytes + 2 * c_bytes,
                            k_start=k_start,
                        )
                    )

        return OpenCakeProgram(
            workload={
                **workload_metadata,
                "m": workload.m,
                "n": workload.n,
                "k": workload.k,
                "tile_m": tile_plan.tile_m,
                "tile_n": tile_plan.tile_n,
                "k_tile": tile_plan.tile_k,
                "k_chunks": stream_plan.chunks,
                "streaming_dimension": stream_plan.dimension,
                "double_buffer_ab": int(hardware.double_buffer_ab),
                "ab_buffer_count": stream_plan.ab_buffer_count,
                "bytes_per_element": workload.bytes_per_element,
                "total_ops": workload.total_ops,
                "estimated_sram_bytes_per_rank": tile_plan.sram_bytes_per_rank,
                "estimated_sram_utilization": tile_plan.sram_utilization,
                "active_rank_rows": rank_rows,
                "active_rank_cols": rank_cols,
                "mapping_source": mapping_source
                or ("tiled_mlir" if rank_grid is not None else "workload_policy"),
                "streaming_strategy_source": stream_plan.strategy_source,
                "streaming_strategy_schema": stream_plan.strategy_schema,
                "streaming_strategy_target_mesh": list(stream_plan.strategy_target_mesh)
                if stream_plan.strategy_target_mesh is not None
                else None,
                "streaming_strategy_objective_cycles": stream_plan.strategy_objective_cycles,
                "streaming_strategy_optimality_gap": stream_plan.strategy_optimality_gap,
                "a_stream_direction": stream_plan.a_direction,
                "b_stream_direction": stream_plan.b_direction,
                "a_injection_boundary": (
                    "west" if stream_plan.a_direction == "east" else "east"
                ),
                "b_injection_boundary": (
                    "north" if stream_plan.b_direction == "south" else "south"
                ),
                "communication_materialization": "adjacent_hop_color_multicast",
            },
            hardware={
                **asdict(hardware),
                "rank_model": hardware.rank_model,
                "rank_rows": rank_rows,
                "rank_cols": rank_cols,
                "ranks": hardware.ranks,
                "pes_per_rank": hardware.pes_per_rank,
                "sram_banks_per_rank": hardware.sram_banks_per_rank,
                "sram_bytes_per_bank": hardware.sram_bytes_per_bank,
            },
            ops=tuple(ops),
        )

    @staticmethod
    def _tile_extent(total: int, tile: int, index: int) -> int:
        start = index * tile
        return max(0, min(tile, total - start))

    @staticmethod
    def _horizontal_route(row: int, src_col: int, dst_col: int) -> tuple[Coord, ...]:
        step = 1 if dst_col >= src_col else -1
        return tuple((row, col) for col in range(src_col, dst_col + step, step))

    @staticmethod
    def _vertical_route(col: int, src_row: int, dst_row: int) -> tuple[Coord, ...]:
        step = 1 if dst_row >= src_row else -1
        return tuple((row, col) for row in range(src_row, dst_row + step, step))
