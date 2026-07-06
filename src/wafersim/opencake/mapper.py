from __future__ import annotations

from dataclasses import asdict
from math import ceil

from wafersim.opencake.ir import AllocOp, ComputeOp, DefineColorOp, OpenCakeProgram, SendOp
from wafersim.opencake.workload import GemmWorkload, HardwareSpec

Coord = tuple[int, int]


class OpenCakeMapper:
    """Lower high-level workloads into simple wafer spatial IR."""

    def map_gemm(self, workload: GemmWorkload, hardware: HardwareSpec) -> OpenCakeProgram:
        workload.validate()
        hardware.validate()

        rank_rows = hardware.rank_rows
        rank_cols = hardware.rank_cols
        tile_m = ceil(workload.m / rank_rows)
        tile_n = ceil(workload.n / rank_cols)
        k_tile = min(hardware.gemm_k_tile, workload.k)
        ab_buffer_count = 2 if hardware.double_buffer_ab else 1
        ops = [
            DefineColorOp(op="define_color", color="a_east", direction="east"),
            DefineColorOp(op="define_color", color="b_south", direction="south"),
        ]

        for rank_r in range(rank_rows):
            for rank_c in range(rank_cols):
                rows = self._tile_extent(workload.m, tile_m, rank_r)
                cols = self._tile_extent(workload.n, tile_n, rank_c)
                a_buffer_bytes = rows * k_tile * workload.bytes_per_element
                b_buffer_bytes = k_tile * cols * workload.bytes_per_element
                c_bytes = rows * cols * workload.bytes_per_element
                next_addr = 0

                ops.append(
                    AllocOp(op="alloc", rank=(rank_r, rank_c), name="c_tile", addr=next_addr, bytes=c_bytes)
                )
                next_addr += c_bytes
                for buffer_index in range(ab_buffer_count):
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

                for k_start in range(0, workload.k, k_tile):
                    current_k = self._tile_extent(workload.k, k_tile, k_start // k_tile)
                    a_chunk_bytes = rows * current_k * workload.bytes_per_element
                    b_chunk_bytes = current_k * cols * workload.bytes_per_element

                    if rank_c > 0:
                        ops.append(
                            SendOp(
                                op="send",
                                src_rank=(rank_r, 0),
                                dst_rank=(rank_r, rank_c),
                                color="a_east",
                                bytes=a_chunk_bytes,
                                route=self._horizontal_route(rank_r, 0, rank_c),
                                tensor="A",
                                k_start=k_start,
                            )
                        )
                    if rank_r > 0:
                        ops.append(
                            SendOp(
                                op="send",
                                src_rank=(0, rank_c),
                                dst_rank=(rank_r, rank_c),
                                color="b_south",
                                bytes=b_chunk_bytes,
                                route=self._vertical_route(rank_c, 0, rank_r),
                                tensor="B",
                                k_start=k_start,
                            )
                        )

                    memory_bytes = a_chunk_bytes + b_chunk_bytes + 2 * c_bytes
                    compute_ops = workload.ops_per_mac * rows * cols * current_k
                    ops.append(
                        ComputeOp(
                            op="compute",
                            rank=(rank_r, rank_c),
                            kernel="gemm_k_chunk",
                            m=rows,
                            n=cols,
                            k=current_k,
                            ops=compute_ops,
                            memory_bytes=memory_bytes,
                            k_start=k_start,
                        )
                    )

        return OpenCakeProgram(
            workload={
                "type": "gemm",
                "m": workload.m,
                "n": workload.n,
                "k": workload.k,
                "k_tile": k_tile,
                "double_buffer_ab": int(hardware.double_buffer_ab),
                "bytes_per_element": workload.bytes_per_element,
                "total_ops": workload.total_ops,
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
