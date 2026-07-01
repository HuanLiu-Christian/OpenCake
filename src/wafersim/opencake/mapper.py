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
        ops = [
            DefineColorOp(op="define_color", color="a_east", direction="east"),
            DefineColorOp(op="define_color", color="b_south", direction="south"),
        ]

        for rank_r in range(rank_rows):
            for rank_c in range(rank_cols):
                rows = self._tile_extent(workload.m, tile_m, rank_r)
                cols = self._tile_extent(workload.n, tile_n, rank_c)
                a_bytes = rows * workload.k * workload.bytes_per_element
                b_bytes = workload.k * cols * workload.bytes_per_element
                c_bytes = rows * cols * workload.bytes_per_element

                ops.extend(
                    [
                        AllocOp(op="alloc", rank=(rank_r, rank_c), name="a_tile", addr=0, bytes=a_bytes),
                        AllocOp(op="alloc", rank=(rank_r, rank_c), name="b_tile", addr=a_bytes, bytes=b_bytes),
                        AllocOp(
                            op="alloc",
                            rank=(rank_r, rank_c),
                            name="c_tile",
                            addr=a_bytes + b_bytes,
                            bytes=c_bytes,
                        ),
                    ]
                )

                if rank_c > 0:
                    ops.append(
                        SendOp(
                            op="send",
                            src_rank=(rank_r, 0),
                            dst_rank=(rank_r, rank_c),
                            color="a_east",
                            bytes=a_bytes,
                            route=self._horizontal_route(rank_r, 0, rank_c),
                        )
                    )
                if rank_r > 0:
                    ops.append(
                        SendOp(
                            op="send",
                            src_rank=(0, rank_c),
                            dst_rank=(rank_r, rank_c),
                            color="b_south",
                            bytes=b_bytes,
                            route=self._vertical_route(rank_c, 0, rank_r),
                        )
                    )

                memory_bytes = a_bytes + b_bytes + c_bytes
                compute_ops = workload.ops_per_mac * rows * cols * workload.k
                ops.append(
                    ComputeOp(
                        op="compute",
                        rank=(rank_r, rank_c),
                        kernel="gemm_tile",
                        m=rows,
                        n=cols,
                        k=workload.k,
                        ops=compute_ops,
                        memory_bytes=memory_bytes,
                    )
                )

        return OpenCakeProgram(
            workload={
                "type": "gemm",
                "m": workload.m,
                "n": workload.n,
                "k": workload.k,
                "bytes_per_element": workload.bytes_per_element,
                "total_ops": workload.total_ops,
            },
            hardware={
                **asdict(hardware),
                "rank_rows": rank_rows,
                "rank_cols": rank_cols,
                "ranks": hardware.ranks,
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
