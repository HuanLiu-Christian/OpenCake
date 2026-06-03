from __future__ import annotations

from wafersim.program import Direction, LayoutSpec


LAYOUT = LayoutSpec(
    mesh_rows=32,
    mesh_cols=32,
    kernel_rows=3,
    kernel_cols=4,
    routes={
        "MEMCPYH2D_DATA_1": None,
        "MEMCPYH2D_DATA_2": None,
        "x_color": Direction.SOUTH,
        "ax_color_1": Direction.EAST,
        "ax_color_2": Direction.EAST,
        "MEMCPYD2H_DATA_1": None,
    },
)
