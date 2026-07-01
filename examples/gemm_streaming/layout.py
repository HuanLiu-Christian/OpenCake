from __future__ import annotations

from wafersim.program import Direction, LayoutSpec


LAYOUT = LayoutSpec(
    mesh_rows=32,
    mesh_cols=32,
    kernel_rows=2,
    kernel_cols=2,
    routes={
        "MEMCPYH2D_A": None,
        "MEMCPYH2D_B": None,
        "a_color": Direction.EAST,
        "b_color": Direction.SOUTH,
        "MEMCPYD2H_C": None,
    },
)
