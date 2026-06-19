from __future__ import annotations

from wafersim.program import Direction, LayoutSpec


LAYOUT = LayoutSpec(
    mesh_rows=32,
    mesh_cols=32,
    kernel_rows=1,
    kernel_cols=4,
    routes={
        "A_H2D": None,
        "B_H2D": None,
        "partial_sum_color": Direction.EAST,
        "Y_D2H": None,
    },
)
