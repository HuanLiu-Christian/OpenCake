from __future__ import annotations

from wafersim.program import PeProgramSpec, PeTaskSpec


PE_PROGRAM = PeProgramSpec(
    kernel="gemv_09_streaming",
    tasks=(
        PeTaskSpec(
            name="recv_x",
            description="Receive an x chunk from host or the PE above.",
        ),
        PeTaskSpec(
            name="compute_local_ax",
            description="Compute the local A tile times the local x chunk.",
        ),
        PeTaskSpec(
            name="recv_b",
            description="Receive the row bias chunk on the west edge.",
        ),
        PeTaskSpec(
            name="reduce_ax_east",
            description="Accumulate partial y chunks from west to east.",
        ),
        PeTaskSpec(
            name="send_y_d2h",
            description="Return the final y chunk from the east edge to host.",
        ),
    ),
)
