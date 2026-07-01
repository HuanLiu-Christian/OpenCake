"""Python/SimPy wafer-scale accelerator simulation toolkit."""

from wafersim.config import Coord, MeshConfig

__all__ = [
    "Coord",
    "MeshConfig",
    "LayoutSpec",
    "PeProgramSpec",
    "PeTaskSpec",
    "GemvProblemSpec",
]


def __getattr__(name: str):
    if name in {"GemvProblemSpec", "LayoutSpec", "PeProgramSpec", "PeTaskSpec"}:
        from wafersim.program import GemvProblemSpec, LayoutSpec, PeProgramSpec, PeTaskSpec

        exports = {
            "GemvProblemSpec": GemvProblemSpec,
            "LayoutSpec": LayoutSpec,
            "PeProgramSpec": PeProgramSpec,
            "PeTaskSpec": PeTaskSpec,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
