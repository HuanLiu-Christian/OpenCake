from __future__ import annotations

from dataclasses import dataclass

from wafersim.opencake.backend import OpenCakeBackend, OpenCakeBackendResult
from wafersim.opencake.ir import OpenCakeProgram
from wafersim.opencake.workload import HardwareSpec


@dataclass(frozen=True, slots=True)
class AstraSimWaferResult:
    """Result wrapper for the Astra-Sim-style wafer backend."""

    backend: str
    timing_model: str
    result: OpenCakeBackendResult

    def summary(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "timing_model": self.timing_model,
            **self.result.summary(),
        }


class AstraSimWaferSimulator:
    """Astra-Sim-style wafer simulator backend.

    This is the local Python prototype of the backend contract we eventually
    want to port into Astra-Sim: ranks are PE blocks, send ops consume route
    links, and compute/memory ops are reduced into cycle bottlenecks.
    """

    def __init__(self) -> None:
        self._backend = OpenCakeBackend()

    def run(self, program: OpenCakeProgram, hardware: HardwareSpec) -> AstraSimWaferResult:
        return AstraSimWaferResult(
            backend="opencake_astrasim_wafer_python",
            timing_model="timestamp_boundary_rank_memory_compute_network",
            result=self._backend.run(program, hardware),
        )
