from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wafersim.opencake.chakra import ChakraTrace
from wafersim.opencake.ir import OpenCakeProgram
from wafersim.opencake.mapper import OpenCakeMapper
from wafersim.opencake.simulator import AstraSimWaferResult, AstraSimWaferSimulator
from wafersim.opencake.workload import HardwareSpec


@dataclass(frozen=True, slots=True)
class OpenCakePipelineResult:
    chakra_nodes: int
    spatial_ir_ops: int
    program: OpenCakeProgram
    simulator_result: AstraSimWaferResult

    def summary(self) -> dict[str, object]:
        return {
            "chakra_nodes": self.chakra_nodes,
            "workload": self.program.workload,
            "hardware": self.program.hardware,
            "spatial_ir_ops": self.spatial_ir_ops,
            "simulator": self.simulator_result.summary(),
        }


def run_chakra_to_wafer_sim(
    *,
    chakra_trace: str | Path,
    hardware: HardwareSpec,
    spatial_ir_out: str | Path | None = None,
) -> OpenCakePipelineResult:
    trace = ChakraTrace.read(chakra_trace)
    workload = trace.gemm_workload()
    program = OpenCakeMapper().map_gemm(workload, hardware)
    if spatial_ir_out is not None:
        program.write_jsonl(spatial_ir_out)
    simulator_result = AstraSimWaferSimulator().run(program, hardware)
    return OpenCakePipelineResult(
        chakra_nodes=len(trace.nodes),
        spatial_ir_ops=len(program.ops),
        program=program,
        simulator_result=simulator_result,
    )
