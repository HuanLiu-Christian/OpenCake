from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wafersim.opencake.astrasim import AstraSimExperimentExporter, AstraSimExperimentPackage
from wafersim.opencake.backend import OpenCakeBackend
from wafersim.opencake.chakra import ChakraTrace
from wafersim.opencake.ir import OpenCakeProgram
from wafersim.opencake.mapper import OpenCakeMapper
from wafersim.opencake.schema import SpatialIrValidation, validate_spatial_program
from wafersim.opencake.workload import HardwareSpec


@dataclass(frozen=True, slots=True)
class FormalPipelineResult:
    program: OpenCakeProgram
    validation: SpatialIrValidation
    package: AstraSimExperimentPackage
    reference_summary: dict[str, float | int]

    def summary(self) -> dict[str, object]:
        return {
            "workload": self.program.workload,
            "hardware": self.program.hardware,
            "validation": self.validation.summary(),
            "package": self.package.summary(),
            "python_reference": self.reference_summary,
        }


def run_formal_opencake_pipeline(
    *,
    chakra_trace: str | Path,
    out_dir: str | Path,
    hardware: HardwareSpec,
) -> FormalPipelineResult:
    trace = ChakraTrace.read(chakra_trace)
    workload = trace.gemm_workload()
    program = OpenCakeMapper().map_gemm(workload, hardware)
    validation = validate_spatial_program(program)
    reference = OpenCakeBackend().run(program, hardware)
    package = AstraSimExperimentExporter().export(
        program=program,
        hardware=hardware,
        validation=validation,
        reference_result=reference,
        out_dir=out_dir,
    )
    return FormalPipelineResult(
        program=program,
        validation=validation,
        package=package,
        reference_summary=reference.summary(),
    )
