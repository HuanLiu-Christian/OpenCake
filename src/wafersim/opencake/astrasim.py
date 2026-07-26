from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from wafersim.opencake.backend import OpenCakeBackendResult
from wafersim.opencake.ir import OpenCakeProgram
from wafersim.opencake.schema import SPATIAL_IR_SCHEMA_VERSION, SpatialIrValidation
from wafersim.opencake.stages import ChakraExecutionTrace
from wafersim.opencake.workload import HardwareSpec


@dataclass(frozen=True, slots=True)
class AstraSimExperimentPackage:
    root: Path
    manifest: Path
    system: Path
    network: Path
    spatial_ir: Path
    chakra_trace: Path | None
    chakra_et: Path | None
    chakra_workload_prefix: Path | None
    chakra_rank_files: tuple[Path, ...]
    reference_result: Path

    def summary(self) -> dict[str, object]:
        result = {
            "root": str(self.root),
            "manifest": str(self.manifest),
            "system": str(self.system),
            "network": str(self.network),
            "spatial_ir": str(self.spatial_ir),
            "reference_result": str(self.reference_result),
        }
        if self.chakra_trace is not None:
            result["chakra_trace_jsonl"] = str(self.chakra_trace)
        if self.chakra_et is not None:
            result["chakra_et"] = str(self.chakra_et)
        if self.chakra_workload_prefix is not None:
            result["chakra_workload_prefix"] = str(self.chakra_workload_prefix)
            result["chakra_rank_files"] = [str(path) for path in self.chakra_rank_files]
        return result


class AstraSimExperimentExporter:
    """Export a formal OpenCake experiment package for Astra-Sim integration."""

    def export(
        self,
        *,
        program: OpenCakeProgram,
        hardware: HardwareSpec,
        validation: SpatialIrValidation,
        reference_result: OpenCakeBackendResult,
        out_dir: str | Path,
        execution_trace: ChakraExecutionTrace | None = None,
    ) -> AstraSimExperimentPackage:
        root = Path(out_dir)
        workload_dir = root / "workload"
        workload_dir.mkdir(parents=True, exist_ok=True)

        spatial_ir = workload_dir / "opencake_spatial_ir.jsonl"
        chakra_trace = (
            workload_dir / "chakra_execution_trace.jsonl"
            if execution_trace is not None
            else None
        )
        chakra_et = (
            workload_dir / "chakra_execution_trace.et"
            if execution_trace is not None
            else None
        )
        chakra_workload_prefix = (
            workload_dir / "opencake"
            if execution_trace is not None
            else None
        )
        chakra_rank_files: tuple[Path, ...] = ()
        system = root / "system.json"
        network = root / "network.yml"
        reference = root / "reference_result.json"
        manifest = root / "manifest.json"

        program.write_jsonl(spatial_ir)
        if execution_trace is not None and chakra_trace is not None:
            execution_trace.write_jsonl(chakra_trace)
            assert chakra_et is not None
            assert chakra_workload_prefix is not None
            execution_trace.write_et(chakra_et)
            chakra_rank_files = execution_trace.write_ranked_et(
                chakra_workload_prefix,
                rank_rows=hardware.rank_rows,
                rank_cols=hardware.rank_cols,
            ).rank_files
        _write_json(system, self._system_config(program, hardware, validation))
        network.write_text(self._network_config(hardware), encoding="utf-8")
        _write_json(reference, reference_result.summary())
        _write_json(
            manifest,
            {
                "schema": SPATIAL_IR_SCHEMA_VERSION,
                "backend_contract": "Astra-Sim C++ OpenCake wafer backend input package",
                "workload": program.workload,
                "hardware": program.hardware,
                "validation": validation.summary(),
                "files": {
                    "system": system.name,
                    "network": network.name,
                    "spatial_ir": str(spatial_ir.relative_to(root)),
                    **(
                        {
                            "chakra_trace_jsonl": str(chakra_trace.relative_to(root)),
                            "chakra_et": str(chakra_et.relative_to(root)),
                            "chakra_workload_prefix": str(
                                chakra_workload_prefix.relative_to(root)
                            ),
                            "chakra_rank_files": [
                                str(path.relative_to(root)) for path in chakra_rank_files
                            ],
                        }
                        if chakra_trace is not None
                        else {}
                    ),
                    "reference_result": reference.name,
                },
            },
        )
        return AstraSimExperimentPackage(
            root=root,
            manifest=manifest,
            system=system,
            network=network,
            spatial_ir=spatial_ir,
            chakra_trace=chakra_trace,
            chakra_et=chakra_et,
            chakra_workload_prefix=chakra_workload_prefix,
            chakra_rank_files=chakra_rank_files,
            reference_result=reference,
        )

    @staticmethod
    def _system_config(
        program: OpenCakeProgram,
        hardware: HardwareSpec,
        validation: SpatialIrValidation,
    ) -> dict[str, object]:
        return {
            "schema": SPATIAL_IR_SCHEMA_VERSION,
            "workload_type": program.workload["type"],
            "rank_model": hardware.rank_model,
            "rank_rows": hardware.rank_rows,
            "rank_cols": hardware.rank_cols,
            "ranks": hardware.ranks,
            "pes_per_rank": hardware.pes_per_rank,
            "sram_banks_per_rank": hardware.sram_banks_per_rank,
            "sram_bytes_per_bank": hardware.sram_bytes_per_bank,
            "compute_ops_per_cycle_per_rank": hardware.compute_ops_per_cycle_per_rank,
            "sram_bandwidth_bytes_per_cycle_per_rank": hardware.sram_bandwidth_bytes_per_cycle_per_rank,
            "sram_bytes_per_rank": hardware.sram_bytes_per_rank,
            "spatial_ir_ops": validation.ops,
            "colors": list(validation.colors),
        }

    @staticmethod
    def _network_config(hardware: HardwareSpec) -> str:
        lines = [
            "topology: [ OpenCake2DMesh ]",
            f"rank_rows: [ {hardware.rank_rows} ]",
            f"rank_cols: [ {hardware.rank_cols} ]",
            f"npus_count: [ {hardware.ranks} ]",
            f"link_bandwidth_bytes_per_cycle: [ {hardware.link_bandwidth_bytes_per_cycle} ]",
            f"router_latency_cycles: [ {hardware.router_latency_cycles} ]",
            f"queue_capacity: [ {hardware.queue_capacity} ]",
            "routing: [ color_or_xy ]",
            "",
        ]
        return "\n".join(lines)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
