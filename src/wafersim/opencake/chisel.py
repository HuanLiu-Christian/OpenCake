from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from wafersim.opencake.ir import OpenCakeProgram
from wafersim.opencake.schema import SPATIAL_IR_SCHEMA_VERSION, SpatialIrValidation
from wafersim.opencake.workload import HardwareSpec


@dataclass(frozen=True, slots=True)
class ChiselExportPackage:
    root: Path
    config: Path
    spatial_ir: Path

    def summary(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "config": str(self.config),
            "spatial_ir": str(self.spatial_ir),
        }


class ChiselTraceExporter:
    """Write the same Spatial IR contract for the Chisel-side functional model."""

    def export(
        self,
        *,
        program: OpenCakeProgram,
        hardware: HardwareSpec,
        validation: SpatialIrValidation,
        out_dir: str | Path,
    ) -> ChiselExportPackage:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)

        spatial_ir = root / "opencake_spatial_ir.jsonl"
        config = root / "chisel_config.json"
        active_rank_rows = int(program.workload.get("active_rank_rows", hardware.rank_rows))
        active_rank_cols = int(program.workload.get("active_rank_cols", hardware.rank_cols))

        program.write_jsonl(spatial_ir)
        config.write_text(
            json.dumps(
                {
                    "schema": SPATIAL_IR_SCHEMA_VERSION,
                    "contract": "OpenCake Spatial IR replay for Chisel functional simulation",
                    "workload": program.workload,
                    "hardware": {
                        "rank_model": hardware.rank_model,
                        "rank_rows": active_rank_rows,
                        "rank_cols": active_rank_cols,
                        "ranks": active_rank_rows * active_rank_cols,
                        "physical_rank_rows": hardware.rank_rows,
                        "physical_rank_cols": hardware.rank_cols,
                        "physical_ranks": hardware.ranks,
                        "pes_per_rank": hardware.pes_per_rank,
                        "sram_banks_per_rank": hardware.sram_banks_per_rank,
                        "sram_bytes_per_bank": hardware.sram_bytes_per_bank,
                        "sram_bytes_per_rank": hardware.sram_bytes_per_rank,
                    },
                    "functional_model": {
                        "a_pattern": "row_major_iota",
                        "b_pattern": "ones",
                    },
                    "validation": validation.summary(),
                    "files": {
                        "spatial_ir": spatial_ir.name,
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return ChiselExportPackage(root=root, config=config, spatial_ir=spatial_ir)
