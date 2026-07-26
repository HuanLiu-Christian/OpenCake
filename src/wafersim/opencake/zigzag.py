from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from wafersim.opencake.mapper.tiler import GemmTilePlan, SramAwareTiler
from wafersim.opencake.workload import GemmWorkload, HardwareSpec


@dataclass(frozen=True, slots=True)
class ZigZagGemmSearchResult:
    """Auditable result of a real ZigZag PE-local mapping search."""

    schema: str
    source: str
    version: str
    optimization: str
    temporal_engine: str
    tile_plan: GemmTilePlan
    candidate_plans: tuple[GemmTilePlan, ...]
    zigzag_resident_k_upper_bound: int
    energy_pj: float
    latency_cycles: float
    ideal_compute_cycles: float
    temporal_mapping: dict[str, list[list[tuple[str, int]]]]
    spatial_mapping: dict[str, list[list[tuple[str, int]]]]
    workload_path: str
    accelerator_path: str
    mapping_path: str

    def summary(self) -> dict[str, Any]:
        result = asdict(self)
        result["tile_plan"] = asdict(self.tile_plan)
        return result

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return output


class ZigZagGemmMapper:
    """Use ZigZag LOMA to search a PE-local GEMM temporal mapping.

    Global M/N partitioning remains a wafer placement decision: one output tile
    maps to one rank/PE. ZigZag sees that PE-local MxN tile and the complete K
    reduction, then searches how K and the operand tensors are allocated across
    the PE SRAM and the streaming memory level.
    """

    def __init__(
        self,
        *,
        optimization: str = "latency",
        temporal_engine: str = "loma",
        lpf_limit: int = 8,
    ) -> None:
        if optimization not in {"latency", "energy", "EDP"}:
            raise ValueError("ZigZag optimization must be latency, energy, or EDP")
        if temporal_engine not in {"loma", "salsa"}:
            raise ValueError("ZigZag temporal engine must be loma or salsa")
        self.optimization = optimization
        self.temporal_engine = temporal_engine
        self.lpf_limit = lpf_limit

    def search(
        self,
        workload: GemmWorkload,
        hardware: HardwareSpec,
        *,
        output_dir: str | Path,
    ) -> ZigZagGemmSearchResult:
        workload.validate()
        hardware.validate()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
        try:
            import zigzag
            from zigzag.api import get_hardware_performance_zigzag
        except ImportError as exc:
            raise RuntimeError(
                "ZigZag is not installed; run `python -m pip install -e '.[zigzag]'`"
            ) from exc

        tile_m, tile_n = SramAwareTiler.output_tile_shape(workload, hardware)
        workload_path = output / "zigzag_workload.yaml"
        accelerator_path = output / "zigzag_accelerator.yaml"
        mapping_path = output / "zigzag_mapping.yaml"
        self._write_json_yaml(
            workload_path,
            self._workload_description(tile_m, tile_n, workload),
        )
        self._write_json_yaml(
            accelerator_path,
            self._accelerator_description(tile_m, tile_n, workload, hardware),
        )
        self._write_json_yaml(
            mapping_path,
            self._mapping_description(workload, hardware),
        )

        energy, latency, cmes = get_hardware_performance_zigzag(
            workload=str(workload_path),
            accelerator=str(accelerator_path),
            mapping=str(mapping_path),
            temporal_mapping_search_engine=self.temporal_engine,
            temporal_mapping_type="even",
            opt=self.optimization,
            dump_folder=str(output / "cost_model"),
            lpf_limit=self.lpf_limit,
            nb_spatial_mappings_generated=1,
            loma_show_progress_bar=False,
        )
        cme = cmes[0][1][0][0]
        temporal_mapping = self._mapping_record(cme.temporal_mapping.mapping_dic_origin)
        spatial_mapping = self._mapping_record(cme.spatial_mapping_dict_int)
        resident_k_upper_bound = self._resident_k_tile(cme, workload)
        feasible_plans = tuple(
            plan
            for plan in SramAwareTiler().feasible_gemm_tiles(workload, hardware)
            if plan.tile_k <= resident_k_upper_bound and workload.k % plan.tile_k == 0
        )
        if not feasible_plans:
            raise RuntimeError(
                "ZigZag mapping has no compatible K tile under the OpenCake shared-SRAM contract"
            )
        tile_plan = max(feasible_plans, key=lambda plan: plan.tile_k)

        result = ZigZagGemmSearchResult(
            schema="opencake.zigzag_gemm_search.v1",
            source="ZigZag-DSE official API",
            version=getattr(zigzag, "__version__", "unknown"),
            optimization=self.optimization,
            temporal_engine=self.temporal_engine,
            tile_plan=tile_plan,
            candidate_plans=feasible_plans,
            zigzag_resident_k_upper_bound=resident_k_upper_bound,
            energy_pj=float(energy),
            latency_cycles=float(latency),
            ideal_compute_cycles=float(cme.ideal_temporal_cycle),
            temporal_mapping=temporal_mapping,
            spatial_mapping=spatial_mapping,
            workload_path=str(workload_path),
            accelerator_path=str(accelerator_path),
            mapping_path=str(mapping_path),
        )
        result.write(output / "zigzag_search_result.json")
        return result

    @staticmethod
    def _workload_description(
        tile_m: int,
        tile_n: int,
        workload: GemmWorkload,
    ) -> list[dict[str, Any]]:
        precision = workload.bytes_per_element * 8
        return [
            {
                "id": 0,
                "operator_type": "Gemm",
                "equation": "O[d0][d1]+=I[d0][d2]*W[d2][d1]",
                "dimension_relations": [],
                "loop_dims": ["D0", "D1", "D2"],
                "loop_sizes": [tile_m, tile_n, workload.k],
                "operand_precision": {
                    "W": precision,
                    "I": precision,
                    "O": precision,
                    "O_final": precision,
                },
                "operand_source": {"I": 0, "W": 0},
            }
        ]

    @staticmethod
    def _accelerator_description(
        tile_m: int,
        tile_n: int,
        workload: GemmWorkload,
        hardware: HardwareSpec,
    ) -> dict[str, Any]:
        precision = workload.bytes_per_element * 8
        c_bits = tile_m * tile_n * precision
        if c_bits >= hardware.sram_bytes_per_rank * 8:
            raise ValueError("C tile alone does not fit PE-local SRAM")
        ab_bits = hardware.sram_bytes_per_rank * 8 - c_bits
        macs_per_cycle = max(1, int(hardware.compute_ops_per_cycle_per_rank // 2))
        stream_capacity = max(
            1 << 40,
            (tile_m * workload.k + workload.k * tile_n + tile_m * tile_n) * precision,
        )
        sram_bw_bits = max(precision, int(hardware.sram_bandwidth_bytes_per_cycle_per_rank * 8))
        stream_bw_bits = max(precision, int(hardware.link_bandwidth_bytes_per_cycle * 8))

        def rw_port(name: str, operands: tuple[str, ...], bandwidth: int) -> dict[str, Any]:
            transfers = ("fh", "tl", "fl", "th")
            return {
                "name": name,
                "type": "read_write",
                "bandwidth_min": precision,
                "bandwidth_max": bandwidth,
                "allocation": [f"{operand}, {transfer}" for operand in operands for transfer in transfers],
            }

        return {
            "name": "opencake_pe_48kib",
            "memories": {
                "a_rf": {
                    "size": macs_per_cycle * precision,
                    "r_cost": 0.1,
                    "w_cost": 0.1,
                    "area": 0.0,
                    "latency": 1,
                    "operands": ["I1"],
                    "ports": [rw_port("rw_port_1", ("I1",), macs_per_cycle * precision)],
                    "served_dimensions": [],
                },
                "b_rf": {
                    "size": macs_per_cycle * precision,
                    "r_cost": 0.1,
                    "w_cost": 0.1,
                    "area": 0.0,
                    "latency": 1,
                    "operands": ["I2"],
                    "ports": [rw_port("rw_port_1", ("I2",), macs_per_cycle * precision)],
                    "served_dimensions": [],
                },
                "acc_rf": {
                    "size": precision,
                    "r_cost": 0.1,
                    "w_cost": 0.1,
                    "area": 0.0,
                    "latency": 1,
                    "operands": ["O"],
                    "ports": [rw_port("rw_port_1", ("O",), precision)],
                    "served_dimensions": [],
                },
                "ab_sram": {
                    "size": ab_bits,
                    "r_cost": 1.0,
                    "w_cost": 1.0,
                    "area": 0.0,
                    "latency": 1,
                    "operands": ["I1", "I2"],
                    "ports": [rw_port("rw_port_1", ("I1", "I2"), sram_bw_bits)],
                    "served_dimensions": ["D1"],
                    "force_double_buffering": hardware.double_buffer_ab,
                },
                "c_sram": {
                    "size": c_bits,
                    "r_cost": 1.0,
                    "w_cost": 1.0,
                    "area": 0.0,
                    "latency": 1,
                    "operands": ["O"],
                    "ports": [rw_port("rw_port_1", ("O",), sram_bw_bits)],
                    "served_dimensions": ["D1"],
                },
                "stream_memory": {
                    "size": stream_capacity,
                    "r_cost": 10.0,
                    "w_cost": 10.0,
                    "area": 0.0,
                    "latency": max(1, int(hardware.router_latency_cycles)),
                    "operands": ["I1", "I2", "O"],
                    "ports": [rw_port("rw_port_1", ("I1", "I2", "O"), stream_bw_bits)],
                    "served_dimensions": ["D1"],
                },
            },
            "operational_array": {
                "input_precision": [precision, precision],
                "unit_energy": 1.0,
                "unit_area": 1.0,
                "dimensions": ["D1"],
                "sizes": [macs_per_cycle],
            },
        }

    @staticmethod
    def _mapping_description(
        workload: GemmWorkload,
        hardware: HardwareSpec,
    ) -> list[dict[str, Any]]:
        spatial_k = min(workload.k, max(1, int(hardware.compute_ops_per_cycle_per_rank // 2)))
        return [
            {
                "name": "default",
                "spatial_mapping": {"D1": [f"D2, {spatial_k}"]},
                "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
            },
            {
                "name": "Gemm",
                "spatial_mapping": {"D1": [f"D2, {spatial_k}"]},
                "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
            },
        ]

    @staticmethod
    def _resident_k_tile(cme: Any, workload: GemmWorkload) -> int:
        spatial_k = 1
        for levels in cme.spatial_mapping_dict_int.values():
            for level in levels:
                for dim, factor in level:
                    if str(dim) == "D2":
                        spatial_k = max(spatial_k, int(factor))
        mapping = cme.temporal_mapping.mapping_dic_origin
        input_operand = next((operand for operand in mapping if str(operand) == "I"), None)
        weight_operand = next((operand for operand in mapping if str(operand) == "W"), None)
        if input_operand is None or weight_operand is None:
            raise RuntimeError("ZigZag temporal mapping is missing GEMM I/W operands")

        # Level zero is an operand RF and level one is the shared A/B SRAM.
        # Count loop-prime factors allocated at or below that SRAM for both
        # operands. Counter intersection preserves duplicate prime factors.
        input_local = Counter(
            (str(dim), int(factor))
            for level in mapping[input_operand][:2]
            for dim, factor in level
        )
        weight_local = Counter(
            (str(dim), int(factor))
            for level in mapping[weight_operand][:2]
            for dim, factor in level
        )
        common = input_local & weight_local
        temporal_k = 1
        for (dim, factor), count in common.items():
            if dim == "D2":
                temporal_k *= factor**count
        return min(workload.k, spatial_k * temporal_k)

    @staticmethod
    def _mapping_record(mapping: Any) -> dict[str, list[list[tuple[str, int]]]]:
        return {
            str(operand): [
                [(str(dim), int(factor)) for dim, factor in level]
                for level in levels
            ]
            for operand, levels in mapping.items()
        }

    @staticmethod
    def _write_json_yaml(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
