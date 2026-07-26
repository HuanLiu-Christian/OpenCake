from __future__ import annotations

from dataclasses import dataclass
import gzip
import importlib.util
import json
from math import ceil
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, BinaryIO, Iterable, Mapping

from wafersim.opencake.workload import GemmWorkload

DEFAULT_CHAKRA_PROTOBUF = Path(
    "/tmp/astra-sim-micro2024/micro2024/astra-sim/extern/graph_frontend/"
    "chakra/schema/protobuf/et_def_pb2.py"
)
EMBEDDED_CHAKRA_PROTOBUF = Path(__file__).with_name("proto") / "et_def_pb2.py"
CHAKRA_PROTOBUF_CANDIDATES = (
    EMBEDDED_CHAKRA_PROTOBUF,
    DEFAULT_CHAKRA_PROTOBUF,
    Path(
        "/tmp/astra-sim-micro2024/micro2024/astra-sim/extern/graph_frontend/"
        "chakra/build/lib/chakra/schema/protobuf/et_def_pb2.py"
    ),
    Path(
        "/tmp/astra-sim-micro2024/micro2024/symbolic_tensor_graph/symbolic_tensor_graph/"
        "chakra/backends/chakra_00_4_backend/et_def/et_def_pb2.py"
    ),
)


@dataclass(frozen=True, slots=True)
class ChakraEtPackage:
    """Astra-Sim native rank-sharded Chakra ET workload."""

    prefix: Path
    rank_files: tuple[Path, ...]

    def summary(self) -> dict[str, object]:
        return {
            "prefix": str(self.prefix),
            "rank_files": [str(path) for path in self.rank_files],
            "ranks": len(self.rank_files),
        }


@dataclass(frozen=True, slots=True)
class ChakraNode:
    """Normalized workload node consumed by the OpenCake mapper."""

    id: int
    type: str
    op: str
    attrs: dict[str, Any]
    deps: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ChakraTrace:
    """Normalized workload graph input for OpenCake.

    The mapper consumes this small Python representation. The front door can be
    either the readable JSONL format used by the tests or a real Chakra ET
    protobuf file from the Chakra/Astra-Sim toolchain.
    """

    nodes: tuple[ChakraNode, ...]

    @classmethod
    def read(cls, path: str | Path) -> "ChakraTrace":
        trace_path = Path(path)
        if trace_path.suffix == ".et":
            return cls.read_et(trace_path)
        return cls.read_jsonl(trace_path)

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "ChakraTrace":
        records = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid Chakra JSONL on line {line_no}: {exc}") from exc
                records.append(_node_from_record(record, line_no))
        trace = cls(nodes=tuple(records))
        trace.validate()
        return trace

    @classmethod
    def read_et(
        cls,
        path: str | Path,
        *,
        proto_module_path: str | Path | None = None,
    ) -> "ChakraTrace":
        """Read a real Chakra execution trace protobuf file.

        Chakra ET files are length-delimited protobuf streams. Official Chakra
        writes one GlobalMetadata message followed by Node messages. This reader
        imports the generated `et_def_pb2.py` dynamically so protobuf remains an
        optional dependency for the rest of the simulator.
        """

        pb2 = _load_et_def_pb2(proto_module_path)
        records = []
        with _open_binary(path) as handle:
            payloads = list(_read_length_delimited_payloads(handle))

        if not payloads:
            raise ValueError(f"empty Chakra ET trace: {path}")

        node_payloads = payloads[1:]
        if not node_payloads:
            maybe_node = pb2.Node()
            maybe_node.ParseFromString(payloads[0])
            node_payloads = payloads if _looks_like_node(maybe_node) else []

        for index, payload in enumerate(node_payloads, start=1):
            node = pb2.Node()
            node.ParseFromString(payload)
            if not _looks_like_node(node):
                continue
            records.append(_node_from_et_proto(node, pb2, index))

        trace = cls(nodes=tuple(records))
        trace.validate()
        return trace

    def validate(self) -> None:
        ids = set()
        for node in self.nodes:
            if node.id in ids:
                raise ValueError(f"duplicate Chakra node id {node.id}")
            ids.add(node.id)
        for node in self.nodes:
            missing = [dep for dep in node.deps if dep not in ids]
            if missing:
                raise ValueError(f"node {node.id} depends on missing nodes {missing}")

    def gemm_workload(self) -> GemmWorkload:
        gemm_nodes = [node for node in self.nodes if node.type == "compute" and node.op == "gemm"]
        if len(gemm_nodes) != 1:
            raise ValueError(f"expected exactly one compute/gemm node, found {len(gemm_nodes)}")
        node = gemm_nodes[0]
        return GemmWorkload(
            m=_positive_int(node.attrs, "m", node.id),
            n=_positive_int(node.attrs, "n", node.id),
            k=_positive_int(node.attrs, "k", node.id),
            bytes_per_element=int(node.attrs.get("bytes_per_element", 4)),
            ops_per_mac=int(node.attrs.get("ops_per_mac", 2)),
        )


def write_gemm_et_trace(
    path: str | Path,
    workload: GemmWorkload,
    *,
    proto_module_path: str | Path | None = None,
) -> None:
    """Write a small real Chakra ET protobuf trace for a GEMM workload."""

    pb2 = _load_et_def_pb2(proto_module_path)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = pb2.GlobalMetadata(
        version="opencake.chakra.et.v1",
        attr=[
            pb2.AttributeProto(name="schema", string_val="chakra.et"),
            pb2.AttributeProto(name="producer", string_val="wafersim.opencake"),
        ],
    )
    gemm = pb2.Node(
        id=1,
        name="gemm",
        type=pb2.COMP_NODE,
        attr=[
            pb2.AttributeProto(name="op", string_val="gemm"),
            pb2.AttributeProto(name="m", int64_val=workload.m),
            pb2.AttributeProto(name="n", int64_val=workload.n),
            pb2.AttributeProto(name="k", int64_val=workload.k),
            pb2.AttributeProto(name="bytes_per_element", int64_val=workload.bytes_per_element),
            pb2.AttributeProto(name="ops_per_mac", int64_val=workload.ops_per_mac),
        ],
    )
    with output.open("wb") as handle:
        _write_length_delimited_message(handle, metadata)
        _write_length_delimited_message(handle, gemm)


def write_execution_trace_et(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    proto_module_path: str | Path | None = None,
) -> Path:
    """Write one formal Chakra ET containing the complete scheduled DAG.

    This aggregate file is useful for inspection and round-trip validation.
    Astra-Sim's native multi-rank loader uses :func:`write_ranked_execution_trace_et`
    instead because it expects one ``<prefix>.<rank>.et`` file per rank.
    """

    pb2 = _load_et_def_pb2(proto_module_path)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = tuple(dict(record) for record in records)
    rank_cols = _infer_rank_cols(normalized)
    for record in normalized:
        if record.get("type") == "communication":
            record["rank_cols"] = rank_cols
    _write_et_file(output, normalized, pb2, rank_id=None)
    return output


def write_ranked_execution_trace_et(
    prefix: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    rank_rows: int,
    rank_cols: int,
    proto_module_path: str | Path | None = None,
) -> ChakraEtPackage:
    """Project a scheduled global DAG into Astra-Sim rank-local ET files.

    Communication is represented as a matched ``COMM_SEND_NODE`` in the
    source rank and a synthesized ``COMM_RECV_NODE`` in the destination rank.
    Dependencies on a remote send are rewritten to the local receive node.
    """

    if rank_rows <= 0 or rank_cols <= 0:
        raise ValueError("rank_rows and rank_cols must be positive")

    pb2 = _load_et_def_pb2(proto_module_path)
    output_prefix = Path(prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    normalized = tuple(dict(record) for record in records)
    records_by_id = {int(record["id"]): record for record in normalized}
    if len(records_by_id) != len(normalized):
        raise ValueError("Chakra records contain duplicate node IDs")

    rank_coords = tuple(
        (row, col) for row in range(rank_rows) for col in range(rank_cols)
    )
    per_rank: dict[tuple[int, int], list[dict[str, Any]]] = {
        rank: [] for rank in rank_coords
    }
    global_records = [record for record in normalized if _record_rank(record) is None]
    for rank in rank_coords:
        per_rank[rank].extend(dict(record) for record in global_records)
    for record in normalized:
        rank = _record_rank(record)
        if rank is None:
            continue
        if rank not in per_rank:
            raise ValueError(f"record {record['id']} references rank {rank} outside the mesh")
        per_rank[rank].append(dict(record))

    next_id = max(records_by_id, default=0) + 1
    recv_for_send: dict[tuple[tuple[int, int], int], int] = {}
    for record in normalized:
        if record.get("type") != "communication" or record.get("op") != "send":
            continue
        src = _coord(record.get("src_rank"), "src_rank", int(record["id"]))
        dst = _coord(record.get("dst_rank"), "dst_rank", int(record["id"]))
        if src not in per_rank or dst not in per_rank:
            raise ValueError(f"send {record['id']} has endpoint outside the rank mesh")
        recv_id = next_id
        next_id += 1
        recv_for_send[(dst, int(record["id"]))] = recv_id
        receive = dict(record)
        receive.update(
            {
                "id": recv_id,
                "op": "recv",
                "rank": list(dst),
                "deps": [],
                "source_send_id": int(record["id"]),
            }
        )
        per_rank[dst].append(receive)

    rank_files: list[Path] = []
    for rank in rank_coords:
        local_records = per_rank[rank]
        local_ids = {int(record["id"]) for record in local_records}
        rewritten: list[dict[str, Any]] = []
        for record in local_records:
            local = dict(record)
            if local.get("type") == "communication":
                local["rank_cols"] = rank_cols
            deps: list[int] = []
            for raw_dep in record.get("deps", ()):
                dep = int(raw_dep)
                if dep in local_ids:
                    deps.append(dep)
                    continue
                recv_id = recv_for_send.get((rank, dep))
                if recv_id is not None:
                    deps.append(recv_id)
            local["deps"] = sorted(set(deps))
            rewritten.append(local)

        rank_id = rank[0] * rank_cols + rank[1]
        rank_path = Path(f"{output_prefix}.{rank_id}.et")
        _write_et_file(rank_path, rewritten, pb2, rank_id=rank_id)
        rank_files.append(rank_path)

    return ChakraEtPackage(prefix=output_prefix, rank_files=tuple(rank_files))


def _write_et_file(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    pb2: ModuleType,
    *,
    rank_id: int | None,
) -> None:
    metadata_attrs = [
        pb2.AttributeProto(name="schema", string_val="chakra.et"),
        pb2.AttributeProto(name="producer", string_val="wafersim.opencake"),
        pb2.AttributeProto(name="time_unit", string_val="cycle"),
    ]
    if rank_id is not None:
        metadata_attrs.append(pb2.AttributeProto(name="rank", uint32_val=rank_id))
    metadata = pb2.GlobalMetadata(
        version="opencake.chakra.et.v1",
        attr=metadata_attrs,
    )
    ordered = sorted(
        records,
        key=lambda record: (float(record.get("start_cycle", 0.0)), int(record["id"])),
    )
    with path.open("wb") as handle:
        _write_length_delimited_message(handle, metadata)
        for record in ordered:
            _write_length_delimited_message(handle, _record_to_et_node(record, pb2))


def _record_to_et_node(record: Mapping[str, Any], pb2: ModuleType) -> Any:
    node_id = int(record["id"])
    op = str(record.get("op", "unknown"))
    node_type = _record_node_type(record, pb2)
    attrs = {
        key: value
        for key, value in record.items()
        if key not in {"id", "type", "op", "deps"}
    }
    attrs["op"] = op
    attrs["opencake_kind"] = str(record.get("type", "unknown"))

    rank = _record_rank(record)
    if rank is not None:
        attrs["rank_row"] = rank[0]
        attrs["rank_col"] = rank[1]

    if node_type == pb2.COMP_NODE:
        attrs["num_ops"] = int(record.get("ops", attrs.get("num_ops", 0)))
        attrs["tensor_size"] = _compute_tensor_size(record)
        attrs.setdefault("is_cpu_op", False)
    elif node_type in {pb2.MEM_LOAD_NODE, pb2.MEM_STORE_NODE}:
        attrs["tensor_size"] = int(record.get("bytes", attrs.get("tensor_size", 0)))
    elif node_type in {pb2.COMM_SEND_NODE, pb2.COMM_RECV_NODE}:
        src = _coord(record.get("src_rank"), "src_rank", node_id)
        dst = _coord(record.get("dst_rank"), "dst_rank", node_id)
        rank_cols = int(record.get("rank_cols", 0))
        if rank_cols <= 0:
            rank_cols = max(src[1], dst[1]) + 1
        attrs["comm_src"] = src[0] * rank_cols + src[1]
        attrs["comm_dst"] = dst[0] * rank_cols + dst[1]
        attrs["comm_size"] = int(record.get("bytes", 0))
        attrs["comm_tag"] = int(record.get("source_send_id", node_id))
        attrs.setdefault("is_cpu_op", False)

    start_cycle = float(record.get("start_cycle", 0.0))
    duration_cycles = float(
        record.get(
            "duration_cycles",
            float(record.get("end_cycle", start_cycle)) - start_cycle,
        )
    )
    return pb2.Node(
        id=node_id,
        name=op,
        type=node_type,
        data_deps=[int(dep) for dep in record.get("deps", ())],
        start_time_micros=max(0, ceil(start_cycle / 1000.0)),
        duration_micros=max(0, ceil(duration_cycles / 1000.0)),
        attr=[_attribute_proto(pb2, key, value) for key, value in attrs.items()],
    )


def _record_node_type(record: Mapping[str, Any], pb2: ModuleType) -> int:
    kind = str(record.get("type", "unknown"))
    op = str(record.get("op", "unknown"))
    if kind in {"control", "metadata"}:
        return pb2.METADATA_NODE
    if kind == "compute":
        return pb2.COMP_NODE
    if kind == "memory":
        return pb2.MEM_STORE_NODE if op == "alloc" else pb2.MEM_LOAD_NODE
    if kind == "communication":
        return pb2.COMM_RECV_NODE if op == "recv" else pb2.COMM_SEND_NODE
    return pb2.INVALID_NODE


def _attribute_proto(pb2: ModuleType, name: str, value: Any) -> Any:
    if isinstance(value, bool):
        return pb2.AttributeProto(name=name, bool_val=value)
    if isinstance(value, int):
        if value >= 0:
            return pb2.AttributeProto(name=name, uint64_val=value)
        return pb2.AttributeProto(name=name, int64_val=value)
    if isinstance(value, float):
        return pb2.AttributeProto(name=name, double_val=value)
    if isinstance(value, str):
        return pb2.AttributeProto(name=name, string_val=value)
    return pb2.AttributeProto(
        name=name,
        string_val=json.dumps(value, sort_keys=True, separators=(",", ":")),
    )


def _compute_tensor_size(record: Mapping[str, Any]) -> int:
    if int(record.get("bytes", 0)) > 0:
        return int(record["bytes"])
    m = int(record.get("m", 0))
    n = int(record.get("n", 0))
    k = int(record.get("k", 0))
    if min(m, n, k) <= 0:
        return 0
    return (m * k + k * n + m * n) * int(record.get("bytes_per_element", 4))


def _record_rank(record: Mapping[str, Any]) -> tuple[int, int] | None:
    raw = record.get("rank")
    if raw is None:
        return None
    return _coord(raw, "rank", int(record["id"]))


def _infer_rank_cols(records: Iterable[Mapping[str, Any]]) -> int:
    max_col = -1
    for record in records:
        for key in ("rank", "src_rank", "dst_rank"):
            raw = record.get(key)
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                max_col = max(max_col, int(raw[1]))
    return max_col + 1 if max_col >= 0 else 1


def _coord(value: Any, name: str, node_id: int) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"record {node_id} has invalid {name}: {value!r}")
    return int(value[0]), int(value[1])


def _node_from_record(record: dict[str, Any], line_no: int) -> ChakraNode:
    if "id" not in record or "type" not in record or "op" not in record:
        raise ValueError(f"Chakra record on line {line_no} must contain id, type, and op")
    attrs = {
        key: value
        for key, value in record.items()
        if key not in {"id", "type", "op", "deps"}
    }
    return ChakraNode(
        id=int(record["id"]),
        type=str(record["type"]),
        op=str(record["op"]),
        attrs=attrs,
        deps=tuple(int(dep) for dep in record.get("deps", ())),
    )


def _node_from_et_proto(node: Any, pb2: ModuleType, index: int) -> ChakraNode:
    attrs = {attr.name: _attribute_value(attr) for attr in node.attr}
    node_type = _node_type_name(node.type, pb2)
    op = _infer_op(node.name, node_type, attrs)
    attrs.setdefault("name", node.name)
    if getattr(node, "duration_micros", 0):
        attrs.setdefault("duration_micros", int(node.duration_micros))
    if getattr(node, "start_time_micros", 0):
        attrs.setdefault("start_time_micros", int(node.start_time_micros))
    return ChakraNode(
        id=int(node.id),
        type=node_type,
        op=op,
        attrs=attrs,
        deps=tuple(int(dep) for dep in tuple(node.ctrl_deps) + tuple(node.data_deps)),
    )


def _load_et_def_pb2(proto_module_path: str | Path | None) -> ModuleType:
    try:
        import google.protobuf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading Chakra .et files requires the protobuf Python package. "
            "Install the optional Chakra dependency or run inside the Astra-Sim "
            "Chakra environment. JSONL traces continue to work without it."
        ) from exc

    env_path = os.environ.get("WAFERSIM_CHAKRA_ET_DEF_PB2")
    if proto_module_path is not None:
        candidates = (Path(proto_module_path),)
    elif env_path:
        candidates = (Path(env_path), *CHAKRA_PROTOBUF_CANDIDATES)
    else:
        candidates = CHAKRA_PROTOBUF_CANDIDATES

    errors: list[str] = []
    for index, module_path in enumerate(candidates):
        if not module_path.exists():
            errors.append(f"{module_path}: missing")
            continue
        module_name = f"wafersim_chakra_et_def_pb2_{index}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            errors.append(f"{module_path}: cannot create import spec")
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - depends on external protobuf versions
            errors.append(f"{module_path}: {type(exc).__name__}: {exc}")
            sys.modules.pop(spec.name, None)
            continue
        return module

    raise ImportError(
        "cannot load Chakra generated protobuf module. Tried:\n  "
        + "\n  ".join(errors)
        + "\nSet WAFERSIM_CHAKRA_ET_DEF_PB2 or pass proto_module_path=..."
    )


def _open_binary(path: str | Path) -> BinaryIO:
    trace_path = Path(path)
    handle = trace_path.open("rb")
    magic = handle.read(2)
    handle.seek(0)
    if magic == b"\x1f\x8b":
        handle.close()
        return gzip.open(trace_path, "rb")
    return handle


def _read_length_delimited_payloads(handle: BinaryIO):
    while True:
        size = _decode_varint32(handle)
        if size is None:
            return
        if size <= 0:
            raise ValueError(f"invalid Chakra protobuf message size {size}")
        payload = handle.read(size)
        if len(payload) != size:
            raise ValueError("truncated Chakra protobuf message")
        yield payload


def _decode_varint32(handle: BinaryIO) -> int | None:
    result = 0
    shift = 0
    while True:
        raw = handle.read(1)
        if not raw:
            if shift == 0:
                return None
            raise ValueError("truncated Chakra protobuf varint")
        byte = raw[0]
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result & 0xFFFFFFFF
        shift += 7
        if shift >= 64:
            raise ValueError("too many bytes while decoding Chakra protobuf varint")


def _write_length_delimited_message(handle: BinaryIO, message: Any) -> None:
    payload = message.SerializeToString()
    _encode_varint32(handle, len(payload))
    handle.write(payload)


def _encode_varint32(handle: BinaryIO, value: int) -> None:
    bits = value & 0x7F
    value >>= 7
    while value:
        handle.write(bytes((0x80 | bits,)))
        bits = value & 0x7F
        value >>= 7
    handle.write(bytes((bits,)))


def _looks_like_node(node: Any) -> bool:
    return bool(node.id or node.name or node.type or node.attr or node.ctrl_deps or node.data_deps)


def _node_type_name(node_type: int, pb2: ModuleType) -> str:
    if node_type == pb2.COMP_NODE:
        return "compute"
    if node_type in {pb2.MEM_LOAD_NODE, pb2.MEM_STORE_NODE}:
        return "memory"
    if node_type in {pb2.COMM_SEND_NODE, pb2.COMM_RECV_NODE, pb2.COMM_COLL_NODE}:
        return "communication"
    if node_type == pb2.METADATA_NODE:
        return "metadata"
    return "unknown"


def _infer_op(name: str, node_type: str, attrs: dict[str, Any]) -> str:
    explicit = attrs.get("op")
    if explicit is not None:
        return str(explicit)
    lowered = name.lower()
    if node_type == "compute" and "gemm" in lowered:
        return "gemm"
    if node_type == "memory":
        if "store" in lowered or "write" in lowered:
            return "store"
        return "load"
    if node_type == "communication":
        return attrs.get("comm_type", "comm")
    return lowered or node_type


def _attribute_value(attr: Any) -> Any:
    field = attr.WhichOneof("value")
    if field is None:
        return None
    value = getattr(attr, field)
    if field.endswith("_list"):
        return list(value.values)
    if field == "bytes_val":
        return bytes(value).hex()
    if field == "bytes_list":
        return [bytes(item).hex() for item in value.values]
    return value


def _positive_int(attrs: dict[str, Any], key: str, node_id: int) -> int:
    if key not in attrs:
        raise ValueError(f"compute/gemm node {node_id} is missing {key}")
    value = int(attrs[key])
    if value <= 0:
        raise ValueError(f"compute/gemm node {node_id} has non-positive {key}")
    return value
