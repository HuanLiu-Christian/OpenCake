from __future__ import annotations

from dataclasses import dataclass
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, BinaryIO

from wafersim.opencake.workload import GemmWorkload

DEFAULT_CHAKRA_PROTOBUF = Path(
    "/tmp/astra-sim-micro2024/micro2024/astra-sim/extern/graph_frontend/"
    "chakra/schema/protobuf/et_def_pb2.py"
)
CHAKRA_PROTOBUF_CANDIDATES = (
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
