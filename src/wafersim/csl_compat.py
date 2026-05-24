from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable

import simpy

from wafersim.config import Coord
from wafersim.network import WaferMesh


@dataclass(slots=True)
class LayoutRegion:
    rows: int
    cols: int
    origin: Coord = (0, 0)

    def coords(self) -> list[Coord]:
        row0, col0 = self.origin
        return [
            (row0 + row, col0 + col)
            for row in range(self.rows)
            for col in range(self.cols)
        ]

    def linear_id(self, coord: Coord) -> int:
        row0, col0 = self.origin
        row, col = coord
        return (row - row0) * self.cols + (col - col0)


@dataclass(slots=True)
class TraceEvent:
    name: str
    coord: Coord
    value: int | float | str
    cycle: float


@dataclass(slots=True)
class SimulatedPointer:
    buffer: list[int | float]
    index: int = 0

    def load(self) -> int | float:
        return self.buffer[self.index]

    def store(self, value: int | float) -> None:
        self.buffer[self.index] = value


@dataclass(slots=True)
class MallocArena:
    capacity_bytes: int
    used_bytes: int = 0
    allocations: list[int] = field(default_factory=list)

    def malloc(self, size_bytes: int) -> int:
        if size_bytes < 0:
            raise ValueError("allocation size cannot be negative")
        if self.used_bytes + size_bytes > self.capacity_bytes:
            raise MemoryError("simulated CSL arena out of memory")
        address = self.used_bytes
        self.used_bytes += size_bytes
        self.allocations.append(size_bytes)
        return address


class CslBuiltins:
    """Python-side approximations of common CSL builtins.

    These helpers model compile-time/layout behavior as plain Python values.
    They do not parse or compile CSL source.
    """

    @staticmethod
    def set_rectangle(rows: int, cols: int, origin: Coord = (0, 0)) -> LayoutRegion:
        if rows <= 0 or cols <= 0:
            raise ValueError("rectangle dimensions must be positive")
        return LayoutRegion(rows=rows, cols=cols, origin=origin)

    @staticmethod
    def map_builtin(coord: Coord, region: LayoutRegion) -> int:
        return region.linear_id(coord)

    @staticmethod
    def zeros(length: int, value: int | float = 0) -> list[int | float]:
        if length < 0:
            raise ValueError("length cannot be negative")
        return [value for _ in range(length)]

    @staticmethod
    def constant(length: int, value: int | float) -> list[int | float]:
        if length < 0:
            raise ValueError("length cannot be negative")
        return [value for _ in range(length)]

    @staticmethod
    def ptr(buffer: list[int | float], index: int = 0) -> SimulatedPointer:
        return SimulatedPointer(buffer=buffer, index=index)

    @staticmethod
    def sizeof(elements: int, bytes_per_element: int) -> int:
        return elements * bytes_per_element

    @staticmethod
    def bitcast_int(value: int, bits: int) -> int:
        if bits <= 0:
            raise ValueError("bits must be positive")
        mask = (1 << bits) - 1
        return value & mask

    @staticmethod
    def import_module(name: str) -> str:
        if not name.startswith("<") or not name.endswith(">"):
            raise ValueError("CSL library imports use angle brackets")
        return name[1:-1]

    @staticmethod
    def select(condition: bool, if_true: int | float, if_false: int | float) -> int | float:
        return if_true if condition else if_false

    @staticmethod
    def concat(*buffers: list[int | float]) -> list[int | float]:
        output: list[int | float] = []
        for buffer in buffers:
            output.extend(buffer)
        return output

    @staticmethod
    def range_values(stop: int, start: int = 0, step: int = 1) -> list[int]:
        if step == 0:
            raise ValueError("range step cannot be zero")
        return list(range(start, stop, step))


class MathLibrary:
    def sqrt_f32(self, value: float) -> float:
        return math.sqrt(value)

    def exp_f32(self, value: float) -> float:
        return math.exp(value)

    def log_f32(self, value: float) -> float:
        return math.log(value)

    def sin_f32(self, value: float) -> float:
        return math.sin(value)

    def cos_f32(self, value: float) -> float:
        return math.cos(value)

    def abs_f32(self, value: float) -> float:
        return abs(value)

    def estimate_ops(self, calls: int, ops_per_call: int = 8) -> int:
        return calls * ops_per_call


class RandomLibrary:
    def __init__(self, seed: int = 1) -> None:
        self._rng = random.Random(seed)

    def random_f32(self, low: float = 0.0, high: float = 1.0) -> float:
        return self._rng.uniform(low, high)

    def random_i16(self, low: int = 0, high: int = 32767) -> int:
        return self._rng.randint(low, high)


class TimeLibrary:
    def __init__(self, env: simpy.Environment) -> None:
        self.env = env

    def timestamp(self) -> float:
        return float(self.env.now)


class TimerLibrary:
    def __init__(self, env: simpy.Environment) -> None:
        self.env = env
        self._starts: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = float(self.env.now)

    def elapsed(self, name: str) -> float:
        return float(self.env.now) - self._starts[name]


class ComplexLibrary:
    def add(self, a: complex, b: complex) -> complex:
        return a + b

    def mul(self, a: complex, b: complex) -> complex:
        return a * b

    def magnitude(self, value: complex) -> float:
        return abs(value)


class ControlLibrary:
    def __init__(self, env: simpy.Environment) -> None:
        self.env = env
        self.blocked = False
        self.unblocked_at: float | None = None

    def block_cmd_stream(self) -> None:
        self.blocked = True

    def unblock_cmd_stream(self) -> None:
        self.blocked = False
        self.unblocked_at = float(self.env.now)


class DataUtilsLibrary:
    def chunk(self, total_bytes: int, chunk_bytes: int) -> list[int]:
        if total_bytes < 0:
            raise ValueError("total bytes cannot be negative")
        if chunk_bytes <= 0:
            raise ValueError("chunk bytes must be positive")
        chunks = []
        remaining = total_bytes
        while remaining > 0:
            chunk = min(chunk_bytes, remaining)
            chunks.append(chunk)
            remaining -= chunk
        return chunks

    def flatten(self, rows: Iterable[Iterable[int | float]]) -> list[int | float]:
        return [item for row in rows for item in row]

    def reshape(self, values: list[int | float], rows: int, cols: int) -> list[list[int | float]]:
        if rows * cols != len(values):
            raise ValueError("shape does not match value count")
        return [values[row * cols : (row + 1) * cols] for row in range(rows)]


class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    WEST = "west"
    EAST = "east"


class DirectionsLibrary:
    OFFSETS: dict[Direction, Coord] = {
        Direction.NORTH: (-1, 0),
        Direction.SOUTH: (1, 0),
        Direction.WEST: (0, -1),
        Direction.EAST: (0, 1),
    }

    def move(self, coord: Coord, direction: Direction) -> Coord:
        row, col = coord
        d_row, d_col = self.OFFSETS[direction]
        return (row + d_row, col + d_col)

    def opposite(self, direction: Direction) -> Direction:
        return {
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.WEST: Direction.EAST,
            Direction.EAST: Direction.WEST,
        }[direction]


class DsdOpsLibrary:
    """Conceptual data-structure descriptor operations.

    This models strided/gather/scatter address lists, not real hardware DSDs.
    """

    def strided(self, base: int, count: int, stride: int) -> list[int]:
        return [base + index * stride for index in range(count)]

    def gather(self, buffer: list[int | float], indices: list[int]) -> list[int | float]:
        return [buffer[index] for index in indices]

    def scatter(
        self,
        buffer: list[int | float],
        indices: list[int],
        values: list[int | float],
    ) -> None:
        if len(indices) != len(values):
            raise ValueError("indices and values must have the same length")
        for index, value in zip(indices, values, strict=True):
            buffer[index] = value


class EmptyLibrary:
    def noop(self) -> None:
        return None


class StringLibrary:
    def concat(self, *parts: str) -> str:
        return "".join(parts)

    def length(self, value: str) -> int:
        return len(value)


class TypesLibrary:
    BYTES = {
        "i8": 1,
        "u8": 1,
        "i16": 2,
        "u16": 2,
        "i32": 4,
        "u32": 4,
        "f16": 2,
        "f32": 4,
    }

    def sizeof(self, type_name: str) -> int:
        return self.BYTES[type_name]

    def clamp_to_unsigned(self, value: int, bits: int) -> int:
        return value & ((1 << bits) - 1)


class KernelsLibrary:
    def saxpy(self, alpha: float, x: list[float], y: list[float]) -> list[float]:
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        return [alpha * x_value + y_value for x_value, y_value in zip(x, y, strict=True)]

    def dot(self, x: list[float], y: list[float]) -> float:
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        return sum(x_value * y_value for x_value, y_value in zip(x, y, strict=True))


class FftKernelLibrary:
    def estimate_ops(self, points: int) -> int:
        if points <= 0:
            raise ValueError("FFT points must be positive")
        return 5 * points * max(1, math.ceil(math.log2(points)))

    def bit_reverse_indices(self, points: int) -> list[int]:
        bits = max(1, math.ceil(math.log2(points)))
        return [
            int(f"{index:0{bits}b}"[::-1], 2) % points
            for index in range(points)
        ]


class TallyKernelLibrary:
    def histogram(self, values: list[int], bins: int) -> list[int]:
        if bins <= 0:
            raise ValueError("bins must be positive")
        counts = [0 for _ in range(bins)]
        for value in values:
            counts[value % bins] += 1
        return counts


class DebugLibrary:
    def __init__(self, env: simpy.Environment) -> None:
        self.env = env
        self.events: list[TraceEvent] = []

    def trace(self, name: str, coord: Coord, value: int | float | str) -> None:
        self.events.append(TraceEvent(name=name, coord=coord, value=value, cycle=self.env.now))


class SimprintLibrary:
    def __init__(self, env: simpy.Environment) -> None:
        self.env = env
        self.messages: list[str] = []

    def print(self, coord: Coord, message: str) -> None:
        self.messages.append(f"{self.env.now:g} PE{coord}: {message}")


class LayoutLibrary:
    def __init__(self, region: LayoutRegion) -> None:
        self.region = region

    def get_coord(self, linear_id: int) -> Coord:
        row = linear_id // self.region.cols
        col = linear_id % self.region.cols
        row0, col0 = self.region.origin
        return (row0 + row, col0 + col)

    def get_linear_id(self, coord: Coord) -> int:
        return self.region.linear_id(coord)


class MallocLibrary:
    def __init__(self, capacity_bytes: int) -> None:
        self.arena = MallocArena(capacity_bytes)

    def malloc(self, size_bytes: int) -> int:
        return self.arena.malloc(size_bytes)


class MemcpyLibrary:
    def __init__(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        packet_payload_bytes: int,
    ) -> None:
        self.env = env
        self.mesh = mesh
        self.packet_payload_bytes = packet_payload_bytes
        self.bytes = 0
        self.packets = 0

    def h2d(self, dst: Coord, size_bytes: int):
        yield from self._copy(src=(0, 0), dst=dst, size_bytes=size_bytes, kind="compat:memcpy:h2d")

    def d2h(self, src: Coord, size_bytes: int):
        yield from self._copy(src=src, dst=(0, 0), size_bytes=size_bytes, kind="compat:memcpy:d2h")

    def p2p(self, src: Coord, dst: Coord, size_bytes: int):
        yield from self._copy(src=src, dst=dst, size_bytes=size_bytes, kind="compat:memcpy:p2p")

    def _copy(self, src: Coord, dst: Coord, size_bytes: int, kind: str):
        remaining = size_bytes
        events = []
        while remaining > 0:
            chunk = min(self.packet_payload_bytes, remaining)
            events.append(
                self.env.process(
                    self.mesh.send(src=src, dst=dst, size_bytes=chunk, kind=kind)
                )
            )
            self.bytes += chunk
            self.packets += 1
            remaining -= chunk
        if events:
            yield simpy.events.AllOf(self.env, events)


class Collectives2DLibrary:
    def __init__(
        self,
        env: simpy.Environment,
        mesh: WaferMesh,
        coords: list[Coord],
    ) -> None:
        self.env = env
        self.mesh = mesh
        self.coords = coords
        self.packets = 0
        self.bytes = 0

    def broadcast(self, root: Coord, size_bytes: int):
        events = [
            self.env.process(
                self.mesh.send(
                    src=root,
                    dst=coord,
                    size_bytes=size_bytes,
                    kind="compat:collective:broadcast",
                )
            )
            for coord in self.coords
            if coord != root
        ]
        self._count(events, size_bytes)
        if events:
            yield simpy.events.AllOf(self.env, events)

    def reduce(self, root: Coord, size_bytes: int):
        events = [
            self.env.process(
                self.mesh.send(
                    src=coord,
                    dst=root,
                    size_bytes=size_bytes,
                    kind="compat:collective:reduce",
                )
            )
            for coord in self.coords
            if coord != root
        ]
        self._count(events, size_bytes)
        if events:
            yield simpy.events.AllOf(self.env, events)

    def allreduce(self, root: Coord, size_bytes: int):
        yield from self.reduce(root=root, size_bytes=size_bytes)
        yield from self.broadcast(root=root, size_bytes=size_bytes)

    def scatter(self, root: Coord, size_bytes: int):
        yield from self.broadcast(root=root, size_bytes=size_bytes)

    def gather(self, root: Coord, size_bytes: int):
        yield from self.reduce(root=root, size_bytes=size_bytes)

    def _count(self, events: list[simpy.Event], size_bytes: int) -> None:
        self.packets += len(events)
        self.bytes += len(events) * size_bytes


@dataclass(slots=True)
class Collectives2DParams:
    rows: int
    cols: int
    root: Coord = (0, 0)

    @property
    def size(self) -> int:
        return self.rows * self.cols


class Collectives2DPeLibrary:
    def __init__(self, params: Collectives2DParams) -> None:
        self.params = params

    def rank(self, coord: Coord) -> int:
        row, col = coord
        return row * self.params.cols + col

    def coord(self, rank: int) -> Coord:
        return (rank // self.params.cols, rank % self.params.cols)


class TileConfigLibrary:
    def __init__(self, env: simpy.Environment) -> None:
        self.env = env

    def fifo(self, capacity: int) -> simpy.Store:
        if capacity <= 0:
            raise ValueError("FIFO capacity must be positive")
        return simpy.Store(self.env, capacity=capacity)

    def filter_coords(
        self, coords: list[Coord], predicate: Callable[[Coord], bool]
    ) -> list[Coord]:
        return [coord for coord in coords if predicate(coord)]

    def switch_destinations(self, root: Coord, destinations: list[Coord]) -> list[Coord]:
        return [dst for dst in destinations if dst != root]


CSL_COMPAT_COVERAGE: dict[str, dict[str, str]] = {
    "builtins": {
        "layout rectangle / @set_rectangle": "implemented as LayoutRegion",
        "@map-style coordinate mapping": "implemented as linear_id mapping",
        "@zeros / @constant": "implemented as Python buffers",
        "pointer load/store": "implemented as SimulatedPointer",
        "sizeof / bitcast / select / concat / range": "implemented as value helpers",
        "@import_module": "implemented as import-name validation only",
        "@set_tile_code / compiler code placement": "omitted: compiler/layout linking semantics",
        "true fabric DSD builtins": "partial: conceptual address-list helpers only",
    },
    "libraries": {
        "<complex>": "implemented for add/mul/magnitude",
        "<control>": "implemented for command-stream block/unblock state",
        "<data_utils>": "implemented for chunk/flatten/reshape",
        "<debug>": "implemented as in-memory trace events",
        "<directions>": "implemented for N/S/E/W movement and opposites",
        "<dsd_ops>": "partial: strided/gather/scatter address-list model",
        "<empty>": "implemented as noop",
        "<layout>": "implemented for coord/rank mapping",
        "<malloc>": "implemented as PE-local arena allocator",
        "<math>": "implemented for common scalar math plus op estimates",
        "<memcpy>": "implemented as packetized H2D/D2H/P2P events",
        "<random>": "implemented as deterministic Python RNG",
        "<simprint>": "implemented as per-PE message log",
        "<string>": "implemented for concat/length",
        "<tile_config>": "implemented for FIFO/filter/switch abstractions",
        "<time>": "implemented with SimPy env.now",
        "<timer>": "implemented as named elapsed timers",
        "<types>": "implemented for size and unsigned clamping helpers",
        "<kernels>": "implemented for saxpy/dot examples",
        "<kernels/fft>": "implemented for FFT op estimates and bit-reversal",
        "<kernels/tally>": "implemented for histogram tally",
        "<collectives_2d>": "implemented for broadcast/reduce/allreduce/scatter/gather",
        "<collectives_2d/params>": "implemented as group parameter dataclass",
        "<collectives_2d/pe>": "implemented as rank/coord helpers",
        "exact hardware routing trees / colors / task IDs": "omitted: WSE-specific hardware semantics",
    },
}


def csl_compat_coverage() -> dict[str, dict[str, str]]:
    return CSL_COMPAT_COVERAGE


@dataclass
class CslCompatibilityRuntime:
    env: simpy.Environment
    mesh: WaferMesh
    region: LayoutRegion
    packet_payload_bytes: int = 256

    def __post_init__(self) -> None:
        coords = self.region.coords()
        self.builtins = CslBuiltins()
        self.complex = ComplexLibrary()
        self.control = ControlLibrary(self.env)
        self.data_utils = DataUtilsLibrary()
        self.directions = DirectionsLibrary()
        self.dsd_ops = DsdOpsLibrary()
        self.empty = EmptyLibrary()
        self.string = StringLibrary()
        self.timer = TimerLibrary(self.env)
        self.types = TypesLibrary()
        self.kernels = KernelsLibrary()
        self.fft = FftKernelLibrary()
        self.tally = TallyKernelLibrary()
        self.math = MathLibrary()
        self.random = RandomLibrary(seed=1)
        self.time = TimeLibrary(self.env)
        self.debug = DebugLibrary(self.env)
        self.simprint = SimprintLibrary(self.env)
        self.layout = LayoutLibrary(self.region)
        self.malloc = MallocLibrary(self.mesh.config.local_sram_bytes_per_pe)
        self.memcpy = MemcpyLibrary(self.env, self.mesh, self.packet_payload_bytes)
        self.collectives = Collectives2DLibrary(self.env, self.mesh, coords)
        self.collective_params = Collectives2DParams(
            rows=self.region.rows, cols=self.region.cols
        )
        self.collective_pe = Collectives2DPeLibrary(self.collective_params)
        self.tile_config = TileConfigLibrary(self.env)

    def summary(self) -> dict[str, int | float | list[str]]:
        return {
            "debug_events": len(self.debug.events),
            "simprint_messages": len(self.simprint.messages),
            "malloc_used_bytes": self.malloc.arena.used_bytes,
            "memcpy_packets": self.memcpy.packets,
            "memcpy_bytes": self.memcpy.bytes,
            "collective_packets": self.collectives.packets,
            "collective_bytes": self.collectives.bytes,
            "timestamp_cycles": self.time.timestamp(),
            "control_blocked": int(self.control.blocked),
            "simprint_log": self.simprint.messages,
        }
