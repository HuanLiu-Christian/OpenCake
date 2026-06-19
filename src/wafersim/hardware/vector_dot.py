from __future__ import annotations

from dataclasses import dataclass

import simpy

from wafersim.config import Coord, MeshConfig
from wafersim.hardware.pe import ProcessingElement
from wafersim.mapping.vector_dot_ir import VectorDotCommand


@dataclass(frozen=True, slots=True)
class VectorDotResult:
    left: tuple[int, ...]
    right: tuple[int, ...]
    result: int | None
    expected: int | None
    pe_results: dict[Coord, int | None]
    sim_time_cycles: float
    compute_tasks_completed: int
    memory_tasks_completed: int


class VectorDotPe:
    """SimPy golden model for PE-local vector dot compute semantics."""

    def __init__(
        self,
        env: simpy.Environment,
        *,
        max_words: int,
        word_bytes: int = 4,
        ops_per_mac: int = 2,
        coord: Coord = (0, 0),
        config: MeshConfig | None = None,
    ) -> None:
        if max_words <= 0:
            raise ValueError("max_words must be positive")
        if word_bytes <= 0:
            raise ValueError("word_bytes must be positive")
        if ops_per_mac <= 0:
            raise ValueError("ops_per_mac must be positive")

        mesh_config = config or MeshConfig(rows=1, cols=1)
        self.env = env
        self.coord = coord
        self.word_bytes = word_bytes
        self.ops_per_mac = ops_per_mac
        self.left = [0 for _ in range(max_words)]
        self.right = [0 for _ in range(max_words)]
        self.left_count = 0
        self.right_count = 0
        self.result: int | None = None
        self.expected: int | None = None
        self.pe = ProcessingElement(
            env=env,
            coord=coord,
            compute_throughput_ops_per_cycle=mesh_config.compute_throughput_ops_per_cycle,
            local_sram_bytes=max_words * word_bytes * 2,
            memory_bandwidth_bytes_per_cycle=mesh_config.memory_bandwidth_bytes_per_cycle,
        )
        self.pe.reserve_memory(max_words * word_bytes * 2)

    def load_left(self, words: tuple[int, ...]):
        self._check_words(len(words))
        yield from self.pe.access_memory(len(words) * self.word_bytes)
        self.left[: len(words)] = list(words)
        self.left_count = len(words)

    def load_right(self, words: tuple[int, ...]):
        self._check_words(len(words))
        yield from self.pe.access_memory(len(words) * self.word_bytes)
        self.right[: len(words)] = list(words)
        self.right_count = len(words)

    def compute_dot(self, word_count: int):
        self._check_words(word_count)
        if self.left_count < word_count or self.right_count < word_count:
            raise ValueError(
                f"dot needs {word_count} loaded words, "
                f"got left={self.left_count}, right={self.right_count}"
            )
        yield from self.pe.compute(word_count * self.ops_per_mac)
        self.result = sum(
            left_word * right_word
            for left_word, right_word in zip(
                self.left[:word_count],
                self.right[:word_count],
                strict=True,
            )
        )
        return self.result

    def execute(self, command: VectorDotCommand):
        command.validate()
        if command.op == "load_left":
            yield from self.load_left(command.words)
            return None
        if command.op == "load_right":
            yield from self.load_right(command.words)
            return None
        if command.op == "compute_dot":
            return (yield from self.compute_dot(command.word_count))
        if command.op == "expect":
            self.expected = command.expected
            if self.result != command.expected:
                raise AssertionError(
                    f"dot result expected {command.expected}, got {self.result}"
                )
            return None
        raise ValueError(f"unknown vector dot command op: {command.op}")

    def _check_words(self, word_count: int) -> None:
        if word_count < 0:
            raise ValueError("word count must be non-negative")
        if word_count > len(self.left):
            raise ValueError(f"word count {word_count} exceeds max {len(self.left)}")


def run_vector_dot_commands(
    commands: list[VectorDotCommand],
    *,
    max_words: int = 1024,
    word_bytes: int = 4,
    ops_per_mac: int = 2,
) -> VectorDotResult:
    env = simpy.Environment()
    models: dict[Coord, VectorDotPe] = {}
    final_pe: Coord = (0, 0)
    expected: int | None = None

    def _model(coord: Coord) -> VectorDotPe:
        if coord not in models:
            models[coord] = VectorDotPe(
                env,
                max_words=max_words,
                word_bytes=word_bytes,
                ops_per_mac=ops_per_mac,
                coord=coord,
            )
        return models[coord]

    def _dst_for(command: VectorDotCommand) -> Coord:
        if command.dst_pe is not None:
            return command.dst_pe
        if command.color != "partial_sum_color":
            raise ValueError(f"unsupported vector dot reduction color: {command.color}")
        row, col = command.src_pe
        return (row, col + 1)

    def _run():
        nonlocal final_pe, expected
        for command in commands:
            if command.op == "reduce_east":
                src = _model(command.src_pe)
                dst_coord = _dst_for(command)
                dst = _model(dst_coord)
                if src.result is None:
                    raise ValueError(f"PE{command.src_pe} has no partial dot result")
                yield from src.pe.access_memory(word_bytes)
                yield from dst.pe.access_memory(word_bytes)
                dst.result = (dst.result or 0) + src.result
            elif command.op == "expect":
                final_pe = command.pe
                expected = command.expected
                yield env.process(_model(command.pe).execute(command))
            else:
                yield env.process(_model(command.pe).execute(command))

    env.run(until=env.process(_run()))
    if (0, 0) not in models:
        _model((0, 0))
    result_model = _model(final_pe)
    return VectorDotResult(
        left=tuple(result_model.left[: result_model.left_count]),
        right=tuple(result_model.right[: result_model.right_count]),
        result=result_model.result,
        expected=expected,
        pe_results={coord: model.result for coord, model in models.items()},
        sim_time_cycles=env.now,
        compute_tasks_completed=sum(
            model.pe.compute_tasks_completed for model in models.values()
        ),
        memory_tasks_completed=sum(
            model.pe.memory_tasks_completed for model in models.values()
        ),
    )
