from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import simpy

from wafersim.config import Coord, MeshConfig
from wafersim.hardware.pe import ProcessingElement
from wafersim.mapping.memcpy_ir import (
    MemcpyCommand,
    d2h_read,
    expect_word,
    h2d_write,
    local_copy,
)

LocalMemcpyStep = MemcpyCommand


@dataclass(frozen=True, slots=True)
class LocalMemcpyResult:
    final_sram: tuple[int, ...]
    final_srams: dict[Coord, tuple[int, ...]]
    d2h_outputs: tuple[tuple[int, ...], ...]
    sim_time_cycles: float
    memory_tasks_completed: int


class LocalMemcpyPe:
    """SimPy golden model for PE-local SRAM memcpy semantics."""

    def __init__(
        self,
        env: simpy.Environment,
        *,
        depth_words: int,
        word_bytes: int = 4,
        coord: Coord = (0, 0),
        config: MeshConfig | None = None,
    ) -> None:
        if depth_words <= 0:
            raise ValueError("depth_words must be positive")
        if word_bytes <= 0:
            raise ValueError("word_bytes must be positive")

        mesh_config = config or MeshConfig(rows=1, cols=1)
        self.env = env
        self.coord = coord
        self.word_bytes = word_bytes
        self.sram = [0 for _ in range(depth_words)]
        self.pe = ProcessingElement(
            env=env,
            coord=coord,
            compute_throughput_ops_per_cycle=mesh_config.compute_throughput_ops_per_cycle,
            local_sram_bytes=depth_words * word_bytes,
            memory_bandwidth_bytes_per_cycle=mesh_config.memory_bandwidth_bytes_per_cycle,
        )
        self.pe.reserve_memory(depth_words * word_bytes)

    def h2d_write(self, dst_addr: int, words: tuple[int, ...]):
        self._check_range(dst_addr, len(words))
        yield from self.pe.access_memory(len(words) * self.word_bytes)
        for offset, word in enumerate(words):
            self.sram[dst_addr + offset] = word

    def d2h_read(self, src_addr: int, word_count: int):
        self._check_range(src_addr, word_count)
        yield from self.pe.access_memory(word_count * self.word_bytes)
        return tuple(self.sram[src_addr : src_addr + word_count])

    def local_copy(self, src_addr: int, dst_addr: int, word_count: int):
        self._check_range(src_addr, word_count)
        self._check_range(dst_addr, word_count)
        yield from self.pe.access_memory(word_count * self.word_bytes)
        read_data = tuple(self.sram[src_addr : src_addr + word_count])
        yield from self.pe.access_memory(word_count * self.word_bytes)
        for offset, word in enumerate(read_data):
            self.sram[dst_addr + offset] = word

    def execute(self, command: MemcpyCommand):
        """Execute one portable memcpy command on the SimPy PE model."""

        command.validate()
        if command.op == "h2d_write":
            yield from self.h2d_write(command.dst_addr, command.words)
            return None
        if command.op == "local_copy":
            yield from self.local_copy(
                command.src_addr,
                command.dst_addr,
                command.word_count,
            )
            return None
        if command.op == "d2h_read":
            return (yield from self.d2h_read(
                command.src_addr,
                command.effective_word_count,
            ))
        if command.op == "expect":
            actual_word = self.sram[command.src_addr]
            expected_word = command.words[0]
            if actual_word != expected_word:
                raise AssertionError(
                    f"SRAM[{command.src_addr}] expected {expected_word}, "
                    f"got {actual_word}"
                )
            return None
        raise ValueError(f"unknown memcpy command op: {command.op}")

    def _check_range(self, start: int, word_count: int) -> None:
        if start < 0 or word_count < 0:
            raise ValueError("SRAM addresses and lengths must be non-negative")
        if start + word_count > len(self.sram):
            raise ValueError(
                f"SRAM range [{start}, {start + word_count}) exceeds "
                f"depth {len(self.sram)}"
            )


def parse_local_memcpy_trace(path: str | Path) -> list[LocalMemcpyStep]:
    steps: list[LocalMemcpyStep] = []
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        parts = line.split()
        op = parts[0].lower()
        try:
            if op == "h2d":
                if len(parts) < 3:
                    raise ValueError("h2d needs dst_addr and at least one word")
                steps.append(
                    h2d_write(int(parts[1]), tuple(int(word) for word in parts[2:]))
                )
            elif op == "copy":
                if len(parts) != 4:
                    raise ValueError("copy needs src_addr dst_addr word_count")
                steps.append(local_copy(int(parts[1]), int(parts[2]), int(parts[3])))
            elif op == "d2h":
                if len(parts) < 3:
                    raise ValueError("d2h needs src_addr and expected words")
                steps.append(
                    d2h_read(int(parts[1]), tuple(int(word) for word in parts[2:]))
                )
            elif op == "expect":
                if len(parts) != 3:
                    raise ValueError("expect needs addr value")
                steps.append(expect_word(int(parts[1]), int(parts[2])))
            else:
                raise ValueError(f"unknown op: {op}")
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return steps


def run_memcpy_commands(
    commands: list[MemcpyCommand],
    *,
    depth_words: int = 64,
    word_bytes: int = 4,
) -> LocalMemcpyResult:
    env = simpy.Environment()
    models: dict[Coord, LocalMemcpyPe] = {}
    d2h_outputs: list[tuple[int, ...]] = []

    def _model(coord: Coord) -> LocalMemcpyPe:
        if coord not in models:
            models[coord] = LocalMemcpyPe(
                env,
                depth_words=depth_words,
                word_bytes=word_bytes,
                coord=coord,
            )
        return models[coord]

    def _run():
        for command in commands:
            if command.op == "p2p_copy":
                src_model = _model(command.src_pe)
                dst_model = _model(command.dst_pe)
                payload = yield env.process(
                    src_model.d2h_read(command.src_addr, command.word_count)
                )
                yield env.process(dst_model.h2d_write(command.dst_addr, payload))
            else:
                actual = yield env.process(_model(command.pe).execute(command))
            if command.op == "d2h_read":
                if command.words and actual != command.words:
                    raise AssertionError(
                        f"d2h mismatch at addr {command.src_addr}: "
                        f"expected {command.words}, got {actual}"
                    )
                d2h_outputs.append(actual)

    env.run(until=env.process(_run()))
    if (0, 0) not in models:
        _model((0, 0))
    return LocalMemcpyResult(
        final_sram=tuple(models[(0, 0)].sram),
        final_srams={coord: tuple(model.sram) for coord, model in models.items()},
        d2h_outputs=tuple(d2h_outputs),
        sim_time_cycles=env.now,
        memory_tasks_completed=sum(
            model.pe.memory_tasks_completed for model in models.values()
        ),
    )


def run_local_memcpy_trace(
    trace_path: str | Path,
    *,
    depth_words: int = 64,
    word_bytes: int = 4,
) -> LocalMemcpyResult:
    return run_memcpy_commands(
        parse_local_memcpy_trace(trace_path),
        depth_words=depth_words,
        word_bytes=word_bytes,
    )
