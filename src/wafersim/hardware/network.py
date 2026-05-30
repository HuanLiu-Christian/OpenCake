from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable

import simpy

from wafersim.config import Coord, MeshConfig
from wafersim.hardware.pe import ProcessingElement


@dataclass(slots=True)
class Packet:
    packet_id: int
    src: Coord
    dst: Coord
    size_bytes: int
    kind: str
    created_at: float
    route: list[tuple[Coord, Coord]] = field(default_factory=list)


@dataclass(slots=True)
class PacketRecord:
    packet_id: int
    src: Coord
    dst: Coord
    size_bytes: int
    kind: str
    created_at: float
    delivered_at: float
    hops: int

    @property
    def latency(self) -> float:
        return self.delivered_at - self.created_at


@dataclass(slots=True)
class LinkTransfer:
    packet: Packet
    done: simpy.Event


class DirectedLink:
    def __init__(
        self,
        env: simpy.Environment,
        src: Coord,
        dst: Coord,
        bandwidth_bytes_per_cycle: float,
        router_latency_cycles: float,
        queue_capacity: int,
    ) -> None:
        self.env = env
        self.src = src
        self.dst = dst
        self.bandwidth_bytes_per_cycle = bandwidth_bytes_per_cycle
        self.router_latency_cycles = router_latency_cycles
        self.queue = simpy.Store(env, capacity=queue_capacity)
        self.busy_time = 0.0
        self.packets = 0
        self.bytes = 0
        self.max_queue_depth = 0
        self._worker = env.process(self._run())

    def transmit(self, packet: Packet):
        done = self.env.event()
        yield self.queue.put(LinkTransfer(packet=packet, done=done))
        self.max_queue_depth = max(self.max_queue_depth, len(self.queue.items))
        yield done

    def _run(self):
        while True:
            transfer = yield self.queue.get()
            packet = transfer.packet
            duration = self.router_latency_cycles + (
                packet.size_bytes / self.bandwidth_bytes_per_cycle
            )
            self.packets += 1
            self.bytes += packet.size_bytes
            self.busy_time += duration
            if duration > 0:
                yield self.env.timeout(duration)
            transfer.done.succeed()


class WaferMesh:
    def __init__(self, env: simpy.Environment, config: MeshConfig) -> None:
        config.validate()
        self.env = env
        self.config = config
        self.pes = {
            coord: ProcessingElement(
                env=env,
                coord=coord,
                compute_throughput_ops_per_cycle=config.compute_throughput_ops_per_cycle,
                local_sram_bytes=config.local_sram_bytes_per_pe,
                memory_bandwidth_bytes_per_cycle=config.memory_bandwidth_bytes_per_cycle,
            )
            for coord in self.coords()
        }
        self.links = self._build_links()
        self.packet_records: list[PacketRecord] = []
        self._next_packet_id = 0

    def coords(self) -> Iterable[Coord]:
        for row in range(self.config.rows):
            for col in range(self.config.cols):
                yield (row, col)

    def neighbors(self, coord: Coord) -> list[Coord]:
        row, col = coord
        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]
        return [candidate for candidate in candidates if self.contains(candidate)]

    def contains(self, coord: Coord) -> bool:
        row, col = coord
        return 0 <= row < self.config.rows and 0 <= col < self.config.cols

    def route_xy(self, src: Coord, dst: Coord) -> list[tuple[Coord, Coord]]:
        if not self.contains(src):
            raise ValueError(f"source coordinate out of mesh: {src}")
        if not self.contains(dst):
            raise ValueError(f"destination coordinate out of mesh: {dst}")

        route: list[tuple[Coord, Coord]] = []
        cur_row, cur_col = src
        dst_row, dst_col = dst

        while cur_col != dst_col:
            next_col = cur_col + (1 if dst_col > cur_col else -1)
            nxt = (cur_row, next_col)
            route.append(((cur_row, cur_col), nxt))
            cur_col = next_col

        while cur_row != dst_row:
            next_row = cur_row + (1 if dst_row > cur_row else -1)
            nxt = (next_row, cur_col)
            route.append(((cur_row, cur_col), nxt))
            cur_row = next_row

        return route

    def send(self, src: Coord, dst: Coord, size_bytes: int, kind: str = "data"):
        if size_bytes < 0:
            raise ValueError("packet size cannot be negative")

        packet = Packet(
            packet_id=self._next_packet_id,
            src=src,
            dst=dst,
            size_bytes=size_bytes,
            kind=kind,
            created_at=self.env.now,
        )
        self._next_packet_id += 1
        packet.route = self.route_xy(src, dst)

        for edge in packet.route:
            yield from self.links[edge].transmit(packet)

        self.packet_records.append(
            PacketRecord(
                packet_id=packet.packet_id,
                src=packet.src,
                dst=packet.dst,
                size_bytes=packet.size_bytes,
                kind=packet.kind,
                created_at=packet.created_at,
                delivered_at=self.env.now,
                hops=len(packet.route),
            )
        )

    def packet_summary(self) -> dict[str, float | int]:
        latencies = [record.latency for record in self.packet_records]
        return {
            "packets": len(self.packet_records),
            "bytes": sum(record.size_bytes for record in self.packet_records),
            "mean_packet_latency_cycles": mean(latencies) if latencies else 0.0,
            "p95_packet_latency_cycles": percentile(latencies, 95),
            "max_packet_latency_cycles": max(latencies) if latencies else 0.0,
            "mean_hops": mean([record.hops for record in self.packet_records])
            if self.packet_records
            else 0.0,
        }

    def link_summary(self, elapsed_cycles: float) -> dict[str, object]:
        link_rows = []
        for (src, dst), link in self.links.items():
            utilization = link.busy_time / elapsed_cycles if elapsed_cycles > 0 else 0.0
            link_rows.append(
                {
                    "src": src,
                    "dst": dst,
                    "packets": link.packets,
                    "bytes": link.bytes,
                    "busy_cycles": link.busy_time,
                    "utilization": utilization,
                    "max_queue_depth": link.max_queue_depth,
                }
            )

        busiest = sorted(link_rows, key=lambda row: row["utilization"], reverse=True)[:8]
        return {
            "links": len(self.links),
            "max_link_utilization": busiest[0]["utilization"] if busiest else 0.0,
            "busiest_links": busiest,
        }

    def pe_summary(self, elapsed_cycles: float) -> dict[str, float | int]:
        compute_utilizations = [
            pe.compute_busy_time / elapsed_cycles if elapsed_cycles > 0 else 0.0
            for pe in self.pes.values()
        ]
        memory_utilizations = [
            pe.memory_busy_time / elapsed_cycles if elapsed_cycles > 0 else 0.0
            for pe in self.pes.values()
        ]
        active_utilizations = [
            min(
                1.0,
                (pe.compute_busy_time + pe.memory_busy_time) / elapsed_cycles,
            )
            if elapsed_cycles > 0
            else 0.0
            for pe in self.pes.values()
        ]
        max_memory_footprint = max(
            [pe.max_memory_footprint_bytes for pe in self.pes.values()],
            default=0,
        )
        return {
            "pes": len(self.pes),
            "mean_pe_utilization": mean(compute_utilizations)
            if compute_utilizations
            else 0.0,
            "max_pe_utilization": max(compute_utilizations)
            if compute_utilizations
            else 0.0,
            "mean_pe_memory_utilization": mean(memory_utilizations)
            if memory_utilizations
            else 0.0,
            "max_pe_memory_utilization": max(memory_utilizations)
            if memory_utilizations
            else 0.0,
            "mean_pe_active_utilization": mean(active_utilizations)
            if active_utilizations
            else 0.0,
            "max_pe_active_utilization": max(active_utilizations)
            if active_utilizations
            else 0.0,
            "compute_tasks_completed": sum(
                pe.compute_tasks_completed for pe in self.pes.values()
            ),
            "memory_tasks_completed": sum(
                pe.memory_tasks_completed for pe in self.pes.values()
            ),
            "local_sram_bytes_per_pe": self.config.local_sram_bytes_per_pe,
            "memory_bandwidth_bytes_per_cycle": (
                self.config.memory_bandwidth_bytes_per_cycle
            ),
            "max_memory_footprint_bytes_per_pe": max_memory_footprint,
            "max_memory_capacity_utilization": (
                max_memory_footprint / self.config.local_sram_bytes_per_pe
            ),
        }

    def _build_links(self) -> dict[tuple[Coord, Coord], DirectedLink]:
        links: dict[tuple[Coord, Coord], DirectedLink] = {}
        for coord in self.coords():
            for neighbor in self.neighbors(coord):
                links[(coord, neighbor)] = DirectedLink(
                    env=self.env,
                    src=coord,
                    dst=neighbor,
                    bandwidth_bytes_per_cycle=self.config.link_bandwidth_bytes_per_cycle,
                    router_latency_cycles=self.config.router_latency_cycles,
                    queue_capacity=self.config.link_queue_capacity,
                )
        return links


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (pct / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight
