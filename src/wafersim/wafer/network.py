"""Cycle-level packet, directed-link, and per-PE FIFO timing model."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Literal, TypeAlias

from ..specs import WaferSpec

Coord: TypeAlias = tuple[int, int]
DirectedLink: TypeAlias = tuple[Coord, Coord]
Direction: TypeAlias = Literal["east", "west", "south", "north"]
QueueKey: TypeAlias = tuple[int, int, Direction]


@dataclass(slots=True)
class FifoState:
    completions: list[float] = field(default_factory=list)
    max_depth: int = 0
    packets: int = 0
    bytes: int = 0
    wait_cycles: float = 0.0


@dataclass(slots=True)
class LinkState:
    next_free_cycle: float = 0.0
    packets: int = 0
    bytes: int = 0
    busy_cycles: float = 0.0
    queue_wait_cycles: float = 0.0


class PacketNetwork:
    """Models row/column broadcast with independent per-direction FIFOs.

    Implements classic SUMMA panel ownership: at K-phase p, the A panel for
    row r is resident at PE(r, p mod active_cols) -- NOT always column 0 --
    and that owner broadcasts it to the REST of row r in both directions
    (east *and* west). Symmetrically, the B panel for column c is resident
    at PE(p mod active_rows, c) and broadcasts both south and north. Because
    the owner can be any interior PE (not just a boundary one), every PE
    must be able to both send and receive in all four directions over the
    course of a run -- an earlier version of this model injected A/B only
    at the column-0/row-0 edge (as if DRAM always fed the mesh from a fixed
    boundary), which only ever needed east+south and undercounted the
    communication this project's SUMMA schedule actually requires.

    Mirrors Cerebras CSL's fabric queue model
    (https://sdk.cerebras.ai/csl/language/dsds#fabric-queues): each PE there
    has multiple independent input/output queues, and a DSD operation binds
    to one specific queue -- "the programmer is responsible for ensuring no
    two concurrent DSD operations share a queue", meaning two streams on
    *different* queues genuinely proceed concurrently, never blocking on
    each other. Each PE gets one independent send/receive FIFO PER
    DIRECTION (east/west/south/north) rather than a single shared pair, so
    e.g. relaying A east and relaying B north through the same PE in the
    same phase don't serialize behind one shared queue.

    FIFO storage is deliberately treated as fabric state in this project
    version; it is not subtracted from the PE's data SRAM capacity.
    """

    def __init__(self, wafer: WaferSpec, rows: int, cols: int) -> None:
        self.spec = wafer
        self.rows = rows
        self.cols = cols
        directions: tuple[Direction, ...] = ("east", "west", "south", "north")
        self.send_fifos: dict[QueueKey, FifoState] = {
            (r, c, d): FifoState() for r in range(rows) for c in range(cols) for d in directions
        }
        self.receive_fifos: dict[QueueKey, FifoState] = {
            (r, c, d): FifoState() for r in range(rows) for c in range(cols) for d in directions
        }
        # Both directions of every physical edge get their own LinkState,
        # so east-bound and west-bound traffic on the same wire pair (and
        # south-bound/north-bound) don't contend for one shared link slot
        # either -- matching independent per-direction fabric channels.
        self.links: dict[DirectedLink, LinkState] = {}
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    self.links[((r, c), (r, c + 1))] = LinkState()  # east
                    self.links[((r, c + 1), (r, c))] = LinkState()  # west
                if r + 1 < rows:
                    self.links[((r, c), (r + 1, c))] = LinkState()  # south
                    self.links[((r + 1, c), (r, c))] = LinkState()  # north

    @staticmethod
    def _discard_completed(fifo: FifoState, cycle: float) -> None:
        fifo.completions[:] = [end for end in fifo.completions if end > cycle]

    def _admit_fifo(self, fifo: FifoState, ready_cycle: float, size_bytes: int) -> float:
        self._discard_completed(fifo, ready_cycle)
        admission = ready_cycle
        if len(fifo.completions) >= self.spec.fifo_capacity_packets:
            admission = min(fifo.completions)
            self._discard_completed(fifo, admission)
        fifo.wait_cycles += admission - ready_cycle
        fifo.packets += 1
        fifo.bytes += size_bytes
        fifo.max_depth = max(fifo.max_depth, len(fifo.completions) + 1)
        return admission

    def _send_hop(
        self,
        ready_cycle: float,
        size_bytes: int,
        source: Coord,
        destination: Coord,
        direction: Direction,
    ) -> float:
        send_fifo = self.send_fifos[(*source, direction)]
        send_admitted = self._admit_fifo(send_fifo, ready_cycle, size_bytes)
        link = (source, destination)
        state = self.links[link]
        serialization = ceil(size_bytes / self.spec.link_bandwidth_bytes_per_cycle)
        service = self.spec.router_latency_cycles + serialization
        link_start = max(send_admitted, state.next_free_cycle)
        state.queue_wait_cycles += link_start - send_admitted
        link_end = link_start + service
        state.next_free_cycle = link_end
        state.packets += 1
        state.bytes += size_bytes
        state.busy_cycles += service
        send_fifo.completions.append(link_end)

        receive_fifo = self.receive_fifos[(*destination, direction)]
        receive_admitted = self._admit_fifo(receive_fifo, link_end, size_bytes)
        hop_end = receive_admitted + self.spec.receive_dequeue_latency_cycles
        receive_fifo.completions.append(hop_end)
        return hop_end

    def _packet_sizes(self, size_bytes: int) -> list[int]:
        payload = self.spec.packet_payload_bytes
        return [min(payload, size_bytes - offset) for offset in range(0, size_bytes, payload)]

    def broadcast_row(
        self, row: int, ready_cycle: float, size_bytes: int, owner_col: int
    ) -> dict[Coord, float]:
        """A panel for `row` is resident at PE(row, owner_col); relay it to the
        rest of the row in both directions -- east via the 'east' queue,
        west via the 'west' queue -- one hop (and one queue) at a time.
        """
        arrivals = {(row, owner_col): ready_cycle}
        for size in self._packet_sizes(size_bytes):
            ready = ready_cycle
            for col in range(owner_col, self.cols - 1):
                ready = self._send_hop(ready, size, (row, col), (row, col + 1), "east")
                arrivals[(row, col + 1)] = max(arrivals.get((row, col + 1), 0.0), ready)
            ready = ready_cycle
            for col in range(owner_col, 0, -1):
                ready = self._send_hop(ready, size, (row, col), (row, col - 1), "west")
                arrivals[(row, col - 1)] = max(arrivals.get((row, col - 1), 0.0), ready)
        return arrivals

    def broadcast_column(
        self, col: int, ready_cycle: float, size_bytes: int, owner_row: int
    ) -> dict[Coord, float]:
        """B panel for `col` is resident at PE(owner_row, col); relay it to the
        rest of the column in both directions -- south via the 'south' queue,
        north via the 'north' queue -- one hop (and one queue) at a time.
        """
        arrivals = {(owner_row, col): ready_cycle}
        for size in self._packet_sizes(size_bytes):
            ready = ready_cycle
            for row in range(owner_row, self.rows - 1):
                ready = self._send_hop(ready, size, (row, col), (row + 1, col), "south")
                arrivals[(row + 1, col)] = max(arrivals.get((row + 1, col), 0.0), ready)
            ready = ready_cycle
            for row in range(owner_row, 0, -1):
                ready = self._send_hop(ready, size, (row, col), (row - 1, col), "north")
                arrivals[(row - 1, col)] = max(arrivals.get((row - 1, col), 0.0), ready)
        return arrivals

    @property
    def latest_cycle(self) -> float:
        times = [state.next_free_cycle for state in self.links.values()]
        times.extend(end for fifo in self.receive_fifos.values() for end in fifo.completions)
        return max(times, default=0.0)

    def summary(self, final_cycle: float) -> dict[str, float | int]:
        link_states = list(self.links.values())
        send_states = list(self.send_fifos.values())
        receive_states = list(self.receive_fifos.values())
        max_util = max((state.busy_cycles / final_cycle for state in link_states), default=0.0)
        return {
            "links": len(link_states),
            "packet_hops": sum(state.packets for state in link_states),
            "link_bytes": sum(state.bytes for state in link_states),
            "router_byte_hops": sum(state.bytes for state in link_states),
            "wire_byte_hops": sum(state.bytes for state in link_states),
            "fifo_enqueued_bytes": sum(fifo.bytes for fifo in send_states + receive_states),
            "send_fifo_wait_cycles": sum(fifo.wait_cycles for fifo in send_states),
            "receive_fifo_wait_cycles": sum(fifo.wait_cycles for fifo in receive_states),
            "link_queue_wait_cycles": sum(state.queue_wait_cycles for state in link_states),
            "max_send_fifo_depth": max((fifo.max_depth for fifo in send_states), default=0),
            "max_receive_fifo_depth": max((fifo.max_depth for fifo in receive_states), default=0),
            "max_link_utilization": max_util,
        }
