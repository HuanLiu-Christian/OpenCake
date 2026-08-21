from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GemmSpec:
    """Problem C[M, N] += A[M, K] * B[K, N]."""

    m: int
    n: int
    k: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.m, self.n, self.k, self.bytes_per_element) <= 0:
            raise ValueError("GEMM dimensions and element size must be positive")


@dataclass(frozen=True, slots=True)
class WaferSpec:
    """The subset of wafer resources needed by this performance model."""

    mesh_rows: int = 32
    mesh_cols: int = 32
    sram_kib_per_pe: int = 48
    link_bandwidth_bytes_per_cycle: int = 32
    router_latency_cycles: int = 1
    packet_payload_bytes: int = 256
    fifo_capacity_packets: int = 8
    receive_dequeue_latency_cycles: int = 1
    clock_ghz: float = 1.0

    # Process node passed to the real TimeLoop/Accelergy architecture model
    # (compound components require an explicit `technology` attribute).
    technology_node: str = "45nm"

    # Real Cerebras WSE PE tile: "each core has a fabric router within the
    # mesh topology. The fabric routers have a simple 5-port design with
    # 32-bit bi-directional interfaces" (Cerebras Architecture Deep Dive,
    # cerebras.ai/blog/cerebras-architecture-deep-dive-...). 5 ports (4 mesh
    # directions + 1 local injection/ejection into the PE's own compute
    # engine), each 32 bits = 4 bytes wide. Used below to give the router a
    # real-cited (if approximate) SRAM-equivalent byte footprint, since
    # Accelergy has no dedicated router estimator (see fifo_router_bytes_
    # per_pe) -- this is NOT a claim that a router's real silicon area is
    # literally "n_ports * port_bytes" of SRAM, only that this is a
    # real-cited sizing basis for the CactiSRAM approximation used below,
    # in the absence of any real router area/energy estimator plug-in.
    router_ports_per_pe: int = 5
    router_port_bytes: int = 4

    # Real, per-single-instance Accelergy area for the router/FIFO is no
    # longer a configurable coefficient here -- as of this refactor, both
    # are declared as real SRAM-class components inside the PE's own
    # architecture.yaml subtree (see timeloop.py's generate_timeloop_
    # inputs -- `fifo_buffer`/`router_buffer`), so CactiSRAM gives them a
    # real (if approximate, for the router) area number the same way it
    # already does for `pe_scratchpad`; that number flows into
    # GemmMappingContract.pe_internal_area_um2 alongside mac/pe_scratchpad
    # (see contract.py). This also means FIFO/router now really consume
    # part of the PE's declared SRAM depth -- see fifo_router_bytes_per_pe
    # below and timeloop.py's pe_scratchpad_depth_words derivation.
    #
    # Only `link` has no such path: a link is the wire BETWEEN two PEs'
    # routers, not something owned by a single PE's architecture subtree,
    # so there is nothing to declare per-PE in Accelergy for it. It stays a
    # configurable OpenCake placeholder, still 0.0 -- not because no
    # citable data exists for wire area (unlike router/FIFO, real NoC wire
    # area studies do exist), but because no single-hop, per-PE-attributable
    # figure was found precise enough to state as "real" here; fill in a
    # cited value (e.g. from a DSENT/Orion NoC area study, or a real wire
    # pitch x real per-PE tile pitch at your technology node) when you have
    # one.
    link_area_um2_per_pe: float = 0.0

    # fifo_energy_pj_per_byte / router_energy_pj_per_byte_hop /
    # wire_energy_pj_per_byte_hop: TimeLoop/Accelergy structurally CANNOT
    # give these a real number even after the router/FIFO area change above
    # -- Accelergy's real "read"/"write" action counts come from TimeLoop's
    # own dataflow/mapping, and this project's actual inter-PE traffic
    # pattern (fifo_enqueued_bytes, router_byte_hops, ...) is produced by
    # OpenCake's own network.py simulation, which TimeLoop never sees. So
    # these three remain OpenCake's own per-byte-per-hop coefficients, but
    # as of this revision they are anchored to a REAL, silicon-measured
    # 45nm mesh NoC test chip rather than an arbitrary guess: "Approaching
    # the Theoretical Limits of a Mesh NoC with a 16-Node Chip Prototype in
    # 45nm SOI" (Kwon & Chatha/MIT, DAC 2012) reports 892 Gb/s of mixed
    # traffic at 531.4 mW measured total network power -- i.e. a real,
    # measured ~0.596 pJ/bit == ~4.77 pJ/byte of combined router+link+
    # buffer energy at 45nm. That paper does not separately report a
    # router-vs-link-vs-buffer split, so the 3-way split below is this
    # project's own reasoned (NOT independently cited) allocation --
    # buffers/crossbar routing logic are the widely-reported dominant NoC
    # power consumers over pure wire at mesh-scale hop lengths, so FIFO and
    # router are each given a larger share than wire: fifo ~40%, router
    # ~45%, wire ~15% of the real 4.77 pJ/byte total (4.77 * 0.40 = 1.9,
    # 4.77 * 0.45 = 2.1, 4.77 * 0.15 = 0.7 pJ/byte, rounded).
    fifo_energy_pj_per_byte: float = 1.9
    router_energy_pj_per_byte_hop: float = 2.1
    wire_energy_pj_per_byte_hop: float = 0.7

    # static_power_mw_per_pe: no cleanly-separated real leakage-only figure
    # was found in the sources checked. The closest real anchor is DSENT's
    # (NoCS 2012, MIT) validated 45nm SOI router model: a 6-port/64-bit-
    # flit/8-VC/16-buffers-per-port router at 1 GHz and a representative
    # 0.16 injection rate draws a real, tool-validated ~11.2 mW total
    # (dynamic + static combined). Static/leakage is not broken out
    # separately there either, so this value is a reasoned ~20% fraction of
    # that real total router power (11.2 * 0.2 = 2.24, rounded to 2.0) as a
    # conservative order-of-magnitude anchor -- an explicit estimate, not a
    # measured leakage number, and per-PE here covers the whole PE (mac +
    # scratchpad + fifo + router), not just the router.
    static_power_mw_per_pe: float = 2.0

    @property
    def sram_bytes_per_pe(self) -> int:
        return self.sram_kib_per_pe * 1024

    @property
    def fifo_bytes_per_pe(self) -> int:
        """Real, configured FIFO byte capacity for one PE: 8 independent
        FIFOs (4 mesh directions x send+receive -- see wafer/network.py's
        PacketNetwork), each holding up to `fifo_capacity_packets` packets
        of `packet_payload_bytes` bytes. Single source of truth: both
        timeloop.py (declaring the real fifo_buffer Accelergy component and
        reserving this many bytes out of the PE's SRAM budget) and
        wafer/area.py (reporting it) read this same property, instead of
        each recomputing "8 * fifo_capacity_packets * packet_payload_bytes"
        independently and risking the two drifting apart.
        """
        return 8 * self.fifo_capacity_packets * self.packet_payload_bytes

    @property
    def router_bytes_per_pe(self) -> int:
        """Real-cited (if approximate) SRAM-equivalent byte footprint for
        one PE's router -- see router_ports_per_pe/router_port_bytes above.
        """
        return self.router_ports_per_pe * self.router_port_bytes

    @property
    def fifo_router_bytes_per_pe(self) -> int:
        """Total bytes reserved out of the PE's declared SRAM for FIFO +
        router storage -- per the project's decision that these are now
        modeled as physically occupying part of the PE's real SRAM budget,
        not a separate/additional allocation on top of it (see timeloop.py's
        pe_scratchpad_depth_words and the SRAM-feasibility checks in
        feasible_tile_k_candidates/generate_timeloop_inputs, both of which
        subtract this from wafer.sram_bytes_per_pe before budgeting GEMM
        data).
        """
        return self.fifo_bytes_per_pe + self.router_bytes_per_pe

    def validate(self) -> None:
        integer_fields = (
            self.mesh_rows,
            self.mesh_cols,
            self.sram_kib_per_pe,
            self.link_bandwidth_bytes_per_cycle,
            self.router_latency_cycles,
            self.packet_payload_bytes,
            self.fifo_capacity_packets,
            self.receive_dequeue_latency_cycles,
            self.router_ports_per_pe,
            self.router_port_bytes,
        )
        if min(integer_fields) <= 0:
            raise ValueError("wafer integer resources must be positive")
        if self.clock_ghz <= 0:
            raise ValueError("clock_ghz must be positive")
        if self.link_area_um2_per_pe < 0:
            raise ValueError("interconnect area coefficients cannot be negative")
        if self.fifo_router_bytes_per_pe >= self.sram_bytes_per_pe:
            raise ValueError(
                f"FIFO+router reserved bytes ({self.fifo_router_bytes_per_pe}) "
                f"leave no room for GEMM data in the configured PE SRAM "
                f"({self.sram_bytes_per_pe} bytes) -- raise sram_kib_per_pe or "
                f"shrink fifo_capacity_packets/packet_payload_bytes"
            )
