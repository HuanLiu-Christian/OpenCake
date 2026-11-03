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

    # SIMD / vector width of the PE's multiply-accumulate unit. 1 = this project's own architecture.yaml (one scalar MAC
    # per cycle, TimeLoop-measured 1.0 cycles/MAC). N > 1 is an IDEAL-SIMD what-if applied in wafer/simulator.py on top of
    # TimeLoop's measured cycles/MAC (cycles/MAC / N, rounded up per PE per K-phase); it is NOT measured by TimeLoop and
    # assumes every lane is busy (no remainder loss). Real Cerebras WSE-3: "8-way SIMD for 16b data (FP/BF16)" (Hot Chips
    # 2024); WSE-2 had 4 FP16 FMACs. Network time (broadcasts) is NOT scaled.
    mac_lanes: int = 1
    # Bytes per cycle one PE's SRAM can deliver to the MAC unit for the B (weight) operand. 0 = unlimited (the project's
    # historical assumption: SRAM bandwidth never binds). > 0 puts a floor on every PE's per-phase compute time of
    # ceil(B bytes read / this). A is assumed to sit in registers and C to be accumulated there (not modelled).
    # WSE-2: 2 x 64-bit reads + 1 x 64-bit write per cycle (Hot Chips 2022); WSE-3 per-core width is not published.
    sram_read_bytes_per_cycle: int = 0

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
    # Only `link` has no Accelergy path: a link is the wire BETWEEN two PEs'
    # routers, not something owned by a single PE's architecture subtree, so
    # there is nothing to declare per-PE in Accelergy for it. It stays a
    # configurable OpenCake coefficient, but is no longer 0.0 -- neither
    # Cerebras (checked directly: no published router or wire area figure)
    # nor the DSENT NoCS 2012 paper (checked directly: gives a real router
    # area but explicitly states "link area simply is not a primary metric"
    # and reports none) publish a single ready-to-use number, so this value
    # is this project's own composite estimate built ONLY from real, cited
    # inputs, not a fabricated guess:
    #   - wire pitch: real ITRS 2007 Interconnect Roadmap, Table INTC2a
    #     ("MPU Interconnect Technology Requirements -- Near-term Years"),
    #     2010 production year (DRAM 1/2 pitch = 45nm node, matching this
    #     project's technology_node): "Minimum global wiring pitch" =
    #     135 nm = 0.135 um/bit-track.
    #   - bus width: fw=32 bits, matching router_port_bytes=4 bytes above
    #     (the same real-cited Cerebras "32-bit bi-directional interfaces"
    #     figure), using the ORION 2.0 (TVLSI 2012) link-area formula's
    #     leading term Arealink = fw * (ww + sw) + sw with ww+sw = the ITRS
    #     pitch above (the paper gives the formula but no numeric ww/sw
    #     split or worked example, so the +sw remainder term, <2% of the
    #     total, is dropped): 32 bits * 0.135 um = 4.32 um of wire width per
    #     mesh direction.
    #   - per-hop wire length: real Cerebras-cited PE tile area = 38,000
    #     um^2 ("Cerebras Architecture Deep Dive", cerebras.ai) -- Cerebras
    #     does not publish tile width/height separately, so this project
    #     ASSUMES a square tile (side = sqrt(38000) = 194.94 um) as the
    #     center-to-center hop length, and attributes half of each hop
    #     (97.47 um) to this PE, the other half to its neighbor.
    #   - directions: 4 mesh neighbors (N/S/E/W); the router's 5th port is
    #     local injection/ejection into this PE's own compute engine (see
    #     router_ports_per_pe above), not an inter-PE wire, so it is
    #     excluded here.
    #   4 * (32 bits * 0.135 um/bit) * (sqrt(38000 um^2) / 2) = 1684.25 um^2
    # This is a reasoned engineering estimate assembled from real citations,
    # not an independently-published single number -- replace it if a
    # direct real per-PE link/wire area figure is ever found.
    link_area_um2_per_pe: float = 1684.25

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
        if self.mac_lanes < 1:
            raise ValueError("mac_lanes must be >= 1")
        if self.sram_read_bytes_per_cycle < 0:
            raise ValueError("sram_read_bytes_per_cycle must be >= 0 (0 = unlimited)")
        if self.link_area_um2_per_pe < 0:
            raise ValueError("interconnect area coefficients cannot be negative")
        if self.fifo_router_bytes_per_pe >= self.sram_bytes_per_pe:
            raise ValueError(
                f"FIFO+router reserved bytes ({self.fifo_router_bytes_per_pe}) "
                f"leave no room for GEMM data in the configured PE SRAM "
                f"({self.sram_bytes_per_pe} bytes) -- raise sram_kib_per_pe or "
                f"shrink fifo_capacity_packets/packet_payload_bytes"
            )
