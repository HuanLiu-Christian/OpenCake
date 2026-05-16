# Hardware Mapping Roadmap

This simulator is intentionally structured around blocks that can become Chisel
modules later. The first target is not a bit-accurate Cerebras clone; it is a
wafer-scale, dataflow-style mesh accelerator whose timing model can be refined
until it is close enough to drive RTL design choices.

## Simulator Block To Chisel Block

| Simulator block | Chisel block | Notes |
| --- | --- | --- |
| `ProcessingElement` | `ProcessingElement` | Local SRAM, task queue, simple vector/FMA datapath, network interface. |
| `DirectedLink` | `CreditLink` or ready/valid channel | Current SimPy queue maps naturally to credit-based backpressure. |
| `WaferMesh.route_xy` | `Router` | Deterministic XY routing first; virtual channels can be added later. |
| `Packet` | `PacketBundle` | Stable hardware contract: source, destination, opcode, byte count, sequence id. |
| `StencilWorkload` | Chisel testbench traffic | Packet traces can seed RTL tests and latency comparisons. |

## Suggested Chisel Interfaces

Keep the first RTL iteration boring and explicit:

```scala
class Coord(val rowBits: Int, val colBits: Int) extends Bundle {
  val row = UInt(rowBits.W)
  val col = UInt(colBits.W)
}

class PacketBundle(rowBits: Int, colBits: Int, payloadBits: Int) extends Bundle {
  val src = new Coord(rowBits, colBits)
  val dst = new Coord(rowBits, colBits)
  val opcode = UInt(4.W)
  val sizeBytes = UInt(16.W)
  val seq = UInt(32.W)
  val payload = UInt(payloadBits.W)
}
```

Use `Decoupled[PacketBundle]` for router ports first. Once the functionality
matches the SimPy model, split payload into flits and add credits.

## Next Milestones

1. Add a packet/flit mode to the SimPy model so queue occupancy and head-of-line
   blocking are closer to hardware.
2. Add `--trace-jsonl` workloads as golden traffic for Chisel unit tests.
3. Implement a single Chisel router and compare packet latency against a small
   `2x2` or `4x4` SimPy run.
4. Tile the router into a mesh generator, then add a small PE with SRAM and a
   command queue.

