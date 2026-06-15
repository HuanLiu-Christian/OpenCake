# Hardware Mapping Roadmap

This simulator is structured around Python blocks that can later become Chisel
or another hardware implementation. The current clean target is the GEMV9-style
streaming example, not a bit-accurate Cerebras WSE clone.

## Simulator Block To Hardware Block

| Simulator block | Future hardware block | Notes |
| --- | --- | --- |
| `GemvStreamingPeProgram` | PE-local GEMV kernel | Owns local A tile, x input, y buffer, and eastward reduction state. |
| `ProcessingElement` | PE timing/datapath model | Tracks compute cycles, memory cycles, SRAM footprint, and completed tasks. |
| `DirectedLink` | Credit link or ready/valid channel | Current SimPy queue maps naturally to link backpressure. |
| `WaferMesh` | Mesh generator plus routers | Provides coordinates, neighbors, links, routing, and utilization metrics. |
| `CslCompatibilityRuntime.colors` | Color/routing metadata | Models named communication paths such as `x_color` and `ax_color_1`. |
| `MemcpyCommand` | Memcpy command interface | Shared mapping IR for PE-local and two-PE memcpy, consumed by SimPy and translated to Chisel opcodes. |

## Suggested First RTL Boundary

Keep the first hardware contract small:

```scala
class Coord(val rowBits: Int, val colBits: Int) extends Bundle {
  val row = UInt(rowBits.W)
  val col = UInt(colBits.W)
}

class PacketBundle(rowBits: Int, colBits: Int, payloadBits: Int) extends Bundle {
  val src = new Coord(rowBits, colBits)
  val dst = new Coord(rowBits, colBits)
  val color = UInt(8.W)
  val sizeBytes = UInt(16.W)
  val payload = UInt(payloadBits.W)
}
```

For the first GEMV mapping, model:

- `x_color`: x chunks stream south through a column.
- `ax_color_1` / `ax_color_2`: partial y chunks reduce east through a row.
- H2D/D2H colors: host input and output movement.
- PE-local memory: A tile, x chunk, and y buffer.

## Next Milestones

1. Add flit-level packet modeling to the SimPy links.
2. Emit packet traces from `examples/gemv_09_streaming/run.py`.
3. Build a small Chisel router and compare against a `2x2` or `4x4` SimPy run.
4. Add a simple PE with local SRAM and a GEMV FMA datapath.
5. Scale the mesh generator after small-router latency and backpressure match.
