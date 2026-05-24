# CSL Compatibility Mapping

This project does not implement a CSL compiler or the real Cerebras SDK runtime.
Instead, it provides a Python/SimPy compatibility layer that maps useful CSL
builtins and libraries to simulator primitives.

## Implemented Builtin Mappings

- Rectangle layout / `@set_rectangle`: `LayoutRegion`
- `@map`-style coordinate mapping: PE coordinate to linear ID
- Zero and constant buffers: Python lists
- Pointer load/store: `SimulatedPointer`
- `sizeof`: element count times bytes per element
- Integer bitcast: bit mask approximation
- Select, concat, range: Python value helpers
- `@import_module`: library-name validation only

## Implemented Library Mappings

- `<complex>`: add, multiply, magnitude
- `<control>`: command stream block/unblock state
- `<data_utils>`: chunk, flatten, reshape
- `<debug>`: in-memory trace events
- `<directions>`: north/south/east/west movement and opposites
- `<empty>`: no-op
- `<layout>`: coordinate/rank conversion
- `<malloc>`: PE-local arena allocator
- `<math>`: common scalar math and operation estimates
- `<memcpy>`: packetized H2D, D2H, and P2P copies
- `<random>`: deterministic Python random stream
- `<simprint>`: per-PE simulator print log
- `<string>`: concat and length
- `<tile_config>`: FIFO, filter, and switch helpers
- `<time>`: SimPy cycle timestamp
- `<timer>`: named elapsed timers
- `<types>`: type sizes and unsigned clamping helpers
- `<kernels>`: SAXPY and dot-product examples
- `<kernels/fft>`: FFT operation estimate and bit-reversal helper
- `<kernels/tally>`: histogram tally helper
- `<collectives_2d>`: broadcast, reduce, allreduce, scatter, gather
- `<collectives_2d/params>`: PE-group parameter object
- `<collectives_2d/pe>`: rank/coordinate helper

## Partially Implemented

- `<dsd_ops>`: implemented as conceptual strided/gather/scatter address-list
  operations. It does not model real WSE DSD descriptors, hardware data movers,
  or descriptor scheduling.
- `<memcpy>`: implemented as packetized mesh traffic. It does not reserve the
  exact SDK memcpy infrastructure resources.
- `<collectives_2d>`: implements communication patterns, not exact Cerebras
  collective algorithms, route trees, colors, or task IDs.
- `<tile_config>`: implements FIFO/filter/switch concepts, not exact hardware
  queue descriptor semantics.

## Intentionally Omitted

- Real CSL parsing, type checking, compilation, and linking
- `@set_tile_code` compiler placement semantics
- Real fabric DSD behavior
- Exact task ID, color, routing, and queue descriptor hardware behavior
- WSE microthread scheduler accuracy
- SDK `SdkRuntime` launch/memcpy command-stream semantics
- Hardware resource reservation and placement constraints

These are omitted because they require compiler/runtime/hardware semantics rather
than architecture-level simulation. The current goal is to support application
workloads with reusable SimPy abstractions before deciding whether to target CSL,
Chisel, or another hardware path.

