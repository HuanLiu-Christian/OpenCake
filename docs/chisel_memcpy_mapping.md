# Chisel Mapping

This is the first RTL-level mapping target for the Python/SimPy simulator. It
models PE-local memcpy, a first two-PE memcpy path, and a multi-PE vector-dot
compute/reduction block so simulator semantics can be replayed and checked
against Chisel hardware.

## Scope

The current Chisel block maps the simulator's PE-local memcpy intent to a
fixed-function hardware datapath:

```text
host/testbench command + data stream
        |
        v
ProcessingElement / TwoPeMemcpyTop
        |
        v
PeMemcpyEngine <-> PE-local SRAM
```

Supported operations:

| Operation | Meaning | Current boundary |
| --- | --- | --- |
| `H2D_WRITE` | Host stream writes PE-local SRAM | single PE |
| `D2H_READ` | PE-local SRAM streams back to host | single PE |
| `LOCAL_COPY` | PE-local SRAM source range copies to destination range | single PE |
| `P2P_COPY` | Source PE SRAM range streams along a named color route into the next PE | two PEs |
| `VECTOR_DOT` | Load vector chunks into multiple PEs, compute local dots, reduce partial sums east | PE row |

This version does not instantiate RISC-V. A fixed-function FSM is enough to
validate the memory-copy contract. A PE-local processor or custom ISA can be
added later if kernels need runtime programmability.

## Files

```text
tests/fixtures/
  pe_local_memcpy.jsonl     shared SimPy/Chisel PE-local memcpy command IR
  pe_to_pe_memcpy.jsonl     shared SimPy/Chisel PE-to-PE memcpy command IR
  pe_local_memcpy.trace     legacy text trace converted into the same IR

examples/pe_local_memcpy/
  run.py                    builds API calls, emits command IR, runs SimPy
examples/pe_to_pe_memcpy/
  run.py                    builds color-routed two-PE API calls, emits command IR, runs SimPy

src/wafersim/
  mapping/
    memcpy_ir.py            portable memcpy command dataclass and program API
    trace_io.py             JSONL IR reader/writer
    vector_dot_ir.py        portable multi-PE vector-dot command dataclass and program API
    vector_dot_trace_io.py  vector-dot JSONL IR reader/writer
  hardware/
    local_memcpy.py         SimPy PE SRAM memcpy golden model plus color p2p routing
    vector_dot.py           SimPy multi-PE vector-dot golden model

chisel/
  build.sbt
  project/build.properties
  src/main/scala/wafersim/rtl/
    MemcpyCmd.scala         command bundle and opcodes
    PeMemcpyEngine.scala    FSM + PE-local SRAM
    PeToPeMemcpyCmd.scala   two-PE copy command bundle with route direction
    TwoPeMemcpyTop.scala    two PE engines plus color-routed p2p streaming controller
    VectorDotEngine.scala   PE-local vector-dot FSM and SRAMs
    VectorDotRowTop.scala   multi-PE row wrapper and eastward reduction FSM
    ProcessingElement.scala PE wrapper exposing PE-level memcpy ports
  src/test/scala/wafersim/rtl/
    PeMemcpyEngineTest.scala
                            H2D, D2H, local-copy, and SimPy trace replay tests
    PeToPeMemcpyTest.scala  two-PE command IR replay test
    VectorDotEngineTest.scala
                            PE-local dot and multi-PE vector-dot command IR replay test
```

## SimPy-To-Chisel Mapping Pipeline

The mapping validation now uses one command IR as the shared contract:

```text
MemcpyProgram API calls
VectorDotProgram API calls
        |
        v
MemcpyCommand IR / VectorDotCommand IR
        |
        +--> JSONL fixture:
        |       tests/fixtures/pe_local_memcpy.jsonl
        |       tests/fixtures/pe_to_pe_memcpy.jsonl
        |       tests/fixtures/vector_dot.jsonl
        |
        +--> SimPy LocalMemcpyPe / run_memcpy_commands(command)
        |       executes h2d_write / local_copy / d2h_read on Python SRAM
        |       resolves p2p_copy color through color_routes, then streams
        |       source PE SRAM into the routed destination PE SRAM
        |       checks expected final SRAM/output
        |
        +--> SimPy VectorDotPe / run_vector_dot_commands(command)
        |       loads left/right chunks into Python PE-local storage
        |       applies local compute timing for word_count * ops_per_mac
        |       replays reduce_east commands as PE-to-PE partial-sum accumulation
        |       checks the expected dot-product scalar at the east edge PE
        |
        +--> Chisel PeMemcpyEngineTest / PeToPeMemcpyTest
        +--> Chisel VectorDotEngineTest
                reads the same JSONL IR
                translates op strings to MemcpyOpcode values
                translates p2p color names to routeDir values
                streams vector chunks into VectorDotRowTop for multi-PE compute IR
                drives the RTL command/data interfaces
                checks RTL SRAM/output against the IR expectations
```

The RTL test is therefore not only a standalone Chisel unit test. It checks that
the Chisel engine behaves like the SimPy memcpy model for the same command
sequence. For PE-to-PE copies, the JSONL command carries a color such as
`east_memcpy`; the SimPy runner uses `COLOR_ROUTES` to resolve that color to a
direction, while the Chisel test lowers the same color to the RTL `routeDir`.
This is still not a full compiler, but it gives the simulator a real mapping
boundary: workload code can emit portable commands, and each backend decides how
to execute them.

For a custom command sequence, generate a JSONL file from the Python API and
point the Chisel test at it:

```bash
python examples/pe_local_memcpy/run.py --ir-out /tmp/pe_local_memcpy.jsonl
python examples/pe_to_pe_memcpy/run.py --ir-out /tmp/pe_to_pe_memcpy.jsonl
python examples/vector_vector_dot/run.py --ir-out /tmp/vector_dot.jsonl
cd chisel
WAFERSIM_MEMCPY_IR=/tmp/pe_local_memcpy.jsonl \
WAFERSIM_P2P_MEMCPY_IR=/tmp/pe_to_pe_memcpy.jsonl \
WAFERSIM_VECTOR_DOT_IR=/tmp/vector_dot.jsonl \
sbt test
```

## Cycle Model

The engine uses one SRAM word per transfer step. With the default 32-bit data
path, one word is 4 bytes:

```text
H2D_WRITE cycles  ~= bytes / 4
D2H_READ cycles   ~= bytes / 4 plus synchronous-read bubbles
LOCAL_COPY cycles ~= 2 * bytes / 4 because it reads then writes each word
P2P_COPY cycles   ~= source D2H stream plus destination H2D stream in the
                    two-PE Chisel controller
VECTOR_DOT cycles ~= per-PE chunk synchronous SRAM read/accumulate steps plus
                    one eastward row-reduction step per participating PE
```

This is a deliberately simple hardware baseline. Later versions can increase
`dataBits`, add burst interfaces, insert a FIFO, or connect the path to a
router for PE-to-PE memcpy.

## Run

From the repository root:

```bash
python examples/pe_local_memcpy/run.py
python examples/pe_to_pe_memcpy/run.py
python examples/vector_vector_dot/run.py --ir-out /tmp/vector_dot.jsonl
python -m unittest discover -s tests
cd chisel
sbt test
```

Emit SystemVerilog:

```bash
cd chisel
sbt "runMain wafersim.rtl.EmitPeMemcpyEngine"
```
