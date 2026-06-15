# Chisel Memcpy Mapping

This is the first RTL-level mapping target for the Python/SimPy simulator. It
models PE-local memcpy plus a first two-PE memcpy path so the simulator's
memory-copy semantics can be replayed and checked against Chisel hardware.

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

Supported PE-local operations:

| Operation | Meaning | Current boundary |
| --- | --- | --- |
| `H2D_WRITE` | Host stream writes PE-local SRAM | single PE |
| `D2H_READ` | PE-local SRAM streams back to host | single PE |
| `LOCAL_COPY` | PE-local SRAM source range copies to destination range | single PE |
| `P2P_COPY` | Source PE SRAM range streams into destination PE SRAM | two PEs |

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
  run.py                    builds two-PE API calls, emits command IR, runs SimPy

src/wafersim/
  mapping/
    memcpy_ir.py            portable memcpy command dataclass and program API
    trace_io.py             JSONL IR reader/writer
  hardware/
    local_memcpy.py         SimPy PE-local SRAM memcpy golden model

chisel/
  build.sbt
  project/build.properties
  src/main/scala/wafersim/rtl/
    MemcpyCmd.scala         command bundle and opcodes
    PeMemcpyEngine.scala    FSM + PE-local SRAM
    PeToPeMemcpyCmd.scala   two-PE copy command bundle
    TwoPeMemcpyTop.scala    two PE engines plus p2p streaming controller
    ProcessingElement.scala PE wrapper exposing PE-level memcpy ports
  src/test/scala/wafersim/rtl/
    PeMemcpyEngineTest.scala
                            H2D, D2H, local-copy, and SimPy trace replay tests
    PeToPeMemcpyTest.scala  two-PE command IR replay test
```

## SimPy-To-Chisel Mapping Pipeline

The mapping validation now uses one command IR as the shared contract:

```text
MemcpyProgram API calls
        |
        v
MemcpyCommand IR
        |
        +--> JSONL fixture:
        |       tests/fixtures/pe_local_memcpy.jsonl
        |       tests/fixtures/pe_to_pe_memcpy.jsonl
        |
        +--> SimPy LocalMemcpyPe.execute(command)
        |       executes h2d_write / local_copy / d2h_read on Python SRAM
        |       executes p2p_copy by reading source PE SRAM and writing dst PE SRAM
        |       checks expected final SRAM/output
        |
        +--> Chisel PeMemcpyEngineTest / PeToPeMemcpyTest
                reads the same JSONL IR
                translates op strings to MemcpyOpcode values
                drives the RTL command/data interfaces
                checks RTL SRAM/output against the IR expectations
```

The RTL test is therefore not only a standalone Chisel unit test. It checks that
the Chisel engine behaves like the SimPy PE-local memcpy model for the same
command sequence. This is still not a full compiler, but it gives the simulator
a real mapping boundary: workload code can emit portable commands, and each
backend decides how to execute them.

For a custom command sequence, generate a JSONL file from the Python API and
point the Chisel test at it:

```bash
python examples/pe_local_memcpy/run.py --ir-out /tmp/pe_local_memcpy.jsonl
python examples/pe_to_pe_memcpy/run.py --ir-out /tmp/pe_to_pe_memcpy.jsonl
cd chisel
WAFERSIM_MEMCPY_IR=/tmp/pe_local_memcpy.jsonl \
WAFERSIM_P2P_MEMCPY_IR=/tmp/pe_to_pe_memcpy.jsonl \
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
```

This is a deliberately simple hardware baseline. Later versions can increase
`dataBits`, add burst interfaces, insert a FIFO, or connect the path to a
router for PE-to-PE memcpy.

## Run

From the repository root:

```bash
python examples/pe_local_memcpy/run.py
python examples/pe_to_pe_memcpy/run.py
python -m unittest discover -s tests
cd chisel
sbt test
```

Emit SystemVerilog:

```bash
cd chisel
sbt "runMain wafersim.rtl.EmitPeMemcpyEngine"
```
