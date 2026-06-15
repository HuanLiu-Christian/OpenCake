package wafersim.rtl

import chisel3._
import chisel3.util._

class PeMemcpyEngine(
    depthWords: Int = 1024,
    dataBits: Int = 32,
    lenBits: Int = 32
) extends Module {
  require(depthWords > 0, "depthWords must be positive")
  require(dataBits % 8 == 0, "dataBits must be byte-addressable")
  require(isPow2(dataBits / 8), "dataBits / 8 must be a power of two")

  private val addrBits = log2Ceil(depthWords)
  private val wordBytes = dataBits / 8
  private val wordBytesLog2 = log2Ceil(wordBytes)

  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new MemcpyCmd(addrBits, lenBits)))
    val dataIn = Flipped(Decoupled(UInt(dataBits.W)))
    val dataOut = Decoupled(UInt(dataBits.W))
    val done = Output(Bool())
    val busy = Output(Bool())
    val wordsRemaining = Output(UInt(lenBits.W))

    val debug = new Bundle {
      val readAddr = Input(UInt(addrBits.W))
      val readData = Output(UInt(dataBits.W))
    }
  })

  private val (
    sIdle :: sH2dWrite :: sD2hRead :: sD2hCapture :: sD2hSend ::
      sCopyRead :: sCopyCapture :: sCopyWrite :: sDone :: Nil
  ) = Enum(9)

  private val mem = SyncReadMem(depthWords, UInt(dataBits.W))

  private val state = RegInit(sIdle)
  private val srcAddrReg = RegInit(0.U(addrBits.W))
  private val dstAddrReg = RegInit(0.U(addrBits.W))
  private val wordIndexReg = RegInit(0.U(lenBits.W))
  private val totalWordsReg = RegInit(0.U(lenBits.W))
  private val readDataReg = RegInit(0.U(dataBits.W))
  private val readAddr = WireDefault(0.U(addrBits.W))
  private val readEnable = WireDefault(false.B)
  private val readData = mem.read(readAddr, readEnable)

  private val roundedWords =
    (io.cmd.bits.bytes + (wordBytes - 1).U(lenBits.W)) >> wordBytesLog2
  private val lastWord = wordIndexReg === (totalWordsReg - 1.U)

  io.cmd.ready := state === sIdle
  io.dataIn.ready := state === sH2dWrite
  io.dataOut.valid := state === sD2hSend
  io.dataOut.bits := readDataReg
  io.done := state === sDone
  io.busy := state =/= sIdle
  io.wordsRemaining := totalWordsReg - wordIndexReg

  io.debug.readData := mem.read(io.debug.readAddr, true.B)

  when(io.cmd.fire) {
    srcAddrReg := io.cmd.bits.srcAddr
    dstAddrReg := io.cmd.bits.dstAddr
    wordIndexReg := 0.U
    totalWordsReg := roundedWords

    when(roundedWords === 0.U) {
      state := sDone
    }.otherwise {
      switch(io.cmd.bits.op) {
        is(MemcpyOpcode.h2dWrite) {
          state := sH2dWrite
        }
        is(MemcpyOpcode.d2hRead) {
          state := sD2hRead
        }
        is(MemcpyOpcode.localCopy) {
          state := sCopyRead
        }
        is(3.U) {
          state := sDone
        }
      }
    }
  }

  switch(state) {
    is(sH2dWrite) {
      when(io.dataIn.fire) {
        mem.write(dstAddrReg + wordIndexReg, io.dataIn.bits)
        when(lastWord) {
          state := sDone
        }.otherwise {
          wordIndexReg := wordIndexReg + 1.U
        }
      }
    }

    is(sD2hRead) {
      readAddr := srcAddrReg + wordIndexReg
      readEnable := true.B
      state := sD2hCapture
    }

    is(sD2hCapture) {
      readDataReg := readData
      state := sD2hSend
    }

    is(sD2hSend) {
      when(io.dataOut.fire) {
        when(lastWord) {
          state := sDone
        }.otherwise {
          wordIndexReg := wordIndexReg + 1.U
          state := sD2hRead
        }
      }
    }

    is(sCopyRead) {
      readAddr := srcAddrReg + wordIndexReg
      readEnable := true.B
      state := sCopyCapture
    }

    is(sCopyCapture) {
      readDataReg := readData
      state := sCopyWrite
    }

    is(sCopyWrite) {
      mem.write(dstAddrReg + wordIndexReg, readDataReg)
      when(lastWord) {
        state := sDone
      }.otherwise {
        wordIndexReg := wordIndexReg + 1.U
        state := sCopyRead
      }
    }

    is(sDone) {
      state := sIdle
    }
  }
}

object EmitPeMemcpyEngine extends App {
  println(
    _root_.circt.stage.ChiselStage.emitSystemVerilog(
      new PeMemcpyEngine(),
      Array.empty,
      Array("-disable-all-randomization", "-strip-debug-info")
    )
  )
}
