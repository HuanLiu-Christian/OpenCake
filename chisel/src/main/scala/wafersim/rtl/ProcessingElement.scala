package wafersim.rtl

import chisel3._
import chisel3.util._

class ProcessingElement(
    depthWords: Int = 1024,
    dataBits: Int = 32,
    lenBits: Int = 32
) extends Module {
  private val addrBits = log2Ceil(depthWords)

  val io = IO(new Bundle {
    val memcpyCmd = Flipped(Decoupled(new MemcpyCmd(addrBits, lenBits)))
    val hostDataIn = Flipped(Decoupled(UInt(dataBits.W)))
    val hostDataOut = Decoupled(UInt(dataBits.W))
    val memcpyDone = Output(Bool())
    val busy = Output(Bool())

    val debug = new Bundle {
      val readAddr = Input(UInt(addrBits.W))
      val readData = Output(UInt(dataBits.W))
    }
  })

  private val memcpy = Module(new PeMemcpyEngine(depthWords, dataBits, lenBits))

  memcpy.io.cmd <> io.memcpyCmd
  memcpy.io.dataIn <> io.hostDataIn
  io.hostDataOut <> memcpy.io.dataOut

  io.memcpyDone := memcpy.io.done
  io.busy := memcpy.io.busy
  memcpy.io.debug.readAddr := io.debug.readAddr
  io.debug.readData := memcpy.io.debug.readData
}

