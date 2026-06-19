package wafersim.rtl

import chisel3._
import chisel3.util._

class VectorDotEngine(
    maxWords: Int = 1024,
    dataBits: Int = 32,
    accBits: Int = 64,
    lenBits: Int = 32
) extends Module {
  require(maxWords > 0, "maxWords must be positive")
  require(accBits >= dataBits * 2, "accBits must hold one product")

  private val addrBits = log2Ceil(maxWords)

  val io = IO(new Bundle {
    val leftIn = Flipped(Decoupled(UInt(dataBits.W)))
    val rightIn = Flipped(Decoupled(UInt(dataBits.W)))
    val start = Flipped(Decoupled(UInt(lenBits.W)))
    val result = Decoupled(UInt(accBits.W))
    val busy = Output(Bool())

    val debug = new Bundle {
      val leftCount = Output(UInt(lenBits.W))
      val rightCount = Output(UInt(lenBits.W))
      val index = Output(UInt(lenBits.W))
      val accumulator = Output(UInt(accBits.W))
    }
  })

  private val (
    sIdle :: sRead :: sAccumulate :: sResult :: Nil
  ) = Enum(4)

  private val leftMem = SyncReadMem(maxWords, UInt(dataBits.W))
  private val rightMem = SyncReadMem(maxWords, UInt(dataBits.W))

  private val state = RegInit(sIdle)
  private val leftWriteIndex = RegInit(0.U(lenBits.W))
  private val rightWriteIndex = RegInit(0.U(lenBits.W))
  private val computeIndex = RegInit(0.U(lenBits.W))
  private val totalWords = RegInit(0.U(lenBits.W))
  private val accumulator = RegInit(0.U(accBits.W))
  private val readAddr = WireDefault(0.U(addrBits.W))
  private val readEnable = WireDefault(false.B)
  private val leftReadData = leftMem.read(readAddr, readEnable)
  private val rightReadData = rightMem.read(readAddr, readEnable)
  private val lastWord = computeIndex === (totalWords - 1.U)

  io.leftIn.ready := true.B
  io.rightIn.ready := true.B
  io.start.ready := state === sIdle
  io.result.valid := state === sResult
  io.result.bits := accumulator
  io.busy := state =/= sIdle
  io.debug.leftCount := leftWriteIndex
  io.debug.rightCount := rightWriteIndex
  io.debug.index := computeIndex
  io.debug.accumulator := accumulator

  when(io.leftIn.fire) {
    leftMem.write(leftWriteIndex(addrBits - 1, 0), io.leftIn.bits)
    leftWriteIndex := leftWriteIndex + 1.U
  }

  when(io.rightIn.fire) {
    rightMem.write(rightWriteIndex(addrBits - 1, 0), io.rightIn.bits)
    rightWriteIndex := rightWriteIndex + 1.U
  }

  when(io.start.fire) {
    computeIndex := 0.U
    totalWords := io.start.bits
    accumulator := 0.U
    state := Mux(io.start.bits === 0.U, sResult, sRead)
  }

  switch(state) {
    is(sRead) {
      readAddr := computeIndex(addrBits - 1, 0)
      readEnable := true.B
      state := sAccumulate
    }

    is(sAccumulate) {
      accumulator := accumulator + (leftReadData * rightReadData)
      when(lastWord) {
        state := sResult
      }.otherwise {
        computeIndex := computeIndex + 1.U
        state := sRead
      }
    }

    is(sResult) {
      when(io.result.fire) {
        state := sIdle
      }
    }
  }
}

object EmitVectorDotEngine extends App {
  println(
    _root_.circt.stage.ChiselStage.emitSystemVerilog(
      new VectorDotEngine(),
      Array.empty,
      Array("-disable-all-randomization", "-strip-debug-info")
    )
  )
}
