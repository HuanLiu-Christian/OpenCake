package wafersim.rtl

import chisel3._
import chisel3.util._

class VectorDotRowTop(
    peCount: Int = 4,
    maxWordsPerPe: Int = 1024,
    dataBits: Int = 32,
    accBits: Int = 64,
    lenBits: Int = 32
) extends Module {
  require(peCount > 0, "peCount must be positive")

  private val peSelBits = math.max(1, log2Ceil(peCount))
  private val reduceIndexBits = math.max(1, log2Ceil(peCount))

  val io = IO(new Bundle {
    val peSel = Input(UInt(peSelBits.W))
    val leftIn = Flipped(Decoupled(UInt(dataBits.W)))
    val rightIn = Flipped(Decoupled(UInt(dataBits.W)))
    val start = Flipped(Decoupled(UInt(lenBits.W)))
    val result = Decoupled(UInt(accBits.W))
    val busy = Output(Bool())

    val debug = new Bundle {
      val reductionIndex = Output(UInt(reduceIndexBits.W))
      val accumulator = Output(UInt(accBits.W))
      val partials = Output(Vec(peCount, UInt(accBits.W)))
    }
  })

  private val engines = Seq.fill(peCount)(
    Module(new VectorDotEngine(maxWordsPerPe, dataBits, accBits, lenBits))
  )

  private val (sIdle :: sCompute :: sReduce :: sResult :: Nil) = Enum(4)
  private val state = RegInit(sIdle)
  private val partials = RegInit(VecInit(Seq.fill(peCount)(0.U(accBits.W))))
  private val reduceIndex = RegInit(0.U(reduceIndexBits.W))
  private val accumulator = RegInit(0.U(accBits.W))
  private val allStartReady = engines.map(_.io.start.ready).reduce(_ && _)
  private val allResultsValid = engines.map(_.io.result.valid).reduce(_ && _)
  private val lastReducePe = reduceIndex === (peCount - 1).U

  for ((engine, index) <- engines.zipWithIndex) {
    val selected = io.peSel === index.U

    engine.io.leftIn.valid := io.leftIn.valid && selected
    engine.io.leftIn.bits := io.leftIn.bits
    engine.io.rightIn.valid := io.rightIn.valid && selected
    engine.io.rightIn.bits := io.rightIn.bits
    engine.io.start.valid := io.start.valid && io.start.ready
    engine.io.start.bits := io.start.bits
    engine.io.result.ready := state === sCompute && allResultsValid
  }

  io.leftIn.ready := MuxLookup(io.peSel, false.B)(
    engines.zipWithIndex.map { case (engine, index) =>
      index.U -> engine.io.leftIn.ready
    }
  )
  io.rightIn.ready := MuxLookup(io.peSel, false.B)(
    engines.zipWithIndex.map { case (engine, index) =>
      index.U -> engine.io.rightIn.ready
    }
  )
  io.start.ready := state === sIdle && allStartReady
  io.result.valid := state === sResult
  io.result.bits := accumulator
  io.busy := state =/= sIdle
  io.debug.reductionIndex := reduceIndex
  io.debug.accumulator := accumulator
  io.debug.partials := partials

  switch(state) {
    is(sIdle) {
      when(io.start.fire) {
        state := sCompute
      }
    }

    is(sCompute) {
      when(allResultsValid) {
        for ((engine, index) <- engines.zipWithIndex) {
          partials(index) := engine.io.result.bits
        }
        accumulator := 0.U
        reduceIndex := 0.U
        state := sReduce
      }
    }

    is(sReduce) {
      accumulator := accumulator + partials(reduceIndex)
      when(lastReducePe) {
        state := sResult
      }.otherwise {
        reduceIndex := reduceIndex + 1.U
      }
    }

    is(sResult) {
      when(io.result.fire) {
        state := sIdle
      }
    }
  }
}

object EmitVectorDotRowTop extends App {
  println(
    _root_.circt.stage.ChiselStage.emitSystemVerilog(
      new VectorDotRowTop(),
      Array.empty,
      Array("-disable-all-randomization", "-strip-debug-info")
    )
  )
}
