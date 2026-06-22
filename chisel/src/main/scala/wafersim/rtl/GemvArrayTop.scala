package wafersim.rtl

import chisel3._
import chisel3.util._

class GemvArrayTop(
    kernelRows: Int = 3,
    kernelCols: Int = 4,
    rowsPerPe: Int = 2,
    colsPerPe: Int = 2,
    dataBits: Int = 32,
    accBits: Int = 64
) extends Module {
  require(kernelRows > 0, "kernelRows must be positive")
  require(kernelCols > 0, "kernelCols must be positive")

  private val peCount = kernelRows * kernelCols
  private val peSelBits = math.max(1, log2Ceil(peCount))
  private val peIndexBits = math.max(1, log2Ceil(peCount))
  private val rowBits = math.max(1, log2Ceil(kernelRows))
  private val colBits = math.max(1, log2Ceil(kernelCols))
  private val yWords = kernelRows * rowsPerPe
  private val reduceOps = kernelRows * math.max(kernelCols - 1, 0)
  private val reduceCountBits = math.max(1, log2Ceil(reduceOps + 1))

  class ForwardXCmd extends Bundle {
    val row = UInt(rowBits.W)
    val col = UInt(colBits.W)
  }

  class ReduceEastCmd extends Bundle {
    val row = UInt(rowBits.W)
    val col = UInt(colBits.W)
  }

  val io = IO(new Bundle {
    val peSel = Input(UInt(peSelBits.W))
    val aIn = Flipped(Decoupled(UInt(dataBits.W)))
    val xIn = Flipped(Decoupled(UInt(dataBits.W)))
    val biasIn = Flipped(Decoupled(UInt(dataBits.W)))
    val forwardX = Flipped(Decoupled(new ForwardXCmd))
    val reduceEast = Flipped(Decoupled(new ReduceEastCmd))
    val start = Flipped(Decoupled(Bool()))
    val result = Decoupled(Vec(yWords, UInt(accBits.W)))
    val busy = Output(Bool())
  })

  private val engines = Seq.fill(kernelRows, kernelCols)(
    Module(new GemvPeEngine(rowsPerPe, colsPerPe, dataBits, accBits))
  )
  private val (sIdle :: sCompute :: sReduce :: sResult :: Nil) = Enum(4)
  private val state = RegInit(sIdle)
  private val peYRegs = RegInit(
    VecInit(Seq.fill(peCount)(VecInit(Seq.fill(rowsPerPe)(0.U(accBits.W)))))
  )
  private val reduceCount = RegInit(0.U(reduceCountBits.W))
  private val allStartReady = engines.flatten.map(_.io.start.ready).reduce(_ && _)
  private val allResultsValid = engines.flatten.map(_.io.result.valid).reduce(_ && _)

  for (row <- 0 until kernelRows; col <- 0 until kernelCols) {
    val engine = engines(row)(col)
    val selected = io.peSel === (row * kernelCols + col).U

    engine.io.aIn.valid := io.aIn.valid && selected
    engine.io.aIn.bits := io.aIn.bits
    engine.io.xIn.valid := io.xIn.valid && selected
    engine.io.xIn.bits := io.xIn.bits
    engine.io.biasIn.valid := io.biasIn.valid && selected
    engine.io.biasIn.bits := io.biasIn.bits
    engine.io.start.valid := io.start.valid && io.start.ready
    engine.io.start.bits := true.B
    engine.io.result.ready := state === sCompute && allResultsValid

    val forwardToThis =
      if (row == 0) {
        false.B
      } else {
        io.forwardX.valid &&
          (io.forwardX.bits.row === (row - 1).U) &&
          (io.forwardX.bits.col === col.U)
      }
    engine.io.xCopyIn.valid := forwardToThis
    engine.io.xCopyIn.bits := (if (row == 0) {
        VecInit(Seq.fill(colsPerPe)(0.U(dataBits.W)))
      } else {
        engines(row - 1)(col).io.xDebug
      })
  }

  io.aIn.ready := MuxLookup(io.peSel, false.B)(
    engines.flatten.zipWithIndex.map { case (engine, index) =>
      index.U -> engine.io.aIn.ready
    }
  )
  io.xIn.ready := MuxLookup(io.peSel, false.B)(
    engines.flatten.zipWithIndex.map { case (engine, index) =>
      index.U -> engine.io.xIn.ready
    }
  )
  io.biasIn.ready := MuxLookup(io.peSel, false.B)(
    engines.flatten.zipWithIndex.map { case (engine, index) =>
      index.U -> engine.io.biasIn.ready
    }
  )
  io.forwardX.ready := state === sIdle
  io.reduceEast.ready := state === sReduce && reduceCount =/= reduceOps.U
  io.start.ready := state === sIdle && allStartReady
  io.result.valid := state === sResult
  io.busy := state =/= sIdle
  for (row <- 0 until kernelRows; localRow <- 0 until rowsPerPe) {
    io.result.bits(row * rowsPerPe + localRow) :=
      peYRegs(row * kernelCols + kernelCols - 1)(localRow)
  }

  when(io.start.fire) {
    reduceCount := 0.U
    state := sCompute
  }

  switch(state) {
    is(sCompute) {
      when(allResultsValid) {
        for (row <- 0 until kernelRows; col <- 0 until kernelCols; localRow <- 0 until rowsPerPe) {
          peYRegs(row * kernelCols + col)(localRow) :=
            engines(row)(col).io.result.bits(localRow)
        }
        state := Mux(reduceOps.U === 0.U, sResult, sReduce)
      }
    }

    is(sReduce) {
      when(io.reduceEast.fire) {
        val srcIndex =
          (io.reduceEast.bits.row * kernelCols.U + io.reduceEast.bits.col)(
            peIndexBits - 1,
            0,
          )
        val dstIndex = (srcIndex + 1.U)(peIndexBits - 1, 0)
        for (localRow <- 0 until rowsPerPe) {
          peYRegs(dstIndex)(localRow) :=
            peYRegs(dstIndex)(localRow) + peYRegs(srcIndex)(localRow)
        }
        reduceCount := reduceCount + 1.U
        when(reduceCount + 1.U === reduceOps.U) {
          state := sResult
        }
      }
    }

    is(sResult) {
      when(io.result.fire) {
        state := sIdle
      }
    }
  }
}

object EmitGemvArrayTop extends App {
  println(
    _root_.circt.stage.ChiselStage.emitSystemVerilog(
      new GemvArrayTop(),
      Array.empty,
      Array("-disable-all-randomization", "-strip-debug-info")
    )
  )
}
