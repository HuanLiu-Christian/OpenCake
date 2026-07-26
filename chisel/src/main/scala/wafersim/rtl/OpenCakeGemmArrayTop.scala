package wafersim.rtl

import chisel3._
import chisel3.util._

class OpenCakeGemmArrayTop(
    rankRows: Int,
    rankCols: Int,
    tileM: Int,
    tileN: Int,
    tileK: Int,
    dataBits: Int = 32,
    accBits: Int = 64
) extends Module {
  require(rankRows > 0, "rankRows must be positive")
  require(rankCols > 0, "rankCols must be positive")
  require(tileM > 0, "tileM must be positive")
  require(tileN > 0, "tileN must be positive")
  require(tileK > 0, "tileK must be positive")

  private val rankCount = rankRows * rankCols
  private val peSelBits = math.max(1, log2Ceil(rankCount))
  private val rowBits = math.max(1, log2Ceil(rankRows))
  private val colBits = math.max(1, log2Ceil(rankCols))
  private val aWords = tileM * tileK
  private val bWords = tileK * tileN
  private val cWordsPerRank = tileM * tileN
  private val cWords = rankCount * cWordsPerRank

  class ForwardCmd extends Bundle {
    val row = UInt(rowBits.W)
    val col = UInt(colBits.W)
  }

  val io = IO(new Bundle {
    val peSel = Input(UInt(peSelBits.W))
    val aIn = Flipped(Decoupled(UInt(dataBits.W)))
    val bIn = Flipped(Decoupled(UInt(dataBits.W)))
    val forwardA = Flipped(Decoupled(new ForwardCmd))
    val forwardB = Flipped(Decoupled(new ForwardCmd))
    val start = Flipped(Decoupled(Bool()))
    val result = Decoupled(Vec(cWords, UInt(accBits.W)))
    val busy = Output(Bool())
  })

  private val engines = Seq.fill(rankRows, rankCols)(
    Module(new GemmPeEngine(tileM, tileN, tileK, dataBits, accBits))
  )
  private val (sIdle :: sCompute :: sResult :: Nil) = Enum(3)
  private val state = RegInit(sIdle)
  private val cRegs = RegInit(VecInit(Seq.fill(cWords)(0.U(accBits.W))))
  private val clearAccum = RegInit(false.B)
  private val allStartReady = engines.flatten.map(_.io.start.ready).reduce(_ && _)
  private val allResultsValid = engines.flatten.map(_.io.result.valid).reduce(_ && _)

  for (row <- 0 until rankRows; col <- 0 until rankCols) {
    val engine = engines(row)(col)
    val selected = io.peSel === (row * rankCols + col).U

    engine.io.aIn.valid := io.aIn.valid && selected
    engine.io.aIn.bits := io.aIn.bits
    engine.io.bIn.valid := io.bIn.valid && selected
    engine.io.bIn.bits := io.bIn.bits
    engine.io.start.valid := io.start.valid && io.start.ready
    engine.io.start.bits := true.B
    engine.io.result.ready := state === sCompute && allResultsValid

    val forwardAToThis =
      if (col == 0) {
        false.B
      } else {
        io.forwardA.valid &&
          (io.forwardA.bits.row === row.U) &&
          (io.forwardA.bits.col === (col - 1).U)
      }
    engine.io.aCopyIn.valid := forwardAToThis
    engine.io.aCopyIn.bits := (if (col == 0) {
        VecInit(Seq.fill(aWords)(0.U(dataBits.W)))
      } else {
        engines(row)(col - 1).io.aDebug
      })

    val forwardBToThis =
      if (row == 0) {
        false.B
      } else {
        io.forwardB.valid &&
          (io.forwardB.bits.row === (row - 1).U) &&
          (io.forwardB.bits.col === col.U)
      }
    engine.io.bCopyIn.valid := forwardBToThis
    engine.io.bCopyIn.bits := (if (row == 0) {
        VecInit(Seq.fill(bWords)(0.U(dataBits.W)))
      } else {
        engines(row - 1)(col).io.bDebug
      })
  }

  io.aIn.ready := MuxLookup(io.peSel, false.B)(
    engines.flatten.zipWithIndex.map { case (engine, index) =>
      index.U -> engine.io.aIn.ready
    }
  )
  io.bIn.ready := MuxLookup(io.peSel, false.B)(
    engines.flatten.zipWithIndex.map { case (engine, index) =>
      index.U -> engine.io.bIn.ready
    }
  )
  io.forwardA.ready := state === sIdle
  io.forwardB.ready := state === sIdle
  io.start.ready := state === sIdle && allStartReady
  io.result.valid := state === sResult
  io.result.bits := cRegs
  io.busy := state =/= sIdle

  when(io.start.fire) {
    clearAccum := io.start.bits
    state := sCompute
  }

  switch(state) {
    is(sCompute) {
      when(allResultsValid) {
        for (row <- 0 until rankRows; col <- 0 until rankCols; index <- 0 until cWordsPerRank) {
          val flatIndex = (row * rankCols + col) * cWordsPerRank + index
          val chunkValue = engines(row)(col).io.result.bits(index)
          cRegs(flatIndex) := Mux(clearAccum, chunkValue, cRegs(flatIndex) + chunkValue)
        }
        state := sResult
      }
    }

    is(sResult) {
      when(io.result.fire) {
        state := sIdle
      }
    }
  }
}

