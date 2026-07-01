package wafersim.rtl

import chisel3._
import chisel3.util._

class GemmPeEngine(
    rowsPerPe: Int = 2,
    colsPerPe: Int = 2,
    kDim: Int = 4,
    dataBits: Int = 32,
    accBits: Int = 64
) extends Module {
  require(rowsPerPe > 0, "rowsPerPe must be positive")
  require(colsPerPe > 0, "colsPerPe must be positive")
  require(kDim > 0, "kDim must be positive")

  private val aWords = rowsPerPe * kDim
  private val bWords = kDim * colsPerPe
  private val cWords = rowsPerPe * colsPerPe
  private val aIndexBits = math.max(1, log2Ceil(aWords))
  private val bIndexBits = math.max(1, log2Ceil(bWords))
  private val rowBits = math.max(1, log2Ceil(rowsPerPe))
  private val colBits = math.max(1, log2Ceil(colsPerPe))
  private val kBits = math.max(1, log2Ceil(kDim))

  val io = IO(new Bundle {
    val aIn = Flipped(Decoupled(UInt(dataBits.W)))
    val bIn = Flipped(Decoupled(UInt(dataBits.W)))
    val aCopyIn = Flipped(Decoupled(Vec(aWords, UInt(dataBits.W))))
    val bCopyIn = Flipped(Decoupled(Vec(bWords, UInt(dataBits.W))))
    val start = Flipped(Decoupled(Bool()))
    val result = Decoupled(Vec(cWords, UInt(accBits.W)))
    val aDebug = Output(Vec(aWords, UInt(dataBits.W)))
    val bDebug = Output(Vec(bWords, UInt(dataBits.W)))
    val busy = Output(Bool())
  })

  private val (sIdle :: sCompute :: sResult :: Nil) = Enum(3)
  private val aRegs = RegInit(VecInit(Seq.fill(aWords)(0.U(dataBits.W))))
  private val bRegs = RegInit(VecInit(Seq.fill(bWords)(0.U(dataBits.W))))
  private val cRegs = RegInit(VecInit(Seq.fill(cWords)(0.U(accBits.W))))
  private val aWriteIndex = RegInit(0.U(aIndexBits.W))
  private val bWriteIndex = RegInit(0.U(bIndexBits.W))
  private val rowIndex = RegInit(0.U(rowBits.W))
  private val colIndex = RegInit(0.U(colBits.W))
  private val kIndex = RegInit(0.U(kBits.W))
  private val state = RegInit(sIdle)

  private val lastRow = rowIndex === (rowsPerPe - 1).U
  private val lastCol = colIndex === (colsPerPe - 1).U
  private val lastK = kIndex === (kDim - 1).U
  private val aIndex = (rowIndex * kDim.U + kIndex)(aIndexBits - 1, 0)
  private val bIndex = (kIndex * colsPerPe.U + colIndex)(bIndexBits - 1, 0)
  private val cIndex = (rowIndex * colsPerPe.U + colIndex)(math.max(1, log2Ceil(cWords)) - 1, 0)

  io.aIn.ready := true.B
  io.bIn.ready := true.B
  io.aCopyIn.ready := true.B
  io.bCopyIn.ready := true.B
  io.start.ready := state === sIdle
  io.result.valid := state === sResult
  io.result.bits := cRegs
  io.aDebug := aRegs
  io.bDebug := bRegs
  io.busy := state =/= sIdle

  when(io.aIn.fire) {
    aRegs(aWriteIndex) := io.aIn.bits
    aWriteIndex := Mux(aWriteIndex === (aWords - 1).U, 0.U, aWriteIndex + 1.U)
  }

  when(io.bIn.fire) {
    bRegs(bWriteIndex) := io.bIn.bits
    bWriteIndex := Mux(bWriteIndex === (bWords - 1).U, 0.U, bWriteIndex + 1.U)
  }

  when(io.aCopyIn.fire) {
    aRegs := io.aCopyIn.bits
  }

  when(io.bCopyIn.fire) {
    bRegs := io.bCopyIn.bits
  }

  when(io.start.fire) {
    cRegs := VecInit(Seq.fill(cWords)(0.U(accBits.W)))
    rowIndex := 0.U
    colIndex := 0.U
    kIndex := 0.U
    state := sCompute
  }

  switch(state) {
    is(sCompute) {
      cRegs(cIndex) := cRegs(cIndex) + (aRegs(aIndex) * bRegs(bIndex))
      when(lastK) {
        kIndex := 0.U
        when(lastCol) {
          colIndex := 0.U
          when(lastRow) {
            state := sResult
          }.otherwise {
            rowIndex := rowIndex + 1.U
          }
        }.otherwise {
          colIndex := colIndex + 1.U
        }
      }.otherwise {
        kIndex := kIndex + 1.U
      }
    }

    is(sResult) {
      when(io.result.fire) {
        state := sIdle
      }
    }
  }
}
