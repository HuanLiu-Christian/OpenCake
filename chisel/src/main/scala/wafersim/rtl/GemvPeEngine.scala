package wafersim.rtl

import chisel3._
import chisel3.util._

class GemvPeEngine(
    rowsPerPe: Int = 2,
    colsPerPe: Int = 2,
    dataBits: Int = 32,
    accBits: Int = 64
) extends Module {
  require(rowsPerPe > 0, "rowsPerPe must be positive")
  require(colsPerPe > 0, "colsPerPe must be positive")

  private val aWords = rowsPerPe * colsPerPe
  private val aIndexBits = math.max(1, log2Ceil(aWords))
  private val rowBits = math.max(1, log2Ceil(rowsPerPe))
  private val colBits = math.max(1, log2Ceil(colsPerPe))

  val io = IO(new Bundle {
    val aIn = Flipped(Decoupled(UInt(dataBits.W)))
    val xIn = Flipped(Decoupled(UInt(dataBits.W)))
    val biasIn = Flipped(Decoupled(UInt(dataBits.W)))
    val xCopyIn = Flipped(Decoupled(Vec(colsPerPe, UInt(dataBits.W))))
    val start = Flipped(Decoupled(Bool()))
    val result = Decoupled(Vec(rowsPerPe, UInt(accBits.W)))
    val xDebug = Output(Vec(colsPerPe, UInt(dataBits.W)))
    val busy = Output(Bool())
  })

  private val (
    sIdle :: sCompute :: sResult :: Nil
  ) = Enum(3)

  private val aRegs = RegInit(VecInit(Seq.fill(aWords)(0.U(dataBits.W))))
  private val xRegs = RegInit(VecInit(Seq.fill(colsPerPe)(0.U(dataBits.W))))
  private val yRegs = RegInit(VecInit(Seq.fill(rowsPerPe)(0.U(accBits.W))))
  private val aWriteIndex = RegInit(0.U(aIndexBits.W))
  private val xWriteIndex = RegInit(0.U(colBits.W))
  private val biasWriteIndex = RegInit(0.U(rowBits.W))
  private val rowIndex = RegInit(0.U(rowBits.W))
  private val colIndex = RegInit(0.U(colBits.W))
  private val state = RegInit(sIdle)

  private val lastRow = rowIndex === (rowsPerPe - 1).U
  private val lastCol = colIndex === (colsPerPe - 1).U
  private val aIndex = (rowIndex * colsPerPe.U + colIndex)(aIndexBits - 1, 0)

  io.aIn.ready := true.B
  io.xIn.ready := true.B
  io.biasIn.ready := true.B
  io.xCopyIn.ready := true.B
  io.start.ready := state === sIdle
  io.result.valid := state === sResult
  io.result.bits := yRegs
  io.xDebug := xRegs
  io.busy := state =/= sIdle

  when(io.aIn.fire) {
    aRegs(aWriteIndex) := io.aIn.bits
    aWriteIndex := Mux(aWriteIndex === (aWords - 1).U, 0.U, aWriteIndex + 1.U)
  }

  when(io.xIn.fire) {
    xRegs(xWriteIndex) := io.xIn.bits
    xWriteIndex := Mux(xWriteIndex === (colsPerPe - 1).U, 0.U, xWriteIndex + 1.U)
  }

  when(io.biasIn.fire) {
    yRegs(biasWriteIndex) := io.biasIn.bits
    biasWriteIndex := Mux(biasWriteIndex === (rowsPerPe - 1).U, 0.U, biasWriteIndex + 1.U)
  }

  when(io.xCopyIn.fire) {
    xRegs := io.xCopyIn.bits
  }

  when(io.start.fire) {
    rowIndex := 0.U
    colIndex := 0.U
    state := sCompute
  }

  switch(state) {
    is(sCompute) {
      yRegs(rowIndex) := yRegs(rowIndex) + (aRegs(aIndex) * xRegs(colIndex))
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
    }

    is(sResult) {
      when(io.result.fire) {
        state := sIdle
      }
    }
  }
}
