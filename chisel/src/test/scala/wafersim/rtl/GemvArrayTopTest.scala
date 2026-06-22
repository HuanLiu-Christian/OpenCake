package wafersim.rtl

import chisel3._
import chiseltest._
import java.nio.file.Paths
import scala.io.Source
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GemvArrayTopTest extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  behavior of "GemvArrayTop"

  private sealed trait IrStep
  private case class LoadATile(pe: (Int, Int), rows: Int, cols: Int, values: Seq[BigInt]) extends IrStep
  private case class LoadXChunk(pe: (Int, Int), values: Seq[BigInt]) extends IrStep
  private case class LoadBias(pe: (Int, Int), values: Seq[BigInt]) extends IrStep
  private case class ForwardXSouth(srcPe: (Int, Int), dstPe: (Int, Int), color: String) extends IrStep
  private case class ComputeAx(pe: (Int, Int), rows: Int, cols: Int) extends IrStep
  private case class ReduceEast(srcPe: (Int, Int), dstPe: (Int, Int), color: String) extends IrStep
  private case class ExpectY(pe: (Int, Int), expected: Seq[BigInt]) extends IrStep

  private val kernelCols = 4
  private val rowsPerPe = 2

  private def stringField(line: String, key: String): String = {
    val pattern = ("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"").r
    pattern
      .findFirstMatchIn(line)
      .map(_.group(1))
      .getOrElse(throw new IllegalArgumentException(s"missing string field: $key"))
  }

  private def intField(line: String, key: String, default: Int = 0): Int = {
    val pattern = ("\"" + key + "\"\\s*:\\s*(-?\\d+)").r
    pattern.findFirstMatchIn(line).map(_.group(1).toInt).getOrElse(default)
  }

  private def valuesField(line: String, key: String): Seq[BigInt] = {
    val pattern = ("\"" + key + "\"\\s*:\\s*\\[(.*?)\\]").r
    pattern.findFirstMatchIn(line) match {
      case Some(matched) =>
        val body = matched.group(1).trim
        if (body.isEmpty) Seq.empty else body.split(",").toSeq.map(value => BigInt(value.trim))
      case None =>
        Seq.empty
    }
  }

  private def peCoordField(line: String, key: String): (Int, Int) = {
    val pattern = ("\"" + key + "\"\\s*:\\s*\\[(\\d+)\\s*,\\s*(\\d+)\\]").r
    pattern.findFirstMatchIn(line) match {
      case Some(matched) =>
        (matched.group(1).toInt, matched.group(2).toInt)
      case None =>
        throw new IllegalArgumentException(s"missing PE coordinate field: $key")
    }
  }

  private def peIndex(pe: (Int, Int)): Int = pe._1 * kernelCols + pe._2

  private def parseGemvIr(): Seq[IrStep] = {
    val tracePath = sys.props
      .get("wafersim.gemv.ir")
      .orElse(sys.env.get("WAFERSIM_GEMV_IR"))
      .map(Paths.get(_))
      .getOrElse(Paths.get("..", "tests", "fixtures", "gemv_09_streaming.jsonl"))
    val source = Source.fromFile(tracePath.toFile)
    try {
      source
        .getLines()
        .map(_.trim)
        .filter(line => line.nonEmpty && !line.startsWith("#"))
        .map { line =>
          stringField(line, "op") match {
            case "load_a_tile" =>
              LoadATile(peCoordField(line, "pe"), intField(line, "rows"), intField(line, "cols"), valuesField(line, "values"))
            case "load_x_chunk" =>
              LoadXChunk(peCoordField(line, "pe"), valuesField(line, "values"))
            case "load_bias" =>
              LoadBias(peCoordField(line, "pe"), valuesField(line, "values"))
            case "forward_x_south" =>
              ForwardXSouth(peCoordField(line, "src_pe"), peCoordField(line, "dst_pe"), stringField(line, "color"))
            case "compute_ax" =>
              ComputeAx(peCoordField(line, "pe"), intField(line, "rows"), intField(line, "cols"))
            case "reduce_east" =>
              ReduceEast(peCoordField(line, "src_pe"), peCoordField(line, "dst_pe"), stringField(line, "color"))
            case "expect_y" =>
              ExpectY(peCoordField(line, "pe"), valuesField(line, "expected"))
            case other =>
              throw new IllegalArgumentException(s"unknown GEMV IR op: $other")
          }
        }
        .toSeq
    } finally {
      source.close()
    }
  }

  private def idleInputs(dut: GemvArrayTop): Unit = {
    dut.io.peSel.poke(0.U)
    dut.io.aIn.valid.poke(false.B)
    dut.io.xIn.valid.poke(false.B)
    dut.io.biasIn.valid.poke(false.B)
    dut.io.forwardX.valid.poke(false.B)
    dut.io.reduceEast.valid.poke(false.B)
    dut.io.start.valid.poke(false.B)
    dut.io.result.ready.poke(false.B)
  }

  private def writeA(dut: GemvArrayTop, pe: (Int, Int), values: Seq[BigInt]): Unit = {
    dut.io.peSel.poke(peIndex(pe).U)
    for (value <- values) {
      dut.io.aIn.valid.poke(true.B)
      dut.io.aIn.bits.poke(value.U)
      while (!dut.io.aIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.aIn.valid.poke(false.B)
  }

  private def writeX(dut: GemvArrayTop, pe: (Int, Int), values: Seq[BigInt]): Unit = {
    dut.io.peSel.poke(peIndex(pe).U)
    for (value <- values) {
      dut.io.xIn.valid.poke(true.B)
      dut.io.xIn.bits.poke(value.U)
      while (!dut.io.xIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.xIn.valid.poke(false.B)
  }

  private def writeBias(dut: GemvArrayTop, pe: (Int, Int), values: Seq[BigInt]): Unit = {
    dut.io.peSel.poke(peIndex(pe).U)
    for (value <- values) {
      dut.io.biasIn.valid.poke(true.B)
      dut.io.biasIn.bits.poke(value.U)
      while (!dut.io.biasIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.biasIn.valid.poke(false.B)
  }

  private def forwardX(dut: GemvArrayTop, srcPe: (Int, Int)): Unit = {
    dut.io.forwardX.valid.poke(true.B)
    dut.io.forwardX.bits.row.poke(srcPe._1.U)
    dut.io.forwardX.bits.col.poke(srcPe._2.U)
    while (!dut.io.forwardX.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.forwardX.valid.poke(false.B)
  }

  private def startCompute(dut: GemvArrayTop): Unit = {
    dut.io.start.valid.poke(true.B)
    dut.io.start.bits.poke(true.B)
    while (!dut.io.start.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.start.valid.poke(false.B)
    while (!dut.io.reduceEast.ready.peekBoolean() && !dut.io.result.valid.peekBoolean()) {
      dut.clock.step()
    }
  }

  private def reduceEast(dut: GemvArrayTop, srcPe: (Int, Int)): Unit = {
    dut.io.reduceEast.valid.poke(true.B)
    dut.io.reduceEast.bits.row.poke(srcPe._1.U)
    dut.io.reduceEast.bits.col.poke(srcPe._2.U)
    while (!dut.io.reduceEast.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.reduceEast.valid.poke(false.B)
  }

  private def readResult(dut: GemvArrayTop): Seq[BigInt] = {
    dut.io.result.ready.poke(true.B)
    while (!dut.io.result.valid.peekBoolean()) {
      dut.clock.step()
    }
    val result = dut.io.result.bits.map(_.peek().litValue)
    dut.clock.step()
    dut.io.result.ready.poke(false.B)
    result
  }

  it should "replay the SimPy multi-PE GEMV command IR" in {
    test(new GemvArrayTop(kernelRows = 3, kernelCols = 4, rowsPerPe = 2, colsPerPe = 2)) { dut =>
      idleInputs(dut)
      var computeStarted = false
      var arrayResult = Seq.empty[BigInt]
      for (step <- parseGemvIr()) {
        step match {
          case LoadATile(pe, rows, cols, values) =>
            rows shouldBe rowsPerPe
            cols shouldBe 2
            writeA(dut, pe, values)
          case LoadXChunk(pe, values) =>
            pe._1 shouldBe 0
            writeX(dut, pe, values)
          case LoadBias(pe, values) =>
            pe._2 shouldBe 0
            writeBias(dut, pe, values)
          case ForwardXSouth(srcPe, dstPe, color) =>
            color shouldBe "x_color"
            dstPe shouldBe (srcPe._1 + 1, srcPe._2)
            forwardX(dut, srcPe)
          case ComputeAx(_, rows, cols) =>
            rows shouldBe rowsPerPe
            cols shouldBe 2
            if (!computeStarted) {
              startCompute(dut)
              computeStarted = true
            }
          case ReduceEast(srcPe, dstPe, color) =>
            Set("ax_color_1", "ax_color_2") should contain(color)
            dstPe shouldBe (srcPe._1, srcPe._2 + 1)
            reduceEast(dut, srcPe)
          case ExpectY(pe, expected) =>
            pe._2 shouldBe 3
            if (arrayResult.isEmpty) {
              arrayResult = readResult(dut)
            }
            arrayResult.slice(pe._1 * rowsPerPe, pe._1 * rowsPerPe + rowsPerPe) shouldBe expected
        }
      }
    }
  }
}
