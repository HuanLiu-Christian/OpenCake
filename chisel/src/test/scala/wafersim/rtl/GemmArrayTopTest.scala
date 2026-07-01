package wafersim.rtl

import chisel3._
import chiseltest._
import java.nio.file.Paths
import scala.io.Source
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GemmArrayTopTest extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  behavior of "GemmArrayTop"

  private sealed trait IrStep
  private case class LoadAPanel(pe: (Int, Int), rows: Int, k: Int, values: Seq[BigInt]) extends IrStep
  private case class LoadBPanel(pe: (Int, Int), cols: Int, k: Int, values: Seq[BigInt]) extends IrStep
  private case class ForwardAEast(srcPe: (Int, Int), dstPe: (Int, Int), color: String) extends IrStep
  private case class ForwardBSouth(srcPe: (Int, Int), dstPe: (Int, Int), color: String) extends IrStep
  private case class ComputeC(pe: (Int, Int), rows: Int, cols: Int, k: Int) extends IrStep
  private case class ExpectC(pe: (Int, Int), rows: Int, cols: Int, expected: Seq[BigInt]) extends IrStep

  private val kernelCols = 2
  private val rowsPerPe = 2
  private val colsPerPe = 2
  private val kDim = 4
  private val cWordsPerPe = rowsPerPe * colsPerPe

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

  private def parseGemmIr(): Seq[IrStep] = {
    val tracePath = sys.props
      .get("wafersim.gemm.ir")
      .orElse(sys.env.get("WAFERSIM_GEMM_IR"))
      .map(Paths.get(_))
      .getOrElse(Paths.get("..", "tests", "fixtures", "gemm_streaming.jsonl"))
    val source = Source.fromFile(tracePath.toFile)
    try {
      source
        .getLines()
        .map(_.trim)
        .filter(line => line.nonEmpty && !line.startsWith("#"))
        .map { line =>
          stringField(line, "op") match {
            case "load_a_panel" =>
              LoadAPanel(peCoordField(line, "pe"), intField(line, "rows"), intField(line, "k"), valuesField(line, "values"))
            case "load_b_panel" =>
              LoadBPanel(peCoordField(line, "pe"), intField(line, "cols"), intField(line, "k"), valuesField(line, "values"))
            case "forward_a_east" =>
              ForwardAEast(peCoordField(line, "src_pe"), peCoordField(line, "dst_pe"), stringField(line, "color"))
            case "forward_b_south" =>
              ForwardBSouth(peCoordField(line, "src_pe"), peCoordField(line, "dst_pe"), stringField(line, "color"))
            case "compute_c" =>
              ComputeC(peCoordField(line, "pe"), intField(line, "rows"), intField(line, "cols"), intField(line, "k"))
            case "expect_c" =>
              ExpectC(peCoordField(line, "pe"), intField(line, "rows"), intField(line, "cols"), valuesField(line, "expected"))
            case other =>
              throw new IllegalArgumentException(s"unknown GEMM IR op: $other")
          }
        }
        .toSeq
    } finally {
      source.close()
    }
  }

  private def idleInputs(dut: GemmArrayTop): Unit = {
    dut.io.peSel.poke(0.U)
    dut.io.aIn.valid.poke(false.B)
    dut.io.bIn.valid.poke(false.B)
    dut.io.forwardA.valid.poke(false.B)
    dut.io.forwardB.valid.poke(false.B)
    dut.io.start.valid.poke(false.B)
    dut.io.result.ready.poke(false.B)
  }

  private def writeA(dut: GemmArrayTop, pe: (Int, Int), values: Seq[BigInt]): Unit = {
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

  private def writeB(dut: GemmArrayTop, pe: (Int, Int), values: Seq[BigInt]): Unit = {
    dut.io.peSel.poke(peIndex(pe).U)
    for (value <- values) {
      dut.io.bIn.valid.poke(true.B)
      dut.io.bIn.bits.poke(value.U)
      while (!dut.io.bIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.bIn.valid.poke(false.B)
  }

  private def forwardA(dut: GemmArrayTop, srcPe: (Int, Int)): Unit = {
    dut.io.forwardA.valid.poke(true.B)
    dut.io.forwardA.bits.row.poke(srcPe._1.U)
    dut.io.forwardA.bits.col.poke(srcPe._2.U)
    while (!dut.io.forwardA.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.forwardA.valid.poke(false.B)
  }

  private def forwardB(dut: GemmArrayTop, srcPe: (Int, Int)): Unit = {
    dut.io.forwardB.valid.poke(true.B)
    dut.io.forwardB.bits.row.poke(srcPe._1.U)
    dut.io.forwardB.bits.col.poke(srcPe._2.U)
    while (!dut.io.forwardB.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.forwardB.valid.poke(false.B)
  }

  private def startCompute(dut: GemmArrayTop): Seq[BigInt] = {
    dut.io.start.valid.poke(true.B)
    dut.io.start.bits.poke(true.B)
    while (!dut.io.start.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.start.valid.poke(false.B)
    dut.io.result.ready.poke(true.B)
    while (!dut.io.result.valid.peekBoolean()) {
      dut.clock.step()
    }
    val result = dut.io.result.bits.map(_.peek().litValue)
    dut.clock.step()
    dut.io.result.ready.poke(false.B)
    result
  }

  it should "replay the SimPy multi-PE GEMM command IR" in {
    test(new GemmArrayTop(kernelRows = 2, kernelCols = 2, rowsPerPe = 2, colsPerPe = 2, kDim = 4)) { dut =>
      idleInputs(dut)
      var computeStarted = false
      var arrayResult = Seq.empty[BigInt]
      for (step <- parseGemmIr()) {
        step match {
          case LoadAPanel(pe, rows, k, values) =>
            rows shouldBe rowsPerPe
            k shouldBe kDim
            writeA(dut, pe, values)
          case LoadBPanel(pe, cols, k, values) =>
            cols shouldBe colsPerPe
            k shouldBe kDim
            writeB(dut, pe, values)
          case ForwardAEast(srcPe, dstPe, color) =>
            color shouldBe "a_color"
            dstPe shouldBe (srcPe._1, srcPe._2 + 1)
            forwardA(dut, srcPe)
          case ForwardBSouth(srcPe, dstPe, color) =>
            color shouldBe "b_color"
            dstPe shouldBe (srcPe._1 + 1, srcPe._2)
            forwardB(dut, srcPe)
          case ComputeC(_, rows, cols, k) =>
            rows shouldBe rowsPerPe
            cols shouldBe colsPerPe
            k shouldBe kDim
            if (!computeStarted) {
              arrayResult = startCompute(dut)
              computeStarted = true
            }
          case ExpectC(pe, rows, cols, expected) =>
            rows shouldBe rowsPerPe
            cols shouldBe colsPerPe
            val offset = peIndex(pe) * cWordsPerPe
            arrayResult.slice(offset, offset + cWordsPerPe) shouldBe expected
        }
      }
    }
  }
}
