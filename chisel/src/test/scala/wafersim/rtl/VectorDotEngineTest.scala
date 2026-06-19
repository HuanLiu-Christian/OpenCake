package wafersim.rtl

import chisel3._
import chiseltest._
import java.nio.file.Paths
import scala.io.Source
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class VectorDotEngineTest extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  behavior of "VectorDotEngine"

  private sealed trait IrStep
  private case class LoadLeft(pe: Int, words: Seq[BigInt]) extends IrStep
  private case class LoadRight(pe: Int, words: Seq[BigInt]) extends IrStep
  private case class ComputeDot(pe: Int, wordCount: Int) extends IrStep
  private case class ReduceEast(srcPe: Int, dstPe: Int, color: String) extends IrStep
  private case class Expect(expected: BigInt) extends IrStep

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

  private def wordsField(line: String): Seq[BigInt] = {
    val pattern = "\"words\"\\s*:\\s*\\[(.*?)\\]".r
    pattern.findFirstMatchIn(line) match {
      case Some(matched) =>
        val body = matched.group(1).trim
        if (body.isEmpty) Seq.empty else body.split(",").toSeq.map(value => BigInt(value.trim))
      case None =>
        Seq.empty
    }
  }

  private def peColField(line: String, key: String, default: Int = 0): Int = {
    val pattern = ("\"" + key + "\"\\s*:\\s*\\[(\\d+)\\s*,\\s*(\\d+)\\]").r
    pattern.findFirstMatchIn(line) match {
      case Some(matched) =>
        val row = matched.group(1).toInt
        val col = matched.group(2).toInt
        if (row != 0) {
          throw new IllegalArgumentException(s"vector-dot row top only supports row 0, got row $row")
        }
        col
      case None =>
        default
    }
  }

  private def parseVectorDotIr(): Seq[IrStep] = {
    val tracePath = sys.props
      .get("wafersim.vector_dot.ir")
      .orElse(sys.env.get("WAFERSIM_VECTOR_DOT_IR"))
      .map(Paths.get(_))
      .getOrElse(Paths.get("..", "tests", "fixtures", "vector_dot.jsonl"))
    val source = Source.fromFile(tracePath.toFile)
    try {
      source
        .getLines()
        .map(_.trim)
        .filter(line => line.nonEmpty && !line.startsWith("#"))
        .map { line =>
          stringField(line, "op") match {
            case "load_left" =>
              LoadLeft(peColField(line, "pe"), wordsField(line))
            case "load_right" =>
              LoadRight(peColField(line, "pe"), wordsField(line))
            case "compute_dot" =>
              ComputeDot(peColField(line, "pe"), intField(line, "word_count"))
            case "reduce_east" =>
              ReduceEast(
                peColField(line, "src_pe"),
                peColField(line, "dst_pe"),
                stringField(line, "color"),
              )
            case "expect" =>
              Expect(BigInt(intField(line, "expected")))
            case other =>
              throw new IllegalArgumentException(s"unknown vector dot IR op: $other")
          }
        }
        .toSeq
    } finally {
      source.close()
    }
  }

  private def idleInputs(dut: VectorDotEngine): Unit = {
    dut.io.leftIn.valid.poke(false.B)
    dut.io.rightIn.valid.poke(false.B)
    dut.io.start.valid.poke(false.B)
    dut.io.result.ready.poke(false.B)
  }

  private def writeLeftWords(dut: VectorDotEngine, words: Seq[BigInt]): Unit = {
    for (word <- words) {
      dut.io.leftIn.valid.poke(true.B)
      dut.io.leftIn.bits.poke(word.U)
      while (!dut.io.leftIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.leftIn.valid.poke(false.B)
  }

  private def writeRightWords(dut: VectorDotEngine, words: Seq[BigInt]): Unit = {
    for (word <- words) {
      dut.io.rightIn.valid.poke(true.B)
      dut.io.rightIn.bits.poke(word.U)
      while (!dut.io.rightIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.rightIn.valid.poke(false.B)
  }

  private def startCompute(dut: VectorDotEngine, wordCount: Int): BigInt = {
    dut.io.start.valid.poke(true.B)
    dut.io.start.bits.poke(wordCount.U)
    while (!dut.io.start.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.start.valid.poke(false.B)

    dut.io.result.ready.poke(true.B)
    while (!dut.io.result.valid.peekBoolean()) {
      dut.clock.step()
    }
    val result = dut.io.result.bits.peek().litValue
    dut.clock.step()
    dut.io.result.ready.poke(false.B)
    result
  }

  private def idleRowInputs(dut: VectorDotRowTop): Unit = {
    dut.io.peSel.poke(0.U)
    dut.io.leftIn.valid.poke(false.B)
    dut.io.rightIn.valid.poke(false.B)
    dut.io.start.valid.poke(false.B)
    dut.io.result.ready.poke(false.B)
  }

  private def writeRowLeftWords(dut: VectorDotRowTop, pe: Int, words: Seq[BigInt]): Unit = {
    dut.io.peSel.poke(pe.U)
    for (word <- words) {
      dut.io.leftIn.valid.poke(true.B)
      dut.io.leftIn.bits.poke(word.U)
      while (!dut.io.leftIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.leftIn.valid.poke(false.B)
  }

  private def writeRowRightWords(dut: VectorDotRowTop, pe: Int, words: Seq[BigInt]): Unit = {
    dut.io.peSel.poke(pe.U)
    for (word <- words) {
      dut.io.rightIn.valid.poke(true.B)
      dut.io.rightIn.bits.poke(word.U)
      while (!dut.io.rightIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.rightIn.valid.poke(false.B)
  }

  private def startRowCompute(dut: VectorDotRowTop, wordCount: Int): BigInt = {
    dut.io.start.valid.poke(true.B)
    dut.io.start.bits.poke(wordCount.U)
    while (!dut.io.start.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.start.valid.poke(false.B)

    dut.io.result.ready.poke(true.B)
    while (!dut.io.result.valid.peekBoolean()) {
      dut.clock.step()
    }
    val result = dut.io.result.bits.peek().litValue
    dut.clock.step()
    dut.io.result.ready.poke(false.B)
    result
  }

  it should "compute a local vector dot product" in {
    test(new VectorDotEngine(maxWords = 64, dataBits = 32, accBits = 64)) { dut =>
      idleInputs(dut)
      writeLeftWords(dut, Seq(BigInt(2), BigInt(3), BigInt(4)))
      writeRightWords(dut, Seq(BigInt(10), BigInt(20), BigInt(30)))

      startCompute(dut, 3) shouldBe BigInt(200)
    }
  }

  it should "replay the SimPy multi-PE vector dot command IR" in {
    test(new VectorDotRowTop(peCount = 4, maxWordsPerPe = 1024, dataBits = 32, accBits = 64)) { dut =>
      idleRowInputs(dut)
      var lastResult = BigInt(0)
      var rowStarted = false
      var rowWordCount = 0
      for (step <- parseVectorDotIr()) {
        step match {
          case LoadLeft(pe, words) =>
            writeRowLeftWords(dut, pe, words)
          case LoadRight(pe, words) =>
            writeRowRightWords(dut, pe, words)
          case ComputeDot(_, wordCount) =>
            if (!rowStarted) {
              rowWordCount = wordCount
              lastResult = startRowCompute(dut, wordCount)
              rowStarted = true
            } else {
              wordCount shouldBe rowWordCount
            }
          case ReduceEast(srcPe, dstPe, color) =>
            color shouldBe "partial_sum_color"
            dstPe shouldBe (srcPe + 1)
          case Expect(expected) =>
            lastResult shouldBe expected
        }
      }
    }
  }
}
