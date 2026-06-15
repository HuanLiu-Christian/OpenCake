package wafersim.rtl

import chisel3._
import chiseltest._
import java.nio.file.Paths
import scala.io.Source
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PeMemcpyEngineTest extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  behavior of "PeMemcpyEngine"

  private sealed trait TraceStep
  private case class H2d(dstAddr: Int, words: Seq[BigInt]) extends TraceStep
  private case class Copy(srcAddr: Int, dstAddr: Int, wordCount: Int) extends TraceStep
  private case class D2h(srcAddr: Int, expected: Seq[BigInt]) extends TraceStep
  private case class Expect(addr: Int, expected: BigInt) extends TraceStep

  private def sendCmd(
      dut: PeMemcpyEngine,
      op: UInt,
      srcAddr: Int,
      dstAddr: Int,
      bytes: Int
  ): Unit = {
    dut.io.cmd.valid.poke(true.B)
    dut.io.cmd.bits.op.poke(op)
    dut.io.cmd.bits.srcAddr.poke(srcAddr.U)
    dut.io.cmd.bits.dstAddr.poke(dstAddr.U)
    dut.io.cmd.bits.bytes.poke(bytes.U)
    dut.io.dataIn.valid.poke(false.B)
    dut.io.dataOut.ready.poke(false.B)
    dut.clock.step()
    dut.io.cmd.valid.poke(false.B)
  }

  private def waitDone(dut: PeMemcpyEngine): Unit = {
    while (!dut.io.done.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
  }

  private def writeWords(dut: PeMemcpyEngine, words: Seq[BigInt]): Unit = {
    for (word <- words) {
      dut.io.dataIn.valid.poke(true.B)
      dut.io.dataIn.bits.poke(word.U)
      while (!dut.io.dataIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.dataIn.valid.poke(false.B)
  }

  private def readDebugWord(dut: PeMemcpyEngine, addr: Int): BigInt = {
    dut.io.debug.readAddr.poke(addr.U)
    dut.clock.step()
    dut.io.debug.readData.peek().litValue
  }

  private def readHostWords(dut: PeMemcpyEngine): Seq[BigInt] = {
    val output = scala.collection.mutable.ArrayBuffer.empty[BigInt]
    dut.io.dataOut.ready.poke(true.B)
    while (!dut.io.done.peekBoolean()) {
      if (dut.io.dataOut.valid.peekBoolean()) {
        output += dut.io.dataOut.bits.peek().litValue
      }
      dut.clock.step()
    }
    dut.io.dataOut.ready.poke(false.B)
    dut.clock.step()
    output.toSeq
  }

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

  private def parseCommandIr(): Seq[TraceStep] = {
    val tracePath = sys.props
      .get("wafersim.memcpy.ir")
      .orElse(sys.env.get("WAFERSIM_MEMCPY_IR"))
      .map(Paths.get(_))
      .getOrElse(Paths.get("..", "tests", "fixtures", "pe_local_memcpy.jsonl"))
    val source = Source.fromFile(tracePath.toFile)
    try {
      source
        .getLines()
        .map(_.trim)
        .filter(line => line.nonEmpty && !line.startsWith("#"))
        .map { line =>
          stringField(line, "op") match {
            case "h2d_write" =>
              H2d(intField(line, "dst_addr"), wordsField(line))
            case "local_copy" =>
              Copy(
                intField(line, "src_addr"),
                intField(line, "dst_addr"),
                intField(line, "word_count"),
              )
            case "d2h_read" =>
              D2h(intField(line, "src_addr"), wordsField(line))
            case "expect" =>
              Expect(intField(line, "src_addr"), wordsField(line).head)
            case other =>
              throw new IllegalArgumentException(s"unknown memcpy IR op: $other")
          }
        }
        .toSeq
    } finally {
      source.close()
    }
  }

  it should "write a host stream into PE-local SRAM" in {
    test(new PeMemcpyEngine(depthWords = 64, dataBits = 32)) { dut =>
      val words = Seq(BigInt(0x11), BigInt(0x22), BigInt(0x33), BigInt(0x44))

      sendCmd(dut, MemcpyOpcode.h2dWrite, srcAddr = 0, dstAddr = 8, bytes = 16)
      writeWords(dut, words)
      waitDone(dut)

      readDebugWord(dut, 8) shouldBe BigInt(0x11)
      readDebugWord(dut, 9) shouldBe BigInt(0x22)
      readDebugWord(dut, 10) shouldBe BigInt(0x33)
      readDebugWord(dut, 11) shouldBe BigInt(0x44)
    }
  }

  it should "copy data inside PE-local SRAM" in {
    test(new PeMemcpyEngine(depthWords = 64, dataBits = 32)) { dut =>
      val words = Seq(BigInt(7), BigInt(8), BigInt(9))

      sendCmd(dut, MemcpyOpcode.h2dWrite, srcAddr = 0, dstAddr = 4, bytes = 12)
      writeWords(dut, words)
      waitDone(dut)

      sendCmd(dut, MemcpyOpcode.localCopy, srcAddr = 4, dstAddr = 20, bytes = 12)
      waitDone(dut)

      readDebugWord(dut, 20) shouldBe BigInt(7)
      readDebugWord(dut, 21) shouldBe BigInt(8)
      readDebugWord(dut, 22) shouldBe BigInt(9)
    }
  }

  it should "read PE-local SRAM back to the host stream" in {
    test(new PeMemcpyEngine(depthWords = 64, dataBits = 32)) { dut =>
      val words = Seq(BigInt(101), BigInt(202), BigInt(303), BigInt(404))

      sendCmd(dut, MemcpyOpcode.h2dWrite, srcAddr = 0, dstAddr = 12, bytes = 16)
      writeWords(dut, words)
      waitDone(dut)

      sendCmd(dut, MemcpyOpcode.d2hRead, srcAddr = 12, dstAddr = 0, bytes = 16)
      readHostWords(dut) shouldBe words
    }
  }

  it should "replay the SimPy PE-local memcpy command IR" in {
    test(new PeMemcpyEngine(depthWords = 64, dataBits = 32)) { dut =>
      for (step <- parseCommandIr()) {
        step match {
          case H2d(dstAddr, words) =>
            sendCmd(
              dut,
              MemcpyOpcode.h2dWrite,
              srcAddr = 0,
              dstAddr = dstAddr,
              bytes = words.length * 4,
            )
            writeWords(dut, words)
            waitDone(dut)
          case Copy(srcAddr, dstAddr, wordCount) =>
            sendCmd(
              dut,
              MemcpyOpcode.localCopy,
              srcAddr = srcAddr,
              dstAddr = dstAddr,
              bytes = wordCount * 4,
            )
            waitDone(dut)
          case D2h(srcAddr, expected) =>
            sendCmd(
              dut,
              MemcpyOpcode.d2hRead,
              srcAddr = srcAddr,
              dstAddr = 0,
              bytes = expected.length * 4,
            )
            readHostWords(dut) shouldBe expected
          case Expect(addr, expected) =>
            readDebugWord(dut, addr) shouldBe expected
        }
      }
    }
  }
}
