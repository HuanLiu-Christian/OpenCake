package wafersim.rtl

import chisel3._
import chiseltest._
import java.nio.file.Paths
import scala.io.Source
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PeToPeMemcpyTest extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  behavior of "TwoPeMemcpyTop"

  private sealed trait IrStep
  private case class H2d(pe: Int, dstAddr: Int, words: Seq[BigInt]) extends IrStep
  private case class P2p(
      srcPe: Int,
      direction: Int,
      srcAddr: Int,
      dstAddr: Int,
      wordCount: Int
  ) extends IrStep
  private case class D2h(pe: Int, srcAddr: Int, expected: Seq[BigInt]) extends IrStep
  private case class Expect(pe: Int, addr: Int, expected: BigInt) extends IrStep

  private def intField(line: String, key: String, default: Int = 0): Int = {
    val pattern = ("\"" + key + "\"\\s*:\\s*(-?\\d+)").r
    pattern.findFirstMatchIn(line).map(_.group(1).toInt).getOrElse(default)
  }

  private def stringField(line: String, key: String): String = {
    val pattern = ("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"").r
    pattern
      .findFirstMatchIn(line)
      .map(_.group(1))
      .getOrElse(throw new IllegalArgumentException(s"missing string field: $key"))
  }

  private val colorDirections = Map(
    "north_memcpy" -> 0,
    "south_memcpy" -> 1,
    "west_memcpy" -> 2,
    "east_memcpy" -> 3,
  )

  private def directionFromColor(line: String): Int = {
    val color = stringField(line, "color")
    colorDirections.getOrElse(
      color,
      throw new IllegalArgumentException(s"unknown memcpy color route: $color"),
    )
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

  private def peField(line: String, key: String, default: Int = 0): Int = {
    val pattern = ("\"" + key + "\"\\s*:\\s*\\[(\\d+)\\s*,\\s*(\\d+)\\]").r
    pattern.findFirstMatchIn(line) match {
      case Some(matched) =>
        (matched.group(1).toInt, matched.group(2).toInt) match {
          case (0, 0) => 0
          case (0, 1) => 1
          case coord =>
            throw new IllegalArgumentException(s"unsupported two-PE coord: $coord")
        }
      case None =>
        default
    }
  }

  private def parseP2pIr(): Seq[IrStep] = {
    val tracePath = sys.props
      .get("wafersim.p2p.memcpy.ir")
      .orElse(sys.env.get("WAFERSIM_P2P_MEMCPY_IR"))
      .map(Paths.get(_))
      .getOrElse(Paths.get("..", "tests", "fixtures", "pe_to_pe_memcpy.jsonl"))
    val source = Source.fromFile(tracePath.toFile)
    try {
      source
        .getLines()
        .map(_.trim)
        .filter(line => line.nonEmpty && !line.startsWith("#"))
        .map { line =>
          stringField(line, "op") match {
            case "h2d_write" =>
              H2d(peField(line, "pe"), intField(line, "dst_addr"), wordsField(line))
            case "p2p_copy" =>
              P2p(
                peField(line, "src_pe"),
                directionFromColor(line),
                intField(line, "src_addr"),
                intField(line, "dst_addr"),
                intField(line, "word_count"),
              )
            case "d2h_read" =>
              D2h(peField(line, "pe"), intField(line, "src_addr"), wordsField(line))
            case "expect" =>
              Expect(peField(line, "pe"), intField(line, "src_addr"), wordsField(line).head)
            case other =>
              throw new IllegalArgumentException(s"unknown p2p memcpy IR op: $other")
          }
        }
        .toSeq
    } finally {
      source.close()
    }
  }

  private def idleInputs(dut: TwoPeMemcpyTop): Unit = {
    dut.io.pe0HostCmd.valid.poke(false.B)
    dut.io.pe1HostCmd.valid.poke(false.B)
    dut.io.pe0HostDataIn.valid.poke(false.B)
    dut.io.pe1HostDataIn.valid.poke(false.B)
    dut.io.pe0HostDataOut.ready.poke(false.B)
    dut.io.pe1HostDataOut.ready.poke(false.B)
    dut.io.p2pCmd.valid.poke(false.B)
  }

  private def sendHostCmd(
      dut: TwoPeMemcpyTop,
      pe: Int,
      op: UInt,
      srcAddr: Int,
      dstAddr: Int,
      bytes: Int
  ): Unit = {
    val cmd = if (pe == 0) dut.io.pe0HostCmd else dut.io.pe1HostCmd
    cmd.valid.poke(true.B)
    cmd.bits.op.poke(op)
    cmd.bits.srcAddr.poke(srcAddr.U)
    cmd.bits.dstAddr.poke(dstAddr.U)
    cmd.bits.bytes.poke(bytes.U)
    while (!cmd.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    cmd.valid.poke(false.B)
  }

  private def waitHostDone(dut: TwoPeMemcpyTop, pe: Int): Unit = {
    val done = if (pe == 0) dut.io.pe0Done else dut.io.pe1Done
    while (!done.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
  }

  private def writeHostWords(dut: TwoPeMemcpyTop, pe: Int, words: Seq[BigInt]): Unit = {
    val dataIn = if (pe == 0) dut.io.pe0HostDataIn else dut.io.pe1HostDataIn
    for (word <- words) {
      dataIn.valid.poke(true.B)
      dataIn.bits.poke(word.U)
      while (!dataIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dataIn.valid.poke(false.B)
  }

  private def readHostWords(dut: TwoPeMemcpyTop, pe: Int): Seq[BigInt] = {
    val dataOut = if (pe == 0) dut.io.pe0HostDataOut else dut.io.pe1HostDataOut
    val done = if (pe == 0) dut.io.pe0Done else dut.io.pe1Done
    val output = scala.collection.mutable.ArrayBuffer.empty[BigInt]
    dataOut.ready.poke(true.B)
    while (!done.peekBoolean()) {
      if (dataOut.valid.peekBoolean()) {
        output += dataOut.bits.peek().litValue
      }
      dut.clock.step()
    }
    dataOut.ready.poke(false.B)
    dut.clock.step()
    output.toSeq
  }

  private def sendP2pCmd(
      dut: TwoPeMemcpyTop,
      srcPe: Int,
      direction: Int,
      srcAddr: Int,
      dstAddr: Int,
      bytes: Int
  ): Unit = {
    dut.io.p2pCmd.valid.poke(true.B)
    dut.io.p2pCmd.bits.srcPe.poke(srcPe.U)
    dut.io.p2pCmd.bits.routeDir.poke(direction.U)
    dut.io.p2pCmd.bits.srcAddr.poke(srcAddr.U)
    dut.io.p2pCmd.bits.dstAddr.poke(dstAddr.U)
    dut.io.p2pCmd.bits.bytes.poke(bytes.U)
    while (!dut.io.p2pCmd.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.p2pCmd.valid.poke(false.B)
  }

  private def waitP2pDone(dut: TwoPeMemcpyTop): Unit = {
    while (!dut.io.p2pDone.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
  }

  private def readDebugWord(dut: TwoPeMemcpyTop, pe: Int, addr: Int): BigInt = {
    dut.io.debug.pe.poke(pe.U)
    dut.io.debug.readAddr.poke(addr.U)
    dut.clock.step()
    dut.io.debug.readData.peek().litValue
  }

  it should "replay the SimPy PE-to-PE memcpy command IR" in {
    test(new TwoPeMemcpyTop(depthWords = 64, dataBits = 32)) { dut =>
      idleInputs(dut)
      for (step <- parseP2pIr()) {
        step match {
          case H2d(pe, dstAddr, words) =>
            sendHostCmd(
              dut,
              pe,
              MemcpyOpcode.h2dWrite,
              srcAddr = 0,
              dstAddr = dstAddr,
              bytes = words.length * 4,
            )
            writeHostWords(dut, pe, words)
            waitHostDone(dut, pe)
          case P2p(srcPe, direction, srcAddr, dstAddr, wordCount) =>
            sendP2pCmd(
              dut,
              srcPe,
              direction,
              srcAddr = srcAddr,
              dstAddr = dstAddr,
              bytes = wordCount * 4,
            )
            waitP2pDone(dut)
          case D2h(pe, srcAddr, expected) =>
            sendHostCmd(
              dut,
              pe,
              MemcpyOpcode.d2hRead,
              srcAddr = srcAddr,
              dstAddr = 0,
              bytes = expected.length * 4,
            )
            readHostWords(dut, pe) shouldBe expected
          case Expect(pe, addr, expected) =>
            readDebugWord(dut, pe, addr) shouldBe expected
        }
      }
    }
  }
}
