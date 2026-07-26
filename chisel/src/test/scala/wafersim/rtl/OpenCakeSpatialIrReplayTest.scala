package wafersim.rtl

import chisel3._
import chiseltest._
import java.nio.file.{Path, Paths}
import scala.io.Source
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class OpenCakeSpatialIrReplayTest extends AnyFlatSpec with ChiselScalatestTester with Matchers {
  behavior of "OpenCakeGemmArrayTop"

  private case class Config(
      rankRows: Int,
      rankCols: Int,
      tileM: Int,
      tileN: Int,
      tileK: Int,
      m: Int,
      n: Int,
      k: Int
  )

  private sealed trait SpatialStep
  private case class SendStep(
      tensor: String,
      color: String,
      kStart: Int,
      route: Seq[(Int, Int)]
  ) extends SpatialStep
  private case class ComputeStep(rank: (Int, Int), kStart: Int, k: Int) extends SpatialStep

  private def intField(text: String, key: String): Int = {
    val pattern = ("\"" + key + "\"\\s*:\\s*(-?\\d+)").r
    pattern
      .findFirstMatchIn(text)
      .map(_.group(1).toInt)
      .getOrElse(throw new IllegalArgumentException(s"missing int field: $key"))
  }

  private def stringField(line: String, key: String): String = {
    val pattern = ("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"").r
    pattern
      .findFirstMatchIn(line)
      .map(_.group(1))
      .getOrElse(throw new IllegalArgumentException(s"missing string field: $key"))
  }

  private def coordField(line: String, key: String): (Int, Int) = {
    val pattern = ("\"" + key + "\"\\s*:\\s*\\[(\\d+)\\s*,\\s*(\\d+)\\]").r
    pattern.findFirstMatchIn(line) match {
      case Some(matched) => (matched.group(1).toInt, matched.group(2).toInt)
      case None => throw new IllegalArgumentException(s"missing coordinate field: $key")
    }
  }

  private def routeField(line: String): Seq[(Int, Int)] = {
    val start = line.indexOf("\"route\"")
    if (start < 0) {
      throw new IllegalArgumentException("missing route field")
    }
    val bodyStart = line.indexOf("[[", start)
    val bodyEnd = line.indexOf("]]", bodyStart)
    if (bodyStart < 0 || bodyEnd < 0) {
      throw new IllegalArgumentException("malformed route field")
    }
    val body = line.substring(bodyStart + 2, bodyEnd)
    body
      .split("\\],\\s*\\[")
      .toSeq
      .map { pair =>
        val values = pair.split(",").map(_.trim.toInt)
        (values(0), values(1))
      }
  }

  private def packageRoot(): Path = {
    sys.props
      .get("wafersim.opencake.chisel.package")
      .orElse(sys.env.get("WAFERSIM_OPENCAKE_CHISEL_PACKAGE"))
      .map(Paths.get(_))
      .getOrElse(Paths.get("..", "tests", "fixtures", "opencake_chisel_gemm"))
  }

  private def readConfig(root: Path): Config = {
    val text = Source.fromFile(root.resolve("chisel_config.json").toFile).mkString
    Config(
      rankRows = intField(text, "rank_rows"),
      rankCols = intField(text, "rank_cols"),
      tileM = intField(text, "tile_m"),
      tileN = intField(text, "tile_n"),
      tileK = intField(text, "k_tile"),
      m = intField(text, "m"),
      n = intField(text, "n"),
      k = intField(text, "k")
    )
  }

  private def readSpatialSteps(root: Path): Seq[SpatialStep] = {
    val source = Source.fromFile(root.resolve("opencake_spatial_ir.jsonl").toFile)
    try {
      source
        .getLines()
        .map(_.trim)
        .filter(_.nonEmpty)
        .flatMap { line =>
          stringField(line, "op") match {
            case "send" =>
              Some(
                SendStep(
                  tensor = stringField(line, "tensor"),
                  color = stringField(line, "color"),
                  kStart = intField(line, "k_start"),
                  route = routeField(line)
                )
              )
            case "compute" =>
              Some(ComputeStep(coordField(line, "rank"), intField(line, "k_start"), intField(line, "k")))
            case "define_color" | "alloc" =>
              None
            case other =>
              throw new IllegalArgumentException(s"unknown Spatial IR op: $other")
          }
        }
        .toSeq
    } finally {
      source.close()
    }
  }

  private def peIndex(config: Config, pe: (Int, Int)): Int = pe._1 * config.rankCols + pe._2

  private def idleInputs(dut: OpenCakeGemmArrayTop): Unit = {
    dut.io.peSel.poke(0.U)
    dut.io.aIn.valid.poke(false.B)
    dut.io.bIn.valid.poke(false.B)
    dut.io.forwardA.valid.poke(false.B)
    dut.io.forwardB.valid.poke(false.B)
    dut.io.start.valid.poke(false.B)
    dut.io.result.ready.poke(false.B)
  }

  private def writeA(dut: OpenCakeGemmArrayTop, config: Config, pe: (Int, Int), kStart: Int): Unit = {
    dut.io.peSel.poke(peIndex(config, pe).U)
    val rowBase = pe._1 * config.tileM
    for (localRow <- 0 until config.tileM; localK <- 0 until config.tileK) {
      val value = BigInt((rowBase + localRow) * config.k + kStart + localK)
      dut.io.aIn.valid.poke(true.B)
      dut.io.aIn.bits.poke(value.U)
      while (!dut.io.aIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.aIn.valid.poke(false.B)
  }

  private def writeB(dut: OpenCakeGemmArrayTop, config: Config, pe: (Int, Int)): Unit = {
    dut.io.peSel.poke(peIndex(config, pe).U)
    for (_ <- 0 until config.tileK * config.tileN) {
      dut.io.bIn.valid.poke(true.B)
      dut.io.bIn.bits.poke(1.U)
      while (!dut.io.bIn.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
    }
    dut.io.bIn.valid.poke(false.B)
  }

  private def forwardA(dut: OpenCakeGemmArrayTop, src: (Int, Int)): Unit = {
    dut.io.forwardA.valid.poke(true.B)
    dut.io.forwardA.bits.row.poke(src._1.U)
    dut.io.forwardA.bits.col.poke(src._2.U)
    while (!dut.io.forwardA.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.forwardA.valid.poke(false.B)
  }

  private def forwardB(dut: OpenCakeGemmArrayTop, src: (Int, Int)): Unit = {
    dut.io.forwardB.valid.poke(true.B)
    dut.io.forwardB.bits.row.poke(src._1.U)
    dut.io.forwardB.bits.col.poke(src._2.U)
    while (!dut.io.forwardB.ready.peekBoolean()) {
      dut.clock.step()
    }
    dut.clock.step()
    dut.io.forwardB.valid.poke(false.B)
  }

  private def startChunk(dut: OpenCakeGemmArrayTop, clearAccum: Boolean): Seq[BigInt] = {
    dut.io.start.valid.poke(true.B)
    dut.io.start.bits.poke(clearAccum.B)
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

  private def expectedC(config: Config): Seq[BigInt] = {
    val values = for {
      rankRow <- 0 until config.rankRows
      rankCol <- 0 until config.rankCols
      localRow <- 0 until config.tileM
      _ <- 0 until config.tileN
    } yield {
      val globalRow = rankRow * config.tileM + localRow
      (0 until config.k).map(kIndex => BigInt(globalRow * config.k + kIndex)).sum
    }
    values.toSeq
  }

  private def replayPackage(root: Path): Unit = {
    val config = readConfig(root)
    val steps = readSpatialSteps(root)
    val computeByK = steps.collect { case compute: ComputeStep => compute }.groupBy(_.kStart)
    val sendsByK = steps.collect { case send: SendStep => send }.groupBy(_.kStart)
    val kStarts = computeByK.keys.toSeq.sorted

    test(
      new OpenCakeGemmArrayTop(
        rankRows = config.rankRows,
        rankCols = config.rankCols,
        tileM = config.tileM,
        tileN = config.tileN,
        tileK = config.tileK
      )
    ) { dut =>
      idleInputs(dut)
      var finalResult = Seq.empty[BigInt]

      for ((kStart, chunkIndex) <- kStarts.zipWithIndex) {
        for (row <- 0 until config.rankRows) {
          writeA(dut, config, (row, 0), kStart)
        }
        for (col <- 0 until config.rankCols) {
          writeB(dut, config, (0, col))
        }

        for (send <- sendsByK.getOrElse(kStart, Seq.empty)) {
          for ((src, dst) <- send.route.zip(send.route.drop(1))) {
            send.tensor match {
              case "A" =>
                send.color shouldBe "a_east"
                dst shouldBe (src._1, src._2 + 1)
                forwardA(dut, src)
              case "B" =>
                send.color shouldBe "b_south"
                dst shouldBe (src._1 + 1, src._2)
                forwardB(dut, src)
              case other =>
                throw new IllegalArgumentException(s"unexpected tensor: $other")
            }
          }
        }

        computeByK(kStart).size shouldBe config.rankRows * config.rankCols
        finalResult = startChunk(dut, clearAccum = chunkIndex == 0)
      }

      finalResult shouldBe expectedC(config)
    }
  }

  it should "replay the small GEMM OpenCake Spatial IR fixture" in {
    replayPackage(packageRoot())
  }
}
