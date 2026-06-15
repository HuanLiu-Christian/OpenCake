package wafersim.rtl

import chisel3._
import chisel3.util._

class TwoPeMemcpyTop(
    depthWords: Int = 1024,
    dataBits: Int = 32,
    lenBits: Int = 32
) extends Module {
  require(depthWords > 0, "depthWords must be positive")

  private val addrBits = log2Ceil(depthWords)
  private val peBits = 1

  val io = IO(new Bundle {
    val pe0HostCmd = Flipped(Decoupled(new MemcpyCmd(addrBits, lenBits)))
    val pe0HostDataIn = Flipped(Decoupled(UInt(dataBits.W)))
    val pe0HostDataOut = Decoupled(UInt(dataBits.W))
    val pe0Done = Output(Bool())

    val pe1HostCmd = Flipped(Decoupled(new MemcpyCmd(addrBits, lenBits)))
    val pe1HostDataIn = Flipped(Decoupled(UInt(dataBits.W)))
    val pe1HostDataOut = Decoupled(UInt(dataBits.W))
    val pe1Done = Output(Bool())

    val p2pCmd = Flipped(Decoupled(new PeToPeMemcpyCmd(peBits, addrBits, lenBits)))
    val p2pDone = Output(Bool())
    val p2pBusy = Output(Bool())

    val debug = new Bundle {
      val pe = Input(UInt(peBits.W))
      val readAddr = Input(UInt(addrBits.W))
      val readData = Output(UInt(dataBits.W))
    }
  })

  private val pe0 = Module(new PeMemcpyEngine(depthWords, dataBits, lenBits))
  private val pe1 = Module(new PeMemcpyEngine(depthWords, dataBits, lenBits))

  private val (sIdle :: sIssue :: sStream :: sDone :: Nil) = Enum(4)
  private val state = RegInit(sIdle)
  private val srcPeReg = RegInit(0.U(peBits.W))
  private val dstPeReg = RegInit(0.U(peBits.W))
  private val srcAddrReg = RegInit(0.U(addrBits.W))
  private val dstAddrReg = RegInit(0.U(addrBits.W))
  private val bytesReg = RegInit(0.U(lenBits.W))
  private val readIssued = RegInit(false.B)
  private val writeIssued = RegInit(false.B)
  private val srcDoneSeen = RegInit(false.B)
  private val dstDoneSeen = RegInit(false.B)

  private val p2pActive = state =/= sIdle
  private val srcIsPe0 = srcPeReg === 0.U
  private val dstIsPe0 = dstPeReg === 0.U

  private val pe0CmdBits = Wire(new MemcpyCmd(addrBits, lenBits))
  private val pe1CmdBits = Wire(new MemcpyCmd(addrBits, lenBits))
  pe0CmdBits := io.pe0HostCmd.bits
  pe1CmdBits := io.pe1HostCmd.bits
  private val pe0CmdValid = WireDefault(io.pe0HostCmd.valid && !p2pActive)
  private val pe1CmdValid = WireDefault(io.pe1HostCmd.valid && !p2pActive)

  private val pe0DataInBits = WireDefault(io.pe0HostDataIn.bits)
  private val pe1DataInBits = WireDefault(io.pe1HostDataIn.bits)
  private val pe0DataInValid = WireDefault(io.pe0HostDataIn.valid && !p2pActive)
  private val pe1DataInValid = WireDefault(io.pe1HostDataIn.valid && !p2pActive)
  private val pe0DataOutReady = WireDefault(io.pe0HostDataOut.ready && !p2pActive)
  private val pe1DataOutReady = WireDefault(io.pe1HostDataOut.ready && !p2pActive)

  private val readCmdBits = Wire(new MemcpyCmd(addrBits, lenBits))
  readCmdBits.op := MemcpyOpcode.d2hRead
  readCmdBits.srcAddr := srcAddrReg
  readCmdBits.dstAddr := 0.U
  readCmdBits.bytes := bytesReg

  private val writeCmdBits = Wire(new MemcpyCmd(addrBits, lenBits))
  writeCmdBits.op := MemcpyOpcode.h2dWrite
  writeCmdBits.srcAddr := 0.U
  writeCmdBits.dstAddr := dstAddrReg
  writeCmdBits.bytes := bytesReg

  private val issueRead = state === sIssue && !readIssued
  private val issueWrite = state === sIssue && !writeIssued
  private val readReady = Mux(srcIsPe0, pe0.io.cmd.ready, pe1.io.cmd.ready)
  private val writeReady = Mux(dstIsPe0, pe0.io.cmd.ready, pe1.io.cmd.ready)
  private val readFire = issueRead && readReady
  private val writeFire = issueWrite && writeReady

  when(issueRead && srcIsPe0) {
    pe0CmdValid := true.B
    pe0CmdBits := readCmdBits
  }
  when(issueRead && !srcIsPe0) {
    pe1CmdValid := true.B
    pe1CmdBits := readCmdBits
  }
  when(issueWrite && dstIsPe0) {
    pe0CmdValid := true.B
    pe0CmdBits := writeCmdBits
  }
  when(issueWrite && !dstIsPe0) {
    pe1CmdValid := true.B
    pe1CmdBits := writeCmdBits
  }

  when(state === sStream && srcIsPe0 && !dstIsPe0) {
    pe0DataOutReady := pe1.io.dataIn.ready
    pe1DataInValid := pe0.io.dataOut.valid
    pe1DataInBits := pe0.io.dataOut.bits
  }
  when(state === sStream && !srcIsPe0 && dstIsPe0) {
    pe1DataOutReady := pe0.io.dataIn.ready
    pe0DataInValid := pe1.io.dataOut.valid
    pe0DataInBits := pe1.io.dataOut.bits
  }

  pe0.io.cmd.valid := pe0CmdValid
  pe0.io.cmd.bits := pe0CmdBits
  pe0.io.dataIn.valid := pe0DataInValid
  pe0.io.dataIn.bits := pe0DataInBits
  pe0.io.dataOut.ready := pe0DataOutReady

  pe1.io.cmd.valid := pe1CmdValid
  pe1.io.cmd.bits := pe1CmdBits
  pe1.io.dataIn.valid := pe1DataInValid
  pe1.io.dataIn.bits := pe1DataInBits
  pe1.io.dataOut.ready := pe1DataOutReady

  io.pe0HostCmd.ready := pe0.io.cmd.ready && !p2pActive
  io.pe1HostCmd.ready := pe1.io.cmd.ready && !p2pActive
  io.pe0HostDataIn.ready := pe0.io.dataIn.ready && !p2pActive
  io.pe1HostDataIn.ready := pe1.io.dataIn.ready && !p2pActive
  io.pe0HostDataOut.valid := pe0.io.dataOut.valid && !p2pActive
  io.pe0HostDataOut.bits := pe0.io.dataOut.bits
  io.pe1HostDataOut.valid := pe1.io.dataOut.valid && !p2pActive
  io.pe1HostDataOut.bits := pe1.io.dataOut.bits
  io.pe0Done := pe0.io.done
  io.pe1Done := pe1.io.done

  pe0.io.debug.readAddr := io.debug.readAddr
  pe1.io.debug.readAddr := io.debug.readAddr
  io.debug.readData := Mux(io.debug.pe === 0.U, pe0.io.debug.readData, pe1.io.debug.readData)

  io.p2pCmd.ready := state === sIdle
  io.p2pDone := state === sDone
  io.p2pBusy := p2pActive

  private val srcDone = Mux(srcIsPe0, pe0.io.done, pe1.io.done)
  private val dstDone = Mux(dstIsPe0, pe0.io.done, pe1.io.done)
  private val srcComplete = srcDoneSeen || srcDone
  private val dstComplete = dstDoneSeen || dstDone

  when(io.p2pCmd.fire) {
    srcPeReg := io.p2pCmd.bits.srcPe
    dstPeReg := io.p2pCmd.bits.dstPe
    srcAddrReg := io.p2pCmd.bits.srcAddr
    dstAddrReg := io.p2pCmd.bits.dstAddr
    bytesReg := io.p2pCmd.bits.bytes
    readIssued := false.B
    writeIssued := false.B
    srcDoneSeen := false.B
    dstDoneSeen := false.B
    state := sIssue
  }

  when(state === sIssue) {
    when(readFire) {
      readIssued := true.B
    }
    when(writeFire) {
      writeIssued := true.B
    }
    when((readIssued || readFire) && (writeIssued || writeFire)) {
      state := sStream
    }
  }

  when(state === sStream) {
    when(srcDone) {
      srcDoneSeen := true.B
    }
    when(dstDone) {
      dstDoneSeen := true.B
    }
    when(srcComplete && dstComplete) {
      state := sDone
    }
  }

  when(state === sDone) {
    state := sIdle
  }
}
