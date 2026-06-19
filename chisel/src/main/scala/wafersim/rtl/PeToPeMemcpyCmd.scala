package wafersim.rtl

import chisel3._

object MemcpyDirection {
  val north = 0.U(2.W)
  val south = 1.U(2.W)
  val west = 2.U(2.W)
  val east = 3.U(2.W)
}

class PeToPeMemcpyCmd(peBits: Int, addrBits: Int, lenBits: Int) extends Bundle {
  val srcPe = UInt(peBits.W)
  val routeDir = UInt(2.W)
  val srcAddr = UInt(addrBits.W)
  val dstAddr = UInt(addrBits.W)
  val bytes = UInt(lenBits.W)
}
