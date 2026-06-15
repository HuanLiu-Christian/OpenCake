package wafersim.rtl

import chisel3._

object MemcpyOpcode {
  val h2dWrite: UInt = 0.U(2.W)
  val d2hRead: UInt = 1.U(2.W)
  val localCopy: UInt = 2.U(2.W)
}

class MemcpyCmd(addrBits: Int, lenBits: Int) extends Bundle {
  val op = UInt(2.W)
  val srcAddr = UInt(addrBits.W)
  val dstAddr = UInt(addrBits.W)
  val bytes = UInt(lenBits.W)
}

