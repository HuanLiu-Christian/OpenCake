package wafersim.rtl

import chisel3._

class PeToPeMemcpyCmd(peBits: Int, addrBits: Int, lenBits: Int) extends Bundle {
  val srcPe = UInt(peBits.W)
  val dstPe = UInt(peBits.W)
  val srcAddr = UInt(addrBits.W)
  val dstAddr = UInt(addrBits.W)
  val bytes = UInt(lenBits.W)
}
