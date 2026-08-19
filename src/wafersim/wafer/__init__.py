"""Wafer-scale packet-network and analytical GEMM simulation."""

from .simulator import WaferGemmResult, simulate_gemm

__all__ = ["WaferGemmResult", "simulate_gemm"]
