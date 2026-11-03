#!/usr/bin/env python3
"""One CS-3's share of Llama-3.3-70B, shrunk to 9,000 PEs x 48 KiB: 20 layers, hidden 704, weights AND KV in SRAM, batch 1.

WHY. Cerebras serves Llama-70B by splitting the 80 layers at layer boundaries across (at least) 4 CS-3 wafers (Cerebras blog;
Next Platform). Assuming an even split, one wafer holds 20 layers at FULL width (hidden 8192): 70.55 B / 4 = 17.6 B parameters =
35.25 GB, about 39 KB per PE on 900,000 PEs. We model one such wafer at 1/100 of the PE count (9,000 PEs on a 90x100 mesh, same
48 KiB SRAM per PE), so the 20 layers must be narrowed until weights + KV + GEMM tiles fit in the free 37.98 KiB/PE (48 KiB minus
the 10,260 B FIFO + router reserve). Block parameters scale with hidden^2, so hidden 8192 -> 704 (11.6x narrower):

  real, one wafer : hidden 8192, 20 layers, 64 Q / 8 KV heads of 128, FFN 28672
  this model      : hidden  704, 20 layers, 11 Q / 1 KV head  of  64, FFN  2464, vocab 12800 (tied)
  parameters ~135 M, ~29 KiB/PE of weights, KV 5 KiB/token (~1.1 KiB/PE at 2048 tokens), ~7.6 KiB/PE left for GEMM tiles.

The 80 / 8192 / 28672 / 64 / 8 dimensions are Meta's published Llama-3 70B shape, recalled from memory (not read from the gated
config.json); that the 4 wafers get exactly 20 layers each is an ASSUMPTION (the sources only say layer boundaries, >= 4 systems).
The vocabulary is cut ~10x (128256 -> 12800) so the LM head stays near its real ~6% share of one wafer's work. Heads are reduced
(GQA 8:1 cannot be kept at this width) and nothing is trained: only the cost structure matters. Everything else (RMSNorm eps
1e-5, rope theta 500000, SwiGLU, no biases, no q/k norm) is Llama-3.3's, as in experiment_llama33_scaled.

COMPARISON TARGET. 2,500 tok/s at 80 layers = 0.4 ms/token = 5 us/layer (5,000 cycles at 1 GHz). For these 20 layers that is
100 us/token (10,000 tok/s) if the 3 inter-wafer hops and the LM head are ignored. Compare PER-LAYER latency (ITL / 20), not tok/s.
The mesh side here is ~95 vs ~950 hops on a WSE-3, so a latency-bound step is easier to hit here than on the real machine.
NOT modelled: the on-wafer network that moves weights / K / V to the PEs; --mac-lanes > 1 is an ideal-SIMD what-if.

Usage (project root, env active):
  python3 experiment_llama33_slice/llama33_slice.py --dry-run --skip-numeric                       # no Docker
  python3 experiment_llama33_slice/llama33_slice.py --batches 1 --modes optimistic --skip-numeric --tag lanes1
  python3 experiment_llama33_slice/llama33_slice.py --batches 1 --modes optimistic --skip-numeric --mac-lanes 8 --sram-read-bytes 16 --tag lanes8
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load("dense_llm_throughput", "experiment_dense_llm_throughput/dense_llm_throughput.py")
LS = _load("llama33_scaled", "experiment_llama33_scaled/llama33_70b_scaled.py")

# 20 layers (one of 4 wafers), hidden 704 = 11 heads x 64, 1 KV head, FFN 2464 (3.5 x hidden), vocab 12800 tied
LLAMA33_SLICE = D.DenseConfig("Llama-3.3-70B/slice20-d704", 704, 20, 11, 1, 2464, 12800, norm_eps=1e-5, rope_theta=500000.0,
                              rope_interleaved=False, tied_embeddings=True)
MODELS = {"llama33-slice-d704": (LLAMA33_SLICE, "llama33_slice_d704")}
MESH = "90x100"   # 9,000 PEs, near-square like the WSE-3

NOTE = ("One CS-3's 20-layer share of Llama-3.3-70B shrunk to hidden 704 on 9,000 PEs (90x100) x 48 KiB, weights and KV cache in SRAM,\n"
        "batch 1. The even 20-layers-per-wafer split and the 70B dimensions are assumptions; the on-wafer network is not modelled;\n"
        "--mac-lanes > 1 is an ideal-SIMD what-if. Compare per-layer latency (ITL / 20) with the 5 us/layer implied by 2,500 tok/s.")


def numeric_check() -> None:
    LS.numeric_check()


def build_parser():
    ap = D.build_parser(list(MODELS), "llama33-slice-d704", __doc__.splitlines()[0], in_len=1024, out_len=1024)
    ap.set_defaults(sram_kib=48, weights_dram=False, carve_weights=True, kv_placement="auto", max_tile_k_candidates=3,
                    batches="1", mesh=MESH, kv_sram_fraction=0.8)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg, slug = MODELS[args.model]
    D.run_model(args, cfg, slug, {}, numeric_check, note=NOTE)


if __name__ == "__main__":
    main()
