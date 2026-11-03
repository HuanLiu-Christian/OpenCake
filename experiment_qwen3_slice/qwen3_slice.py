#!/usr/bin/env python3
"""One CS-3's share of Qwen3-32B, shrunk to 9,000 PEs x 48 KiB: 32 layers, hidden 460, weights AND KV in SRAM, batch 1.

WHY. Qwen3-32B has 64 layers, hidden 5120, 32.76 B parameters = 65.5 GB in bf16 (Qwen/Qwen3-32B config.json and file listing), which
does not fit one WSE-3 (44 GB of SRAM). ASSUMPTION (Cerebras publishes no system count for Qwen3-32B): it runs on 2 CS-3 wafers, split
at layer boundaries like Llama-70B on 4 (Hot Chips 2024, slide "Naturally scales to multiple wafers"), so one wafer holds 32 layers
at FULL width: 32.76 B / 2 = 16.4 B parameters = 32.8 GB, 36 KB per PE on 900,000 PEs. We model one such wafer at 1/100 of the PE
count (9,000 PEs, 90x100 mesh, 48 KiB per PE). The proportional target is 0.164 B parameters / 0.328 GB, but that needs hidden ~510
and does not fit (the free SRAM is 37.98 KiB/PE = 48 KiB minus the 10,260 B FIFO + router reserve), so hidden is cut to 460:

  real, one wafer : hidden 5120, 32 layers, 64 Q / 8 KV heads of 128 (Q width 8192 = 1.6 x hidden), FFN 25600, vocab 151936, untied
  this model      : hidden  460, 32 layers, 11 Q / 1 KV head  of  64 (Q width  704),               FFN  2300, vocab  15200, untied
  parameters ~138 M (0.276 GB), 30 KiB/PE of weights, KV 8 KiB/token (~2 KiB/PE at 2047 tokens), 6,124 B/PE left for GEMM tiles.

Qwen3 specifics kept: q/k RMSNorm over the head (before RoPE), rotate_half RoPE with theta 1e6, eps 1e-6, SwiGLU, no biases, no sliding
window, untied LM head, head_dim not equal to hidden / n_heads (Q projection hidden -> 704, o_proj 704 -> hidden). The head counts
(11 Q / 1 KV, head_dim 64) are OUR choice, the same attention shape as experiment_llama33_slice (q_mult 11); GQA 8:1 with head_dim 128
does not fit at this width. The vocabulary is cut ~10x (151936 -> 15200). Nothing is trained: only the cost structure matters.

COMPARISON TARGET. Cerebras reports ~2,400 tok/s for Qwen3-32B (Artificial Analysis; number of systems and test conditions NOT
published). At 64 layers that is 0.417 ms/token = 6.5 us/layer; for these 32 layers 208 us/token (4,800 tok/s) if inter-wafer hops and
the LM head are ignored. Compare PER-LAYER latency (ITL / 32), not tok/s.
NOT modelled: the on-wafer network that moves weights / K / V to the PEs; --mac-lanes > 1 is an ideal-SIMD what-if.

Usage (project root, env active):
  python3 experiment_qwen3_slice/qwen3_slice.py --dry-run --skip-numeric                       # no Docker
  python3 experiment_qwen3_slice/qwen3_slice.py --batches 1 --modes optimistic --skip-numeric --tag lanes1
  python3 experiment_qwen3_slice/qwen3_slice.py --batches 1 --modes optimistic --skip-numeric --mac-lanes 8 --sram-read-bytes 16 --tag lanes8
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
Q32 = _load("qwen3_32b", "experiment_qwen3_32b_dram_kv/qwen3_32b.py")

# 32 layers (one of 2 wafers), hidden 460, 11 Q heads / 1 KV head of 64, FFN 2300 (5 x hidden), vocab 15200, untied
QWEN3_SLICE = D.DenseConfig("Qwen3-32B/slice32-d460", 460, 32, 11, 1, 2300, 15200, norm_eps=1e-6, rope_theta=1_000_000.0,
                            sliding_window=0, head_dim_override=64, qk_norm=True, rope_interleaved=False, tied_embeddings=False)
MODELS = {"qwen3-slice-d460": (QWEN3_SLICE, "qwen3_slice_d460")}
MESH = "90x100"   # 9,000 PEs, near-square like the WSE-3

NOTE = ("One CS-3's 32-layer share of Qwen3-32B shrunk to hidden 460 on 9,000 PEs (90x100) x 48 KiB, weights and KV cache in SRAM,\n"
        "batch 1. The 2-wafer even split is an assumption (Cerebras publishes no system count); the on-wafer network is not modelled;\n"
        "--mac-lanes > 1 is an ideal-SIMD what-if. Compare per-layer latency (ITL / 32) with the 6.5 us/layer implied by 2,400 tok/s.")


def numeric_check() -> None:
    Q32.numeric_check()


def build_parser():
    ap = D.build_parser(list(MODELS), "qwen3-slice-d460", __doc__.splitlines()[0], in_len=1024, out_len=1024)
    ap.set_defaults(sram_kib=48, weights_dram=False, carve_weights=True, kv_placement="auto", max_tile_k_candidates=3,
                    batches="1", mesh=MESH, kv_sram_fraction=0.8)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg, slug = MODELS[args.model]
    D.run_model(args, cfg, slug, {}, numeric_check, note=NOTE)


if __name__ == "__main__":
    main()
