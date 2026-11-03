#!/usr/bin/env python3
"""One of FOUR CS-3 wafers' share of Qwen3-32B, shrunk to 9,000 PEs x 48 KiB: 16 layers, hidden 480, weights AND KV in SRAM, batch 1.

WHY. Qwen3-32B has 64 layers, hidden 5120, 32.76 B parameters = 65.5 GB in bf16 (Qwen/Qwen3-32B config.json and file listing).
ASSUMPTION (Cerebras publishes no system count for Qwen3-32B): it runs on 4 CS-3 wafers, split at layer boundaries and evenly like
Llama-70B in Hot Chips 2024 (slide "Naturally scales to multiple wafers"): 16 layers per wafer, 8.2 GB per wafer on 44 GB of SRAM.
We model one wafer at 1/100 of the PE count (9,000 PEs, 90x100, 48 KiB each). The proportional target is 0.082 B parameters /
0.164 GB; unlike the 2-wafer / 32-layer split (experiment_qwen3_slice) it FITS (about 18 KiB of weights per PE), so hidden is set
by matching the target: hidden 480.

  real, one wafer : hidden 5120, 16 layers, 64 Q / 8 KV heads of 128 (Q width 8192), FFN 25600, vocab 151936, untied
  this model      : hidden  480, 16 layers, 11 Q / 1 KV head  of  64 (Q width  704), FFN  2400, vocab  15200, untied
  parameters 81.7 M (0.163 GB), 18 KiB/PE of weights, KV 4 KiB/token (~1 KiB/PE at 2047 tokens), ~19 KB/PE left for GEMM tiles.

The parameter count includes BOTH vocabulary matrices (14.6 M, 18 %), which a real wafer would not hold (embedding on the first wafer,
LM head on the last): matching only the transformer blocks to 1/4 of the real model / 100 would give hidden ~520.
Qwen3 specifics kept: q/k RMSNorm over the head, rotate_half RoPE theta 1e6, eps 1e-6, SwiGLU, no biases, no sliding window, untied LM head,
head_dim != hidden / n_heads. The head counts (11 Q / 1 KV, head_dim 64) are OUR choice (same attention shape as experiment_llama33_slice).

COMPARISON TARGET. Cerebras reports ~2,400 tok/s for Qwen3-32B (Artificial Analysis; number of systems and test conditions NOT published).
At 64 layers that is 0.417 ms/token = 6.5 us/layer; for these 16 layers 104 us/token. Compare PER-LAYER latency (ITL / 16); the whole
model is 4 of these in series plus 3 inter-wafer hops (about 5 us each, Hot Chips 2024).
NOT modelled: the on-wafer network that moves weights / K / V to the PEs; --mac-lanes > 1 is an ideal-SIMD what-if.

Usage (project root, env active):
  python3 experiment_qwen3_slice4w/qwen3_slice4w.py --dry-run --skip-numeric                       # no Docker
  python3 experiment_qwen3_slice4w/qwen3_slice4w.py --batches 1 --modes optimistic --skip-numeric --mac-lanes 8 --sram-read-bytes 16 --tag lanes8
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

# 16 layers (one of 4 wafers), hidden 480, 11 Q heads / 1 KV head of 64, FFN 2400 (5 x hidden), vocab 15200, untied
QWEN3_SLICE4W = D.DenseConfig("Qwen3-32B/slice16-d480", 480, 16, 11, 1, 2400, 15200, norm_eps=1e-6, rope_theta=1_000_000.0,
                              sliding_window=0, head_dim_override=64, qk_norm=True, rope_interleaved=False, tied_embeddings=False)
MODELS = {"qwen3-slice4w-d480": (QWEN3_SLICE4W, "qwen3_slice4w_d480")}
MESH = "90x100"   # 9,000 PEs, near-square like the WSE-3

NOTE = ("One of 4 CS-3 wafers' 16-layer share of Qwen3-32B shrunk to hidden 480 on 9,000 PEs (90x100) x 48 KiB, weights and KV in SRAM,\n"
        "batch 1. The 4-wafer even split is an assumption (Cerebras publishes no system count); the on-wafer network is not modelled;\n"
        "--mac-lanes > 1 is an ideal-SIMD what-if. Compare per-layer latency (ITL / 16) with the 6.5 us/layer implied by 2,400 tok/s.")


def numeric_check() -> None:
    Q32.numeric_check()


def build_parser():
    ap = D.build_parser(list(MODELS), "qwen3-slice4w-d480", __doc__.splitlines()[0], in_len=1024, out_len=1024)
    ap.set_defaults(sram_kib=48, weights_dram=False, carve_weights=True, kv_placement="auto", max_tile_k_candidates=3,
                    batches="1", mesh=MESH, kv_sram_fraction=0.8)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg, slug = MODELS[args.model]
    D.run_model(args, cfg, slug, {}, numeric_check, note=NOTE)


if __name__ == "__main__":
    main()
