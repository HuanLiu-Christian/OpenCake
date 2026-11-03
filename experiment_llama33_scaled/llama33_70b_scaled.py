#!/usr/bin/env python3
"""Llama-3.3-70B shrunk 163.3x, on the 12,800-PE x 48 KiB wafer: weights in DRAM, KV cache on-chip or in DRAM per batch.

WHY SHRINK: the real model has 70.55 B parameters = 141 GB at fp16 (a Cerebras WSE-3 holds 44 GB, so Cerebras needs several
wafers for it). Our wafer is the 1/70-scale machine (12,800 PEs x 48 KiB = 600 MiB). So the model is cut by 163.3x too.

HOW (the design chosen with the user): keep the DEPTH (80 layers: decode latency is a chain of 80 sequential layers, which is
what makes it behave like the 70B) and narrow the WIDTH by ~12.8x (sqrt(163.3)):
  real 70B   : hidden 8192, 80 layers, 64 query / 8 KV heads of 128, FFN 28672, vocab 128256, untied
  this model : hidden  640, 80 layers, 10 query / 2 KV heads of  64, FFN  2240, vocab 128256, TIED
  block parameters 422.8 M vs 68.45 B -> 1/161.9 (target 1/163.3);  KV cache 40 KiB/token vs 320 KiB/token (1/8).
The vocabulary is NOT shrunk (it keeps the real tokenizer's size) and the head is tied to the embedding table, so the table
(82 M parameters) is 16% of this model but only 3% of the real one: the LM head is over-weighted here. Total parameters:
504.9 M = 1/139.7 of 70.55 B. Everything else (RMSNorm eps 1e-5, rope theta 500000, SwiGLU, GQA, no biases, no q/k norm) is
Llama-3.3's, read from transformers' modeling_llama.py; the 70B dimensions are recalled from memory and are NOT checked
against Meta's config.json (the repo is gated).

THE MACHINE / DRAM: as in the Qwen3-0.6B script (see experiment_qwen3_scaled_70x): weights (0.94 GiB, 1.6x the SRAM) are
streamed from DRAM every pass; the KV cache goes on the wafer only if its per-PE share is <= half the free data bytes
(--kv-placement auto): batch 1 and 2 on-chip (6.4 KiB per PE per sequence), batch 3+ in DRAM. DRAM bandwidth / latency /
energy are PLACEHOLDERS (--dram-bw-gbs); the on-wafer network and K/V layout are NOT modelled.

Usage (project root, venv active):
  python3 experiment_llama33_scaled/llama33_70b_scaled.py --dry-run --skip-numeric
  python3 experiment_llama33_scaled/llama33_70b_scaled.py --batches 1,16,32,64 --modes optimistic --dram-bw-gbs 3000 --skip-numeric   # Docker
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

# Meta-Llama-3.3-70B-Instruct (HF LlamaForCausalLM), dimensions from memory -- unverified, the config.json is gated.
LLAMA33_70B = D.DenseConfig("Llama-3.3-70B", 8192, 80, 64, 8, 28672, 128256, norm_eps=1e-5, rope_theta=500000.0,
                            rope_interleaved=False)
SCALE = 163.3
# the 163.3x-smaller model: 80 layers, hidden 640 = 10 heads x 64, 2 KV heads (GQA 5:1), FFN 2240, real vocab, tied head
LLAMA33_70B_S163 = D.DenseConfig("Llama-3.3-70B/163.3", 640, 80, 10, 2, 2240, 128256, norm_eps=1e-5, rope_theta=500000.0,
                                 rope_interleaved=False, tied_embeddings=True)
# toys for the numeric check: HF layout (half-split RoPE), GQA and MHA
TOY_LLAMA_HF = D.DenseConfig("toy-llama-hf-gqa", 32, 2, 8, 2, 48, 50, norm_eps=1e-5, rope_theta=500000.0, rope_interleaved=False)
TOY_LLAMA_HF_MHA = D.DenseConfig("toy-llama-hf-mha", 32, 2, 4, 4, 48, 50, norm_eps=1e-5, rope_theta=500000.0, rope_interleaved=False)

MODELS = {"llama33-70b-s163": (LLAMA33_70B_S163, "llama33_70b_s163")}

NOTE = ("Llama-3.3-70B shrunk 163.3x (80 layers, hidden 640, vocab 128256 tied) on 12,800 PEs x 48 KiB, weights streamed from DRAM,\n"
        "KV cache on-chip or in DRAM per batch. DRAM parameters are placeholders; the on-wafer network is not modelled; the 70B\n"
        "dimensions are from memory. A scaled comparison with Cerebras' 70B, not the same machine or the same model.")


def numeric_check() -> None:
    D.numeric_check((TOY_LLAMA_HF, TOY_LLAMA_HF_MHA), "transformers' modeling_llama.py (rotate_half RoPE, GQA, SwiGLU)")


def build_parser():
    ap = D.build_parser(list(MODELS), "llama33-70b-s163", __doc__.splitlines()[0], in_len=1024, out_len=1024)
    ap.add_argument("--no-weights-dram", dest="weights_dram", action="store_false",
                    help="do NOT bill weight streaming (weights treated as free)")
    ap.set_defaults(sram_kib=48, weights_dram=True, kv_placement="auto", max_tile_k_candidates=3, batches="1,16,32,64")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg, slug = MODELS[args.model]
    D.run_model(args, cfg, slug, {}, numeric_check, note=NOTE)


if __name__ == "__main__":
    main()
