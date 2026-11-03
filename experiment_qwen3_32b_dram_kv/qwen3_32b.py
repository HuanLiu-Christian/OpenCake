#!/usr/bin/env python3
"""Qwen3-32B inference on OpenCake: 12,800 PEs x 48 KiB SRAM, KV cache in off-wafer DRAM.

Architecture, read from transformers ``models/qwen3/modeling_qwen3.py`` and the ``Qwen/Qwen3-32B`` config.json:
  per layer (pre-RMSNorm):  x + o_proj( attention( RoPE(q_norm(q)), RoPE(k_norm(k)), v ) )
                            h + down( silu(gate(RMSNorm(h))) * up(RMSNorm(h)) )
  q = x Wq (5120 -> 64 heads x 128 = 8192), k, v = x Wk, x Wv (5120 -> 8 KV heads x 128 = 1024)  [GQA, 8 query heads/KV head]
  q_norm / k_norm: RMSNorm over the 128-dim head, applied BEFORE RoPE (V is not normalised)
  RoPE: rotate_half (half-split), theta 1e6; no biases; sliding window off (all 64 layers full attention)
  final RMSNorm -> untied LM head (vocab 151936).   dim 5120, FFN 25600, 64 layers, eps 1e-6, 32.76 B parameters.
head_dim (128) is NOT dim / n_heads (80): the Q projection is 5120 -> 8192 and o_proj 8192 -> 5120.

THE MACHINE: the derived mesh (10 x 1280 = 12,800 PEs) with every PE's SRAM FIXED at 48 KiB (--sram-kib 48, the
default here), 10,260 B of it reserved for FIFO + router. The KV cache is NOT kept in that SRAM: it lives in DRAM
(--kv-dram, the default; --kv-on-chip turns it off). The weights are not on the wafer either (61 GiB vs 600 MiB of
SRAM) and are NOT charged -- the script prints what streaming them would cost per decode step, for scale.

HOW THE DRAM IS MODELLED (see ``OffchipKV`` in the LLaMA script): per layer and step, attention reads all cached K/V
and writes the new tokens' K/V; time = bytes / bandwidth (+ one access latency per layer), energy = bytes x pJ/byte.
'optimistic' hides this under compute (step = max(compute, DRAM)); 'conservative' adds it. DRAM bandwidth / latency /
energy are PLACEHOLDERS (1000 GB/s, 100 ns, 40 pJ/byte, unlimited capacity): change them, do not quote them.
NOT modelled: the on-wafer network that carries the K/V to the PEs, K/V placement, DRAM banking.

RUN TIME: every GEMM shape runs one Docker timeloop-model and one Python PE simulation PER candidate K tile; the
simulation costs ~13 s of fixed set-up plus ~0.08 s per K phase on 10k active PEs. With 48 KiB many K tiles are
feasible (down_proj has K = 25600), so --max-tile-k-candidates defaults to 3 here: only the three LARGEST feasible
tiles (fewest phases) are evaluated. Pass 0 for the exhaustive search (hours per shape for K = 25600).

Usage (project root, venv active):
  python3 experiment_qwen3_32b_dram_kv/qwen3_32b.py --dry-run --skip-numeric                  # no Docker: footprint + DRAM time
  python3 experiment_qwen3_32b_dram_kv/qwen3_32b.py --batches 1 --modes optimistic --skip-numeric   # needs Docker
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

# Qwen/Qwen3-32B config.json: hidden 5120, 64 layers, 64 heads, 8 KV heads, head_dim 128, FFN 25600, vocab 151936,
# rms_norm_eps 1e-6, rope_theta 1e6, no attention bias, untied embeddings, sliding window off.
QWEN3_32B = D.DenseConfig("Qwen3-32B", 5120, 64, 64, 8, 25600, 151936, norm_eps=1e-6, rope_theta=1_000_000.0,
                          sliding_window=0, head_dim_override=128, qk_norm=True, rope_interleaved=False)
# toys: a head_dim that differs from dim // n_heads, GQA + q/k norm (toy-qwen3) and plain MHA (toy-qwen3-mha)
TOY_QWEN3 = D.DenseConfig("toy-qwen3", 32, 2, 8, 2, 48, 50, norm_eps=1e-6, rope_theta=1_000_000.0,
                          head_dim_override=6, qk_norm=True, rope_interleaved=False)
TOY_QWEN3_MHA = D.DenseConfig("toy-qwen3-mha", 16, 2, 4, 4, 24, 50, norm_eps=1e-6, rope_theta=1_000_000.0,
                              head_dim_override=8, qk_norm=True, rope_interleaved=False)

MODELS = {"qwen3-32b": (QWEN3_32B, "qwen3_32b")}

NOTE = ("Qwen3-32B on 12,800 PEs x 48 KiB with the KV cache in DRAM. Weights (61 GiB) are NOT on the wafer and NOT charged;\n"
        "the DRAM bandwidth / latency / energy are placeholders; the on-wafer network is not modelled. The cycles are therefore\n"
        "'compute + KV traffic if the weights were free', an upper bound on throughput.")


def numeric_check() -> None:
    D.numeric_check((TOY_QWEN3, TOY_QWEN3_MHA), "transformers' modeling_qwen3.py (rotate_half RoPE, per-head q_norm / k_norm)")


def build_parser():
    ap = D.build_parser(list(MODELS), "qwen3-32b", __doc__.splitlines()[0], in_len=1024, out_len=1024)
    ap.add_argument("--kv-on-chip", dest="kv_dram", action="store_false", help="do NOT move the KV cache to DRAM")
    ap.set_defaults(sram_kib=48, kv_dram=True, max_tile_k_candidates=3)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg, slug = MODELS[args.model]
    D.run_model(args, cfg, slug, {}, numeric_check, note=NOTE)


if __name__ == "__main__":
    main()
