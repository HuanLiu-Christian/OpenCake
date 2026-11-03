#!/usr/bin/env python3
"""Dense decoder LLM (LLaMA-2-7B first) on OpenCake: numeric forward + KV cache,
op-by-op hardware cost, and THROUGHPUT in the metric of the LLM-Inference-Bench
paper (arXiv 2411.00136): batch * (input + output tokens) / end-to-end latency.

Reference being modeled: Meta's ``llama/model.py`` (LLaMA-2). Per layer:
  x -> RMSNorm -> wq/wk/wv (no bias) -> RoPE on q,k -> KV-cache write ->
  softmax(q k^T / sqrt(d) + causal mask) v -> wo -> +x ->
  RMSNorm -> w2( silu(w1 h) * w3 h ) -> +h
then final RMSNorm and the LM head. LLaMA-2-7B: dim 4096, 32 layers, 32 heads
(MHA, n_kv_heads == n_heads), SwiGLU hidden 11008, vocab 32000, theta 10000.
The same code handles GQA (n_kv_heads < n_heads) so LLaMA-3-8B / Mistral-7B are
a config change. LLaMA-2-7B and LLaMA-3-8B (llama/model.py of both, identical but for rope_theta) are checked.

TWO LAYERS, KEPT IN SYNC BY A TEST
  * numeric path  -- real numpy through ``Tensor``; checked against a literal
    numpy transcription of the reference (complex-number RoPE, cache arrays,
    repeat_kv, fp32 softmax) at toy dims, MHA and GQA, batch > 1, prefill +
    token-by-token decode.
  * cost-only path -- the same op sequence expressed as element counts / GEMM
    shapes, so real dims (M = batch*1024 rows!) never materialise a tensor.
    ``tests/test_dense_llm_throughput.py`` asserts the two emit IDENTICAL
    (op, key) logs on a dry-run tracker, so they cannot drift apart.

MODELING CHOICES (each one named, because each moves the answer)
  RoPE. The reference rotates ADJACENT pairs (x[2i], x[2i+1]); this project's
    apply_rope rotates halves. They are equal after permuting wq/wk output
    columns per head to [even..., odd...] (what HF's llama converter does);
    q.k is invariant because q and k share the permutation. Costs are
    identical; the numeric test proves the equivalence.
  Batch. Linear layers / norms / RoPE / FFN act on the stacked (batch*S, .)
    rows in single calls. Attention is per (sequence, KV head) instance. Their
    LATENCY is bracketed: "serial" = every instance one after another (the
    convention of the gpt-oss scripts), "concurrent" = all instances at once
    on disjoint PE regions (ideal lower bound). Energy always counts every
    instance. Cost of one instance is simulated once and multiplied.
  Data movement (concat / gather / copying reshape). Two bracketed modes:
    "mesh" = split over PEs like elementwise ops (assumes aligned layouts,
    no network cost: optimistic); "serial" = one PE (the project's old
    convention). One PE cannot hold more than 2**21 elements per move (CACTI
    ceiling in timeloop.py), so a move is charged as ceil(n/2**20) chunks;
    move cost is linear in elements, so chunking is cost-neutral.
  LM head runs on the LAST row of each sequence only (what vLLM/TRT-LLM do;
    the reference computes all rows). Greedy sampling is charged as one max
    over the vocab per sequence (argmax index extraction not modeled).
  Embedding lookup is a gather of batch*S rows.
  Precision. Cost pass uses 2-byte elements (fp16, as the paper); numeric
    pass fp32.
  Decode cost is sampled at 3 cache lengths and averaged with Simpson's rule
    (cost is ~linear in cache length). Prefill runs in passes of C tokens (C = the largest divisor of the
    input length whose every GEMM has a feasible K tile in the PE SRAM -- the derived wafer has only
    13 KiB/PE, so batch*1024-row GEMMs cannot be tiled); passes are costed at first/middle/last chunk
    (Simpson), the LM head runs once after the last pass. Override with --prefill-chunk.

KNOWN LIMITATIONS (not buried)
  - Cross-PE movement has no network model: the "mesh" move mode is an
    optimistic bound, the "serial" mode a pessimistic one.
  - Wafer is derived from the fixed area budget; LLaMA-2-7B weights (13.5 GB
    in fp16) do not fit its SRAM -- the printout shows the footprint; GEMM
    operand streaming is whatever the mesh GEMM simulator models.
  - Clock = wafer.clock_ghz (default 1.0 GHz, an assumption).
  - Sampling beyond greedy, speculative decoding, paged-KV effects: not modeled.
  - Paper numbers are GPU measurements; comparing absolute values needs the
    caveats in the printout (definition, batch sizes, hardware scale).

Docker/TimeLoop failures RAISE -- no fallback cost numbers.

Usage (project root, venv active):
  python3 experiment_dense_llm_throughput/dense_llm_throughput.py --dry-run
  python3 experiment_dense_llm_throughput/dense_llm_throughput.py          # needs Docker
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from wafersim.primitives import (
    AreaBudget,
    GemmSpec,
    Tensor,
    WaferSpec,
    derive_wafer_spec_from_area_budget,
    slice_ as p_slice,
    transpose as p_transpose,
)
from wafersim.tiling import Tiling
from wafersim.timeloop import _largest_divisor_leq, feasible_tile_k_candidates

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    """Load a sibling experiment script by path (they are scripts, not packages)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses inside need their module registered
    spec.loader.exec_module(mod)
    return mod


moe = _load("gptoss_moe_mlp", "experiment_gptoss_moe_mlp/gptoss_moe_mlp.py")
attn_lib = _load("gptoss_attention_kvcache", "experiment_gptoss_attention_kvcache/gptoss_attention_kvcache.py")

LayerKVCache, append_kv = attn_lib.LayerKVCache, attn_lib.append_kv
rope_tables, apply_rope, build_mask = attn_lib.rope_tables, attn_lib.apply_rope, attn_lib.build_mask
op_add, op_mul, op_sum, op_max = moe.op_add, moe.op_mul, moe.op_sum, moe.op_max
op_reshape, op_gather, op_concat, op_matmul, op_silu = moe.op_reshape, moe.op_gather, moe.op_concat, moe.op_matmul, moe.op_silu
rms_norm, softmax_lastdim, _const = moe.rms_norm, moe.softmax_lastdim, moe._const

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DTYPE = np.float32  # numeric pass
COST_BPE = 2  # cost pass: fp16, as in the paper
SEED = 0
MAX_MOVE_ELEMS = 1 << 20  # per-chunk cap for one-PE data moves (hard limit is 2**21)
FIXED_AREA_BUDGET_UM2 = moe.FIXED_AREA_BUDGET_UM2
AREA_COMPUTE_FRACTION = moe.AREA_COMPUTE_FRACTION
AREA_MEMORY_FRACTION = moe.AREA_MEMORY_FRACTION
AREA_NETWORK_FRACTION = moe.AREA_NETWORK_FRACTION
# THE MACHINE. The wafer is derived from the 800 mm^2 budget by a real-Docker search whose answer depends on
# (a) the representative GEMM's m,n (mesh rows|m x cols|n) and (b) the MAC width: the compute budget buys
# floor(0.15*800e6 / mac_area) PEs, and a 2-byte MAC is much smaller than a 4-byte one. Deriving with the
# model's own GEMM at 2 bytes gave a 10x4096 = 40,960-PE mesh whose per-PE memory budget (3,320 um^2) cannot hold
# any SRAM -> "no sram_kib_per_pe fits". So the machine is fixed to the SAME one the gpt-oss attention run used
# (m=10, n=5120, k=2880, 4-byte MAC -> 10x1280 = 12,800 PEs, 13 KiB/PE), for every model, and cached on disk.
# The fp16 COST model then runs on that machine; the machine itself does not change with the model or precision.
REP_GEMM = (10, 5120, 2880)
WAFER_DERIVE_BPE = 4
PAPER_BATCHES = (1, 16, 32, 64)
PAPER_IN_OUT = 1024
OUT_ROOT = REPO_ROOT / "outputs" / "dense_llm_throughput"


@dataclass(frozen=True)
class DenseConfig:
    name: str
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    vocab: int
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 0  # 0 = full attention
    head_dim_override: int = 0  # > 0: head size is NOT dim // n_heads (Qwen3-32B: 128 with dim 5120 and 64 heads)
    qk_norm: bool = False  # Qwen3: RMSNorm over head_dim on Q and K, after the projection, before RoPE
    rope_interleaved: bool = True  # Meta LLaMA rotates adjacent pairs; HF / Qwen3 use rotate_half (half-split)
    tied_embeddings: bool = False  # LM head shares the embedding table (Qwen3-0.6B/1.7B/4B): stored once

    @property
    def head_dim(self) -> int:
        return self.head_dim_override or self.dim // self.n_heads

    @property
    def q_dim(self) -> int:
        """Width of the Q projection output = the attention output fed to ``wo`` (== dim for LLaMA)."""
        return self.n_heads * self.head_dim

    @property
    def q_mult(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def per_layer_params(self) -> int:
        h, f = self.dim, self.ffn_hidden
        qd = self.q_dim
        per_layer = h * qd + 2 * h * self.kv_dim + qd * h + 3 * h * f + 2 * h
        if self.qk_norm:
            per_layer += 2 * self.head_dim
        return per_layer

    def n_params(self) -> int:
        h, v = self.dim, self.vocab
        return v * h + self.n_layers * self.per_layer_params() + h + (0 if self.tied_embeddings else v * h)

    def weight_stream_bytes(self, tokens: int, bpe: int, tail: bool = True) -> int:
        """Bytes of weights one forward pass reads from a weight store: every block's weights and the final norm,
        the ``tokens`` embedding rows it looks up, and (if ``tail``) the whole LM head. A tied head is the same table
        as the embedding, but it is a full read of vocab x dim, so it is counted once per tail either way."""
        blocks = self.n_layers * self.per_layer_params() + self.dim
        return (blocks + tokens * self.dim + (self.vocab * self.dim if tail else 0)) * bpe

    def kv_bytes_per_token(self, bpe: int) -> int:
        return 2 * self.n_layers * self.kv_dim * bpe


def llama_ffn_hidden(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]) -> int:
    """Exactly the arithmetic in the reference ``FeedForward.__init__`` (hidden_dim = 4*dim)."""
    hidden = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


LLAMA2_7B = DenseConfig("LLaMA-2-7B", 4096, 32, 32, 32, llama_ffn_hidden(4096, 256, None), 32000, 1e-5, 10000.0)
# LLaMA-3-8B: GQA (8 KV heads), SwiGLU hidden 14336, vocab 128256, theta 500000. The ModelArgs defaults in
# llama/model.py do NOT contain these (vocab_size=-1, n_kv_heads=None, ffn_dim_multiplier=None); they come from the
# checkpoint's params.json (dim 4096, n_layers 32, n_heads 32, n_kv_heads 8, multiple_of 1024, ffn_dim_multiplier 1.3,
# norm_eps 1e-5, rope_theta 500000) and the tokenizer (128256). Check them against the params.json you downloaded.
LLAMA3_8B = DenseConfig("LLaMA-3-8B", 4096, 32, 32, 8, llama_ffn_hidden(4096, 1024, 1.3), 128256, 1e-5, 500000.0)
TOY_MHA = DenseConfig("toy-mha", 32, 2, 4, 4, 48, 50)
TOY_GQA = DenseConfig("toy-gqa", 32, 2, 8, 2, 48, 50)
TOY_GQA_L3 = DenseConfig("toy-gqa-l3", 32, 2, 8, 2, 48, 50, 1e-5, 500000.0)  # LLaMA-3 style rope theta

# --------------------------------------------------------------------------
# Weights. Reference layout = torch Linear (out, in). Hardware layout = x @ W.
# --------------------------------------------------------------------------


def deinterleave_perm(n_heads: int, head_dim: int) -> np.ndarray:
    """Per head, reorder features [0..d) -> [0,2,4,..., 1,3,5,...]: turns the
    reference's adjacent-pair RoPE into this project's half-split RoPE."""
    order = np.concatenate([np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)])
    return np.concatenate([h * head_dim + order for h in range(n_heads)])


def random_reference_weights(cfg: DenseConfig, rng: np.random.Generator, scale: float = 0.2) -> dict:
    h, kvd, f, v = cfg.dim, cfg.kv_dim, cfg.ffn_hidden, cfg.vocab
    r = lambda *s: (rng.standard_normal(s) * scale).astype(DTYPE)
    qd, layers = cfg.q_dim, []
    for _ in range(cfg.n_layers):
        lw = {
            "attention_norm": (1.0 + 0.1 * rng.standard_normal(h)).astype(DTYPE),
            "wq": r(qd, h), "wk": r(kvd, h), "wv": r(kvd, h), "wo": r(h, qd),
            "ffn_norm": (1.0 + 0.1 * rng.standard_normal(h)).astype(DTYPE),
            "w1": r(f, h), "w2": r(h, f), "w3": r(f, h),
        }
        if cfg.qk_norm:
            lw["q_norm"] = (1.0 + 0.1 * rng.standard_normal(cfg.head_dim)).astype(DTYPE)
            lw["k_norm"] = (1.0 + 0.1 * rng.standard_normal(cfg.head_dim)).astype(DTYPE)
        layers.append(lw)
    return {"tok_embeddings": r(v, h), "layers": layers,
            "norm": (1.0 + 0.1 * rng.standard_normal(h)).astype(DTYPE), "output": r(v, h)}


@dataclass
class LayerWeights:
    attn_norm: Tensor
    wq: Tensor  # (H, q_dim) columns de-interleaved per head (LLaMA) or as stored (HF-style rope)
    wk: Tensor  # (H, kv_dim) columns de-interleaved per head
    wv: Tensor
    wo: Tensor
    ffn_norm: Tensor
    w1: Tensor
    w2: Tensor
    w3: Tensor
    q_norm: Optional[Tensor] = None  # (head_dim,) -- Qwen3 only
    k_norm: Optional[Tensor] = None


@dataclass
class ModelWeights:
    embed: Tensor  # (V, H)
    layers: list
    norm: Tensor
    output: Tensor  # (H, V)

    @classmethod
    def from_reference(cls, ref: dict, cfg: DenseConfig) -> "ModelWeights":
        f = lambda a: Tensor.from_values(np.ascontiguousarray(a, dtype=DTYPE))
        if cfg.qk_norm and cfg.rope_interleaved:
            raise ValueError("qk_norm weights are per head feature: they would need the same de-interleave; use rope_interleaved=False")
        if cfg.rope_interleaved:  # the reference rotates adjacent pairs; this project rotates halves
            pq = deinterleave_perm(cfg.n_heads, cfg.head_dim)
            pk = deinterleave_perm(cfg.n_kv_heads, cfg.head_dim)
        else:  # already rotate_half layout
            pq, pk = np.arange(cfg.q_dim), np.arange(cfg.kv_dim)
        layers = [LayerWeights(
            f(lw["attention_norm"]), f(lw["wq"][pq, :].T), f(lw["wk"][pk, :].T), f(lw["wv"].T), f(lw["wo"].T),
            f(lw["ffn_norm"]), f(lw["w1"].T), f(lw["w2"].T), f(lw["w3"].T),
            q_norm=f(lw["q_norm"]) if cfg.qk_norm else None, k_norm=f(lw["k_norm"]) if cfg.qk_norm else None)
            for lw in ref["layers"]]
        return cls(f(ref["tok_embeddings"]), layers, f(ref["norm"]), f(ref["output"].T))


# --------------------------------------------------------------------------
# Cost tracker: the shared one plus (a) tiled / chunked data moves and
# (b) a shape-only reshape entry for the cost-only path.
# --------------------------------------------------------------------------




@dataclass
class DenseCostTracker(moe.CostTracker):
    """``tiling`` (elementwise / reduction lanes) is inherited. ``move_tiling``
    decides how concat / gather / copying-reshape are split over PEs:
    ``Tiling()`` = mesh (optimistic), ``Tiling.serial()`` = one PE."""

    move_tiling: Tiling = field(default_factory=Tiling.serial)
    max_tile_k_candidates: Optional[int] = None  # None = evaluate every SRAM-feasible tile_k (the full search)

    def op_gemm(self, m: int, n: int, k: int, bpe: int = 2) -> None:
        key = ("gemm", m, n, k, bpe)  # same memo key as the base tracker
        cap = self.max_tile_k_candidates

        def build(d):
            spec = GemmSpec(m=m, n=n, k=k, bytes_per_element=bpe)
            return moe.best_by_sim_time(moe.search_tile_k(spec, self.wafer, d, max_candidates=cap))

        cand, hit = self._lookup(key, "gemm", build, moe._DRY_GEMM)
        self._record("gemm", key, cand.result.sim_time_cycles, cand.result.energy.total_uj, hit)

    def _move(self, name: str, n: int, bpe: int, factory: Callable, sim_fn: Callable, extra_key: tuple = ()) -> None:
        if n <= 0:
            per_chunk, chunks, e_mult = 0, 1, 1.0
        else:
            sh = self.move_tiling.elements(n, self.total_pes)
            chunks = -(-sh.per_pe // MAX_MOVE_ELEMS)
            per_chunk = -(-sh.per_pe // chunks)
            e_mult = sh.energy_multiplier * chunks
        key = (name, per_chunk, bpe) + extra_key
        res, hit = self._lookup(key, name, lambda d: sim_fn(factory(per_chunk), self.wafer, d), moe._DRY_ELEMENTWISE)
        self._record(name, key, res.local_compute_cycles, res.local_energy_uj, hit,
                     cycle_repeats=chunks, energy_repeats=e_mult, elements=n)

    def op_concat(self, out: Tensor) -> None:
        bpe = out.bytes_per_element
        self._move("concat", out.size, bpe, lambda k: moe.ConcatStackSpec(elements_per_pe=k, bytes_per_element=bpe),
                   moe.simulate_concat_stack)

    def op_gather(self, out: Tensor) -> None:
        bpe = out.bytes_per_element
        self._move("gather", out.size, bpe, lambda k: moe.GatherSpec(elements_per_pe=k, bytes_per_element=bpe),
                   moe.simulate_gather)

    def op_reshape_flag(self, n_out: int, bpe: int, copied: bool) -> None:
        """Shape-only reshape: ``n_out`` elements, physically copied or a free view."""
        self._move("reshape", n_out if copied else 0, bpe,
                   lambda k: moe.ReshapeSpec(elements_per_pe=k, bytes_per_element=bpe, copied=copied),
                   moe.simulate_reshape, extra_key=(copied,))

    def op_reshape(self, tensor_before: Tensor, shape: tuple) -> None:
        spec = moe.ReshapeSpec.from_call(tensor_before, shape)  # real numpy: copied or view?
        self.op_reshape_flag(spec.elements_per_pe if spec.copied else 0, spec.bytes_per_element, spec.copied)


def phantom(shape: tuple, bpe: int) -> Tensor:
    """A Tensor of the given shape with ZERO memory (stride-0 broadcast); only
    ``.size`` / ``.bytes_per_element`` are ever read by the cost methods."""
    dt = {2: np.float16, 4: np.float32}[bpe]
    return Tensor(np.broadcast_to(np.zeros((), dtype=dt), shape))


# --------------------------------------------------------------------------
# NUMERIC forward (real numpy through Tensor; optional cost accounting)
# --------------------------------------------------------------------------


def new_caches(cfg: DenseConfig, b: int) -> list:
    """caches[sequence][layer]"""
    return [[LayerKVCache() for _ in range(cfg.n_layers)] for _ in range(b)]


def layer_forward(x: Tensor, lw: LayerWeights, cfg: DenseConfig, caches: list, b: int, s: int, cost) -> Tensor:
    """One decoder layer on ``b`` sequences x ``s`` new tokens each, stacked as
    (b*s, H). ``caches``: one LayerKVCache per sequence (all the same length)."""
    nh, nkv, d, qm, kvd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.q_mult, cfg.kv_dim
    m, window, t_old = b * s, cfg.sliding_window, caches[0].length
    assert all(c.length == t_old for c in caches), "all sequences must share one cache length"

    normed = rms_norm(x, lw.attn_norm, cfg.norm_eps, cost)
    q = op_matmul(normed, lw.wq, cost)
    k = op_matmul(normed, lw.wk, cost)
    v = op_matmul(normed, lw.wv, cost)

    cos_np, sin_np = rope_tables(np.tile(t_old + np.arange(s), b), d, cfg.rope_theta)
    cos, sin = Tensor.from_values(cos_np[:, None, :]), Tensor.from_values(sin_np[:, None, :])
    if cfg.qk_norm:  # Qwen3: per-head RMSNorm over head_dim, then RoPE (V is not normalised)
        q = op_reshape(rms_norm(op_reshape(q, (m * nh, d), cost), lw.q_norm, cfg.norm_eps, cost), (m, nh, d), cost)
        q = apply_rope(q, cos, sin, cost)
        k = op_reshape(rms_norm(op_reshape(k, (m * nkv, d), cost), lw.k_norm, cfg.norm_eps, cost), (m, nkv, d), cost)
        k_rot = apply_rope(k, cos, sin, cost)
    else:
        q = apply_rope(op_reshape(q, (m, nh, d), cost), cos, sin, cost)
        k_rot = apply_rope(op_reshape(k, (m, nkv, d), cost), cos, sin, cost)
    k_rows = op_reshape(k_rot, (m, kvd), cost)

    for bi in range(b):  # real per-sequence cache update; the write is BILLED once, stacked
        append_kv(caches[bi], p_slice(k_rows, 0, bi * s, (bi + 1) * s), p_slice(v, 0, bi * s, (bi + 1) * s), None)
    if cost is not None:
        cost.op_concat(k_rows)
        cost.op_concat(v)

    t = caches[0].length
    key_start = max(0, t_old - window + 1) if window > 0 else 0
    t_eff = t - key_start
    rows = s * qm
    mask_np = build_mask(s, qm, t_old, key_start, t_eff, window)
    mask = None if mask_np is None else Tensor.from_values(mask_np)
    scale = _const((rows, t_eff), 1.0 / math.sqrt(d))

    seq_outs = []
    for bi in range(b):
        q_b = p_slice(q, 0, bi * s, (bi + 1) * s)  # (s, nh, d)
        k_win = p_slice(caches[bi].k, 0, key_start, t)
        v_win = p_slice(caches[bi].v, 0, key_start, t)
        outs = []
        for g in range(nkv):
            qg = op_reshape(p_slice(q_b, 1, g * qm, (g + 1) * qm), (rows, d), cost)
            kg, vg = p_slice(k_win, 1, g * d, (g + 1) * d), p_slice(v_win, 1, g * d, (g + 1) * d)
            scores = op_mul(op_matmul(qg, p_transpose(kg, 0, 1), cost), scale, cost)
            if mask is not None:
                scores = op_add(scores, mask, cost)
            probs = softmax_lastdim(scores, cost)
            outs.append(op_reshape(op_matmul(probs, vg, cost), (s, qm * d), cost))
        seq_outs.append(op_concat(outs, 1, cost))
    # stacking the sequences' outputs is free: each writes into its rows of a preallocated buffer
    attn = seq_outs[0] if b == 1 else Tensor.from_values(np.concatenate([o.data for o in seq_outs], axis=0))

    h = op_add(x, op_matmul(attn, lw.wo, cost), cost)
    n2 = rms_norm(h, lw.ffn_norm, cfg.norm_eps, cost)
    act = op_silu(op_matmul(n2, lw.w1, cost), cost)
    ff = op_matmul(op_mul(act, op_matmul(n2, lw.w3, cost), cost), lw.w2, cost)
    return op_add(h, ff, cost)


def model_forward(tokens: np.ndarray, w: ModelWeights, cfg: DenseConfig, caches: list, cost) -> Tensor:
    """tokens (b, s) int -> logits (b, vocab) of the LAST position of each sequence."""
    b, s = tokens.shape
    h = op_gather(w.embed, 0, tokens.reshape(-1), cost)
    for l, lw in enumerate(w.layers):
        h = layer_forward(h, lw, cfg, [caches[bi][l] for bi in range(b)], b, s, cost)
    if s > 1:  # keep only each sequence's last row for the LM head
        h = op_gather(h, 0, np.arange(b) * s + (s - 1), cost)
    return op_matmul(rms_norm(h, w.norm, cfg.norm_eps, cost), w.output, cost)


# --------------------------------------------------------------------------
# Literal numpy transcription of the reference (llama/model.py)
# --------------------------------------------------------------------------


def _ref_freqs_cis(head_dim: int, end: int, theta: float) -> np.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2)[: head_dim // 2].astype(np.float32) / head_dim))
    ang = np.outer(np.arange(end), freqs).astype(np.float32)
    return np.exp(1j * ang).astype(np.complex64)  # torch.polar(ones, ang)


def _ref_rope(x: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """x (b, s, h, d): view_as_complex on adjacent pairs, multiply, view_as_real."""
    xc = (x[..., 0::2] + 1j * x[..., 1::2]).astype(np.complex64)
    out = xc * fc[None, :, None, :]
    res = np.empty_like(x)
    res[..., 0::2], res[..., 1::2] = out.real, out.imag
    return res


def _ref_rope_half(x: np.ndarray, start_pos: int, theta: float) -> np.ndarray:
    """transformers' ``apply_rotary_pos_emb`` with ``rotate_half``: x (b, s, h, d)."""
    b, s, h, d = x.shape
    inv_freq = 1.0 / (theta ** (np.arange(0, d, 2).astype(np.float32) / d))
    freqs = (start_pos + np.arange(s)).astype(np.float32)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb)[None, :, None, :], np.sin(emb)[None, :, None, :]
    rot = np.concatenate([-x[..., d // 2:], x[..., : d // 2]], axis=-1)
    return (x * cos + rot * sin).astype(x.dtype)


def _ref_rms(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    xf = x.astype(np.float32)
    return (xf * (1.0 / np.sqrt((xf ** 2).mean(-1, keepdims=True) + eps))).astype(x.dtype) * weight


def reference_forward(tokens: np.ndarray, start_pos: int, ref: dict, cfg: DenseConfig, cache: dict) -> np.ndarray:
    """Reference ``Transformer.forward`` + ``Attention`` + ``FeedForward``, with its own
    preallocated cache arrays. Returns logits (b, s, vocab)."""
    b, s = tokens.shape
    nh, nkv, d, n_rep = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.q_mult
    h = ref["tok_embeddings"][tokens].astype(np.float32)
    fc = _ref_freqs_cis(d, start_pos + s, cfg.rope_theta)[start_pos: start_pos + s]
    mask = None
    if s > 1:
        causal = np.triu(np.full((s, s), -np.inf, dtype=np.float32), k=1)
        mask = np.hstack([np.zeros((s, start_pos), dtype=np.float32), causal])
    for l, lw in enumerate(ref["layers"]):
        x = _ref_rms(h, lw["attention_norm"], cfg.norm_eps)
        xq = (x @ lw["wq"].T).reshape(b, s, nh, d)
        xk = (x @ lw["wk"].T).reshape(b, s, nkv, d)
        xv = (x @ lw["wv"].T).reshape(b, s, nkv, d)
        if cfg.qk_norm:
            xq, xk = _ref_rms(xq, lw["q_norm"], cfg.norm_eps), _ref_rms(xk, lw["k_norm"], cfg.norm_eps)
        if cfg.rope_interleaved:
            xq, xk = _ref_rope(xq, fc), _ref_rope(xk, fc)
        else:
            xq, xk = _ref_rope_half(xq, start_pos, cfg.rope_theta), _ref_rope_half(xk, start_pos, cfg.rope_theta)
        cache["k"][l][:b, start_pos: start_pos + s] = xk
        cache["v"][l][:b, start_pos: start_pos + s] = xv
        keys, vals = cache["k"][l][:b, : start_pos + s], cache["v"][l][:b, : start_pos + s]
        keys, vals = np.repeat(keys, n_rep, axis=2), np.repeat(vals, n_rep, axis=2)  # repeat_kv
        scores = np.einsum("bqhd,bkhd->bhqk", xq, keys) / math.sqrt(d)
        if mask is not None:
            scores = scores + mask
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        scores = scores / scores.sum(-1, keepdims=True)
        out = np.einsum("bhqk,bkhd->bqhd", scores, vals).reshape(b, s, nh * d)
        h = h + out @ lw["wo"].T
        x = _ref_rms(h, lw["ffn_norm"], cfg.norm_eps)
        silu = lambda z: z / (1.0 + np.exp(-z))
        h = h + (silu(x @ lw["w1"].T) * (x @ lw["w3"].T)) @ lw["w2"].T
    return (_ref_rms(h, ref["norm"], cfg.norm_eps) @ ref["output"].T).astype(np.float32)


def new_reference_cache(cfg: DenseConfig, b: int, max_len: int) -> dict:
    z = lambda: [np.zeros((b, max_len, cfg.n_kv_heads, cfg.head_dim), np.float32) for _ in range(cfg.n_layers)]
    return {"k": z(), "v": z()}


# --------------------------------------------------------------------------
# COST-ONLY path: the same op sequence as layer_forward / model_forward,
# expressed as element counts and GEMM shapes (no tensors materialised).
# --------------------------------------------------------------------------


@dataclass
class Section:
    """Cycles / energy plus a per-op split. cycles = latency, energy = total."""
    cycles: float = 0.0
    energy_uj: float = 0.0
    by_op: dict = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0]))  # op -> [cycles, energy_uj]


class SectionRecorder:
    """Attributes the cost-log entries emitted inside ``with rec(name):`` to a named section."""

    def __init__(self, cost) -> None:
        self.cost = cost
        self.sections: dict = defaultdict(Section)

    @contextmanager
    def __call__(self, name: str):
        i0 = len(self.cost.log)
        yield
        sec = self.sections[name]
        for e in self.cost.log[i0:]:
            c, en = e["cycles"] * e["cycle_repeats"], e["energy_uj"] * e["energy_repeats"]
            sec.cycles += c
            sec.energy_uj += en
            sec.by_op[e["op"]][0] += c
            sec.by_op[e["op"]][1] += en


def combine(parts: list) -> Section:
    """sum of (section, cycle_multiplier, energy_multiplier)"""
    out = Section()
    for sec, kc, ke in parts:
        out.cycles += sec.cycles * kc
        out.energy_uj += sec.energy_uj * ke
        for op, (c, e) in sec.by_op.items():
            out.by_op[op][0] += c * kc
            out.by_op[op][1] += e * ke
    return out


def _view(cost, bpe: int) -> None:
    cost.op_reshape_flag(0, bpe, False)


def emit_rms_norm(cost, m: int, h: int, bpe: int) -> None:
    cost.op_mul(m * h, bpe)
    cost.op_sum(h, m, bpe)
    cost.op_mul(m, bpe)
    cost.op_add(m, bpe)
    cost.op_rsqrt(m, bpe)
    cost.op_mul(m * h, bpe)
    cost.op_mul(m * h, bpe)


def emit_rope(cost, m: int, nheads: int, d: int, bpe: int) -> None:
    half = m * nheads * d // 2
    cost.op_mul(half, bpe)
    cost.op_mul(half, bpe)
    cost.op_sub(half, bpe)
    cost.op_mul(half, bpe)
    cost.op_mul(half, bpe)
    cost.op_add(half, bpe)
    cost.op_concat(phantom((m, nheads, d), bpe))


def q_group_reshape_copied(s: int, qm: int, nkv: int) -> bool:
    """(s, qm, d) column-slice of the contiguous (s, nh, d) q, reshaped to (s*qm, d): numpy can
    only view it if s == 1, qm == 1, or the slice already spans every head (nkv == 1)."""
    return s > 1 and qm > 1 and nkv > 1


def emit_core(cost, cfg: DenseConfig, s: int, t_eff: int, has_mask: bool, bpe: int, rec) -> None:
    """ONE attention instance = one (sequence, KV head) pair."""
    qm, d = cfg.q_mult, cfg.head_dim
    rows = s * qm
    with rec("core"):
        copied = q_group_reshape_copied(s, qm, cfg.n_kv_heads)
        cost.op_reshape_flag(rows * d if copied else 0, bpe, copied)
        cost.op_gemm(rows, t_eff, d, bpe)
        cost.op_mul(rows * t_eff, bpe)
        if has_mask:
            cost.op_add(rows * t_eff, bpe)
        cost.op_max(t_eff, rows, bpe)
        cost.op_sub(rows * t_eff, bpe)
        cost.op_exp(rows * t_eff, bpe)
        cost.op_sum(t_eff, rows, bpe)
        cost.op_div(rows * t_eff, bpe)
        cost.op_gemm(rows, d, t_eff, bpe)
        _view(cost, bpe)


def emit_layer(cost, cfg: DenseConfig, b: int, s: int, t_old: int, bpe: int, rec, expand: bool = False) -> None:
    """``expand=True`` emits every (sequence, KV head) instance literally (what the numeric path
    does -- used by the sync test); ``expand=False`` emits ONE instance (the real runs multiply)."""
    nh, nkv, d, kvd, H, F = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.kv_dim, cfg.dim, cfg.ffn_hidden
    qd = cfg.q_dim
    m, window = b * s, cfg.sliding_window
    with rec("stacked"):
        emit_rms_norm(cost, m, H, bpe)
        for n_out in (qd, kvd, kvd):
            cost.op_gemm(m, n_out, H, bpe)
        if cfg.qk_norm:
            _view(cost, bpe)
            emit_rms_norm(cost, m * nh, d, bpe)
            _view(cost, bpe)
            emit_rope(cost, m, nh, d, bpe)
            _view(cost, bpe)
            emit_rms_norm(cost, m * nkv, d, bpe)
            _view(cost, bpe)
            emit_rope(cost, m, nkv, d, bpe)
            _view(cost, bpe)
        else:
            _view(cost, bpe)
            emit_rope(cost, m, nh, d, bpe)
            _view(cost, bpe)
            emit_rope(cost, m, nkv, d, bpe)
            _view(cost, bpe)
        cost.op_concat(phantom((m, kvd), bpe))
        cost.op_concat(phantom((m, kvd), bpe))
    t = t_old + s
    key_start = max(0, t_old - window + 1) if window > 0 else 0
    t_eff = t - key_start
    has_mask = s > 1  # decode (s == 1) never masks anything, with or without a window
    for _ in range(b if expand else 1):
        for _ in range(nkv if expand else 1):
            emit_core(cost, cfg, s, t_eff, has_mask, bpe, rec)
        with rec("perseq"):
            cost.op_concat(phantom((s, nh * d), bpe))
    with rec("stacked"):
        cost.op_gemm(m, H, qd, bpe)
        cost.op_add(m * H, bpe)
        emit_rms_norm(cost, m, H, bpe)
        cost.op_gemm(m, F, H, bpe)
        cost.op_silu(m * F, bpe)
        cost.op_gemm(m, F, H, bpe)
        cost.op_mul(m * F, bpe)
        cost.op_gemm(m, H, F, bpe)
        cost.op_add(m * H, bpe)


def emit_tail(cost, cfg: DenseConfig, b: int, s: int, bpe: int, sampling: bool) -> None:
    """last-row gather (prefill only) + final RMSNorm + LM head on one row per sequence (+ greedy sampling)."""
    custom = getattr(cfg, "emit_tail", None)  # another architecture (GPT-2) supplies its own tail
    if custom is not None:
        return custom(cost, b, s, bpe, sampling)
    if s > 1:
        cost.op_gather(phantom((b, cfg.dim), bpe))
    emit_rms_norm(cost, b, cfg.dim, bpe)
    cost.op_gemm(b, cfg.vocab, cfg.dim, bpe)
    if sampling:
        cost.op_max(cfg.vocab, b, bpe)


def emit_step(cost, cfg: DenseConfig, b: int, s: int, t_old: int, bpe: int, rec, *, layers: int = 1,
              expand: bool = False, sampling: bool = False, tail: bool = True) -> None:
    """embedding + ``layers`` decoder layers + LM head (+ optional greedy sampling).
    ``tail=False`` drops the LM head etc. -- a NON-final prefill chunk produces no token.
    A config that defines ``emit_step`` (GPT-2) replaces the LLaMA-style body."""
    custom = getattr(cfg, "emit_step", None)
    if custom is not None:
        return custom(cost, b, s, t_old, bpe, rec, layers=layers, expand=expand, sampling=sampling, tail=tail)
    with rec("embed"):
        cost.op_gather(phantom((b * s, cfg.dim), bpe))
    for _ in range(layers):
        emit_layer(cost, cfg, b, s, t_old, bpe, rec, expand)
    if tail:
        with rec("tail"):
            emit_tail(cost, cfg, b, s, bpe, sampling)


# --------------------------------------------------------------------------
# Aggregation: one layer's sections -> a full step, in a latency mode
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Mode:
    name: str
    moves: str  # "mesh" (data moves split over PEs) | "serial" (one PE, chunked)
    concurrent: bool  # attention instances: all at once vs one after another
    kv_overlap: bool = True  # off-chip KV traffic hidden under compute (True) or added to it (False)

    def move_tiling(self) -> Tiling:
        return Tiling() if self.moves == "mesh" else Tiling.serial()


MODES = (Mode("optimistic", "mesh", True, True), Mode("conservative", "serial", False, False))


@dataclass(frozen=True)
class OffchipKV:
    """Off-wafer DRAM channel: the KV cache and/or the model weights live there instead of PE SRAM.
    (``kv`` / ``weights`` choose which of the two this run bills; the weights are streamed in full every pass.)

    Per layer and step, attention needs every cached K/V of each sequence on the wafer, and the new tokens' K/V
    are written back: read = b * (cached tokens in the window) * 2 * kv_dim * bpe, write = b * s * 2 * kv_dim * bpe.
    Time = bytes / bandwidth + one access latency per layer; energy = bytes * pJ/byte. A mode either overlaps this
    with the step's compute (optimistic: the step takes max(compute, DRAM)) or adds it (conservative).

    This is a bandwidth model, not a layout model: where the K/V sit on the wafer while attention runs, and the
    on-wafer network traffic that delivers them to the PEs, are NOT modelled. The numbers are placeholders for a
    real DRAM/HBM part -- set them with --dram-bw-gbs / --dram-latency-ns / --dram-pj-per-byte.
    """

    bytes_per_cycle: float
    latency_cycles: float
    pj_per_byte: float
    capacity_bytes: int = 0  # 0 = unlimited ("a sufficiently large DRAM")
    bw_gbs: float = 0.0
    kv: bool = True  # the KV cache is in this DRAM
    weights: bool = False  # the model weights are in this DRAM and are re-read by every forward pass

    def with_kv(self, flag: bool) -> "OffchipKV":
        return replace(self, kv=flag)

    @classmethod
    def make(cls, bw_gbs: float, latency_ns: float, pj_per_byte: float, clock_ghz: float,
             capacity_gib: float = 0.0) -> "OffchipKV":
        if bw_gbs <= 0 or clock_ghz <= 0:
            raise ValueError("DRAM bandwidth and clock must be positive")
        return cls(bytes_per_cycle=bw_gbs / clock_ghz, latency_cycles=latency_ns * clock_ghz,
                   pj_per_byte=pj_per_byte, capacity_bytes=int(capacity_gib * 2 ** 30), bw_gbs=bw_gbs)

    @staticmethod
    def traffic(cfg, b: int, s: int, t_old: int, bpe: int) -> tuple:
        """(read_bytes, write_bytes) for ONE step over ALL layers: s new tokens per sequence on top of t_old cached."""
        window = cfg.sliding_window
        key_start = max(0, t_old - window + 1) if window > 0 else 0
        per_token = 2 * cfg.kv_dim * bpe  # K and V
        return cfg.n_layers * b * (t_old - key_start) * per_token, cfg.n_layers * b * s * per_token

    def step_bytes(self, cfg, b: int, s: int, t_old: int, bpe: int, tail: bool = True) -> int:
        total = 0
        if self.kv:
            total += sum(self.traffic(cfg, b, s, t_old, bpe))
        if self.weights:
            total += cfg.weight_stream_bytes(b * s, bpe, tail)
        return total

    def step(self, cfg, b: int, s: int, t_old: int, bpe: int, tail: bool = True) -> tuple:
        """(cycles, energy_uJ) of the step's DRAM traffic (KV and/or weights) if it ran alone."""
        if not (self.kv or self.weights):
            return 0.0, 0.0
        total = self.step_bytes(cfg, b, s, t_old, bpe, tail)
        return total / self.bytes_per_cycle + cfg.n_layers * self.latency_cycles, total * self.pj_per_byte * 1e-6

    def head_step(self, cfg, bpe: int) -> tuple:
        """(cycles, energy_uJ) of reading the LM head once (the prefill tail); zero if the weights are on the wafer."""
        if not self.weights:
            return 0.0, 0.0
        total = cfg.vocab * cfg.dim * bpe
        return total / self.bytes_per_cycle + self.latency_cycles, total * self.pj_per_byte * 1e-6


def step_totals(secs: dict, cfg: DenseConfig, b: int, concurrent: bool) -> Section:
    L, inst = cfg.n_layers, b * cfg.n_kv_heads
    kc_core = 1 if concurrent else inst
    kc_seq = 1 if concurrent else b
    g = lambda name: secs.get(name, Section())
    return combine([(g("embed"), 1, 1), (g("tail"), 1, 1), (g("stacked"), L, L),
                    (g("core"), L * kc_core, L * inst), (g("perseq"), L * kc_seq, L * b)])


def run_step(cfg: DenseConfig, b: int, s: int, t_old: int, cost, bpe: int, concurrent: bool,
             tail: bool = True, offchip: Optional[OffchipKV] = None, kv_overlap: bool = True,
             kv_exposed: bool = False) -> Section:
    """One forward pass. DRAM traffic (``offchip``) is billed on top of the compute time: with ``kv_overlap`` it is hidden
    under the compute (the step takes max(compute, DRAM)), without it the DRAM time is added. ``kv_exposed`` splits the
    traffic: the WEIGHT stream follows ``kv_overlap`` (it can be prefetched, layer N+1's weights arrive while layer N
    computes) but the KV-cache traffic is always ADDED (attention cannot start before its K/V arrive)."""
    rec = SectionRecorder(cost)
    emit_step(cost, cfg, b, s, t_old, bpe, rec, layers=1, expand=False, sampling=True, tail=tail)
    out = step_totals(dict(rec.sections), cfg, b, concurrent)
    if offchip is not None:  # data in DRAM: bill the transfers
        if kv_exposed:
            parts = []
            if offchip.weights:
                parts.append((replace(offchip, kv=False), "dram", kv_overlap))
            if offchip.kv:
                parts.append((replace(offchip, weights=False), "kv_dram", False))
        else:  # legacy: one DRAM stream, "dram" = KV and/or weights
            parts = [(offchip, "dram" if offchip.weights else "kv_dram", kv_overlap)]
        compute = out.cycles
        for part, bucket, hidden in parts:
            dram_cycles, dram_uj = part.step(cfg, b, s, t_old, bpe, tail)
            exposed = max(0.0, dram_cycles - compute) if hidden else dram_cycles
            out.cycles += exposed
            out.energy_uj += dram_uj
            out.by_op[bucket][0] += exposed
            out.by_op[bucket][1] += dram_uj
    cost.peak_per_pe = max(getattr(cost, "peak_per_pe", 0), max((e.get("per_pe", 0) for e in cost.log), default=0))
    cost.log.clear()
    return out


def run_tail(cfg: DenseConfig, b: int, s: int, cost, bpe: int, offchip: Optional[OffchipKV] = None,
             kv_overlap: bool = True) -> Section:
    """Cost of the LM head / sampling tail alone (once per prefill, after the last chunk). With the weights in DRAM
    the whole LM head is read once here (the prefill passes themselves are costed with tail=False)."""
    rec = SectionRecorder(cost)
    with rec("tail"):
        emit_tail(cost, cfg, b, s, bpe, True)
    out = combine([(rec.sections["tail"], 1, 1)])
    if offchip is not None and offchip.weights:
        dram_cycles, dram_uj = offchip.head_step(cfg, bpe)
        exposed = max(0.0, dram_cycles - out.cycles) if kv_overlap else dram_cycles
        out.cycles += exposed
        out.energy_uj += dram_uj
        out.by_op["dram"][0] += exposed
        out.by_op["dram"][1] += dram_uj
    cost.log.clear()
    return out


# --------------------------------------------------------------------------
# Chunked prefill: choose a chunk size the machine can actually tile
# --------------------------------------------------------------------------


class FitProbe:
    """Stands in for a cost tracker: answers 'does every GEMM of this step have a feasible K tile on this wafer?'
    with the SAME rule the real GEMM search applies (tile_m/tile_n from the largest divisors of m,n that fit the
    mesh; tile_k must divide K and the single-buffered tiles must fit the PE SRAM). Every other op is ignored
    (elementwise / data-move ops are tiled by wafersim.tiling and never hit this limit)."""

    def __init__(self, wafer) -> None:
        self.wafer, self.bad = wafer, []

    def op_gemm(self, m: int, n: int, k: int, bpe: int) -> None:
        g = GemmSpec(m=m, n=n, k=k, bytes_per_element=bpe)
        ar = _largest_divisor_leq(m, self.wafer.mesh_rows)
        ac = _largest_divisor_leq(n, self.wafer.mesh_cols)
        try:
            feasible_tile_k_candidates(g, self.wafer, m // ar, n // ac)
        except ValueError:
            self.bad.append((m, n, k))

    def __getattr__(self, name: str):
        if name.startswith("op_"):
            return lambda *a, **kw: None
        raise AttributeError(name)


def _null_rec(name: str):
    return nullcontext()


def step_fits(cfg: DenseConfig, b: int, s: int, t_old: int, wafer, bpe: int, tail: bool = True) -> list:
    """GEMM shapes of one step that have NO feasible tile on ``wafer`` (empty list = the step fits)."""
    probe = FitProbe(wafer)
    emit_step(probe, cfg, b, s, t_old, bpe, _null_rec, layers=1, expand=False, sampling=True, tail=tail)
    return sorted(set(probe.bad))


def chunk_starts(n_in: int, chunk: int) -> tuple:
    """Chunk start offsets that get costed: (first, middle, last) -- or the distinct ones if there are fewer."""
    n = -(-n_in // chunk)
    return tuple(sorted({0, ((n - 1) // 2) * chunk, (n - 1) * chunk}))


def prefill_chunk_size(cfg: DenseConfig, b: int, n_in: int, wafer, bpe: int) -> int:
    """Largest chunk C dividing n_in such that, at every costed chunk start, all GEMMs fit the PE SRAM.
    (Larger C = fewer passes over the weights.) The tail (LM head, b rows) is checked with the last chunk."""
    cands = [c for c in range(n_in, 0, -1) if n_in % c == 0]
    for c in cands:
        starts = chunk_starts(n_in, c)
        if all(not step_fits(cfg, b, c, t0, wafer, bpe, tail=(t0 == starts[-1])) for t0 in starts):
            return c
    raise ValueError(f"{cfg.name} batch {b}: not even a 1-token prefill chunk has tileable GEMMs on this wafer")


def run_prefill(cfg: DenseConfig, b: int, n_in: int, chunk: int, cost, bpe: int, concurrent: bool,
                offchip: Optional[OffchipKV] = None, kv_overlap: bool = True, kv_exposed: bool = False) -> Section:
    """Prefill as n_in/chunk passes of ``chunk`` tokens each (the KV cache grows between passes).
    Pass cost depends on position only through attention (~linear in cache length), so the n passes are
    costed at the first/middle/last chunk and Simpson-averaged; the LM head runs once, after the last pass."""
    if n_in % chunk:
        raise ValueError("prefill chunk must divide the input length")
    n = n_in // chunk
    pts = chunk_starts(n_in, chunk)
    secs = [run_step(cfg, b, chunk, t0, cost, bpe, concurrent, tail=False, offchip=offchip, kv_overlap=kv_overlap,
                     kv_exposed=kv_exposed) for t0 in pts]
    if len(secs) == 3:
        mean = simpson_mean(*secs)
    else:  # 1 or 2 distinct points: plain average
        mean = combine([(sec, 1 / len(secs), 1 / len(secs)) for sec in secs])
    return combine([(mean, n, n), (run_tail(cfg, b, chunk, cost, bpe, offchip, kv_overlap), 1, 1)])


def decode_points(n_in: int, n_out: int) -> tuple:
    """Cache lengths before the n_out-1 decode steps run from n_in to n_in+n_out-2: (first, mid, last)."""
    a, z = n_in, n_in + n_out - 2
    return a, (a + z) // 2, z


def simpson_mean(f_a: Section, f_m: Section, f_z: Section) -> Section:
    return combine([(f_a, 1 / 6, 1 / 6), (f_m, 4 / 6, 4 / 6), (f_z, 1 / 6, 1 / 6)])


def metrics(prefill: Section, decode_mean: Section, b: int, n_in: int, n_out: int, clock_ghz: float) -> dict:
    """The paper's definition: throughput = batch * (input + output) / end-to-end latency."""
    hz = clock_ghz * 1e9
    ttft, itl = prefill.cycles / hz, decode_mean.cycles / hz
    e2e = ttft + (n_out - 1) * itl
    energy_j = (prefill.energy_uj + (n_out - 1) * decode_mean.energy_uj) * 1e-6
    return {"batch": b, "ttft_s": ttft, "itl_s": itl, "e2e_s": e2e,
            "paper_throughput_tok_s": b * (n_in + n_out) / e2e,
            "output_throughput_tok_s": b * n_out / e2e,
            "decode_throughput_tok_s": b / itl,
            "energy_j": energy_j}


# --------------------------------------------------------------------------
# Paper reference values (LLM-Inference-Bench, Fig. 25: input/output 1024, fp16,
# peak over batch size, read off a BAR CHART -- about +-5-10%).
# --------------------------------------------------------------------------

PAPER_LLAMA2_7B_PEAK = {
    "1x MI250 (vLLM)": 2300, "1x MI300X (vLLM)": 2900, "1x A100 (TRT-LLM)": 1400,
    "1x Gaudi2 (DeepSpeed)": 2200, "1x H100 (TRT-LLM)": 4000, "1x GH200 (TRT-LLM)": 6100,
    "8x SN40L (SambaFlow)": 4100,
}
PAPER_LLAMA3_8B_PEAK = {
    "1x MI250 (vLLM)": 2400, "1x MI300X (vLLM)": 2100, "1x A100 (TRT-LLM)": 3900,
    "1x Gaudi2 (DeepSpeed)": 3900, "1x H100 (TRT-LLM)": 7500, "1x GH200 (TRT-LLM)": 10500,
    "8x SN40L (SambaFlow)": 11300,
}
# key -> (config, output-dir slug, paper reference values)
MODELS = {
    "llama2-7b": (LLAMA2_7B, "llama2_7b", PAPER_LLAMA2_7B_PEAK),
    "llama3-8b": (LLAMA3_8B, "llama3_8b", PAPER_LLAMA3_8B_PEAK),
}

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def numeric_check(cfgs: tuple = None, title: str = "llama/model.py") -> None:
    print("=" * 78)
    print(f"STEP 1: NUMERIC check vs. a literal numpy transcription of {title} (toy dims)")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    for cfg in (cfgs or (TOY_MHA, TOY_GQA, TOY_GQA_L3)):
        ref = random_reference_weights(cfg, rng)
        w = ModelWeights.from_reference(ref, cfg)
        b, pre, steps = 2, 5, 3
        toks = rng.integers(0, cfg.vocab, size=(b, pre + steps))
        rc = new_reference_cache(cfg, b, pre + steps)
        caches = new_caches(cfg, b)
        worst = 0.0
        got = model_forward(toks[:, :pre], w, cfg, caches, None).data
        worst = max(worst, float(np.abs(got - reference_forward(toks[:, :pre], 0, ref, cfg, rc)[:, -1]).max()))
        for i in range(steps):
            pos = pre + i
            got = model_forward(toks[:, pos:pos + 1], w, cfg, caches, None).data
            worst = max(worst, float(np.abs(got - reference_forward(toks[:, pos:pos + 1], pos, ref, cfg, rc)[:, -1]).max()))
        print(f"{cfg.name:<9} (heads {cfg.n_heads}, kv {cfg.n_kv_heads}) batch 2, prefill 5 + decode 3: max |err| = {worst:.2e}")
        if worst > 1e-4:
            raise RuntimeError(f"{cfg.name}: forward disagrees with the reference -- composition bug")
    print("numeric check PASSED")


def kv_peak_bytes(cfg, n_in: int, n_out: int, b: int, bpe: int = COST_BPE) -> int:
    """KV cache of ``b`` sequences at its peak length n_in + n_out - 1 (the last generated token is never fed back)."""
    return b * (n_in + n_out - 1) * cfg.kv_bytes_per_token(bpe)


def choose_kv_placement(cfg, wafer, n_in: int, n_out: int, b: int, bpe: int, policy: str,
                        weights_offchip: bool, sram_fraction: float = 0.5) -> str:
    """'sram' or 'dram' for one batch. ``policy`` 'sram' / 'dram' force it; 'auto' keeps the KV cache on the wafer only if
    its per-PE share (spread evenly over all PEs) is at most ``sram_fraction`` of the data bytes the PE has left after
    the FIFO+router reservation (and after its share of the weights, if those are on-wafer). The rest of the SRAM must
    stay free for the GEMM tiles."""
    if policy in ("sram", "dram"):
        return policy
    if policy != "auto":
        raise ValueError(f"unknown KV placement {policy!r}")
    pes = wafer.mesh_rows * wafer.mesh_cols
    data_pe = wafer.sram_bytes_per_pe - wafer.fifo_router_bytes_per_pe
    w_pe = 0.0 if weights_offchip else cfg.n_params() * bpe / pes
    kv_pe = kv_peak_bytes(cfg, n_in, n_out, b, bpe) / pes
    return "sram" if kv_pe <= sram_fraction * max(0.0, data_pe - w_pe) else "dram"


def kv_carve_kib(cfg, wafer, n_in: int, n_out: int, b: int, bpe: int = COST_BPE) -> int:
    """KiB of every PE's SRAM taken by an on-wafer KV cache (even spread over all PEs, rounded up). The GEMM tiler then
    only gets what is left, so on-chip KV shrinks the K tiles and costs cycles. (Network cost of reaching the K/V: NOT modelled.)"""
    pes = wafer.mesh_rows * wafer.mesh_cols
    per_pe = -(-kv_peak_bytes(cfg, n_in, n_out, b, bpe) // pes)
    return -(-per_pe // 1024)


def weights_carve_kib(cfg, wafer, bpe: int = COST_BPE) -> int:
    """KiB of every PE's SRAM taken by weights that live ON the wafer (even spread over all PEs, rounded up). Like an
    on-chip KV cache, resident weights shrink what the GEMM tiler may use for its A/B/C tiles. (Network cost of
    distributing the weights to the PEs that need them: NOT modelled.)"""
    pes = wafer.mesh_rows * wafer.mesh_cols
    per_pe = -(-cfg.n_params() * bpe // pes)
    return -(-per_pe // 1024)


def capacity_report(cfg, wafer, n_in: int, n_out: int, batches, bpe: int = COST_BPE,
                    kv_offchip=False, dram_capacity_bytes: int = 0, weights_offchip: bool = False) -> dict:
    """Does the working set fit ON the wafer? Weights + KV cache against the SRAM, in bytes.

    The cost model never charges for storing or fetching data that is assumed to be on the wafer, so a run whose
    on-wafer footprint exceeds the SRAM is only meaningful as "compute time if the data were free". This report says
    which case a run is in. Two capacities are given:
      total  = every PE's whole SRAM (mesh PEs x sram_bytes_per_pe);
      data   = the same minus the FIFO + router bytes the project reserves out of each PE's SRAM
               (wafer.fifo_router_bytes_per_pe), i.e. the budget the GEMM tiler is allowed to fill.
    ``kv_offchip`` is a bool or {batch: bool}: where each batch's KV cache lives. ``weights_offchip``: the weights are in
    DRAM too. Whatever is off-chip is not counted against the SRAM; it is counted against ``dram_capacity_bytes``
    (0 = unlimited). The KV cache peaks at n_in + n_out - 1 tokens. Every layer keeps the full cache (no sliding-window saving).
    """
    pes = wafer.mesh_rows * wafer.mesh_cols
    total = pes * wafer.sram_bytes_per_pe
    data = pes * (wafer.sram_bytes_per_pe - wafer.fifo_router_bytes_per_pe)
    weights = cfg.n_params() * bpe
    w_on = 0 if weights_offchip else weights
    kv_tok = cfg.kv_bytes_per_token(bpe)
    peak_len = n_in + n_out - 1
    rows = []
    for b in batches:
        kv = kv_peak_bytes(cfg, n_in, n_out, b, bpe)
        kv_off = kv_offchip.get(b, False) if isinstance(kv_offchip, dict) else bool(kv_offchip)
        need = w_on + (0 if kv_off else kv)
        dram_need = (weights if weights_offchip else 0) + (kv if kv_off else 0)
        rows.append({"batch": b, "kv_bytes": kv, "kv_offchip": kv_off, "need_bytes": need,
                     "kv_in_dram_bytes": kv if kv_off else 0, "dram_need_bytes": dram_need,
                     "fits_dram": dram_capacity_bytes == 0 or dram_need <= dram_capacity_bytes,
                     "need_over_total_sram": need / total, "need_over_data_sram": need / data,
                     "fits_total_sram": need <= total, "fits_data_sram": need <= data})
    return {"kv_offchip": any(r["kv_offchip"] for r in rows), "weights_offchip": weights_offchip,
            "dram_capacity_bytes": dram_capacity_bytes, "bytes_per_elem": bpe, "pes": pes,
            "sram_bytes_per_pe": wafer.sram_bytes_per_pe,
            "fifo_router_bytes_per_pe": wafer.fifo_router_bytes_per_pe,
            "total_sram_bytes": total, "data_sram_bytes": data,
            "weights_bytes": weights, "weights_over_total_sram": weights / total,
            "kv_bytes_per_token": kv_tok, "peak_cache_tokens": peak_len,
            "kv_tokens_that_fit_total_after_weights": max(0, (total - w_on) // kv_tok),
            "kv_tokens_that_fit_data_after_weights": max(0, (data - w_on) // kv_tok),
            "per_batch": rows,
            "all_fit_total_sram": all(r["fits_total_sram"] for r in rows),
            "all_fit_data_sram": all(r["fits_data_sram"] for r in rows),
            "all_fit_dram": all(r["fits_dram"] for r in rows)}


def print_capacity(cfg, rep: dict) -> None:
    mib, gib = 2 ** 20, 2 ** 30
    woff = rep.get("weights_offchip", False)
    print(f"{cfg.name}: {cfg.n_params() / 1e9:.3f} B params -> weights {rep['weights_bytes'] / gib:.3f} GiB "
          f"at {rep['bytes_per_elem']} B/elem; KV cache {rep['kv_bytes_per_token'] / 1024:.0f} KiB/token")
    print(f"on-wafer SRAM: {rep['pes']:,} PEs x {rep['sram_bytes_per_pe']} B = {rep['total_sram_bytes'] / mib:.1f} MiB total; "
          f"{rep['data_sram_bytes'] / mib:.1f} MiB left for data after the {rep['fifo_router_bytes_per_pe']} B/PE "
          f"FIFO+router reservation")
    print(f"  weights: {'in DRAM (streamed every pass)' if woff else 'on the wafer'}; "
          f"they are {rep['weights_over_total_sram']:.2f}x the total SRAM")
    anykv = rep["kv_offchip"]
    print(f"  {'batch':>5} {'KV cache':>12} {'KV in':>6} {'on wafer':>13} {'x total SRAM':>13} {'x data SRAM':>12}  "
          f"fits on wafer?" + ("   DRAM need   fits DRAM?" if (anykv or woff) else ""))
    for r in rep["per_batch"]:
        verdict = "yes" if r["fits_total_sram"] and r["fits_data_sram"] else "NO"
        dram = (f"   {r['dram_need_bytes'] / gib:>7.2f} GiB   " + ("yes" if r["fits_dram"] else "NO")) if (anykv or woff) else ""
        print(f"  {r['batch']:>5} {r['kv_bytes'] / gib:>9.3f} GiB {'DRAM' if r['kv_offchip'] else 'SRAM':>6} "
              f"{r['need_bytes'] / gib:>9.3f} GiB {r['need_over_total_sram']:>13.2f} {r['need_over_data_sram']:>12.2f}  {verdict:<13}{dram}")
    if not rep["all_fit_total_sram"]:
        print("  WARNING: the on-wafer working set does not fit in SRAM for every batch above. The cost model does not\n"
              "  charge for fetching data that lives off-wafer unless it is declared DRAM (weights-dram / KV in DRAM),\n"
              "  so the cycles below are 'compute time if the data were already on the wafer', an upper bound on\n"
              "  throughput -- not a statement that the model fits.")
    elif not rep["all_fit_data_sram"]:
        print("  NOTE: fits in total SRAM but not in the data budget left after FIFO+router; tiles may not all fit.")
    if not rep.get("all_fit_dram", True):
        print("  WARNING: the DRAM capacity given is smaller than the weights + KV placed there for some batch.")


def print_footprint(cfg, wafer, n_in: int, n_out: int, batches, kv_offchip=False,
                    dram_capacity_bytes: int = 0, weights_offchip: bool = False) -> dict:
    """Capacity check + printout; returns the report (it goes into summary.json)."""
    rep = capacity_report(cfg, wafer, n_in, n_out, batches, kv_offchip=kv_offchip,
                          dram_capacity_bytes=dram_capacity_bytes, weights_offchip=weights_offchip)
    print_capacity(cfg, rep)
    return rep


def _wafer_key() -> dict:
    return {"area_um2": FIXED_AREA_BUDGET_UM2,
            "fractions": [AREA_COMPUTE_FRACTION, AREA_MEMORY_FRACTION, AREA_NETWORK_FRACTION],
            "rep_gemm": list(REP_GEMM), "derive_bpe": WAFER_DERIVE_BPE}


def make_wafer(dry_run: bool, shared_root: Path, derive: Callable = derive_wafer_spec_from_area_budget):
    """Dry run: the placeholder WaferSpec (no derivation). Real run: the machine derived from the area budget,
    cached in ``shared_root/wafer_derived.json`` so every model (and every re-run) uses the identical wafer
    and the multi-hour-scale Docker search runs once."""
    if dry_run:
        return WaferSpec()
    cache, key = shared_root / "wafer_derived.json", _wafer_key()
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("key") == key:
            print(f"using the cached derived wafer {saved['wafer']} from {cache}")
            return replace(WaferSpec(), **saved["wafer"])
        print(f"{cache} was derived under different settings -- re-deriving")
    budget = AreaBudget(wafer_area_budget_um2=FIXED_AREA_BUDGET_UM2, compute_area_fraction=AREA_COMPUTE_FRACTION,
                        memory_area_fraction=AREA_MEMORY_FRACTION, network_area_fraction=AREA_NETWORK_FRACTION)
    rep = GemmSpec(m=REP_GEMM[0], n=REP_GEMM[1], k=REP_GEMM[2], bytes_per_element=WAFER_DERIVE_BPE)
    print(f"deriving WaferSpec from area budget {FIXED_AREA_BUDGET_UM2:,.0f} um^2 "
          f"(representative GEMM {REP_GEMM}, {WAFER_DERIVE_BPE}-byte MAC) ...")
    w = derive(budget, rep, WaferSpec(), shared_root / "area_budget_derivation_fp32").wafer
    fields = {k: getattr(w, k) for k in ("mesh_rows", "mesh_cols", "sram_kib_per_pe", "fifo_capacity_packets")}
    shared_root.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "wafer": fields}, indent=2))
    print(f"derived wafer: {fields} (saved to {cache})")
    return replace(WaferSpec(), **fields)


PINNED_WAFER = {"mesh_rows": 10, "mesh_cols": 1280, "sram_kib_per_pe": 13, "fifo_capacity_packets": 5}


def fit_wafer(wafer, dry_run: bool, shared_root: Path):
    """The wafer used to check which GEMMs can be tiled. A real run uses the derived wafer itself. A dry run
    has only the placeholder WaferSpec (32x32, 48 KiB -- tiles everything), which would hide the very failure
    we are guarding against, so it uses the cached derived wafer, else the pinned one from REP_GEMM."""
    if not dry_run:
        return wafer
    cache = shared_root / "wafer_derived.json"
    fields = json.loads(cache.read_text())["wafer"] if cache.exists() else PINNED_WAFER
    return replace(WaferSpec(), **fields)


def parse_mesh(text: str) -> tuple:
    """'100x128' / '100X128' / '100,128' -> (100, 128). Both must be positive integers."""
    parts = text.lower().replace(",", "x").split("x")
    try:
        rows, cols = (int(p) for p in parts)
    except ValueError:
        raise SystemExit(f"--mesh expects ROWSxCOLS, e.g. 100x128 (got {text!r})")
    if rows < 1 or cols < 1:
        raise SystemExit(f"--mesh needs positive rows and cols (got {text!r})")
    return rows, cols


def apply_wafer_overrides(wafer, args):
    """``--sram-kib N`` fixes every PE's SRAM at N KiB on top of the derived / pinned mesh (e.g. 48)."""
    kib = getattr(args, "sram_kib", 0)
    wafer = replace(wafer, sram_kib_per_pe=kib) if kib else wafer
    mesh = getattr(args, "mesh", "")
    if mesh:
        rows, cols = parse_mesh(mesh)
        wafer = replace(wafer, mesh_rows=rows, mesh_cols=cols)
    lanes, rbw = getattr(args, "mac_lanes", 1), getattr(args, "sram_read_bytes", 0)
    return replace(wafer, mac_lanes=lanes, sram_read_bytes_per_cycle=rbw) if (lanes != 1 or rbw) else wafer


def print_dram_summary(cfg, off: OffchipKV, batches, n_in: int, n_out: int, pts: tuple, bpe: int, clock_ghz: float,
                       placements: Optional[dict] = None) -> dict:
    """What the DRAM traffic costs on its own (pure arithmetic: bytes / bandwidth), shown even in a dry run.
    ``placements`` = {batch: 'sram'|'dram'}: where each batch's KV cache lives (default: in DRAM iff ``off.kv``)."""
    gib, hz = 2 ** 30, clock_ghz * 1e9
    cap = f"{off.capacity_bytes / gib:.0f} GiB" if off.capacity_bytes else "unlimited"
    print(f"DRAM: {off.bw_gbs:g} GB/s, {off.latency_cycles / clock_ghz:g} ns/access (per layer), "
          f"{off.pj_per_byte:g} pJ/byte, capacity {cap}; weights {'STREAMED from DRAM each pass' if off.weights else 'not billed'}")
    w_pass = cfg.weight_stream_bytes(1, bpe, True)
    out = {"bw_gbs": off.bw_gbs, "pj_per_byte": off.pj_per_byte, "weights_in_dram": off.weights, "per_batch": {}}
    print(f"  {'batch':>5} {'KV in':>6} {'KV in DRAM':>11} {'prefill KV write':>17} "
          + " ".join(f"{'DRAM ms/step @' + str(t):>20}" for t in pts))
    for b in batches:
        place = (placements or {}).get(b, "dram" if off.kv else "sram")
        ob = off.with_kv(place == "dram")
        kv_total = kv_peak_bytes(cfg, n_in, n_out, b, bpe) if place == "dram" else 0
        wr = ob.traffic(cfg, b, n_in, 0, bpe)[1] if place == "dram" else 0
        ms = [ob.step(cfg, b, 1, t, bpe)[0] / hz * 1e3 for t in pts]
        print(f"  {b:>5} {place.upper():>6} {kv_total / gib:>8.3f} GiB {wr / gib:>14.3f} GiB " + " ".join(f"{x:>20.3f}" for x in ms))
        out["per_batch"][str(b)] = {"kv_placement": place, "kv_in_dram_gib": kv_total / gib,
                                    "decode_dram_ms": dict(zip(map(str, pts), ms))}
    w = cfg.n_params() * bpe
    t_w = w_pass / off.bytes_per_cycle / hz * 1e3
    if off.weights:
        print(f"  (every decode step reads {w_pass / gib:.3f} GiB of weights = {t_w:.3f} ms at this bandwidth: "
              f"the memory-bound floor, <= {1e3 / t_w:,.0f} steps/s per sequence stream, before any KV traffic.)")
    else:
        print(f"  (weights are NOT charged. For scale: streaming all {w / gib:.1f} GiB of weights once per decode step at this "
              f"bandwidth would take {w / off.bytes_per_cycle / hz * 1e3:.1f} ms per step.)")
    out["weights_stream_ms_per_step"] = t_w if off.weights else w / off.bytes_per_cycle / hz * 1e3
    return out


def check_decode_fits(cfg: DenseConfig, b: int, pts: tuple, fw, bpe: int) -> None:
    for t0 in pts:
        bad = step_fits(cfg, b, 1, t0, fw, bpe)
        if bad:
            raise SystemExit(f"{cfg.name} batch {b}: decode step at cache length {t0} has GEMMs (m,n,k) {bad} with no "
                             f"feasible K tile on the {fw.mesh_rows}x{fw.mesh_cols} / {fw.sram_kib_per_pe} KiB wafer")


def fmt_by_op(sec: Section, top: int = 6) -> str:
    tot = sec.cycles or 1.0
    items = sorted(sec.by_op.items(), key=lambda kv: -kv[1][0])[:top]
    return ", ".join(f"{op} {c / tot:.0%}" for op, (c, _) in items)


def build_parser(models: list, default_model: str, doc: str, in_len: int = PAPER_IN_OUT,
                 out_len: int = PAPER_IN_OUT) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--model", default=default_model, choices=sorted(models), help="which model to cost")
    ap.add_argument("--dry-run", action="store_true", help="no Docker: count unique op shapes, costs are zero")
    ap.add_argument("--skip-numeric", action="store_true")
    ap.add_argument("--batches", default=",".join(map(str, PAPER_BATCHES)))
    ap.add_argument("--in-len", type=int, default=in_len)
    ap.add_argument("--out-len", type=int, default=out_len)
    ap.add_argument("--modes", default="optimistic,conservative")
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="prefill tokens per pass (must divide --in-len); 0 = largest the wafer can tile")
    ap.add_argument("--sram-kib", type=int, default=0,
                    help="fix every PE's SRAM at this many KiB (0 = keep the derived/pinned wafer's 13 KiB); e.g. 48")
    ap.add_argument("--max-tile-k-candidates", type=int, default=0,
                    help="evaluate only the N largest SRAM-feasible K tiles per GEMM (0 = all). Every candidate is a Docker "
                         "run plus a Python PE simulation that scales with PEs x K-phases, so big SRAM budgets need a cap")
    ap.add_argument("--kv-dram", action="store_true", help="keep the KV cache in off-wafer DRAM and bill its traffic")
    ap.add_argument("--kv-placement", default="", choices=["", "auto", "sram", "dram"],
                    help="per-batch KV placement: sram = on the wafer (its share is carved out of every PE's SRAM, so the GEMM "
                         "tiles get less), dram = off-chip, auto = sram if its per-PE share is <= --kv-sram-fraction of the "
                         "free data bytes else dram. Empty (default) = legacy: --kv-dram decides, no SRAM is carved out")
    ap.add_argument("--kv-sram-fraction", type=float, default=0.5,
                    help="auto placement: max share of a PE's free data bytes the KV cache may take (rest stays for tiles)")
    ap.add_argument("--kv-hidden", action="store_true",
                    help="OLD behaviour: let off-chip KV traffic overlap with compute in the optimistic mode (step = max(compute, "
                         "DRAM)). Default: the KV traffic is ADDED to the step time in every mode (attention waits for its K/V); "
                         "the weight stream still overlaps with compute in the optimistic mode")
    ap.add_argument("--weights-dram", action="store_true",
                    help="the weights live in off-wafer DRAM and are streamed in full on every forward pass (shares the "
                         "--dram-* bandwidth with any off-chip KV)")
    ap.add_argument("--carve-weights", action="store_true",
                    help="the weights are resident in the PEs' SRAM (use with no --weights-dram): every PE's share is taken out "
                         "of the SRAM the GEMM tiler may use, together with any on-chip KV cache (implies --kv-placement auto)")
    ap.add_argument("--mac-lanes", type=int, default=1,
                    help="MACs per PE per cycle. 1 = TimeLoop-measured scalar MAC (default); 8 = IDEAL 8-way SIMD what-if "
                         "(Cerebras WSE-3), applied to each PE's compute time in the SUMMA simulation, not measured by TimeLoop")
    ap.add_argument("--sram-read-bytes", type=int, default=0,
                    help="bytes per cycle a PE's SRAM can feed the B (weight) operand; 0 = unlimited. 16 exactly feeds 8 fp16 "
                         "lanes at batch 1 (GEMV); less makes the weight read the floor of each PE's phase time")
    ap.add_argument("--mesh", default="",
                    help="override the PE mesh shape, ROWSxCOLS (e.g. 100x128). Default: the cached derived machine, 10x1280 = 12,800 PEs. "
                         "The shape is not a hardware constraint (it came from the representative GEMM's divisors); keep ROWS*COLS at the "
                         "PE count you want to compare. Use a different --tag per shape so the trace folders do not mix")
    ap.add_argument("--tag", default="", help="suffix for the output folder, so variants do not overwrite each other")
    ap.add_argument("--dram-bw-gbs", type=float, default=1000.0, help="DRAM bandwidth, GB/s (placeholder: HBM-class)")
    ap.add_argument("--dram-latency-ns", type=float, default=100.0, help="DRAM access latency per layer, ns")
    ap.add_argument("--dram-pj-per-byte", type=float, default=40.0, help="DRAM access energy, pJ/byte (~5 pJ/bit)")
    ap.add_argument("--dram-capacity-gib", type=float, default=0.0, help="DRAM capacity for the KV cache; 0 = unlimited")
    return ap


def main() -> None:
    args = build_parser(list(MODELS), "llama2-7b", __doc__.splitlines()[0]).parse_args()
    cfg, slug, paper_peak = MODELS[args.model]
    run_model(args, cfg, slug, paper_peak, numeric_check)


def run_model(args, cfg, slug: str, paper_peak: dict, numeric_fn: Callable, out_root_base: Path = OUT_ROOT,
              note: str = "") -> None:
    out_root = out_root_base / (slug + (f"_{args.tag}" if getattr(args, "tag", "") else ""))  # per-model dir: runs never share trace folders
    batches = [int(x) for x in args.batches.split(",")]
    modes = [m for m in MODES if m.name in args.modes.split(",")]
    n_in, n_out = args.in_len, args.out_len
    if n_out < 2:
        raise SystemExit("--out-len must be >= 2 (first token comes from prefill, the rest from decode steps)")
    max_pos = getattr(cfg, "max_positions", None)
    if max_pos is not None and n_in + n_out > max_pos:
        raise SystemExit(f"{cfg.name} has {max_pos} learned positions; input {n_in} + output {n_out} does not fit")

    if not args.skip_numeric:
        numeric_fn()
    print()
    print("=" * 78)
    print(f"STEP 2: THROUGHPUT of {cfg.name}  (input {n_in} / output {n_out}, batches {batches}"
          f"{', DRY RUN: costs are zero' if args.dry_run else ', real Docker/TimeLoop'})")
    print("=" * 78)
    wafer = apply_wafer_overrides(make_wafer(args.dry_run, OUT_ROOT), args)
    clock = wafer.clock_ghz
    fw = apply_wafer_overrides(fit_wafer(wafer, args.dry_run, OUT_ROOT), args)
    weights_dram = getattr(args, "weights_dram", False)
    policy = getattr(args, "kv_placement", "") or ""
    carve_weights = bool(getattr(args, "carve_weights", False)) and not weights_dram
    if carve_weights and not policy:
        policy = "auto"
    kv_exposed = not getattr(args, "kv_hidden", False)
    carve = bool(policy)  # explicit placement: an on-chip KV cache takes its share of SRAM away from the GEMM tiles
    placements = {b: (choose_kv_placement(cfg, fw, n_in, n_out, b, COST_BPE, policy, weights_dram,
                                          getattr(args, "kv_sram_fraction", 0.5))
                      if policy else ("dram" if getattr(args, "kv_dram", False) else "sram")) for b in batches}
    uses_dram = weights_dram or "dram" in placements.values()
    offchip = (replace(OffchipKV.make(args.dram_bw_gbs, args.dram_latency_ns, args.dram_pj_per_byte, clock, args.dram_capacity_gib),
                       weights=weights_dram) if uses_dram else None)
    capacity = print_footprint(cfg, fw, n_in, n_out, batches, kv_offchip={b: p == "dram" for b, p in placements.items()},
                               dram_capacity_bytes=offchip.capacity_bytes if offchip else 0,
                               weights_offchip=weights_dram)  # fw: real wafer / pinned one in a dry run
    pts = decode_points(n_in, n_out)
    dram_info = (print_dram_summary(cfg, offchip, batches, n_in, n_out, pts, COST_BPE, clock, placements)
                 if offchip else None)
    w_carve = weights_carve_kib(cfg, fw) if carve_weights else 0
    carve_kib = {b: (kv_carve_kib(cfg, fw, n_in, n_out, b) if carve and placements[b] == "sram" else 0) + w_carve for b in batches}
    if offchip is not None and "dram" in placements.values():
        print("KV traffic in DRAM is " + ("ADDED to every step's latency (use --kv-hidden for the old overlapped model)"
              if kv_exposed else "HIDDEN under compute in the optimistic mode (--kv-hidden)"))
    if carve:
        print("KV placement per batch: " + ", ".join(f"b{b}={placements[b].upper()}"
              + (f" (-{carve_kib[b]} KiB/PE for tiles)" if carve_kib[b] else "") for b in batches))
    if carve_weights:
        print(f"WEIGHTS resident in SRAM: {w_carve} KiB/PE ({cfg.n_params() * COST_BPE / 2 ** 20:.0f} MiB over {fw.mesh_rows * fw.mesh_cols:,} PEs); "
              f"free data bytes per PE {(fw.sram_bytes_per_pe - fw.fifo_router_bytes_per_pe) / 1024:.2f} KiB")
    if getattr(fw, "mac_lanes", 1) != 1 or getattr(fw, "sram_read_bytes_per_cycle", 0):
        print(f"PE compute: {fw.mac_lanes} MAC lane(s)/cycle (IDEAL what-if on top of TimeLoop's 1 cycle/MAC), SRAM B-read "
              f"{fw.sram_read_bytes_per_cycle or 'unlimited'} B/cycle")
    print(f"tiling check wafer: {fw.mesh_rows}x{fw.mesh_cols} PEs, {fw.sram_kib_per_pe} KiB/PE")
    if getattr(args, "mesh", ""):
        print(f"MESH OVERRIDE: {fw.mesh_rows}x{fw.mesh_cols} = {fw.mesh_rows * fw.mesh_cols:,} PEs (derived machine was {PINNED_WAFER['mesh_rows']}x{PINNED_WAFER['mesh_cols']})")
    shared_by_kib: dict = {}  # one memo per effective SRAM size: cached GEMM costs depend on the wafer they were run on
    results: dict = {}
    t_start = time.time()
    for mode in modes:
        trackers: dict = {}

        def tracker_for(kib: int, wafer_b):
            if kib not in trackers:
                trackers[kib] = DenseCostTracker(
                    wafer=None if args.dry_run else wafer_b, dry_run=args.dry_run,
                    out_dir=None if args.dry_run else out_root / "primitive_cost_traces" / mode.name
                    / (f"sram{kib}k" if carve else ""),
                    tiling=Tiling(), move_tiling=mode.move_tiling(), _cache=shared_by_kib.setdefault(kib, {}),
                    max_tile_k_candidates=getattr(args, "max_tile_k_candidates", 0) or None)
            return trackers[kib]

        print(f"\n##### mode '{mode.name}': data moves {mode.moves}, attention instances "
              f"{'concurrent' if mode.concurrent else 'serial'}, elementwise tiling mesh ({wafer.mesh_rows * wafer.mesh_cols:,} PEs)")
        for b in batches:
            w0 = time.time()
            eff_kib = wafer.sram_kib_per_pe - carve_kib[b]
            if eff_kib <= 0:
                raise SystemExit(f"batch {b}: on-wafer weights + KV cache would take {carve_kib[b]} KiB of every PE's "
                                 f"{wafer.sram_kib_per_pe} KiB SRAM -- nothing left for tiles; shrink the model or use --kv-placement dram")
            wafer_b, fw_b = replace(wafer, sram_kib_per_pe=eff_kib), replace(fw, sram_kib_per_pe=eff_kib)
            cost = tracker_for(eff_kib, wafer_b)
            off_b = offchip.with_kv(placements[b] == "dram") if offchip else None
            if off_b is not None and not (off_b.kv or off_b.weights):
                off_b = None
            check_decode_fits(cfg, b, pts, fw_b, COST_BPE)
            chunk = args.prefill_chunk or prefill_chunk_size(cfg, b, n_in, fw_b, COST_BPE)
            if args.prefill_chunk:
                bad = [(t0, step_fits(cfg, b, chunk, t0, fw_b, COST_BPE)) for t0 in chunk_starts(n_in, chunk)]
                bad = [x for x in bad if x[1]]
                if bad:
                    raise SystemExit(f"--prefill-chunk {chunk} does not fit the wafer: {bad}")
            pre = run_prefill(cfg, b, n_in, chunk, cost, COST_BPE, mode.concurrent, offchip=off_b, kv_overlap=mode.kv_overlap,
                              kv_exposed=kv_exposed)
            dec = [run_step(cfg, b, 1, t0, cost, COST_BPE, mode.concurrent, offchip=off_b, kv_overlap=mode.kv_overlap,
                            kv_exposed=kv_exposed) for t0 in pts]
            mean = simpson_mean(*dec)
            mt = metrics(pre, mean, b, n_in, n_out, clock) if not args.dry_run else {"batch": b}
            results[(mode.name, b)] = {"metrics": mt, "kv_placement": placements[b], "sram_kib_for_tiles": eff_kib,
                                       "prefill_chunk": chunk, "prefill_passes": n_in // chunk,
                                       "prefill_cycles": pre.cycles, "decode_mean_cycles": mean.cycles,
                                       "decode_cycles_at": dict(zip(map(str, pts), (d.cycles for d in dec))),
                                       "prefill_breakdown": fmt_by_op(pre), "decode_breakdown": fmt_by_op(mean)}
            if args.dry_run:
                print(f"  batch {b:>2}: prefill {n_in // chunk} pass(es) of {chunk} tokens; ops recorded (unique shapes so far {cost.unique_op_shapes}, "
                      f"reuses {cost.memoized_reuses}), wall {time.time() - w0:.1f}s")
            else:
                print(f"  batch {b:>2}: prefill {n_in // chunk} pass(es) of {chunk} tokens")
                print(f"           TTFT {mt['ttft_s']:.4g}s  ITL {mt['itl_s'] * 1e3:.4g}ms  "
                      f"paper-throughput {mt['paper_throughput_tok_s']:,.0f} tok/s  "
                      f"(output {mt['output_throughput_tok_s']:,.0f}, decode-only {mt['decode_throughput_tok_s']:,.0f})  "
                      f"wall {time.time() - w0:.0f}s")
                print(f"           prefill: {fmt_by_op(pre)}")
                print(f"           decode : {fmt_by_op(mean)}")
        print(f"  peak elementwise elements per PE: {max((getattr(c, 'peak_per_pe', 0) for c in trackers.values()), default=0):,} "
              f"(unique op shapes {sum(c.unique_op_shapes for c in trackers.values())} = Docker runs)")

    if args.dry_run:
        print("\nDRY RUN finished: every op shape was emitted without error. Run without --dry-run (Docker) for costs.")
        return
    print()
    print("=" * 78)
    print(f"SUMMARY  {cfg.name}  input/output {n_in}/{n_out}, fp16 cost model, clock {clock} GHz")
    print("=" * 78)
    print(f"{'mode':<13} {'batch':>5} {'TTFT s':>10} {'ITL ms':>10} {'paper tok/s':>13} {'output tok/s':>13} {'decode tok/s':>13}")
    for (mname, b), r in results.items():
        m = r["metrics"]
        print(f"{mname:<13} {b:>5} {m['ttft_s']:>10.4g} {m['itl_s'] * 1e3:>10.4g} {m['paper_throughput_tok_s']:>13,.0f} "
              f"{m['output_throughput_tok_s']:>13,.0f} {m['decode_throughput_tok_s']:>13,.0f}")
    if paper_peak:
        print(f"\nPaper (LLM-Inference-Bench Fig. 25, {cfg.name}, 1024/1024, fp16, peak over batch, bar-chart reading):")
        for k, v in paper_peak.items():
            print(f"  {k:<26} ~{v:,} tok/s")
    if note:
        print(f"\n{note}")
    if not capacity["all_fit_total_sram"]:
        print("CAPACITY: weights + KV exceed the on-wafer SRAM (see the table at the top); off-wafer data movement is NOT in these numbers.")
    if weights_dram:
        print("DRAM: the weights are streamed from DRAM at the --dram-bw-gbs placeholder bandwidth; the on-wafer network that\n"
              "distributes them (and any on-chip K/V) to the PEs is NOT modelled.")
    print("Read this honestly: 'optimistic' assumes free-layout data moves and fully concurrent attention;\n"
          "'conservative' serializes both. The true machine lies between, and the derived wafer is far smaller\n"
          "than a real WSE. The paper metric counts input+output tokens over end-to-end latency.")
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {"model": cfg.name, "input_len": n_in, "output_len": n_out, "clock_ghz": clock, "cost_bpe": COST_BPE,
               "decode_cache_lengths_sampled": pts, "wafer": {"mesh_rows": wafer.mesh_rows, "mesh_cols": wafer.mesh_cols,
                                                                "sram_kib_per_pe": wafer.sram_kib_per_pe},
               "results": {f"{k[0]}/batch{k[1]}": v for k, v in results.items()},
               "capacity": capacity, "offchip_kv": dram_info, "weights_dram": weights_dram,
               "weights_sram_carve_kib": w_carve, "mac_lanes": getattr(fw, "mac_lanes", 1),
               "sram_read_bytes_per_cycle": getattr(fw, "sram_read_bytes_per_cycle", 0),
               "kv_placement": placements, "kv_placement_policy": policy or "legacy", "kv_traffic_added_to_latency": kv_exposed,
               "max_tile_k_candidates": getattr(args, "max_tile_k_candidates", 0), "paper_peak_tok_s": paper_peak, "wall_s": time.time() - t_start}
    (out_root / "summary.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
