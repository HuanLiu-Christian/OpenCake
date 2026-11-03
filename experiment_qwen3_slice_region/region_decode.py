#!/usr/bin/env python3
"""Weight-stationary, K-split decode of the Qwen3 slice (32 layers, hidden 460) on PER-LAYER PE regions.

WHY. experiment_qwen3_slice maps every decode GEMM (M = 1) onto ONE row x ~90 columns of the 90x100 mesh (SUMMA, K phased in time):
~1 % of the PEs work, and the weights are assumed to be wherever they are needed. Cerebras (Hot Chips 2024, slides 40-46) instead gives each
layer its own wafer region, with the weights and KV cache resident there, and the token flows region to region. This script models that:

  * the 90x100 mesh is cut into 32 layer regions of R x C = 14 x 18 PEs (8,064 PEs) plus one 24 x 39 = 936-PE region for the LM head;
  * inside a layer region PE(r, c) permanently holds W[K-block r, N-block c] (K split over the R rows, N over the C columns): no weight traffic;
  * a vector op = scatter the input from the region's corner PE(0,0) (down column 0, then east along each row), local MACs,
    column-chain reduction of the partial sums to row 0 (per packet, with the adds), element-wise work on row 0, gather back to PE(0,0);
  * attention: the layer's KV cache is spread over all PEs of the region (tokens, round robin); q is broadcast, scores / softmax stats /
    P.V are local, the softmax max and sum and the P.V partial outputs are reduced to PE(0,0) over the region (row chains, then column 0);
  * RMSNorm runs on PE(0,0); QKV and gate|up are fused GEMMs sharing one scatter.

All traffic goes through the project's own PacketNetwork (per-link occupancy, per-direction FIFOs with backpressure, router latency,
link bandwidth, packet size, dequeue), the same one the SUMMA simulator uses. Compute = ceil(MACs / mac_lanes) per PE floored by the SRAM read
bandwidth, as in wafersim.wafer.simulator. NOT Timeloop-driven: Timeloop measured exactly 1 cycle / MAC for every GEMM, which is all it
contributes there. NOT modelled: element-wise cost beyond a fixed 1 cycle / lane / op (4 for exp, rsqrt, silu), instruction / task
scheduling, queue-count limits on real PEs, the network between regions beyond one analytic hand-off, prefill.

Layers are identical and strictly sequential, so ONE layer is simulated per decode sample and multiplied by 32.

Usage (project root, env active):
  python3 experiment_qwen3_slice_region/region_decode.py
  python3 experiment_qwen3_slice_region/region_decode.py --region 14x18 --lanes 8
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from math import ceil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load("dense_llm_throughput", "experiment_dense_llm_throughput/dense_llm_throughput.py")
from wafersim.specs import WaferSpec  # noqa: E402
from wafersim.wafer.network import PacketNetwork  # noqa: E402

BPE = 2
NONLINEAR_CYCLES = 4     # cycles per element per lane for exp / rsqrt / silu (assumption)


def make_wafer(rows: int, cols: int, lanes: int, read_bw: int) -> WaferSpec:
    return dataclasses.replace(WaferSpec(), **{**D.PINNED_WAFER, "sram_kib_per_pe": 48, "mesh_rows": rows, "mesh_cols": cols,
                                                "mac_lanes": lanes, "sram_read_bytes_per_cycle": read_bw})


class Region:
    """One R x C PE region with its own network state and per-PE busy times."""

    def __init__(self, wafer: WaferSpec, rows: int, cols: int):
        self.w, self.R, self.C = wafer, rows, cols
        self.net = PacketNetwork(wafer, rows, cols)
        self.free = {(r, c): 0.0 for r in range(rows) for c in range(cols)}
        self.lanes = max(1, wafer.mac_lanes)
        self.read_bw = wafer.sram_read_bytes_per_cycle
        self.parts: dict[str, float] = {}

    def add(self, key: str, cycles: float) -> None:
        self.parts[key] = self.parts.get(key, 0.0) + cycles

    # ---- network helpers (all on PacketNetwork) --------------------------------------------------------------------
    def send(self, path, ready: float, size: int) -> float:
        """Send `size` bytes along `path` (list of adjacent coords), packet by packet; return the last packet's arrival."""
        if len(path) == 1 or size <= 0:
            return ready
        last = ready
        for sz in self.net._packet_sizes(size):
            t = ready
            for a, b in zip(path, path[1:]):
                d = "south" if b[0] > a[0] else "north" if b[0] < a[0] else "east" if b[1] > a[1] else "west"
                t = self.net._send_hop(t, sz, a, b, d)
            last = max(last, t)
        return last

    def bcast(self, ready: float, size: int) -> dict:
        """Broadcast `size` bytes from PE(0,0) to every PE: down column 0, then east along every row."""
        col0 = self.net.broadcast_column(0, ready, size, 0) if self.R > 1 else {(0, 0): ready}
        arr = {}
        for r in range(self.R):
            a = self.net.broadcast_row(r, col0[(r, 0)], size, 0) if self.C > 1 else {(r, 0): col0[(r, 0)]}
            arr.update(a)
        return arr

    def scatter(self, ready: float, seg_bytes: int) -> dict:
        """Row r receives its own `seg_bytes` segment (relayed down column 0), then it is broadcast east along row r."""
        arr = {}
        for r in range(self.R - 1, -1, -1):             # farthest first
            at = self.send([(i, 0) for i in range(r + 1)], ready, seg_bytes)
            arr.update(self.net.broadcast_row(r, at, seg_bytes, 0) if self.C > 1 else {(r, 0): at})
        return arr

    def chain_reduce(self, path, ready_at: dict, size: int, op_cycles) -> float:
        """Reduce partial vectors along `path` (far end first) toward path[0]; per packet: receive, combine with own
        partial (op_cycles(elements) cycles), forward. Returns when the last packet is combined at path[0]."""
        done = 0.0
        for sz in self.net._packet_sizes(size):
            elems = max(1, sz // BPE)
            t = ready_at[path[-1]]
            for i in range(len(path) - 1, 0, -1):
                a, b = path[i], path[i - 1]
                d = "south" if b[0] > a[0] else "north" if b[0] < a[0] else "east" if b[1] > a[1] else "west"
                arr = self.net._send_hop(t, sz, a, b, d)
                t = max(arr, ready_at[b]) + op_cycles(elems)
                self.free[b] = max(self.free[b], t)
            done = max(done, t)
        return done

    def reduce_to_corner(self, ready_at: dict, size: int, op_cycles) -> float:
        """Reduce a `size`-byte vector held (partially) at every PE to PE(0,0): along each row to column 0, then down column 0."""
        col0 = {}
        for r in range(self.R):
            if self.C > 1:
                col0[(r, 0)] = self.chain_reduce([(r, c) for c in range(self.C)], ready_at, size, op_cycles)
            else:
                col0[(r, 0)] = ready_at[(r, 0)]
        if self.R == 1:
            return col0[(0, 0)]
        return self.chain_reduce([(r, 0) for r in range(self.R)], col0, size, op_cycles)

    def alu(self, elems: int, ops: int = 1, nonlinear: int = 0) -> int:
        return ceil(elems * ops / self.lanes) + ceil(elems * nonlinear * NONLINEAR_CYCLES / self.lanes)

    # ---- one weight-stationary GEMM, input at PE(0,0), output gathered back to PE(0,0) -------------------------------
    def gemm(self, t: float, m: int, n: int, k: int, name: str, post_elems_per_col=0, post_ops=0, post_nonlinear=0,
             out_cols_bytes_factor=1.0) -> float:
        R, C = self.R, self.C
        nt, kt = ceil(n / C), ceil(k / R)
        t0 = t
        arr = self.scatter(t, m * kt * BPE)
        t_arr = max(arr.values())
        comp = max(ceil(m * nt * kt / self.lanes), ceil(kt * nt * BPE / self.read_bw) if self.read_bw else 0)
        end = {}
        for pe, a in arr.items():
            s = max(a, self.free[pe])
            end[pe] = s + comp
            self.free[pe] = end[pe]
        t_comp = max(end.values())
        row0 = {}
        for c in range(C):
            col = [(r, c) for r in range(R)]
            row0[c] = self.chain_reduce(col, end, m * nt * BPE, lambda e: ceil(e / self.lanes)) if R > 1 else end[(0, c)]
        t_red = max(row0.values())
        post = self.alu(max(1, post_elems_per_col or 0), post_ops, post_nonlinear) if post_ops or post_nonlinear else 0
        out_bytes = int(m * nt * BPE * out_cols_bytes_factor)
        t_out = 0.0
        for c in range(C):
            ready = row0[c] + post
            t_out = max(t_out, self.send([(0, j) for j in range(c, -1, -1)], ready, out_bytes))
        self.free[(0, 0)] = max(self.free[(0, 0)], t_out)
        self.add(f"{name}:scatter", t_arr - t0)
        self.add(f"{name}:compute", t_comp - t_arr)
        self.add(f"{name}:reduce", t_red - t_comp)
        self.add(f"{name}:post+gather", t_out - t_red)
        return t_out

    def rmsnorm(self, t: float, n: int, name: str) -> float:
        cyc = 2 * ceil(n / self.lanes) + NONLINEAR_CYCLES        # sum of squares, scalar rsqrt, scale
        self.add(name, cyc)
        return t + cyc

    def attention(self, t: float, cache_len: int, q_mult: int, hd: int, name="attn") -> float:
        R, C = self.R, self.C
        P = R * C
        tpp = ceil(cache_len / P)                      # tokens per PE (round robin)
        q_bytes = q_mult * hd * BPE
        t0 = t
        arr = self.bcast(t, q_bytes)
        t_q = max(arr.values())
        comp = {}
        for pe, a in arr.items():
            s = max(a, self.free[pe])
            qk = max(ceil(q_mult * tpp * hd / self.lanes), ceil(tpp * hd * BPE / self.read_bw) if self.read_bw else 0)
            comp[pe] = s + qk + self.alu(q_mult * tpp, 1)                       # scores, local row max
            self.free[pe] = comp[pe]
        t_qk = max(comp.values())
        # global softmax max, then broadcast
        mx = self.reduce_to_corner(comp, q_mult * BPE, lambda e: ceil(e / self.lanes) + 1)
        arr = self.bcast(mx, q_mult * BPE)
        t_mx = max(arr.values())
        e2 = {}
        for pe, a in arr.items():
            e2[pe] = max(a, self.free[pe]) + self.alu(q_mult * tpp, 2, 1)       # exp(x - max), local sum
            self.free[pe] = e2[pe]
        sm = self.reduce_to_corner(e2, q_mult * BPE, lambda e: ceil(e / self.lanes) + 1)
        arr = self.bcast(sm, q_mult * BPE)
        t_sm = max(arr.values())
        pv = {}
        for pe, a in arr.items():
            s = max(a, self.free[pe])
            pv[pe] = s + max(ceil(q_mult * tpp * hd / self.lanes), ceil(tpp * hd * BPE / self.read_bw) if self.read_bw else 0)
            self.free[pe] = pv[pe]
        t_pv = max(pv.values())
        out = self.reduce_to_corner(pv, q_mult * hd * BPE, lambda e: ceil(e / self.lanes))
        out += self.alu(q_mult * hd, 1, 1)                                       # divide by the softmax sum
        self.free[(0, 0)] = max(self.free[(0, 0)], out)
        for key, v in (("q-bcast", t_q - t0), ("QK", t_qk - t_q), ("max allreduce", t_mx - t_qk), ("exp+sum allreduce", t_sm - t_mx),
                       ("PV", t_pv - t_sm), ("PV reduce", out - t_pv)):
            self.add(f"{name}:{key}", v)
        return out


@dataclasses.dataclass(frozen=True)
class Cfg:
    d: int = 460
    layers: int = 32
    q_mult: int = 11
    hd: int = 64
    ffn: int = 2300
    vocab: int = 15200


def layer_cycles(cfg: Cfg, wafer: WaferSpec, R: int, C: int, cache_len: int):
    reg = Region(wafer, R, C)
    qd, kvd = cfg.q_mult * cfg.hd, cfg.hd
    t = 0.0
    t = reg.rmsnorm(t, cfg.d, "rmsnorm1")
    per_col = ceil((qd + 2 * kvd) / C)
    t = reg.gemm(t, 1, qd + 2 * kvd, cfg.d, "qkv", post_elems_per_col=per_col, post_ops=6, post_nonlinear=2)   # q/k norm + rope
    t += 2 * (wafer.router_latency_cycles + wafer.receive_dequeue_latency_cycles + 1)                          # per-head norm across columns
    t = reg.attention(t, cache_len, cfg.q_mult, cfg.hd)
    t = reg.gemm(t, 1, cfg.d, qd, "o_proj", post_elems_per_col=ceil(cfg.d / C), post_ops=1)                  # + residual
    t = reg.rmsnorm(t, cfg.d, "rmsnorm2")
    t = reg.gemm(t, 1, 2 * cfg.ffn, cfg.d, "gate_up", post_elems_per_col=ceil(2 * cfg.ffn / C), post_ops=2, post_nonlinear=1,
                 out_cols_bytes_factor=0.5)                                                                   # silu(gate) * up
    t = reg.gemm(t, 1, cfg.d, cfg.ffn, "down", post_elems_per_col=ceil(cfg.d / C), post_ops=1)               # + residual
    hand = C * (wafer.router_latency_cycles + wafer.receive_dequeue_latency_cycles) + ceil(cfg.d * BPE / wafer.link_bandwidth_bytes_per_cycle)
    reg.add("hand-off to next layer region", hand)
    return t + hand, reg.parts


def lm_head_cycles(cfg: Cfg, wafer: WaferSpec, R: int, C: int):
    reg = Region(wafer, R, C)
    t = reg.rmsnorm(0.0, cfg.d, "final rmsnorm")
    t = reg.gemm(t, 1, cfg.vocab, cfg.d, "lm_head", post_elems_per_col=ceil(cfg.vocab / C), post_ops=1, out_cols_bytes_factor=1.0 / max(1, ceil(cfg.vocab / C)))
    return t, reg.parts


def run(region=(14, 18), head_region=(24, 39), lanes=8, read_bw=16, cfg: Cfg = Cfg(), in_len=1024, out_len=1024):
    R, C = region
    wl = make_wafer(R, C, lanes, read_bw)
    wh = make_wafer(*head_region, lanes, read_bw)
    lens = [in_len, in_len + (out_len - 1) // 2, in_len + out_len - 2]      # 1024, 1535, 2046 as in the Llama / Qwen runs
    head, head_parts = lm_head_cycles(cfg, wh, *head_region)
    res = {}
    for L_ in lens:
        lay, parts = layer_cycles(cfg, wl, R, C, L_)
        res[L_] = {"layer": lay, "itl": cfg.layers * lay + head, "parts": parts}
    itl = (res[lens[0]]["itl"] + 4 * res[lens[1]]["itl"] + res[lens[2]]["itl"]) / 6          # Simpson, same weights as the main runs
    return {"region": region, "head_region": head_region, "lanes": lanes, "lm_head_cycles": head, "head_parts": head_parts,
            "samples": res, "itl_cycles": itl, "itl_ms": itl / 1e6, "decode_tok_s": 1e9 / itl, "us_per_layer": itl / cfg.layers / 1e3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="14x18")
    ap.add_argument("--head-region", default="24x39")
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--sram-read-bytes", type=int, default=16)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    r = tuple(int(x) for x in a.region.split("x"))
    h = tuple(int(x) for x in a.head_region.split("x"))
    out = run(r, h, a.lanes, a.sram_read_bytes)
    print(f"region {r[0]}x{r[1]} per layer ({r[0] * r[1]} PEs x 32 = {32 * r[0] * r[1]}), LM-head region {h[0]}x{h[1]} ({h[0] * h[1]} PEs); "
          f"total {32 * r[0] * r[1] + h[0] * h[1]} of 9,000; lanes {a.lanes}")
    for L_, v in out["samples"].items():
        print(f"  cache {L_}: layer {v['layer']:.0f} cycles, ITL {v['itl']:.0f}")
    mid = out["samples"][1535]["parts"]
    tot = sum(mid.values())
    print("\nper-layer critical-path breakdown at cache 1535 (cycles; they sum to the layer time):")
    for k, v in sorted(mid.items(), key=lambda kv: -kv[1]):
        print(f"  {k:34s} {v:8.0f}  {100 * v / tot:5.1f} %")
    print(f"\nLM head + final norm: {out['lm_head_cycles']:.0f} cycles")
    print(f"ITL (Simpson over 1024/1535/2046): {out['itl_cycles']:,.0f} cycles = {out['itl_ms']:.4f} ms @1 GHz -> {out['us_per_layer']:.2f} us/layer, "
          f"decode {out['decode_tok_s']:,.0f} tok/s (slice)")
    print(f"full-model equivalent (2 wafers in series + one ~5 us hop): {1e9 / (2 * out['itl_cycles'] + 5000):,.0f} tok/s")
    print("reference: SUMMA 1-row run 521,447 cycles = 16.3 us/layer = 1,918 tok/s (slice) = ~960 full-model; Cerebras 2,400 -> 6.5 us/layer")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
