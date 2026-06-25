"""Benchmark runner: cuDNN-FE vs Miles tilelang for DSv4 sparse attention + indexer.

  python -m dsv4_cudnn_bench.bench sparse_attn   # fwd+bwd comparison sweep
  python -m dsv4_cudnn_bench.bench indexer       # fwd+topk comparison sweep (sm100)
  python -m dsv4_cudnn_bench.bench all

Per-shape it prints median CUDA-event latency (ms) and the cuDNN-FE/tilelang
speedup. Backward runtime is shape- (not value-) dependent, so forward outputs
fed to the backward may be synthesized.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from . import data, indexer, sparse_attention as sa


def timed(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    return t[len(t) // 2]


def _dev():
    p = torch.cuda.get_device_properties(0)
    return p.name, (p.major, p.minor)


SPARSE_SHAPES = [(1, sq, skv, 64, 512, tk) for tk in (512, 2048) for (sq, skv) in ((2048, 4096), (4096, 8192), (8192, 16384))]
# Prefix-aligned (training/prefill): sq == s_kv, ratio == 1 so cuDNN's bottom-right
# causal mask (q <= k + (s_kv - sq)) reduces to top-left (q <= k), matching Miles
# TileLang -> apples-to-apples. (Earlier sq != s_kv runs were NOT comparable: the
# two backends masked a different number of valid KV per query.)
INDEXER_SHAPES = [(1, s, s, 64, 128, 1, 2048) for s in (4096, 8192, 16384)]


def run_sparse_attn():
    name, cc = _dev()
    rows = []
    print(f"# sparse attention | {name} sm{cc[0]}{cc[1]}")
    print(f"# {'S':>6} {'S_kv':>6} {'H':>3} {'D':>4} {'topk':>5} {'fwd_tl':>8} {'fwd_fmla':>8} {'bwd_tl':>8} {'bwd_fe':>8} {'fe/tl_bwd':>9}")
    have_fmla = True
    try:
        import flash_mla  # noqa
    except Exception:
        have_fmla = False
    for (B, S, Skv, H, D, tk) in SPARSE_SHAPES:
        try:
            inp = data.make_sparse_attn_inputs(B, S, Skv, H, D, tk)
            o, lse = sa.tilelang_fwd(inp)
            do = torch.randn_like(o)
            fwd_tl = timed(lambda: sa.tilelang_fwd(inp))
            bwd_tl = timed(lambda: sa.tilelang_bwd(inp, o, lse, do))
            lse_fe = torch.full((S * B, H), float(math.log(tk)), dtype=torch.float32, device="cuda")
            sa.cudnn_fe_bwd(inp, o, lse_fe, do)
            torch.cuda.synchronize()
            bwd_fe = timed(lambda: sa.cudnn_fe_bwd(inp, o, lse_fe, do))
            fwd_fmla = float("nan")
            if have_fmla:
                sa.flashmla_fwd(inp); torch.cuda.synchronize()
                fwd_fmla = timed(lambda: sa.flashmla_fwd(inp))
            print(f"  {S:>6} {Skv:>6} {H:>3} {D:>4} {tk:>5} {fwd_tl:>8.3f} {fwd_fmla:>8.3f} {bwd_tl:>8.3f} {bwd_fe:>8.3f} {bwd_tl/bwd_fe:>8.2f}x")
            rows.append(dict(op="sparse_attn", S=S, S_kv=Skv, H=H, D=D, topk=tk,
                             fwd_tilelang_ms=fwd_tl, fwd_flashmla_ms=fwd_fmla,
                             bwd_tilelang_ms=bwd_tl, bwd_cudnn_fe_ms=bwd_fe, bwd_speedup_fe_over_tl=bwd_tl / bwd_fe))
        except Exception as e:
            print(f"  {S:>6} {Skv:>6} {H:>3} {D:>4} {tk:>5}  ERROR: {repr(e)[:80]}")
        torch.cuda.empty_cache()
    return rows


def run_indexer():
    name, cc = _dev()
    rows = []
    print(f"# indexer | {name} sm{cc[0]}{cc[1]}" + ("" if cc[0] == 10 else "  (cuDNN IndexerForward is sm100-only)"))
    print(f"# {'S':>6} {'S_kv':>6} {'Hq':>3} {'Di':>3} {'topk':>5} {'fwd_tl':>8} {'fwd_fe':>8} {'fe/tl':>6} {'topk_torch':>10} {'topk_fe':>8} {'fe/torch':>8}")
    for (B, S, Skv, H, Di, ratio, tk) in INDEXER_SHAPES:
        try:
            inp = data.make_indexer_inputs(B, S, Skv, H, Di, ratio, tk)
            logits = indexer.tilelang_fwd(inp)
            fwd_tl = timed(lambda: indexer.tilelang_fwd(inp))
            topk_torch = timed(lambda: indexer.torch_topk(logits, tk))
            scores = indexer.cudnn_fe_fwd(inp); torch.cuda.synchronize()
            fwd_fe = timed(lambda: indexer.cudnn_fe_fwd(inp))
            indexer.cudnn_fe_topk(scores, tk); torch.cuda.synchronize()
            topk_fe = timed(lambda: indexer.cudnn_fe_topk(scores, tk))
            print(f"  {S:>6} {Skv:>6} {H:>3} {Di:>3} {tk:>5} {fwd_tl:>8.3f} {fwd_fe:>8.3f} {fwd_tl/fwd_fe:>5.2f}x {topk_torch:>10.3f} {topk_fe:>8.3f} {topk_torch/topk_fe:>7.2f}x")
            rows.append(dict(op="indexer", S=S, S_kv=Skv, Hq=H, Didx=Di, ratio=ratio, topk=tk,
                             fwd_tilelang_ms=fwd_tl, fwd_cudnn_fe_ms=fwd_fe, fwd_speedup_fe_over_tl=fwd_tl / fwd_fe,
                             topk_torch_ms=topk_torch, topk_cudnn_fe_ms=topk_fe, topk_speedup_fe_over_torch=topk_torch / topk_fe))
        except Exception as e:
            print(f"  {S:>6} {Skv:>6} {H:>3} {Di:>3} {tk:>5}  ERROR: {repr(e)[:80]}")
        torch.cuda.empty_cache()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["sparse_attn", "indexer", "all"], default="all", nargs="?")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    name, cc = _dev()
    rows = []
    if args.op in ("sparse_attn", "all"):
        rows += run_sparse_attn()
    if args.op in ("indexer", "all"):
        rows += run_indexer()
    if args.output:
        args.output.write_text(json.dumps({"device": name, "cc": cc, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
