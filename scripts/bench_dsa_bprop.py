"""Standalone benchmark for cuDNN-FE DSA SparseAttentionBackward (bprop).

Runtime of the bwd kernel is determined by shapes (topk_idxs / topk_length),
not by tensor values, so we synthesize out/lse instead of running the
expensive PyTorch reference forward. lse is set near log(topk) so any internal
exp() stays bounded.

Usage:
  CUDA_VISIBLE_DEVICES=<idx> python bench_dsa_bprop.py
"""

import math
import time
import torch
from cudnn import DSA
from cuda.bindings import driver as cuda

# Compat shim: installed cutlass-dsl 4.5.0 binds nvvm.atomicrmw with a leading
# `res` (result type) positional arg; the sm90 DSA primitive calls it with only
# op/ptr/a (older signature). Inject res = a.type when missing. Inert on sm100
# (which uses inline-asm atomicAdd, not this binding).
try:
    from cutlass._mlir.dialects import nvvm as _nvvm
    import inspect as _inspect

    if "res" in _inspect.signature(_nvvm.atomicrmw).parameters:
        _orig_atomicrmw = _nvvm.atomicrmw

        def _atomicrmw_compat(*args, **kw):
            if "res" not in kw and "a" in kw and not args:
                kw = {"res": kw["a"].type, **kw}
            return _orig_atomicrmw(*args, **kw)

        _nvvm.atomicrmw = _atomicrmw_compat
except Exception as _e:  # pragma: no cover
    print(f"# atomicrmw shim skipped: {_e!r}")


def make_inputs(s_q, s_kv, h, d, topk, has_topk_length, device="cuda"):
    dt = torch.bfloat16
    q = torch.randn(s_q, h, d, dtype=dt, device=device)
    kv = torch.randn(s_kv, d, dtype=dt, device=device)
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)

    topk_k = min(topk, s_kv)
    # random distinct indices per query
    idx = torch.argsort(torch.rand(s_q, s_kv, device=device), dim=-1)[:, :topk_k].to(torch.int32)
    if topk_k < topk:
        pad = torch.full((s_q, topk - topk_k), -1, dtype=torch.int32, device=device)
        idx = torch.cat([idx, pad], dim=-1)
    topk_idxs = idx.contiguous()

    topk_length = None
    if has_topk_length:
        topk_length = torch.randint(1, topk_k + 1, (s_q,), dtype=torch.int32, device=device)

    # synthetic forward outputs (values irrelevant to bwd timing)
    out = (torch.randn(s_q, h, d, dtype=dt, device=device) * 0.1)
    lse = torch.full((s_q, h), float(math.log(max(topk_k, 2))), dtype=torch.float32, device=device)
    dout = torch.randn(s_q, h, d, dtype=dt, device=device) * 0.1
    return q, kv, attn_sink, topk_idxs, topk_length, out, lse, dout


def bench_one(s_q, s_kv, h, d, topk, has_topk_length, warmup=10, iters=50):
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    scale = 1.0 / math.sqrt(d)
    q, kv, attn_sink, topk_idxs, topk_length, out, lse, dout = make_inputs(
        s_q, s_kv, h, d, topk, has_topk_length
    )

    def call():
        return DSA.sparse_attention_backward_wrapper(
            q, kv, out, dout, lse, attn_sink, topk_idxs,
            softmax_scale=scale, topk_length=topk_length, stream=stream,
        )

    # warmup (first call JIT-compiles)
    t0 = time.perf_counter()
    call()
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start[i].record()
        call()
        end[i].record()
    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(start, end))
    median = times_ms[len(times_ms) // 2]
    p10 = times_ms[max(0, len(times_ms) // 10)]

    # approximate bwd FLOPs: dq,dk,dv each ~ 2 * (S_q * topk * d) MACs * 2 ops,
    # backward ~ 5x fwd attention-style flops over the sparse (S_q*topk) area.
    eff_topk = min(topk, s_kv)
    flops = 5.0 * 2.0 * s_q * eff_topk * h * d * 2.0  # 2 gemms (qk-like + pv-like) rough
    tflops = flops / (median * 1e-3) / 1e12
    return median, p10, compile_s, tflops


if __name__ == "__main__":
    name = torch.cuda.get_device_properties(0).name
    cc = torch.cuda.get_device_capability(0)
    print(f"# device: {name}  sm{cc[0]}{cc[1]}")
    print(f"# {'S_q':>6} {'S_kv':>6} {'H':>4} {'D':>4} {'topk':>5} {'tklen':>5} "
          f"{'median_ms':>10} {'p10_ms':>9} {'TFLOP/s':>9} {'compile_s':>9}")

    H, D = 64, 512
    configs = []
    for topk in (512, 2048):
        for s_q, s_kv in ((2048, 4096), (4096, 8192), (8192, 16384)):
            for tklen in (False, True):
                configs.append((s_q, s_kv, H, D, topk, tklen))

    for (s_q, s_kv, h, d, topk, tklen) in configs:
        try:
            median, p10, comp, tflops = bench_one(s_q, s_kv, h, d, topk, tklen)
            print(f"  {s_q:>6} {s_kv:>6} {h:>4} {d:>4} {topk:>5} {str(tklen):>5} "
                  f"{median:>10.3f} {p10:>9.3f} {tflops:>9.1f} {comp:>9.1f}")
        except Exception as e:
            print(f"  {s_q:>6} {s_kv:>6} {h:>4} {d:>4} {topk:>5} {str(tklen):>5}  ERROR: {repr(e)[:80]}")
        torch.cuda.empty_cache()
