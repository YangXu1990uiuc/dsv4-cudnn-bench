"""Backend adapters for the DSv4 lightning indexer.

Indexer forward computes logits S(q,k) = sum_h ReLU(Q_h . K) * W_h with a
ratio-causal mask; then top-k selects the attended KV per query.

Backends:
  miles_tilelang : tilelang score gemm (batched_indexer_fwd) + torch.topk   [sm90/sm100]
  cudnn_fe       : cuDNN-FE IndexerForward + IndexerTopK (radix)            [sm100 only]

Comparison:
  fwd  : miles_tilelang score gemm  vs  cudnn_fe IndexerForward
  topk : torch.topk (Miles today)   vs  cudnn_fe IndexerTopK
"""
from __future__ import annotations

import torch

from ._compat import miles_kernel

_tl = None


def _miles():
    global _tl
    if _tl is None:
        _tl = miles_kernel("tilelang_indexer_fwd")
    return _tl


def tilelang_fwd(inp):
    """Returns logits [B, S, S_kv] fp32 (SBHD internal layout, B handled by loop)."""
    tl = _miles()
    B, S, H, Didx = inp.q.shape
    S_kv = inp.k.shape[1]
    q_sb = inp.q.permute(1, 0, 2, 3).contiguous()    # [S, B, H, Didx]
    k_sb = inp.k.permute(1, 0, 2).contiguous()       # [S_kv, B, Didx]
    w_sb = inp.weights.permute(1, 0, 2).contiguous()  # [S, B, H]
    cu_ks, cu_ke = tl._make_causal_cu_seqlens(S, S_kv, inp.ratio, inp.q.device)
    return tl.batched_indexer_fwd(q_sb, k_sb, w_sb, cu_ks, cu_ke)


def torch_topk(logits, topk):
    k = min(topk, logits.shape[-1])
    return torch.topk(logits, k, dim=-1)


# ---------------- cuDNN-FE (sm100) ----------------
_dsa = _cuda = False


def _fe():
    global _dsa, _cuda
    if _dsa is False:
        from cudnn import DSA
        import cuda.bindings.driver as cuda
        _dsa, _cuda = DSA, cuda
    return _dsa, _cuda


def cudnn_fe_fwd(inp):
    """Returns scores [B, S, S_kv] fp32."""
    DSA, cuda = _fe()
    B, S, H, Didx = inp.q.shape
    k_b = inp.k.unsqueeze(2).contiguous()            # [B, S_kv, 1, Didx]
    w_b = inp.weights.to(torch.bfloat16).contiguous()  # FE wants bf16 W
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return DSA.indexer_forward_wrapper(
        inp.q, k_b, w_b, ratio=inp.ratio, qhead_per_kv_head=H, sm_scale=inp.sm_scale, stream=stream
    )["scores"]


def cudnn_fe_topk(scores, topk):
    DSA, cuda = _fe()
    sc2d = scores.reshape(-1, scores.shape[-1]).contiguous()
    k = min(topk, sc2d.shape[-1])
    seq_lens = torch.full((sc2d.shape[0],), sc2d.shape[-1], dtype=torch.int32, device=scores.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return DSA.indexer_top_k_wrapper(sc2d, seq_lens, top_k=k, next_n=1, return_val=False, stream=stream)
