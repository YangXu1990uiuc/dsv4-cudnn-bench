"""Backend adapters for DSv4 sparse-MLA attention.

Canonical inputs (see data.SparseAttnInputs). Each adapter converts to its
backend's native layout internally.

Backends:
  miles_tilelang : Miles tilelang sparse MQA (fwd + bwd)              [sm90/sm100]
  cudnn_fe       : cuDNN-FE SparseAttentionBackward (bwd only)        [sm90/sm100]
  flashmla       : FlashMLA sparse prefill (fwd only)                 [sm90/sm100]

cuDNN-FE ships no sparse-attn forward (production fwd = FlashMLA); FlashMLA
ships no sparse bwd here. So the comparison is:
  forward  : miles_tilelang  vs  flashmla
  backward : miles_tilelang  vs  cudnn_fe
"""
from __future__ import annotations

import torch

from ._compat import apply_atomicrmw_shim, miles_kernel


# ---------------- Miles tilelang ----------------
_tl_fwd = _tl_bwd = None


def _miles():
    global _tl_fwd, _tl_bwd
    if _tl_fwd is None:
        _tl_fwd = miles_kernel("tilelang_sparse_mla_fwd")
        _tl_bwd = miles_kernel("tilelang_sparse_mla_bwd")
    return _tl_fwd, _tl_bwd


def tilelang_fwd(inp):
    fwd, _ = _miles()
    return fwd.sparse_mqa_fwd_interface(inp.q, inp.kv, inp.attn_sink, inp.topk_idxs, sm_scale=inp.sm_scale)


def tilelang_bwd(inp, o, lse, do):
    _, bwd = _miles()
    return bwd.sparse_mqa_bwd_interface(inp.q, inp.kv, inp.attn_sink, o, do, inp.topk_idxs, lse, sm_scale=inp.sm_scale)


# ---------------- cuDNN-FE backward ----------------
_dsa = _cuda = False


def _fe():
    global _dsa, _cuda
    if _dsa is False:
        apply_atomicrmw_shim()
        from cudnn import DSA
        import cuda.bindings.driver as cuda
        _dsa, _cuda = DSA, cuda
    return _dsa, _cuda


def cudnn_fe_bwd(inp, o, lse_fe, do):
    """lse_fe must be natural-log KV-only LSE [S*B, H] fp32 (synthesize for
    timing; runtime is value-independent)."""
    DSA, cuda = _fe()
    B, S, H, D = inp.q.shape
    S_kv = inp.kv.shape[1]
    topk = inp.topk_idxs.shape[-1]
    qf = inp.q.reshape(S * B, H, D).contiguous()
    kvf = inp.kv.reshape(S_kv * B, D).contiguous()
    of = o.reshape(S * B, H, D).contiguous()
    dof = do.reshape(S * B, H, D).contiguous()
    idxf = inp.topk_idxs.reshape(S * B, topk).contiguous()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    r = DSA.sparse_attention_backward_wrapper(
        qf, kvf, of, dof, lse_fe, inp.attn_sink, idxf, softmax_scale=inp.sm_scale, stream=stream
    )
    return r["dq"], r["dkv"], r["d_sink"]


# ---------------- FlashMLA forward ----------------
def flashmla_fwd(inp):
    from flash_mla import flash_mla_sparse_fwd

    B, S, H, D = inp.q.shape
    S_kv = inp.kv.shape[1]
    topk = inp.topk_idxs.shape[-1]
    q = inp.q.reshape(S * B, H, D).contiguous()          # [s_q, h_q, d_qk]
    kv = inp.kv.reshape(S_kv * B, 1, D).contiguous()     # [s_kv, h_kv=1, d_qk]
    idx = inp.topk_idxs.reshape(S * B, 1, topk).contiguous()  # [s_q, h_kv=1, topk]
    out, max_logits, lse = flash_mla_sparse_fwd(q, kv, idx, inp.sm_scale, d_v=D, attn_sink=inp.attn_sink)
    return out, lse
