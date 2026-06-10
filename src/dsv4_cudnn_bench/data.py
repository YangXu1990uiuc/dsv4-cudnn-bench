"""Synthetic DSv4 inputs in one canonical layout.

Sparse attention (MQA, K=V):
  q   [B, S, H, D] bf16, kv [B, S_kv, D] bf16, attn_sink [H] fp32,
  topk_idxs [B, S, TopK] int32 (-1 = invalid)
Indexer (lightning indexer, single KV head):
  q [B, S, H, Didx] bf16, k [B, S_kv, Didx] bf16, weights [B, S, H] fp32
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SparseAttnInputs:
    q: torch.Tensor
    kv: torch.Tensor
    attn_sink: torch.Tensor
    topk_idxs: torch.Tensor
    sm_scale: float


def make_sparse_attn_inputs(B, S, S_kv, H, D, topk, *, device="cuda", seed=0) -> SparseAttnInputs:
    g = torch.Generator(device=device).manual_seed(seed)
    dt = torch.bfloat16
    q = torch.randn(B, S, H, D, dtype=dt, device=device, generator=g)
    kv = torch.randn(B, S_kv, D, dtype=dt, device=device, generator=g)
    attn_sink = torch.randn(H, dtype=torch.float32, device=device, generator=g)
    topk_k = min(topk, S_kv)
    idx = torch.argsort(torch.rand(B, S, S_kv, device=device, generator=g), dim=-1)[..., :topk_k].to(torch.int32)
    if topk_k < topk:
        pad = torch.full((B, S, topk - topk_k), -1, dtype=torch.int32, device=device)
        idx = torch.cat([idx, pad], dim=-1)
    return SparseAttnInputs(q.contiguous(), kv.contiguous(), attn_sink, idx.contiguous(), 1.0 / D**0.5)


@dataclass(frozen=True)
class IndexerInputs:
    q: torch.Tensor       # [B, S, H, Didx]
    k: torch.Tensor       # [B, S_kv, Didx]
    weights: torch.Tensor  # [B, S, H]
    ratio: int
    topk: int
    sm_scale: float


def make_indexer_inputs(B, S, S_kv, H, Didx, ratio, topk, *, device="cuda", seed=0) -> IndexerInputs:
    g = torch.Generator(device=device).manual_seed(seed)
    dt = torch.bfloat16
    q = torch.randn(B, S, H, Didx, dtype=dt, device=device, generator=g)
    k = torch.randn(B, S_kv, Didx, dtype=dt, device=device, generator=g)
    w = torch.randn(B, S, H, dtype=torch.float32, device=device, generator=g)
    return IndexerInputs(q.contiguous(), k.contiguous(), w.contiguous(), ratio, topk, 1.0 / Didx**0.5)
