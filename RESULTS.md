# Results: cuDNN-FE vs Miles tilelang (DSv4 kernels)

Measured 2026-06-10 on a single box with both a Blackwell (sm100) and a Hopper
H100 NVL (sm90). bf16, MQA (K=V), head_dim=512 for attention / 128 for the
indexer, num_heads=64. Median CUDA-event latency over 30 iters, JIT/compile
excluded via warmup. Same inputs fed to both backends in one process.

Reproduce: see [README.md](README.md). Raw runs in [`results/`](results/).

---

## TL;DR

| Kernel | Winner | Margin |
|---|---|---|
| **sparse attention — backward** | **cuDNN-FE** | **3.0–3.5× (Blackwell), 4.3–6.1× (Hopper)** |
| sparse attention — forward | ~par (FlashMLA vs tilelang) | see table |
| indexer — score gemm (forward) | tilelang at large S | cuDNN-FE ~2× slower at S=8k |
| **indexer — top-k** | **cuDNN-FE (radix)** | **~5× vs torch.topk** |

**The "Megatron+cuDNN beats tilelang" story is carried by the backward.** Miles'
tilelang sparse-attn *forward* is not slow; its *backward* is (no pipelining
`num_stages=0` + `atomic_addx4` dKV scatter), and that is where cuDNN-FE's
`SparseAttentionBackward` wins 3–6×. A second, zero-friction win: cuDNN-FE's
radix `IndexerTopK` replaces Miles' `torch.topk` for ~5×.

---

## Sparse attention (H=64, D=512, bf16)

### Backward — cuDNN-FE `SparseAttentionBackward` vs tilelang `sparse_mqa_bwd`

**Hopper (H100 NVL, sm90)** — latency ms:

| S / S_kv | topk | tilelang bwd | cuDNN-FE bwd | **FE speedup** |
|---|---|---|---|---|
| 2048/4096 | 2048 | 29.21 | 4.82 | **6.1×** |
| 4096/8192 | 2048 | 60.67 | 9.94 | **6.1×** |
| 8192/16384 | 2048 | 128.17 | 21.96 | **5.8×** |
| 2048/4096 | 512 | 7.43 | 1.52 | 4.9× |
| 4096/8192 | 512 | 15.20 | 2.99 | 5.1× |
| 8192/16384 | 512 | 30.87 | 7.26 | 4.3× |

**Blackwell (sm100)** — latency ms:

| S / S_kv | topk | tilelang bwd | cuDNN-FE bwd | **FE speedup** |
|---|---|---|---|---|
| 2048/4096 | 2048 | 15.45 | 4.73 | **3.3×** |
| 4096/8192 | 2048 | 30.79 | 9.24 | **3.3×** |
| 8192/16384 | 2048 | 61.56 | 18.56 | **3.3×** |
| 2048/4096 | 512 | 4.05 | 1.15 | 3.5× |
| 4096/8192 | 512 | 8.06 | 2.63 | 3.1× |
| 8192/16384 | 512 | 16.38 | 5.49 | 3.0× |

### Forward — FlashMLA sparse prefill vs tilelang `sparse_mqa_fwd`

tilelang forward (ms): Hopper 1.93 / 4.00 / 7.91 (topk=2048, S=2k/4k/8k);
Blackwell 2.27 / 4.65 / 9.13. FlashMLA forward: _see `results/` once built_
(cuDNN-FE ships no sparse-attn forward; production forward = FlashMLA).

---

## Indexer (Hq=64, D=128, ratio=4, topk=2048) — Blackwell sm100 only

cuDNN-FE `IndexerForward` is sm100-only, so this comparison is Blackwell-only.

| S / S_kv | tilelang fwd | cuDNN-FE fwd | FE/tl | torch.topk | cuDNN-FE TopK | **FE/torch** |
|---|---|---|---|---|---|---|
| 2048/4096 | 0.155 | 0.162 | 0.96× | 0.232 | 0.044 | **5.3×** |
| 4096/8192 | 0.502 | 0.542 | 0.93× | 0.647 | 0.120 | **5.4×** |
| 8192/16384 | 1.755 | 3.325 | **0.53×** | 1.865 | 0.380 | **4.9×** |

- **Score gemm (forward):** ~par at small/mid S; at S=8k cuDNN-FE `IndexerForward`
  is ~2× slower than tilelang. Sweeping `q_stage`/`kv_stage` does **not** close
  it (q_stage=1 unsupported; kv_stage 2/3/4 all ~3.1–3.4 ms) — it is a
  structural large-S regression, an optimization item for the FE kernel team.
- **Top-k:** cuDNN-FE's radix `IndexerTopK` beats `torch.topk` (what Miles uses
  today in `tilelang_indexer.py`) by ~5×. Drop-in replacement, no tilelang change.
- Indexer **backward** is not compared: cuDNN-FE `IndexerBackward` fuses the KL
  auxiliary-loss gradient (different semantics than tilelang's plain
  dScore→dq/dk/dw), so it is not a clean drop-in head-to-head.

---

## Notes / caveats

- Backward kernel runtime is determined by shapes (topk_idxs / topk_length),
  not tensor values; forward outputs (o, lse) fed into the backward may be
  synthesized without changing timing.
- Miles tilelang kernels: `radixark/miles@9437366` =byte-identical= internal
  mirror `gitlab-master.nvidia.com/dl/miles/miles`.
- cuDNN-FE: dev tree `cudnn_frontend` `python/cudnn/deepseek_sparse_attention`.
- sm90 cuDNN-FE backward needs a one-line `nvvm.atomicrmw` signature shim under
  cutlass-dsl 4.5.0 (see `_compat.apply_atomicrmw_shim`); not a kernel bug.
