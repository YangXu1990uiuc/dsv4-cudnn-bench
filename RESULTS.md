# Results: cuDNN-FE / FlashMLA vs Miles tilelang (DSv4 kernels)

Measured 2026-06-10 on a single box with both a Blackwell (sm100) and a Hopper
H100 NVL (sm90). bf16, MQA (K=V), head_dim=512 for attention / 128 for the
indexer, num_heads=64. Median CUDA-event latency over 30 iters, JIT/compile
excluded via warmup. Same inputs fed to both backends in one process.

Reproduce: see [README.md](README.md). Raw runs in [`results/`](results/).

---

## TL;DR

The "Megatron + cuDNN" path = **FlashMLA (forward) + cuDNN-FE SparseAttentionBackward
(backward)**. Against Miles' tilelang sparse-MLA kernels, it wins **both** directions:

| Kernel | Winner | Margin (Blackwell / Hopper) |
|---|---|---|
| **sparse attention — forward** | **FlashMLA** | **2.1–3.9× / 1.6–2.0×** |
| **sparse attention — backward** | **cuDNN-FE** | **3.0–3.5× / 4.3–6.2×** |
| indexer — score gemm (forward) | tilelang at large S | cuDNN-FE ~2× slower at S=8k |
| **indexer — top-k** | **cuDNN-FE (radix)** | **~5× vs torch.topk** |

- The backward dominates the wall-clock win (tilelang's bwd uses `num_stages=0`
  no-pipelining + `atomic_addx4` dKV scatter), but FlashMLA also beats tilelang
  on the forward — so the whole sparse-attention op is faster on our stack.
- This reproduces and updates Kaixi's earlier "~1.6× forward" — still true on
  Hopper, and **larger on Blackwell (2–4×)**.
- A second, zero-friction win: cuDNN-FE's radix `IndexerTopK` replaces Miles'
  `torch.topk` selection for ~5×.

---

## Sparse attention (H=64, D=512, bf16) — latency ms

### Hopper (H100 NVL, sm90)

| S / S_kv | topk | **fwd** tilelang | **fwd** FlashMLA | fwd FMLA win | **bwd** tilelang | **bwd** cuDNN-FE | bwd FE win |
|---|---|---|---|---|---|---|---|
| 2048/4096 | 2048 | 2.02 | 1.17 | 1.73× | 29.34 | 4.81 | **6.10×** |
| 4096/8192 | 2048 | 4.07 | 2.35 | 1.73× | 62.32 | 10.14 | **6.15×** |
| 8192/16384 | 2048 | 8.15 | 5.21 | 1.56× | 127.51 | 21.94 | **5.81×** |
| 2048/4096 | 512 | 0.65 | 0.35 | 1.86× | 7.43 | 1.51 | 4.94× |
| 4096/8192 | 512 | 1.37 | 0.68 | 2.00× | 15.21 | 2.96 | 5.13× |
| 8192/16384 | 512 | 2.60 | 1.46 | 1.78× | 30.88 | 7.13 | 4.33× |

### Blackwell (sm100)

| S / S_kv | topk | **fwd** tilelang | **fwd** FlashMLA | fwd FMLA win | **bwd** tilelang | **bwd** cuDNN-FE | bwd FE win |
|---|---|---|---|---|---|---|---|
| 2048/4096 | 2048 | 2.25 | 0.59 | 3.78× | 15.46 | 4.58 | **3.37×** |
| 4096/8192 | 2048 | 4.66 | 1.21 | 3.86× | 30.49 | 9.26 | **3.29×** |
| 8192/16384 | 2048 | 9.36 | 4.43 | 2.12× | 61.89 | 18.99 | **3.26×** |
| 2048/4096 | 512 | 0.66 | 0.19 | 3.40× | 4.02 | 1.15 | 3.51× |
| 4096/8192 | 512 | 1.32 | 0.44 | 3.02× | 8.00 | 2.65 | 3.02× |
| 8192/16384 | 512 | 2.83 | 0.85 | 3.33× | 16.47 | 5.36 | 3.07× |

- forward: FlashMLA `flash_mla_sparse_fwd` vs Miles `sparse_mqa_fwd_interface`.
- backward: cuDNN-FE `SparseAttentionBackward` vs Miles `sparse_mqa_bwd_interface`.
- cuDNN-FE ships no sparse-attn forward; FlashMLA ships no sparse backward here.

---

## Indexer (Hq=64, D=128, ratio=4, topk=2048) — Blackwell sm100 only — latency ms

cuDNN-FE `IndexerForward` is sm100-only, so this comparison is Blackwell-only.

| S / S_kv | tilelang fwd | cuDNN-FE fwd | FE/tl | torch.topk | cuDNN-FE TopK | **FE/torch** |
|---|---|---|---|---|---|---|
| 2048/4096 | 0.161 | 0.164 | 0.98× | 0.237 | 0.052 | **4.6×** |
| 4096/8192 | 0.511 | 0.549 | 0.93× | 0.647 | 0.121 | **5.4×** |
| 8192/16384 | 1.761 | 3.626 | **0.49×** | 1.862 | 0.381 | **4.9×** |

- **Score gemm (forward):** ~par at small/mid S; at S=8k cuDNN-FE `IndexerForward`
  is ~2× slower than tilelang. Sweeping `q_stage`/`kv_stage` does **not** close
  it (q_stage=1 unsupported; kv_stage 2/3/4 all ~3.1–3.4 ms) — a structural
  large-S regression, an optimization item for the FE kernel team.
- **Top-k:** cuDNN-FE's radix `IndexerTopK` beats `torch.topk` (what Miles uses
  today in `tilelang_indexer.py`) by ~5×. Drop-in replacement, no tilelang change.
- Indexer **backward** is not compared: cuDNN-FE `IndexerBackward` fuses the KL
  auxiliary-loss gradient (different semantics than tilelang's plain
  dScore→dq/dk/dw), so it is not a clean drop-in head-to-head.

---

## Takeaways

1. **Adopt the cuDNN-FE backward** — biggest, clearest win (3–6×), and the bulk
   of the end-to-end advantage.
2. **Adopt FlashMLA forward** — 1.6–3.9× over tilelang; confirms & extends the
   earlier "~1.6×" finding (now up to ~4× on Blackwell).
3. **Swap `torch.topk` → cuDNN-FE `IndexerTopK`** — ~5×, zero tilelang change.
4. **FE indexer score-gemm needs work at large S** (~2× behind tilelang at
   S=8k) before it's worth switching that one piece.

## Notes / caveats

- Backward kernel runtime is determined by shapes (topk_idxs / topk_length),
  not tensor values; forward outputs (o, lse) fed into the backward may be
  synthesized without changing timing.
- Miles tilelang kernels: `radixark/miles@9437366` (byte-identical to internal
  mirror `gitlab-master.nvidia.com/dl/miles/miles`). tilelang 0.1.11.
- cuDNN-FE: dev tree `cudnn_frontend/python/cudnn/deepseek_sparse_attention`.
- FlashMLA: `deepseek-ai/FlashMLA` `flash_mla_sparse_fwd`, built for sm90a+sm100f.
- sm90 cuDNN-FE backward needs a one-line `nvvm.atomicrmw` signature shim under
  cutlass-dsl 4.5.0 (see `_compat.apply_atomicrmw_shim`); not a kernel bug.
