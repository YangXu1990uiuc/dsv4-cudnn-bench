# dsv4-cudnn-bench

Head-to-head benchmarks for **DeepSeek-V4 sparse-attention kernels**: NVIDIA
**cuDNN-Frontend** vs the **Miles tilelang** kernels (the ones Miles' forked
Megatron currently runs for DSv4 RL).

Two kernel families are compared:

1. **Sparse MLA attention** (MQA, K=V, head_dim=512)
   - forward:  Miles `sparse_attn_tilelang`  vs  FlashMLA sparse prefill
   - backward: Miles `sparse_mqa_bwd`         vs  cuDNN-FE `SparseAttentionBackward`
2. **Lightning indexer** (`S = Σ_h ReLU(Q_h·K)·W_h`, ratio-causal, head_dim=128)
   - score gemm: Miles `batched_indexer_fwd`  vs  cuDNN-FE `IndexerForward` *(sm100)*
   - top-k:      `torch.topk` (Miles today)   vs  cuDNN-FE `IndexerTopK` *(sm100)*

> cuDNN-FE intentionally ships **no** sparse-attention forward — production
> forward is **FlashMLA**. FlashMLA ships no sparse backward here. So forward is
> FlashMLA-vs-tilelang and backward is cuDNN-FE-vs-tilelang.

**See [RESULTS.md](RESULTS.md) for measured numbers and conclusions.**

## Headline

The "Megatron+cuDNN" path (FlashMLA forward + cuDNN-FE backward) beats Miles
tilelang on **both** directions of the sparse-attention op:

- **sparse-attn backward: cuDNN-FE 3.0–3.5× (Blackwell) / 4.3–6.2× (Hopper)** over
  tilelang at production shapes (topk=2048, S up to 8k) — the bulk of the win.
- **sparse-attn forward: FlashMLA 2.1–3.9× (Blackwell) / 1.6–2.0× (Hopper)** over
  tilelang — reproduces & extends the earlier "~1.6×" finding.
- **cuDNN-FE radix top-k beats `torch.topk` ~5×** — drop-in replacement for the
  indexer selection Miles does today, no tilelang change required.
- Honest gap: the indexer *score gemm* forward regresses ~2× vs tilelang at S=8k
  (FE optimization item).

## Requirements

- A Hopper (sm90) and/or Blackwell (sm100) GPU. The indexer comparison needs
  sm100 (cuDNN-FE `IndexerForward` is sm100-only).
- CUDA 12.9+ toolkit (FlashMLA sm100 needs NVCC ≥ 12.9); these results used CUDA 13.2.
- Python 3.10, PyTorch (cu13 build used here).

### Dependencies

```bash
# 1. Miles tilelang kernels (public; == internal dl/miles/miles mirror)
git clone https://github.com/radixark/miles            # -> $MILES_ROOT
pip install tilelang                                    # 0.1.11 used here

# 2. cuDNN-Frontend with the DeepSeek-sparse (cutedsl) kernels + nvidia-cutlass-dsl==4.5.0
#    (build/install per cudnn_frontend; ensure `import cudnn; cudnn.DSA` works)

# 3. FlashMLA (sparse prefill forward) — optional, only for the forward column
git clone https://github.com/deepseek-ai/FlashMLA && cd FlashMLA
git submodule update --init --depth 1 csrc/cutlass
MAX_JOBS=8 pip install --no-build-isolation -e .

# 4. this repo
pip install -e .
```

Point the harness at your Miles checkout (default `/home/scratch.yanxu_gpu/miles`):

```bash
export MILES_ROOT=/path/to/miles
```

## Run

```bash
# pick the GPU (enumerate; never hardcode the index)
export CUDA_VISIBLE_DEVICES=<sm90 or sm100 index>

python -m dsv4_cudnn_bench.bench sparse_attn          # fwd+bwd sweep
python -m dsv4_cudnn_bench.bench indexer              # fwd+topk sweep (sm100)
python -m dsv4_cudnn_bench.bench all --output results/run.json
```

Each row prints median latency (ms) per backend and the cuDNN-FE speedup.

## Layout

```
src/dsv4_cudnn_bench/
  data.py              canonical synthetic inputs
  sparse_attention.py  adapters: tilelang fwd/bwd, cudnn_fe bwd, flashmla fwd
  indexer.py           adapters: tilelang fwd + torch.topk, cudnn_fe fwd + topk
  bench.py             CLI runner / sweep
  _compat.py           Miles loader + sm90 cutlass-dsl atomicrmw shim
RESULTS.md             measured numbers + analysis
results/               captured runs
```

## Method notes

- Backward kernel runtime depends on shapes (topk indices/length), not tensor
  values, so synthesized forward outputs give correct backward timing.
- One canonical input layout; each adapter does its own layout conversion
  (cuDNN-FE flattens batch into the sequence dim; tilelang uses `[B,S,H,D]`).
- Miles kernels pinned at `radixark/miles@9437366`.
