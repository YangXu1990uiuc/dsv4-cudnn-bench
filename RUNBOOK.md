# RUNBOOK — environment & reproduction (durable)

Everything needed to re-run / revisit the DSv4 cuDNN-FE vs Miles tilelang
benchmarks. All paths are on NFS (`/home/scratch.*`, shared across hosts) so
they survive reboots — **except `/tmp/cudnn_shim`, which gets wiped** (re-created
by `env.sh`). Verify versions/paths still exist before trusting them.

## What is being compared (V4)

DeepSeek **V4** sparse attention family. CSA (Compressed Sparse Attention) and
HCA (Heavily Compressed Attention) **share the same attention kernel**; they
differ only in the upstream compressor ratio (CSA ~4× stride-4; HCA c4a 8→1 /
c128a 128→1) and the indexer scoring over compressed entries. So:

- **sparse attention** (the CSA/HCA attention kernel): fwd = FlashMLA,
  bwd = cuDNN-FE `SparseAttentionBackward`; baseline = Miles tilelang `sparse_mqa`.
- **indexer** (lightning indexer): fwd score gemm + top-k; cuDNN-FE
  `IndexerForward`+`IndexerTopK` (sm100) vs Miles tilelang + `torch.topk`.
- **compressor / mHC**: NOT here — that's deepseek-ai/TileKernels, benchmarked
  separately (Jingqin: Megatron cuTile vs Triton vs TileKernels).

## Components & pinned versions (as run 2026-06-10)

| Component | Path | Pin / version |
|---|---|---|
| repro venv (python3.10) | `/home/scratch.yanxu_gpu/flashinfer-bug-repro/.venv` | torch 2.11+cu130, nvidia-cutlass-dsl 4.5.0, tilelang 0.1.11, flash_mla 1.0.0+9241ae3 |
| CUDA toolkit | `/home/scratch.yanxu_libs/cuda-13.2` | 13.2 (nvcc V13.2.46) |
| cuDNN backend .so | `/home/scratch.yanxu_libs/cudnn/lib` | dev build w/ DSA |
| cuDNN-FE python pkg (DSA) | `/home/scratch.yanxu_libs/fe/.../site-packages/cudnn` | 1.25.0 (shimmed into repro venv) |
| Miles (tilelang V4 kernels) | `/home/scratch.yanxu_gpu/miles` | `radixark/miles@9437366` (== internal `dl/miles/miles`, byte-identical) |
| FlashMLA (sparse fwd) | `/home/scratch.yanxu_gpu/FlashMLA` | `deepseek-ai/FlashMLA@9241ae3`, built editable for sm90a+sm100f |
| this harness | `/home/scratch.yanxu_gpu/dsv4-cudnn-bench` | git repo |

GPUs on the box (re-enumerate; CUDA order ≠ nvidia-smi order): L40S sm89,
Blackwell sm100, H100 NVL sm90, A100 sm80. Use `gpu_idx 90`/`gpu_idx 100`.

## Run

```bash
cd /home/scratch.yanxu_gpu/dsv4-cudnn-bench
source env.sh                                  # recreates /tmp/cudnn_shim, exports all paths
export CUDA_VISIBLE_DEVICES=$(gpu_idx 100)     # 100=Blackwell(sm100), 90=Hopper(sm90)
$VPY -m dsv4_cudnn_bench.bench all --output results/run.json
# or: sparse_attn  |  indexer   (indexer needs sm100)
```

## Gotchas (each one cost time — do not rediscover)

1. **`/tmp/cudnn_shim` vanishes overnight.** The repro venv's own `cudnn` has no
   `DSA`; we symlink the fe-venv's. `env.sh` recreates it. Symptom if missing:
   `ImportError: cannot import name 'DSA' from 'cudnn'`.
2. **sm90 cuDNN-FE backward** hits `TypeError: atomicrmw() missing 'res'` under
   cutlass-dsl 4.5.0 (signature drift). Fixed by `_compat.apply_atomicrmw_shim()`
   (injects `res=a.type`). Not a kernel bug. sm100 unaffected (inline-asm).
3. **FlashMLA build on CUDA 13** needs two include fixes — CUDA 13 split headers:
   ```bash
   export CPATH=$CUDA_HOME/include/cccl:<venv>/lib/python3.10/site-packages/nvidia/cu13/include:$CPATH
   ```
   (else `fatal error: cusparse.h` then `cuda/std/utility: No such file`).
   And `MAX_JOBS=8` (default OOMs the heavy sm100 fmha .cu compiles). Also: a
   piped build can report exit 0 from the pipe even when pip failed — grep the
   log for `Successfully installed` / `Error compiling`. Submodule cutlass needs
   `git submodule update --init --depth 1 csrc/cutlass` (shallow clone omits it).
4. **Indexer forward is only apples-to-apples at `sq == s_kv`.** cuDNN-FE masks
   bottom-right (`q<=k+(s_kv-sq)`), Miles top-left (`q<=k`); when `sq!=s_kv` they
   compute different workloads. Megatron prefill = `sq==s_kv` → masks coincide.
   (h/t Jiayu Sun.) top-k and the sparse-attn fwd/bwd are unaffected (explicit
   topk_idxs, no in-kernel causal mask).

## Results

See [RESULTS.md](RESULTS.md). Raw captures in `results/` (`*_all.{json,txt}` =
full sweeps; `*_indexer_sqeqsk.*` = corrected prefix-aligned indexer).

## Ad-hoc probes

`scripts/bench_dsa_bprop.py` (bwd-only abs perf, both arches),
`scripts/bench_indexer_fwd_tune.py` (IndexerForward q_stage/kv_stage sweep).
