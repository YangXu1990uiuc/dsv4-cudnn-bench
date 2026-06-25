# Source me:  source env.sh
# Sets up the exact environment to run the DSv4 cuDNN-FE vs Miles tilelang benchmarks.
# Re-run after any reboot or overnight /tmp cleanup (the cudnn shim lives in /tmp).

# --- paths (verify these still exist; versions bump over time) ---
export VPY=/home/scratch.yanxu_gpu/flashinfer-bug-repro/.venv/bin/python   # torch2.11+cu130, cutlass-dsl 4.5.0, tilelang 0.1.11, flash_mla
export CUDA_HOME=/home/scratch.yanxu_libs/cuda-13.2
export MILES_ROOT=/home/scratch.yanxu_gpu/miles                           # radixark/miles (V4 tilelang kernels)
export FLASHMLA_ROOT=/home/scratch.yanxu_gpu/FlashMLA                     # deepseek-ai/FlashMLA (sparse fwd), built editable
export BENCH_ROOT=/home/scratch.yanxu_gpu/dsv4-cudnn-bench
FE_CUDNN=/home/scratch.yanxu_libs/fe/lib/python3.10/site-packages/cudnn   # DSA-capable cudnn pkg (1.25.0)

export LD_LIBRARY_PATH=/home/scratch.yanxu_libs/cudnn/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- cudnn shim: the repro venv's own cudnn lacks DSA; shim in the fe-venv one.
#     /tmp is wiped periodically, so (re)create it every time. ---
rm -rf /tmp/cudnn_shim && mkdir -p /tmp/cudnn_shim && ln -s "$FE_CUDNN" /tmp/cudnn_shim/cudnn

export PYTHONPATH=/tmp/cudnn_shim:$BENCH_ROOT/src

# --- GPU pick helper: never hardcode the index (CUDA order != nvidia-smi order) ---
# Usage:  export CUDA_VISIBLE_DEVICES=$(gpu_idx 9)    # sm90 (Hopper)
#         export CUDA_VISIBLE_DEVICES=$(gpu_idx 10)   # sm100 (Blackwell)
gpu_idx() { $VPY -c "import torch,sys
m=int(sys.argv[1])
print(next(i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_properties(i).major==m))" "$1"; }

echo "env ready. VPY=$VPY  MILES_ROOT=$MILES_ROOT"
echo "run e.g.:  export CUDA_VISIBLE_DEVICES=\$(gpu_idx 10); \$VPY -m dsv4_cudnn_bench.bench all"
