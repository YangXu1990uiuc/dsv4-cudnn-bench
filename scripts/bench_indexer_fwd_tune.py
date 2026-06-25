"""Probe cuDNN-FE IndexerForward perf vs (q_stage, kv_stage) at large S, to
explain the large-S regression vs tilelang. Blackwell sm100."""
import importlib.util, math, torch
MILES_K = "/home/scratch.yanxu_gpu/miles/miles_plugins/models/deepseek_v4/ops/kernel"
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
tl = _load("tlifwd", f"{MILES_K}/tilelang_indexer_fwd.py")
from cudnn import DSA
from cuda.bindings import driver as cuda

def timed(fn, w=5, it=30):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    st=[torch.cuda.Event(enable_timing=True) for _ in range(it)]; en=[torch.cuda.Event(enable_timing=True) for _ in range(it)]
    for i in range(it): st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    t=sorted(s.elapsed_time(e) for s,e in zip(st,en)); return t[len(t)//2]

S, Skv, Hq, D, ratio = 8192, 16384, 64, 128, 4
dev="cuda"; scale=1.0/math.sqrt(D)
q_sb=torch.randn(S,1,Hq,D,dtype=torch.bfloat16,device=dev); k_sb=torch.randn(Skv,1,D,dtype=torch.bfloat16,device=dev)
w_sb=torch.randn(S,1,Hq,dtype=torch.float32,device=dev)
cu_ks,cu_ke=tl._make_causal_cu_seqlens(S,Skv,ratio,dev)
tl_ms=timed(lambda: tl.batched_indexer_fwd(q_sb,k_sb,w_sb,cu_ks,cu_ke))
print(f"# S={S} Skv={Skv} Hq={Hq} D={D}  tilelang_fwd={tl_ms:.3f} ms")
q_b=q_sb.permute(1,0,2,3).contiguous(); k_b=k_sb.permute(1,0,2).contiguous().unsqueeze(2); w_b=w_sb.permute(1,0,2).contiguous().to(torch.bfloat16)
stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
print(f"# {'q_stage':>7} {'kv_stage':>8} {'fe_ms':>8} {'vs_tl':>6}")
for qs in (1,2):
    for kvs in (2,3,4):
        try:
            fn=lambda qs=qs,kvs=kvs: DSA.indexer_forward_wrapper(q_b,k_b,w_b,ratio=ratio,qhead_per_kv_head=Hq,sm_scale=scale,q_stage=qs,kv_stage=kvs,stream=stream)
            fn(); torch.cuda.synchronize()
            ms=timed(fn); print(f"  {qs:>7} {kvs:>8} {ms:>8.3f} {tl_ms/ms:>5.2f}x")
        except Exception as e:
            print(f"  {qs:>7} {kvs:>8}  ERR {repr(e)[:60]}")
        torch.cuda.empty_cache()
