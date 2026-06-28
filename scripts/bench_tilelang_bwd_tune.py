"""Is the tilelang bwd really 3-6x slower, or just untuned? Miles ships the bwd
with num_stages=0 (no pipelining). Sweep num_stages/block_size/threads and
compare to cuDNN-FE. Run with env.sh sourced; CUDA_VISIBLE_DEVICES set."""
import importlib.util, math, torch
MILES_K = "/home/scratch.yanxu_gpu/miles/miles_plugins/models/deepseek_v4/ops/kernel"
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
fwd = _load("tlfwd", f"{MILES_K}/tilelang_sparse_mla_fwd.py")
bwd = _load("tlbwd", f"{MILES_K}/tilelang_sparse_mla_bwd.py")
from dsv4_cudnn_bench._compat import apply_atomicrmw_shim
apply_atomicrmw_shim()
from cudnn import DSA
from cuda.bindings import driver as cuda

def timed(fn, w=5, it=30):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    st=[torch.cuda.Event(enable_timing=True) for _ in range(it)]; en=[torch.cuda.Event(enable_timing=True) for _ in range(it)]
    for i in range(it): st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    t=sorted(s.elapsed_time(e) for s,e in zip(st,en)); return t[len(t)//2]

B,S,Skv,H,D,topk = 1,4096,8192,64,512,2048
dev="cuda"; scale=1.0/math.sqrt(D)
q=torch.randn(B,S,H,D,dtype=torch.bfloat16,device=dev); kv=torch.randn(B,Skv,D,dtype=torch.bfloat16,device=dev)
sink=torch.randn(H,dtype=torch.float32,device=dev)
idx=torch.argsort(torch.rand(B,S,Skv,device=dev),dim=-1)[...,:topk].to(torch.int32).contiguous()
o,lse=fwd.sparse_mqa_fwd_interface(q,kv,sink,idx,sm_scale=scale); do=torch.randn_like(o)

pre=bwd.preprocess(B,S,H,D); post=bwd.postprocess(B,Skv,D); delta=pre(o,do)
def run_bwd(bk):
    dkv=torch.zeros_like(kv,dtype=torch.float32); dsink=torch.zeros_like(sink)
    dq=bk(q,kv,do,sink,idx,lse,delta,dkv,dsink); return post(dkv)

# cuDNN-FE baseline
qf=q.reshape(S,H,D).contiguous(); kvf=kv.reshape(Skv,D).contiguous(); of=o.reshape(S,H,D).contiguous()
dof=do.reshape(S,H,D).contiguous(); idxf=idx.reshape(S,topk).contiguous()
lse_fe=torch.full((S,H),float(math.log(topk)),dtype=torch.float32,device=dev)
stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
def fe(): return DSA.sparse_attention_backward_wrapper(qf,kvf,of,dof,lse_fe,sink,idxf,softmax_scale=scale,stream=stream)
fe(); torch.cuda.synchronize(); fe_ms=timed(fe)
print(f"# S={S} Skv={Skv} H={H} D={D} topk={topk}   cuDNN-FE bwd = {fe_ms:.3f} ms")
print(f"# {'num_stages':>10} {'block_size':>10} {'threads':>7} {'tilelang_ms':>11} {'fe/tl':>6}")

configs=[(0,32,128),(1,32,128),(2,32,128),(3,32,128),(2,64,128),(2,32,256),(2,64,256),(3,64,128)]
for ns,bs,th in configs:
    try:
        if topk % bs: print(f"  {ns:>10} {bs:>10} {th:>7}  skip (topk%bs)"); continue
        bk=bwd.bwd(B,S,Skv,H,D,topk,scale,block_size=bs,num_stages=ns,threads=th)
        run_bwd(bk); torch.cuda.synchronize()
        ms=timed(lambda bk=bk: run_bwd(bk))
        print(f"  {ns:>10} {bs:>10} {th:>7} {ms:>11.3f} {ms/fe_ms:>5.2f}x")
    except Exception as e:
        print(f"  {ns:>10} {bs:>10} {th:>7}  ERR {repr(e)[:55]}")
    torch.cuda.empty_cache()
