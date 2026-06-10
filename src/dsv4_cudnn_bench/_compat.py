"""Environment compat helpers: locate Miles tilelang kernels, apply the sm90
cutlass-dsl atomicrmw signature shim used by cuDNN-FE DSA backward."""
from __future__ import annotations

import importlib.util
import os

# Root of a checkout of github.com/radixark/miles (or the internal mirror
# gitlab-master.nvidia.com/dl/miles/miles -- byte-identical for these kernels).
MILES_ROOT = os.environ.get("MILES_ROOT", "/home/scratch.yanxu_gpu/miles")
_KDIR = f"{MILES_ROOT}/miles_plugins/models/deepseek_v4/ops/kernel"


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def miles_kernel(stem: str):
    """Load a Miles tilelang kernel file by stem, bypassing the miles_plugins
    package __init__ chain (the fwd/bwd files only import tilelang + torch)."""
    return load_module(f"miles_{stem}", f"{_KDIR}/{stem}.py")


def apply_atomicrmw_shim() -> None:
    """cutlass-dsl 4.5.0 binds nvvm.atomicrmw(res, op, ptr, a, ...) with a
    leading result-type arg; cuDNN-FE's sm90 atomic_add_fp32 calls it as
    (op=, ptr=, a=). Inject res=a.type when missing. Inert on sm100 (inline-asm
    atomicAdd). Call before the first cuDNN-FE sm90 backward compile."""
    try:
        import inspect

        from cutlass._mlir.dialects import nvvm as _nvvm

        if "res" in inspect.signature(_nvvm.atomicrmw).parameters:
            _orig = _nvvm.atomicrmw

            def _compat(*a, **k):
                if "res" not in k and "a" in k and not a:
                    k = {"res": k["a"].type, **k}
                return _orig(*a, **k)

            _nvvm.atomicrmw = _compat
    except Exception as exc:  # pragma: no cover
        print(f"# atomicrmw shim skipped: {exc!r}")
