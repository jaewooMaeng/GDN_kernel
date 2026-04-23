from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import tvm_ffi.cpp

_MOD = None
_KERNEL = None


def _normalize_scale(scale: Any, head_dim: int) -> float:
    if scale is None:
        return 1.0 / (head_dim**0.5)
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    scale_val = float(scale)
    if scale_val == 0.0:
        return 1.0 / (head_dim**0.5)
    return scale_val


def _load_kernel():
    global _MOD, _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    solution_dir = Path(__file__).resolve().parent
    build_dir = Path(tempfile.mkdtemp(prefix="decode-submit-tvmffi-build-"))
    os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "10.0a")
    _MOD = tvm_ffi.cpp.load(
        name="gdn_decode_submit_tvmffi_ext",
        cuda_files=[str(solution_dir / "kernel.cu")],
        build_directory=str(build_dir),
    )
    _KERNEL = _MOD["kernel"]
    return _KERNEL


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float | torch.Tensor | None,
    output: torch.Tensor,
    new_state: torch.Tensor,
) -> None:
    if state is None:
        raise RuntimeError("decode DPS surface expects an incoming state tensor")
    kernel = _load_kernel()
    scale_val = _normalize_scale(scale, q.shape[-1])
    kernel(q, k, v, state, A_log, a, dt_bias, b, scale_val, output, new_state)
