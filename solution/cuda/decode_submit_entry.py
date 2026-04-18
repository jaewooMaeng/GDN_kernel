from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import tvm_ffi.cpp

_MOD = None
_ANNOUNCED = False


def _normalize_scale(scale: Any, head_dim: int) -> float:
    if scale is None:
        return 1.0 / (head_dim**0.5)
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    scale_val = float(scale)
    if scale_val == 0.0:
        return 1.0 / (head_dim**0.5)
    return scale_val


def _announce_build_surface() -> None:
    global _ANNOUNCED
    if _ANNOUNCED:
        return
    _ANNOUNCED = True
    target = os.environ.get("TVM_FFI_CUDA_ARCH_LIST", "<unset>")
    print(f"[decode-build-surface] python-self-compile target={target}", flush=True)


def _load_mod():
    global _MOD
    if _MOD is not None:
        return _MOD

    solution_dir = Path(__file__).resolve().parent
    build_dir = Path(tempfile.mkdtemp(prefix="decode-submit-tvmffi-build-"))
    os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "10.0a")
    _announce_build_surface()
    _MOD = tvm_ffi.cpp.load(
        name="gdn_decode_submit_tvmffi_ext",
        cuda_files=[str(solution_dir / "kernel.cu")],
        build_directory=str(build_dir),
    )
    return _MOD


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

    mod = _load_mod()
    scale_val = _normalize_scale(scale, q.shape[-1])

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    state = state.contiguous()
    A_log = A_log.contiguous()
    a = a.contiguous()
    dt_bias = dt_bias.contiguous()
    b = b.contiguous()
    output = output.contiguous()
    new_state = new_state.contiguous()

    mod["kernel"](q, k, v, state, A_log, a, dt_bias, b, scale_val, output, new_state)
