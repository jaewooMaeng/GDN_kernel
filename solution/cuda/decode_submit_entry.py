from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tvm_ffi
import tvm_ffi.cpp

_EXACT_CUDA_ARCH_LIST = "10.0a"
_EXACT_CUDA_GENCODE_FLAG = "-gencode=arch=compute_100a,code=sm_100a"
_ANNOUNCED_BUILD_KEYS: set[tuple[str, str, str]] = set()
_SOLUTION_DIR = Path(__file__).resolve().parent
_KERNEL_PATH = _SOLUTION_DIR / "kernel.cu"
_ENTRY_PATH = _SOLUTION_DIR / "decode_submit_entry.py"
_SOURCE_KEY_CACHE: dict[tuple[int, int, int, int, str], tuple[str, str, str]] = {}


@dataclass(frozen=True)
class _BuildArtifact:
    mod: Any
    kernel: Any
    module_name: str
    build_dir: Path
    arch_list: str
    source_key: tuple[str, str, str]
    arch_proof: str
    cuda_home: str


_MODULE_CACHE: dict[tuple[str, str, str], _BuildArtifact] = {}


def _normalize_scale(scale: Any, head_dim: int) -> float:
    if scale is None:
        return 1.0 / (head_dim**0.5)
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    scale_val = float(scale)
    if scale_val == 0.0:
        return 1.0 / (head_dim**0.5)
    return scale_val


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _source_key(active_arch: str) -> tuple[str, str, str]:
    kernel_stat = _KERNEL_PATH.stat()
    entry_stat = _ENTRY_PATH.stat()
    cache_key = (
        kernel_stat.st_mtime_ns,
        kernel_stat.st_size,
        entry_stat.st_mtime_ns,
        entry_stat.st_size,
        active_arch,
    )
    cached = _SOURCE_KEY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    source_key = (_file_digest(_KERNEL_PATH), _file_digest(_ENTRY_PATH), active_arch)
    _SOURCE_KEY_CACHE.clear()
    _SOURCE_KEY_CACHE[cache_key] = source_key
    return source_key


def _force_exact_cuda_arch() -> tuple[str | None, str]:
    previous = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = _EXACT_CUDA_ARCH_LIST
    return previous, _EXACT_CUDA_ARCH_LIST


def _restore_cuda_arch(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        return
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous


def _ensure_cuda_home() -> str:
    existing = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if existing:
        return existing
    default_cuda_home = Path("/usr/local/cuda")
    if default_cuda_home.exists():
        resolved = str(default_cuda_home)
        os.environ["CUDA_HOME"] = resolved
        os.environ["CUDA_PATH"] = resolved
        return resolved
    return "<unset>"


def _classify_arch_proof(build_dir: Path) -> str:
    build_ninja = build_dir / "build.ninja"
    if not build_ninja.exists():
        return "missing-build-ninja"
    build_text = build_ninja.read_text(encoding="utf-8")
    if _EXACT_CUDA_GENCODE_FLAG in build_text:
        return "hard"
    return "soft-env-only"


def _announce_build_surface(artifact: _BuildArtifact, previous_arch: str | None) -> None:
    if artifact.source_key in _ANNOUNCED_BUILD_KEYS:
        return
    _ANNOUNCED_BUILD_KEYS.add(artifact.source_key)
    previous = previous_arch if previous_arch is not None else "<unset>"
    print(
        "[decode-build-surface] "
        f"mode=python-runtime-compile "
        f"target={artifact.arch_list} "
        f"gencode={_EXACT_CUDA_GENCODE_FLAG} "
        f"proof={artifact.arch_proof} "
        f"previous_target={previous} "
        f"cuda_home={artifact.cuda_home} "
        f"module={artifact.module_name} "
        f"build_dir={artifact.build_dir}",
        flush=True,
    )


def _load_mod() -> _BuildArtifact:
    previous_arch, active_arch = _force_exact_cuda_arch()
    try:
        cuda_home = _ensure_cuda_home()
        source_key = _source_key(active_arch)
        cached = _MODULE_CACHE.get(source_key)
        if cached is not None:
            _announce_build_surface(cached, previous_arch)
            return cached

        module_name = (
            "gdn_decode_submit_tvmffi_ext_"
            f"{source_key[0][:8]}_{source_key[1][:8]}_{active_arch.replace('.', '')}"
        )
        try:
            lib_path = tvm_ffi.cpp.build(
                name=module_name,
                cuda_files=[str(_KERNEL_PATH)],
            )
        except RuntimeError as exc:
            if "Could not find CUDA installation" in str(exc):
                raise RuntimeError(
                    "decode exact-surface runtime compile requires CUDA_HOME/nvcc visibility. "
                    f"Resolved CUDA_HOME={cuda_home}. Original error: {exc}"
                ) from exc
            raise
        build_dir = Path(lib_path).resolve().parent
        artifact = _BuildArtifact(
            mod=tvm_ffi.load_module(lib_path),
            kernel=tvm_ffi.load_module(lib_path)["kernel"],
            module_name=module_name,
            build_dir=build_dir,
            arch_list=active_arch,
            source_key=source_key,
            arch_proof=_classify_arch_proof(build_dir),
            cuda_home=cuda_home,
        )
        if artifact.arch_proof != "hard":
            raise RuntimeError(
                "decode exact-surface runtime compile did not produce hard sm_100a proof. "
                f"proof={artifact.arch_proof} build_dir={artifact.build_dir}"
            )
        _MODULE_CACHE[source_key] = artifact
        _announce_build_surface(artifact, previous_arch)
        return artifact
    finally:
        _restore_cuda_arch(previous_arch)


def _ensure_contiguous_output(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tensor.is_contiguous():
        return tensor, False
    return tensor.contiguous(), True


def _ensure_contiguous_input(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


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

    artifact = _load_mod()
    scale_val = _normalize_scale(scale, q.shape[-1])

    q = _ensure_contiguous_input(q)
    k = _ensure_contiguous_input(k)
    v = _ensure_contiguous_input(v)
    state = _ensure_contiguous_input(state)
    A_log = _ensure_contiguous_input(A_log)
    a = _ensure_contiguous_input(a)
    dt_bias = _ensure_contiguous_input(dt_bias)
    b = _ensure_contiguous_input(b)
    output_buffer, copy_output_back = _ensure_contiguous_output(output)
    new_state_buffer, copy_state_back = _ensure_contiguous_output(new_state)

    artifact.kernel(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        scale_val,
        output_buffer,
        new_state_buffer,
    )

    if copy_output_back:
        output.copy_(output_buffer)
    if copy_state_back:
        new_state.copy_(new_state_buffer)
