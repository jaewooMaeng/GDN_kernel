"""
Nsight Compute (NCU) profiling on Modal B200 via flashinfer_bench_run_ncu.

Not relied upon in this repo: FAQ states NCU is not officially available on Modal,
and long runs here often produce no usable log. Prefer `modal run scripts/run_modal.py`
for tuning, or `scripts/run_ncu_local.py` on a Linux box with `ncu` installed.

This image adds cuda-nsight-compute from NVIDIA's apt repo so `ncu` is on PATH.

Usage:
  conda run -n fi-bench modal run scripts/run_ncu_modal.py
  conda run -n fi-bench modal run scripts/run_ncu_modal.py --workload-uuid <uuid>

Requires: modal setup, flashinfer-trace volume at /data (same as run_modal.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench-ncu")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# CUDA 13 Nsight Compute CLI — base nvidia/cuda:devel image already configures NVIDIA apt repo; do not reinstall cuda-keyring (conflicts Signed-By).
_ncu_install = (
    "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-nsight-compute-13-0"
)

ncu_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .run_commands(_ncu_install)
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


def _pick_workload(workloads, workload_uuid: str | None):
    if not workloads:
        raise ValueError("No workloads for definition")

    def as_workload(item):
        return getattr(item, "workload", item)

    if workload_uuid:
        for item in workloads:
            w = as_workload(item)
            if w.uuid == workload_uuid:
                return w
        raise ValueError(f"Workload uuid not found: {workload_uuid}")

    def batch_score(w) -> int:
        w = as_workload(w)
        ax = w.axes
        for k in ("B", "batch", "batch_size"):
            if k in ax:
                return int(ax[k])
        return max(ax.values()) if ax else 0

    return as_workload(max(workloads, key=batch_score))


@app.function(
    image=ncu_image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_ncu_profile_remote(
    solution_json: str,
    workload_uuid: str | None = None,
    ncu_set: str = "detailed",
    page: str = "details",
    timeout_sec: int = 600,
    max_lines: int | None = 800,
) -> str:
    from flashinfer_bench import Solution, TraceSet
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    solution = Solution.model_validate_json(solution_json)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    workloads = trace_set.workloads.get(solution.definition, [])
    workload = _pick_workload(workloads, workload_uuid)

    out = flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        trace_set_path=TRACE_SET_PATH,
        set=ncu_set,
        page=page,
        timeout=timeout_sec,
        max_lines=max_lines,
    )
    if "No kernels were profiled" not in out:
        return out

    # Modal currently misses the helper's NVTX include range. Fall back to the
    # same flashinfer-bench runner, but filter directly on the CUDA kernel name.
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    definition = trace_set.definitions[solution.definition]
    with tempfile.TemporaryDirectory(prefix="fib_ncu_modal_") as build_dir:
        build_path = Path(build_dir)
        (build_path / "definition.json").write_text(definition.model_dump_json())
        (build_path / "solution.json").write_text(solution.model_dump_json())
        (build_path / "workload.json").write_text(workload.model_dump_json())

        cmd = [
            "ncu",
            "--page",
            page,
            "--set",
            ncu_set,
            "--target-processes",
            "all",
            "--kernel-name",
            "regex:.*gdn_decode_kernel.*",
            "--launch-skip",
            "1",
            "--launch-count",
            "1",
            "-f",
            sys.executable,
            "-u",
            "-m",
            "flashinfer_bench.agents._solution_runner",
            "--data-dir",
            str(build_path),
            "--device",
            "cuda:0",
            "--trace-set-path",
            TRACE_SET_PATH,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=timeout_sec,
        )
        fallback_out = result.stdout + result.stderr
        if result.returncode != 0:
            fallback_out = (
                f"ERROR: fallback NCU exited with non-zero return code {result.returncode}:\n"
                f"{fallback_out}"
            )
        if max_lines is not None:
            lines = fallback_out.splitlines()
            if len(lines) > max_lines:
                fallback_out = "\n".join(lines[:max_lines])
                fallback_out += f"\n[Output truncated: {len(lines) - max_lines} more lines]"
        return (
            "README helper output had no profiled kernels; retried without NVTX filter.\n"
            f"Fallback command: {' '.join(cmd)}\n\n"
            f"{fallback_out}"
        )


@app.local_entrypoint()
def main(workload_uuid: str | None = None, ncu_set: str = "detailed"):
    from scripts.pack_solution import pack_solution

    print("Packing solution...")
    solution_path = pack_solution()
    solution_json = solution_path.read_text()
    print("Running NCU on Modal B200 (this may take several minutes)...")
    out = run_ncu_profile_remote.remote(
        solution_json,
        workload_uuid=workload_uuid,
        ncu_set=ncu_set,
    )
    print(out)
