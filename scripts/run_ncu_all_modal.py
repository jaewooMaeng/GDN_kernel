"""
Nsight Compute (NCU) profiling for ALL workloads on Modal B200.

Iterates every workload of the packed solution's definition, profiles each via
`flashinfer_bench_run_ncu` (with the same kernel-name fallback as
``scripts/run_ncu_modal.py``), parses the Duration from the NCU SOL section,
and reports per-workload durations plus their arithmetic mean.

Usage:
  conda run -n fi-bench modal run scripts/run_ncu_all_modal.py
  conda run -n fi-bench modal run scripts/run_ncu_all_modal.py --ncu-set detailed
  conda run -n fi-bench modal run scripts/run_ncu_all_modal.py --save-json ncu_all.json

Requires: modal setup, flashinfer-trace volume at /data (same as run_modal.py).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench-ncu-all")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

_ncu_install = (
    "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-nsight-compute-13-0"
)

ncu_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .run_commands(_ncu_install)
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

# Matches the "Duration" row from NCU's GPU Speed Of Light section, e.g.:
#     Duration                         us        32.51
#     Duration                    usecond        32.51
# Unit column may be short (ns/us/ms/s) or long (nsecond/usecond/msecond/second).
_DURATION_RE = re.compile(
    r"^\s*Duration\s+(nsecond|usecond|msecond|second|ns|us|ms|s)\s+([0-9.,]+)\s*$",
    re.MULTILINE,
)

_UNIT_TO_US = {
    "ns": 1e-3,
    "nsecond": 1e-3,
    "us": 1.0,
    "usecond": 1.0,
    "ms": 1e3,
    "msecond": 1e3,
    "s": 1e6,
    "second": 1e6,
}


def _parse_duration_us(ncu_out: str) -> float | None:
    m = _DURATION_RE.search(ncu_out)
    if not m:
        return None
    unit = m.group(1)
    val = float(m.group(2).replace(",", ""))
    return val * _UNIT_TO_US[unit]


@app.function(
    image=ncu_image,
    gpu="B200:1",
    timeout=7200,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_ncu_all_remote(
    solution_json: str,
    ncu_set: str = "detailed",
    page: str = "details",
    timeout_sec: int = 600,
    display_max_lines: int | None = 400,
) -> dict:
    import os
    import subprocess
    import tempfile

    from flashinfer_bench import Solution, TraceSet
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    solution = Solution.model_validate_json(solution_json)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    items = trace_set.workloads.get(solution.definition, [])
    if not items:
        raise ValueError(f"No workloads for definition '{solution.definition}'")

    definition = trace_set.definitions[solution.definition]

    def _as_workload(item):
        return getattr(item, "workload", item)

    def _fallback_ncu(workload) -> str:
        with tempfile.TemporaryDirectory(prefix="fib_ncu_modal_") as build_dir:
            build_path = Path(build_dir)
            (build_path / "definition.json").write_text(definition.model_dump_json())
            (build_path / "solution.json").write_text(solution.model_dump_json())
            (build_path / "workload.json").write_text(workload.model_dump_json())

            cmd = [
                "ncu",
                "--page", page,
                "--set", ncu_set,
                "--target-processes", "all",
                "--kernel-name", "regex:.*gdn_decode_kernel.*",
                "--launch-skip", "1",
                "--launch-count", "1",
                "-f",
                sys.executable,
                "-u", "-m", "flashinfer_bench.agents._solution_runner",
                "--data-dir", str(build_path),
                "--device", "cuda:0",
                "--trace-set-path", TRACE_SET_PATH,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return f"ERROR: fallback NCU timed out after {timeout_sec}s\nCommand: {' '.join(cmd)}"
            fb_out = (result.stdout or "") + (result.stderr or "")
            if result.returncode != 0:
                fb_out = (
                    f"ERROR: fallback NCU exited with non-zero return code {result.returncode}:\n"
                    f"{fb_out}"
                )
            return (
                "README helper output had no profiled kernels; retried without NVTX filter.\n"
                f"Fallback command: {' '.join(cmd)}\n\n"
                f"{fb_out}"
            )

    per_workload: list[dict] = []
    for idx, item in enumerate(items):
        w = _as_workload(item)
        axes = dict(w.axes) if getattr(w, "axes", None) else {}
        print(f"[{idx + 1}/{len(items)}] profiling workload {w.uuid} axes={axes}", flush=True)

        try:
            out = flashinfer_bench_run_ncu(
                solution=solution,
                workload=w,
                trace_set_path=TRACE_SET_PATH,
                set=ncu_set,
                page=page,
                timeout=timeout_sec,
                max_lines=None,
            )
        except Exception as e:  # noqa: BLE001
            out = f"ERROR: flashinfer_bench_run_ncu raised: {e!r}"

        used_fallback = False
        if "No kernels were profiled" in out or out.startswith("ERROR:"):
            used_fallback = True
            out = _fallback_ncu(w)

        duration_us = _parse_duration_us(out)

        display = out
        if display_max_lines is not None:
            lines = display.splitlines()
            if len(lines) > display_max_lines:
                display = "\n".join(lines[:display_max_lines])
                display += f"\n[Output truncated: {len(lines) - display_max_lines} more lines]"

        per_workload.append(
            {
                "uuid": w.uuid,
                "axes": axes,
                "used_fallback": used_fallback,
                "duration_us": duration_us,
                "output": display,
            }
        )

    durations = [e["duration_us"] for e in per_workload if e["duration_us"] is not None]
    mean_us = sum(durations) / len(durations) if durations else None
    return {
        "definition": solution.definition,
        "ncu_set": ncu_set,
        "page": page,
        "per_workload": per_workload,
        "mean_duration_us": mean_us,
        "parsed_count": len(durations),
        "total_count": len(per_workload),
    }


def _format_axes(axes: dict) -> str:
    if not axes:
        return "-"
    return ",".join(f"{k}={v}" for k, v in sorted(axes.items()))


@app.local_entrypoint()
def main(
    ncu_set: str = "detailed",
    save_json: str | None = None,
    print_outputs: bool = False,
):
    from scripts.pack_solution import pack_solution

    print("Packing solution...")
    solution_path = pack_solution()
    solution_json = solution_path.read_text()

    print("Running NCU on ALL workloads on Modal B200 (this may take a while)...")
    result = run_ncu_all_remote.remote(solution_json, ncu_set=ncu_set)

    header = f"{'UUID':<38} {'axes':<40} {'duration_us':>12} {'fallback':>9}"
    print()
    print("=== NCU Duration per workload ===")
    print(header)
    print("-" * len(header))
    for entry in result["per_workload"]:
        dur = entry["duration_us"]
        dur_str = f"{dur:.3f}" if dur is not None else "N/A"
        axes_str = _format_axes(entry["axes"])
        print(
            f"{entry['uuid']:<38} {axes_str:<40} {dur_str:>12} "
            f"{str(entry['used_fallback']):>9}"
        )

    mean = result["mean_duration_us"]
    parsed, total = result["parsed_count"], result["total_count"]
    print()
    if mean is not None:
        print(f"Arithmetic mean over {parsed}/{total} workloads: {mean:.3f} us")
    else:
        print(f"Could not parse Duration from any of {total} workloads' NCU output.")

    if print_outputs:
        print("\n=== Per-workload NCU outputs ===")
        for entry in result["per_workload"]:
            print(f"\n----- {entry['uuid']} axes={_format_axes(entry['axes'])} -----")
            print(entry["output"])

    if save_json:
        import json

        Path(save_json).write_text(json.dumps(result, indent=2))
        print(f"\nSaved full result to {save_json}")
