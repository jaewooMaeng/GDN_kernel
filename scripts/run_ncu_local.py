#!/usr/bin/env python3
"""Run NCU locally when `ncu` is on PATH and CUDA is available (Linux).

  export FIB_DATASET_PATH=/path/to/flashinfer-trace
  python scripts/run_ncu_local.py [--workload-uuid UUID]

Uses flashinfer_bench.agents.flashinfer_bench_run_ncu — same as README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench.agents import flashinfer_bench_run_ncu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload-uuid", default=None, help="Default: workload with largest B axis")
    p.add_argument("--set", default="detailed", dest="ncu_set")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    from scripts.pack_solution import pack_solution

    sol_path = pack_solution()
    from flashinfer_bench import Solution, TraceSet
    import os

    trace_path = os.environ.get("FIB_DATASET_PATH")
    if not trace_path:
        print("ERROR: Set FIB_DATASET_PATH to your flashinfer-trace directory.", file=sys.stderr)
        sys.exit(1)

    solution = Solution.model_validate_json(sol_path.read_text())
    trace_set = TraceSet.from_path(trace_path)
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        print("ERROR: No workloads", file=sys.stderr)
        sys.exit(1)

    if args.workload_uuid:
        wl = next((w for w in workloads if w.uuid == args.workload_uuid), None)
        if not wl:
            print(f"ERROR: uuid not found: {args.workload_uuid}", file=sys.stderr)
            sys.exit(1)
    else:

        def score(w):
            ax = w.axes
            for k in ("B", "batch", "batch_size"):
                if k in ax:
                    return ax[k]
            return max(ax.values()) if ax else 0

        wl = max(workloads, key=score)

    out = flashinfer_bench_run_ncu(
        solution=solution,
        workload=wl,
        trace_set_path=trace_path,
        set=args.ncu_set,
        timeout=args.timeout,
    )
    print(out)


if __name__ == "__main__":
    main()
