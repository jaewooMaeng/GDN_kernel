# GDN Decode NCU Profiling Log - 2026-04-23

This file records the Modal B200 Nsight Compute result for the current
`solution/cuda/kernel.cu` path so later optimization iterations can use it as
a baseline.

## Context

- Date: 2026-04-23
- Target: Modal B200, compute capability 10.0
- Definition: `gdn_decode_qk4_v8_d128_k_last`
- Solution: `gdn-decode-v1`
- Active config:
  - `language = "cuda"`
  - `entry_point = "kernel.cu::kernel"`
  - `destination_passing_style = true`
- Kernel path:
  - `BLOCK_SIZE = 128`
  - `NUM_WARPS = 4`
  - For `batch_size >= 32`, host dispatch uses `split_factor = 2`
  - For the profiled B=64 workload, this selects `gdn_decode_kernel<16>`

## Commands

Pack:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/python scripts/pack_solution.py
```

Modal benchmark sanity check:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/modal run scripts/run_modal.py
```

NCU profiling:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/modal run scripts/run_ncu_modal.py \
  --workload-uuid eaf0a285-447c-4432-8e68-d287acc3cb08 \
  --ncu-set detailed
```

Important Modal caveat:

- The README helper path `flashinfer_bench_run_ncu(..., set="detailed", page="details")`
  ran first, but returned `No kernels were profiled`.
- The helper uses `--nvtx --nvtx-include flashinfer_bench_ncu_profile`.
- On this Modal run, the NVTX include filter did not match the CUDA launch.
- `scripts/run_ncu_modal.py` now falls back to the same flashinfer-bench runner
  without NVTX filtering and profiles by kernel name:

```bash
ncu --page details --set detailed --target-processes all \
  --kernel-name regex:.*gdn_decode_kernel.* \
  --launch-skip 1 --launch-count 1 -f \
  /usr/local/bin/python -u -m flashinfer_bench.agents._solution_runner \
  --data-dir <tmpdir> --device cuda:0 --trace-set-path /data
```

## Benchmark Result Before Profiling

`modal run scripts/run_modal.py` with the CUDA config passed all workloads:

- Workloads: 54/54 `PASSED`
- Mean latency, arithmetic: `0.012 ms`
- B=64 workloads in this run were around `0.021 ms` to `0.022 ms`

The profiled workload:

- UUID: `eaf0a285-447c-4432-8e68-d287acc3cb08`
- Axes: `batch_size = 64`
- Modal benchmark latency for this workload: `0.021 ms`

## NCU Captured Kernel

NCU successfully profiled:

```text
void gdn_decode_kernel<16>(
  const __nv_bfloat16 *,
  const __nv_bfloat16 *,
  const __nv_bfloat16 *,
  const float *,
  const float *,
  const __nv_bfloat16 *,
  const float *,
  const __nv_bfloat16 *,
  float,
  __nv_bfloat16 *,
  float *,
  int
)
```

Launch shape:

- Grid: `(1024, 1, 1)`
- Block: `(128, 1, 1)`
- Threads: `131072`
- SM count reported by NCU: `148`
- Waves per SM: `0.77`
- NCU replay passes: `21`

## NCU Metrics

Speed of Light:

| Metric | Value |
|---|---:|
| Duration | `33.50 us` |
| Elapsed cycles | `37606` |
| SM frequency | `1.10 GHz` |
| DRAM frequency | `3.99 GHz` |
| Memory throughput | `28.93%` |
| DRAM throughput | `23.55%` |
| L1/TEX throughput | `48.85%` |
| L2 throughput | `19.60%` |
| Compute throughput | `19.88%` |
| SM active cycles | `21930.36` |

Compute workload:

| Metric | Value |
|---|---:|
| Executed IPC active | `1.18 inst/cycle` |
| Executed IPC elapsed | `0.70 inst/cycle` |
| Issue slots busy | `17.51%` |
| SM busy | `17.51%` |

Memory workload:

| Metric | Value |
|---|---:|
| Memory throughput | `1.56 Tbyte/s` |
| Mem busy | `28.93%` |
| Max bandwidth | `23.55%` |
| Mem pipes busy | `19.88%` |
| L1/TEX hit rate | `5.36%` |
| L2 hit rate | `1.48%` |
| L2 persisting size | `82.90 Mbyte` |
| Local memory spilling requests | `0` |
| Local memory spilling overhead | `0%` |
| L2 compression success rate | `0%` |
| L2 compression input sectors | `1074851` |

Launch statistics:

| Metric | Value |
|---|---:|
| Block size | `128` |
| Grid size | `1024` |
| Registers per thread | `56` |
| Static shared memory per block | `512 byte` |
| Dynamic shared memory per block | `0 byte` |
| Driver shared memory per block | `1.02 Kbyte` |
| Shared memory configuration size | `32.77 Kbyte` |
| Block limit registers | `9 blocks/SM` |
| Block limit shared memory | `21 blocks/SM` |
| Block limit warps | `16 blocks/SM` |
| Block limit SM | `32 blocks/SM` |

Occupancy:

| Metric | Value |
|---|---:|
| Theoretical active warps per SM | `36` |
| Theoretical occupancy | `56.25%` |
| Achieved active warps per SM | `18.15` |
| Achieved occupancy | `28.35%` |

Source counters:

| Metric | Value |
|---|---:|
| Branch instructions | `24576` |
| Branch instructions ratio | `0.01%` |
| Branch efficiency | `50%` |
| Avg divergent branches | `6.92` |

## Analysis

The active kernel is not spilling. Register pressure still matters because it
limits theoretical occupancy, but the current `56 registers/thread` is not
causing local memory traffic.

The B=64 grid is still relatively small for B200:

- `1024` blocks over `148` SMs is only `0.77 waves/SM`.
- NCU explicitly reports that the grid is too small to fill the device.
- This limits both compute and memory throughput, even for the largest decode
  batch in the trace set.

The bottleneck is not a single saturated pipeline:

- DRAM throughput is only `23.55%`.
- Memory throughput is only `28.93%`.
- Compute throughput is only `19.88%`.
- Issue slots busy is only `17.51%`.

This points to insufficient parallelism and insufficient bytes in flight, not
classic peak-bandwidth saturation. The kernel does a lot of state traffic, but
the SMs are not generating enough concurrent memory work to drive the memory
system harder.

Cache reuse inside one profiled launch is poor:

- L1/TEX hit rate: `5.36%`
- L2 hit rate: `1.48%`

This matches the expected single-invocation behavior: each state row is largely
streamed once. Cross-invocation cache residency can still matter for benchmark
loops, but this NCU capture should be read as the per-launch hardware profile.

NCU duration is larger than the Modal benchmark latency:

- Modal benchmark for the profiled workload: about `0.021 ms`
- NCU detailed capture: `33.50 us`

This is expected because NCU `--set detailed` replays the kernel across many
passes and changes timing. Use the NCU duration for relative profiling only,
not as the benchmark latency.

## Optimization Implications

Most promising directions:

1. Increase independent work per SM without destroying the 4-row pipeline.
   - More blocks may help only if each block still has enough per-warp ILP.
   - Prior split-factor experiments regressed when rows-per-warp became too
     small, so do not blindly increase `split_factor`.

2. Increase bytes in flight per block or per warp.
   - Current register prefetching is useful but still not enough to saturate
     DRAM.
   - A deeper pipeline could help only if it does not lower occupancy enough to
     offset the gain.

3. Explore a separate B>=32 or B==64 kernel shape.
   - Current B=64 path uses 128-thread blocks and `gdn_decode_kernel<16>`.
   - A B=64-specific variant can be evaluated without disturbing small-batch
     behavior.

4. Consider 256-thread or 8-warp large-batch variants again, but measure in the
   same environment and compare both benchmark latency and NCU occupancy.
   - Previous logs mention 8-warp variants, but the currently profiled code path
     is 128 threads.
   - If trying this, keep the 4-row pipeline invariant and avoid adding spills.

5. If using TMA/cp.async.bulk again, do it only through a build path with full
   control of the Blackwell target flags.
   - Previous attempts crashed under TVM FFI.
   - Treat this as high risk unless the build path is changed.

Directions that look less promising:

1. L1 bypass as a standalone change.
   - Current profile does show low L1 hit rate, but earlier `.cg` experiments
     were neutral or worse.
   - L1/TEX traffic is high, but the core issue is not enough outstanding work.

2. Forcing lower register count with aggressive `__launch_bounds__`.
   - NCU shows no spills now.
   - Previous attempts to force more occupancy caused spills and regressed.

3. Optimizing tiny scalar gate math first.
   - The profile is dominated by launch shape, occupancy, and memory issue rate.
   - Gate math runs once per block and is unlikely to move B=64 latency much.

## Next Iteration Checklist

Before accepting any next optimization:

- Run `python scripts/pack_solution.py`.
- Run `modal run scripts/run_modal.py`.
- Confirm 54/54 workloads are `PASSED`.
- Record arithmetic mean latency.
- For B=64 changes, rerun:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/modal run scripts/run_ncu_modal.py \
  --workload-uuid eaf0a285-447c-4432-8e68-d287acc3cb08 \
  --ncu-set detailed
```

Key NCU fields to compare:

- Grid size and waves/SM
- Registers/thread
- Local memory spilling requests
- Achieved occupancy
- Memory throughput
- DRAM throughput
- Issue slots busy
- L1/L2 hit rates
- Modal benchmark latency for B=64 workloads

