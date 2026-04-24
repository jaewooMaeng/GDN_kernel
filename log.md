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

## iter #1

- 적용한 최적화: D4/D8 일부 실험. `batch_size >= 32`에서 `split_factor`를 2에서 4로 올려 B=64 grid를 1024 blocks에서 2048 blocks로 늘림. Benchmark median 후퇴로 롤백함.
- 측정된 avg latency: 5회 `[0.012853, 0.012956, 0.012696, 0.018082, 0.012920] ms`, median `0.012920 ms`, correctness `PASSED=54/54`.
- NCU Duration: 롤백 후 현재 accepted kernel 기준 `33.89 us`.
- 남아있는 주요 bottleneck: B=64 accepted kernel은 `Grid Size=1024`, `Waves/SM=0.77`, `Issue Slots Busy=17.25%`, `Memory Throughput=28.44%`, `DRAM Throughput=23.65%`, `L1/TEX Throughput=49.20%`, `L2 Throughput=19.27%`, `Achieved Occupancy=30.89%`, `Registers/thread=56`, local spilling `0`, `L1 hit=5.37%`, `L2 hit=1.49%`. 여전히 single pipeline saturation이 아니라 small-grid/low-issue/low-occupancy 성격이 강함.
- 이번에 시도했거나 검토했지만 안 좋다고 판단한 방향:
  - 후보: B>=32 large-batch split 확대 (`split_factor=4`, `ROWS_PER_WARP=8`).
  - 안 좋은 이유: NCU의 parallelism 부족 지표와는 맞았지만 benchmark latency가 후퇴함. B=64 workload가 기존 로그 `0.021~0.022 ms` 수준에서 이번 실험 `0.024~0.029 ms` 수준으로 느려졌고, 전체 median도 기존 `0.012 ms`에서 `0.012920 ms`로 악화됨. block 수 증가보다 q/k/v/gate 중복, launch scheduling overhead, per-block ILP 감소가 더 컸던 것으로 판단.
  - 재시도 가능 조건: q/k/v/gate 중복을 줄이는 cluster/shared broadcast를 함께 넣거나, B=64 전용 커널에서 per-block ILP를 유지하면서 grid만 늘릴 수 있을 때. 단순 split 증가만 재시도하지 않음.
- 다음 iteration 에서 시도할만한 후보 2~3 개:
  - D5 CUDA Graph capture feasibility: avg latency 목표에는 launch overhead 제거가 가장 큰 레버리지. 포인터 안정성/graph update 가능성부터 확인 필요.
  - B1 2-CTA cluster q/k 공유: `V_PER_Q=2` 중복 q/k load를 줄이는 방향. cluster sync overhead와 TVM FFI launch attribute 리스크를 먼저 작게 검증.
  - G5 ptxas register/spill 및 SASS 확인: 현재 56 regs/thread, spill 0이지만 다음 구조 변경 전 기준선을 고정하고 `LDG/FFMA/SHFL` 비율을 확인.

## iter #2

- 적용한 최적화: E1 축소 버전. `q/k/v`를 block shared memory에 1회 staging하고, gate scalar(`g`, `beta`, `beta_g`)를 thread 0에서 한 번만 계산한 뒤 block 전체가 재사용하도록 수정. 첫 benchmark에서 regression 확인 후 즉시 롤백함.
- 측정된 avg latency: 1회 `0.013278 ms`, correctness `PASSED=54/54`. Phase 3 기준선(`0.012920 ms` median)보다 명확히 느려 추가 4회 반복 없이 롤백.
- NCU Duration: 롤백 후 현재 accepted kernel 기준 `32.48 us`.
- 남아있는 주요 bottleneck: accepted kernel은 여전히 `Grid Size=1024`, `Waves/SM=0.77`, `Issue Slots Busy=18.00%`, `Memory Throughput=29.73%`, `DRAM Throughput=24.63%`, `L1/TEX Throughput=49.25%`, `L2 Throughput=20.21%`, `Achieved Occupancy=29.37%`, `Registers/thread=56`, local spilling `0`, `L1 hit=5.36%`, `L2 hit=1.48%`. small-grid / low-issue / low-occupancy 성격은 그대로다.
- 이번에 시도했거나 검토했지만 안 좋다고 판단한 방향:
  - 후보: E1 축소 버전 (`q/k/v` shared staging + gate block 1회 계산).
  - 안 좋은 이유: block-invariant q/k/gate 중복은 줄였지만, 주병목인 state row streaming은 그대로였다. 대신 shared staging과 block-wide barrier cost가 추가되어 전체 avg가 `0.013278 ms`로 악화됐고, B=64 workload `eaf0a285`는 `0.024691 ms`로 baseline(`0.021~0.022 ms`)보다 느려졌다.
  - 재시도 가능 조건: shared staging만으로는 재시도하지 않음. cluster로 q/k를 CTA 간 공유하거나, qk reduction 자체를 1회화하거나, async pipeline으로 state fetch overlap을 키우는 등 더 큰 구조적 중복 제거와 결합될 때만 재검토.
- 다음 iteration 에서 시도할만한 후보 2~3 개:
  - D5 CUDA Graph capture feasibility: kernel Duration보다 benchmark avg/median에 직접적인 레버리지가 큼. 포인터 안정성과 graph cache invalidation 규칙부터 확인.
  - B1 2-CTA cluster q/k 공유: `V_PER_Q=2` 중복 q/k load를 줄이되, 단순 shared staging보다 큰 중복 제거를 노릴 수 있음.
  - G5 + C2 묶음 검토: ptxas/SASS 기준선 확인 후 `__reduce_add_sync` 또는 동등한 warp-reduction 축소가 실제 hot loop shuffle 수를 줄일 수 있는지 검증.

## iter #3

- 적용한 최적화: (1) `__reduce_add_sync(float)` 기반 warp reduction 축소를 시도했으나 현재 Modal `tvm_ffi`/nvcc 경로에서 compile blocked, (2) block이 실제로 사용하는 `v` slice만 shared에 적재하도록 `s_v` 범위를 `ROWS_PER_BLOCK`로 축소, (3) `cudaFuncAttributePreferredSharedMemoryCarveout=0` 적용. 5회 benchmark median 후퇴로 전체 롤백함.
- 측정된 avg latency: 5회 `[0.012954, 0.013378, 0.013142, 0.013244, 0.017654] ms`, median `0.013244 ms`, correctness `PASSED=54/54`.
- NCU Duration: 롤백 후 현재 accepted kernel 기준 `34.43 us`.
- 남아있는 주요 bottleneck: accepted kernel NCU는 `Grid Size=1024`, `Waves/SM=0.77`, `Issue Slots Busy=16.96%`, `Memory Throughput=28.03%`, `DRAM Throughput=22.90%`, `L1/TEX Throughput=48.24%`, `L2 Throughput=19.07%`, `Achieved Occupancy=27.34%`, `Registers/thread=56`, local spilling `0`, `L1 hit=5.38%`, `L2 hit=1.49%`로 계속 small-grid / low-issue / low-occupancy / low-cache-hit 성격이다.
- 이번에 시도했거나 검토했지만 안 좋다고 판단한 방향:
  - 후보: `__reduce_add_sync(float)` 기반 warp reduction 교체.
  - 안 좋은 이유: Modal의 현재 `tvm_ffi`/nvcc build path에서는 `__reduce_add_sync`가 `int`/`unsigned int` overload만 보여 `float` 인자에서 전 workload compile failure가 발생했다.
  - 재시도 가능 조건: inline PTX `redux.sync.add.f32`를 직접 쓰거나, 다른 toolchain/헤더 조합에서 float overload 또는 생성 SASS를 확인할 수 있을 때.
  - 후보: split-local `s_v` staging + `PreferredSharedMemoryCarveout=0`.
  - 안 좋은 이유: shared load 양은 줄었지만 block barrier는 그대로였고, benchmark avg 5회가 모두 baseline(`0.012920 ms` median)보다 나빠졌다. B=64 workload `eaf0a285`도 `0.025133~0.029591 ms`로 baseline(`0.021~0.022 ms`)보다 느려졌다.
  - 재시도 가능 조건: `__syncthreads()` 자체를 warp-local sync로 줄이거나, cluster/D5처럼 더 큰 중복 제거와 결합해서 barrier cost를 상쇄할 수 있을 때.
- 다음 iteration 에서 시도할만한 후보 2~3 개:
  - D5 CUDA Graph capture feasibility: 현재 목표(<0.009 ms)에는 launch overhead 제거가 benchmark median에 가장 직접적인 레버리지다.
  - B1 2-CTA cluster q/k 공유: `V_PER_Q=2` 중복 q/k load를 줄이면서 split 증가 실패(R1)를 보완할 수 있다.
  - G5 + inline PTX 검증: `redux.sync.add.f32`, `LDG/FFMA/SHFL` 비율, register 변화를 먼저 고정해서 실제 reduction 축소가 가능한 경로인지 확인.

## iter #4

- 적용한 최적화: D5 수동 CUDA Graph replay. `solution/cuda/kernel.cu`의 host launch path에 variant별 graph exec cache를 추가해 normal launch 대신 `cudaGraphLaunch`를 쓰도록 했고, 2차 시도에서는 동일 workload 반복 시 `cudaGraphExecKernelNodeSetParams`를 건너뛰는 캐시도 넣었다. 두 시도 모두 benchmark regression으로 전체 롤백함.
- 측정된 avg latency: 1차 `0.014014 ms`, retry `0.017512 ms`, correctness `PASSED=54/54`. Phase 3+는 원칙적으로 5회 median 판정이지만, 둘 다 accepted baseline(`0.012920 ms` median)보다 명확히 느려 workflow의 즉시 롤백 규칙에 따라 추가 5회 측정 없이 중단했다. 롤백 후 accepted latency는 `0.012920 ms`를 유지한다.
- NCU Duration: 롤백 후 현재 accepted kernel 기준 `32.64 us`.
- 남아있는 주요 bottleneck: rollback 후 profiled kernel은 여전히 `Grid Size=1024`, `Waves/SM=0.77`, `Issue Slots Busy=17.82%`, `Memory Throughput=29.37%`, `DRAM Throughput=24.21%`, `L1/TEX Throughput=48.81%`, `L2 Throughput=20.11%`, `Achieved Occupancy=29.36%`, `Registers/thread=56`, local spilling `0`, `L1 hit=5.37%`, `L2 hit=1.48%`다. single-pipeline 포화가 아니라 small-grid / low-issue / low-occupancy / poor-cache-hit 성격이 계속된다.
- 이번에 시도했거나 검토했지만 안 좋다고 판단한 방향:
  - 후보: D5 `kernel.cu` wrapper 내부 수동 CUDA Graph replay.
  - 안 좋은 이유: 현재 flashinfer-bench isolated runner + TVM FFI 경로에서는 graph instantiate/update 관리 비용이 초소형 decode launch savings를 상쇄했다. 첫 full benchmark avg가 `0.014014 ms`, update 생략 캐시를 넣은 retry도 `0.017512 ms`로 더 악화됐고, B=64 workload `eaf0a285`는 각각 `0.030310 ms`, `0.031475 ms`까지 후퇴했다.
  - 재시도 가능 조건: graph를 kernel wrapper 내부가 아니라 benchmark/harness 상위 반복 루프에서 한 번 캡처해 여러 decode call에 재사용할 수 있거나, stable buffer/process reuse가 보장되어 instantiate/update 비용이 측정 구간 밖으로 빠질 때.
- 다음 iteration 에서 시도할만한 후보 2~3 개:
  - B1 2-CTA cluster q/k 공유: `V_PER_Q=2` 중복 q/k load를 줄이면서 kernel body Duration 자체를 줄이는 방향.
  - H1/H2 async state-row prefetch (`cp.async` / `cuda::memcpy_async`): low issue / low bytes-in-flight 문제를 직접 건드릴 수 있음.
  - G5 + SASS/ptxas 기준선 재고정: `Registers/thread=56`, spill 0 상태에서 `LDG/FFMA/SHFL` 비율과 hot loop instruction mix를 확인해 다음 구조 변경 리스크를 줄일 것.

## iter #5

- 적용한 최적화: H4 변형. `ROWS_PER_WARP=16` large-batch 경로만 대상으로 warp-private 2-stage `__pipeline_memcpy_async` state-row prefetch를 넣고, `decode_submit_entry.py`의 TVM FFI JIT path에 `-arch=sm_100a`, `-Xptxas -v`, `-Xptxas -warn-spills`를 명시했다. full benchmark regression으로 전체 롤백함.
- 측정된 avg latency: 1회 `0.013124 ms`, correctness `PASSED=54/54`. Phase 3+는 5회 median 판정이 원칙이지만, accepted baseline(`0.012920 ms` median)보다 명확히 느려 workflow의 즉시 롤백 규칙에 따라 추가 4회 반복 없이 중단했다. 롤백 후 accepted latency는 `0.012920 ms`를 유지한다.
- NCU Duration: 롤백 후 현재 accepted kernel 기준 `32.83 us`.
- 남아있는 주요 bottleneck: rollback 후 profiled kernel은 `Grid Size=1024`, `Waves/SM=0.77`, `Issue Slots Busy=18.00%`, `Memory Throughput=29.76%`, `DRAM Throughput=24.09%`, `L1/TEX Throughput=48.44%`, `L2 Throughput=19.96%`, `Achieved Occupancy=28.79%`, `Registers/thread=56`, local spilling `0`, `L1 hit=5.37%`, `L2 hit=1.51%`다. 여전히 single-pipeline 포화가 아니라 small-grid / low-issue / register-limited occupancy / poor-cache-hit 성격이 강하다.
- 이번에 시도했거나 검토했지만 안 좋다고 판단한 방향:
  - 후보: H4 `ROWS_PER_WARP=16` warp-private `__pipeline_memcpy_async` + `sm_100a` JIT flags.
  - 안 좋은 이유: correctness는 유지됐지만 full benchmark avg가 `0.013124 ms`로 baseline `0.012920 ms`보다 악화됐다. B=64 workload `eaf0a285`도 `0.024348 ms`로 baseline(`0.021~0.022 ms`)보다 느려졌다. per-thread async copy 후 shared reload와 commit/wait overhead가 기존 register prefetch 대비 더 컸고, overlap 이득을 상쇄한 것으로 보인다.
  - 재시도 가능 조건: block/warp 단위 `cuda::pipeline` 또는 `cp.async`로 multiple stages를 더 깊게 유지해 bytes-in-flight를 실제로 늘리거나, B1처럼 q/k 중복 제거와 결합해 per-block duplicated work도 함께 줄일 수 있을 때. 현재의 per-thread 2-stage 4-row staging 단독안은 재시도하지 않음.
  - 후보: C2 `__reduce_add_sync(float)` / `redux.sync.add.f32`.
  - 안 좋은 이유: 공식 CUDA Programming Guide/PTX ISA 기준 하드웨어 warp reduce는 여전히 `int`/`unsigned` 계열만 문서화되어 있어, iter #3의 build blocked 이유가 이번에도 해소되지 않았다.
  - 재시도 가능 조건: 공식 문서 또는 검증된 SASS 경로에서 float reduce 지원이 확인되거나, 다른 합리적 구현 경로가 생길 때.
- 다음 iteration 에서 시도할만한 후보 2~3 개:
  - B1 2-CTA cluster q/k 공유: `V_PER_Q=2` 중복 q/k read와 qk_dot 중복을 줄이면서 kernel body Duration 자체를 줄이는 방향.
  - A2/A3 저위험 메모리 힌트 (`cuda::annotated_ptr`, read-only load path 확인): shared-memory 구조 변경 없이 state read path를 다듬는 방향.
  - G5 + SASS/ptxas 기준선 재고정: accepted kernel의 `Registers/thread=56`, spill 0, `LDG/FFMA/SHFL` 비율을 먼저 고정해 다음 구조 변경 리스크를 낮출 것.

## iter #6

- 적용한 최적화: `batch_size >= 32` large-batch 경로만 256-thread / 8-warp variant로 분기. `split_factor=2`는 유지하고, large-batch path의 `ROWS_PER_WARP`를 8로 낮춰 per-CTA active warps와 bytes-in-flight를 늘리려 했다. 첫 full benchmark regression으로 전체 롤백함.
- 측정된 avg latency: 1회 `0.012979 ms`, correctness `PASSED=54/54`. Phase 3+는 5회 median 판정이 원칙이지만, 핵심 타깃 B=64 workload `eaf0a285`가 `0.023968 ms`로 baseline(`0.021~0.022 ms`)보다 명확히 느려 workflow의 즉시 롤백 규칙에 따라 추가 4회 반복 없이 중단했다. 롤백 후 accepted latency는 `0.012920 ms`를 유지한다.
- NCU Duration: 롤백 후 현재 accepted kernel 기준 `33.06 us`.
- 남아있는 주요 bottleneck: rollback 후 profiled kernel은 `Grid Size=1024`, `Waves/SM=0.77`, `Issue Slots Busy=17.62%`, `Memory Throughput=29.02%`, `DRAM Throughput=23.83%`, `L1/TEX Throughput=46.47%`, `L2 Throughput=19.81%`, `Achieved Occupancy=27.71%`, `Registers/thread=56`, local spilling `0`, `L1 hit=5.37%`, `L2 hit=1.53%`다. 여전히 single-pipeline 포화가 아니라 small-grid / low-issue / register-limited occupancy / poor-cache-hit 성격이 강하다.
- 이번에 시도했거나 검토했지만 안 좋다고 판단한 방향:
  - 후보: `batch_size >= 32` 전용 256-thread / 8-warp large-batch kernel.
  - 안 좋은 이유: NCU의 low-wave / low-issue 지표와는 맞는 접근이었지만, standalone 256-thread 확대만으로는 이득이 나지 않았다. 전체 avg가 `0.012979 ms`로 baseline `0.012920 ms`보다 소폭 악화됐고, 핵심 타깃 B=64 workload `eaf0a285`는 `0.023968 ms`로 baseline(`0.021~0.022 ms`)보다 느려졌다. per-block parallelism 증대 이득보다 q/k load 및 qk reduction의 warp 중복, `ROWS_PER_WARP=8`로 인한 per-warp ILP 감소가 더 컸던 것으로 본다.
  - 재시도 가능 조건: q/k를 block shared 또는 2-CTA cluster에서 1회만 로드/감산하도록 묶어 warp 중복을 줄이거나, ptxas/SASS 기준선에서 register 수를 더 낮춰 256-thread path가 4-blocks/SM 이상으로 안정적으로 올라갈 근거가 생길 때. 현재의 standalone 256-thread 분기 단독안은 재시도하지 않음.
- 다음 iteration 에서 시도할만한 후보 2~3 개:
  - B1 2-CTA cluster q/k 공유: `V_PER_Q=2` 중복 q/k read와 qk_dot 중복 자체를 줄이는 방향이라 이번 실패 이유와 직접 맞물린다.
  - A2/A3 저위험 메모리 힌트 (`cuda::annotated_ptr`, read-only load path 확인): current kernel shape를 바꾸지 않고 state/qk read path를 다듬는 방향.
  - G5 + SASS/ptxas 기준선 재고정: accepted kernel의 `Registers/thread=56`, spill 0, `LDG/FFMA/SHFL` 비율을 먼저 고정해 다음 구조 변경의 실패 이유를 더 명확히 볼 것.

## iter #1 (2026-04-24 session)

- 적용한 최적화: (1) C2 재시도 성격으로 hot loop warp reduction을 inline PTX `redux.sync.add.f32`로 바꾸려 했으나 Modal CUDA 13.0 `ptxas`가 `.add.f32`를 거부해 compile blocked, (2) fallback으로 `state` base pointer에 `cuda::associate_access_property(..., cuda::access_property::persisting{})` 힌트를 주는 A2 계열 저위험 memory-hint를 시도했으나 첫 full benchmark에서 regression이 커 전체 롤백함.
- 측정된 avg latency: 1차는 build failure로 수치 없음. 2차 fallback은 1회 `0.018830 ms`, correctness `PASSED=54/54`; 핵심 B=64 workload `eaf0a285`는 `0.029457 ms`. accepted baseline latency는 롤백 후 `0.012920 ms`를 유지한다.
- NCU Duration: 롤백 후 현재 accepted kernel 기준 `32.51 us`.
- 남아있는 주요 bottleneck: rollback 후 profiled kernel은 `Grid Size=1024`, `Waves/SM=0.77`, `Issue Slots Busy=17.78%`, `Memory Throughput=29.37%`, `DRAM Throughput=24.10%`, `L1/TEX Throughput=49.83%`, `L2 Throughput=20.12%`, `Achieved Occupancy=28.79%`, `Registers/thread=56`, local spilling `0`, `L1 hit=5.37%`, `L2 hit=1.52%`다. 여전히 small-grid / low-issue / register-limited occupancy / poor-cache-hit 성격이 강하다.
- 이번에 시도했거나 검토했지만 안 좋다고 판단한 방향:
  - 후보: C2 inline PTX `redux.sync.add.f32`.
  - 안 좋은 이유: 공식 PTX ISA 9.2 문서에는 `.f32` 지원 메모가 보이지만, 실제 Modal CUDA 13.0 `ptxas`는 `Incorrect type '.f32' for operation '.add' in instruction 'redux'`로 전 workload compile failure를 냈다. 현재 toolchain에서는 float warp-reduce add 경로가 막혀 있다고 보는 편이 맞다.
  - 재시도 가능 조건: Modal/toolkit 쪽 `ptxas`가 `redux.sync.add.f32`를 실제로 수용하는 버전으로 올라가거나, SASS/ptxas 기준으로 동등한 float warp-reduce primitive가 검증될 때.
  - 후보: A2 계열 state access-property persisting hint 단독 적용.
  - 안 좋은 이유: 구조를 전혀 바꾸지 않은 저위험 힌트였지만 full benchmark avg가 `0.018830 ms`로 baseline `0.012920 ms`보다 크게 악화됐고, B=64 `eaf0a285`도 `0.029457 ms`로 baseline(`0.021~0.022 ms`)보다 훨씬 느려졌다. standalone cache-hint만으로는 current bottleneck을 못 움직였고 오히려 load path가 불리해졌을 가능성이 크다.
  - 재시도 가능 조건: 단독 재시도는 하지 않음. 향후 재검토하더라도 `missProp=Streaming`, `__ldg()/ld.global.nc` 검증, 또는 B1처럼 q/k 중복 제거와 결합된 더 큰 메모리 경로 변경 안에서만 본다.
- 다음 iteration 에서 시도할만한 후보 2~3 개:
  - B1 2-CTA cluster q/k 공유: 현재 repeated failure의 공통 원인인 q/k load 및 qk reduction 중복을 직접 줄이는 방향.
  - A3/A4 read-only load path 검증 (`__ldg`, `ld.global.nc.v4.f32`) + SASS 확인: 단, standalone cache-hint 재시도 대신 실제 load opcode 변화가 보일 때만.
  - G5 ptxas/SASS 기준선 재고정: `Registers/thread=56`, spill 0, `LDG/FFMA/SHFL` 비율을 먼저 고정해 다음 구조 변경의 성공/실패 이유를 더 분명히 볼 것.
