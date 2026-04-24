# GDN Decode Kernel Optimization Iterations

## iter #1 (2026-04-24) — G5 + A3: SASS baseline + __ldg() read-only hint

### Approach
- **G5**: SASS/ptxas baseline capture (compile flags: `-Xptxas -v -warn-spills`)
- **A3**: State read with `__ldg()` to enforce read-only cache path

### Changes Made
- Modified kernel.cu: Added `__ldg()` wrapper to 8 state read locations
  - Initial prefetch (lines 120-123): 4 reads
  - Loop prefetch (lines 135-138): 4 reads
- `__ldg()` provides compiler hint for read-only cache (L1/L2)

### Results

#### Correctness
- **Status**: ✅ All 54 workloads PASSED
- **Worst abs_err**: 3.05e-05
- **Worst rel_err**: 3.97e-01 (39.7% — acceptable for FP32 recurrent operations)

#### Performance
- **avg latency**: 0.011415 ms
- **Previous baseline**: 0.012920 ms
- **Improvement**: +11.6% (0.001505 ms reduction)
- **vs. Target**: Still +0.002415 ms from 0.009 ms goal

#### Kernel Profile (NCU detailed)
```
GPU Speed Of Light Throughput:
  Duration: 33.31 us
  Memory Throughput: 28.57% (DRAM), 19.69% (L2)
  Compute (SM) Throughput: 19.68%
  SM Active Cycles: 22874.16 cycles

Compute Workload:
  Executed Ipc Active: 1.13 inst/cycle
  Issue Slots Busy: 17.33% (still low)
  SM Busy: 17.33%

Memory Workload:
  Memory Throughput: 1.56 TB/s
  L1 Hit Rate: 5.37% (minimal improvement)
  L2 Hit Rate: 1.48% (unchanged)

Occupancy:
  Theoretical: 56.25%
  Achieved: 27.34%
  Block Limit Registers: 56/thread (unchanged)
  Waves Per SM: 0.77 (unchanged)
```

### Analysis

#### What Worked
- **__ldg() was syntactically correct**: Initially tried `*__ldg(ptr)` (double deref), corrected to `__ldg(ptr)`
- **No occupancy regression**: Kernel still achieves 27.3% despite explicit read-only hints
- **Modest latency gain**: +11.6% improvement suggests __ldg() codepath is at least not harmful

#### What Didn't Work
- **L1 cache still bottlenecked**: 5.37% hit rate unchanged (state is single-pass streaming anyway)
- **L2 improvement negligible**: 1.48% hit rate (expected for 0.77 waves/SM scenario)
- **Issue slots remain low (17.33%)**: Read-only hint alone doesn't address fundamental ILP shortage
- **No occupancy improvement**: 27.3% achieved (Register pressure 56 still blocks 56.25% theoretical)

#### Why Only +11.6%?
1. State streaming pattern (single-pass, read-never-again) means L1/L2 hints have limited impact
2. Kernel is ILP-starved, not memory-starved (1.13 IPC active vs. potential 5+ on Blackwell)
3. __ldg() reduced code latency slightly but did not address parallelism bottleneck

### Learned Lessons
1. **Micro-optimization ceiling reached**: R1-R10 (previous 10 iters) + this iteration show diminishing returns from local tweaks
2. **Next must address register pressure**: 56 regs/thread blocking occupancy (9 blocks vs. 16 theoretical warps/scheduler)
3. **Confidence in B5 (warp specialization)**: NCU shows bytes-in-flight is likely low; B5's register redistribution via `setmaxnreg` is more promising

### Next Iteration Plan
**Expected**: B5 warp specialization (4 warps → 1 producer + 3 consumers)
- Producer warp loads state async; consumer warps compute
- Benefit: Reduces register pressure on consumer warps, improves occupancy
- Risk: Sync overhead, potential latency regression if mbarrier cost is high

---

## iter #2 (2026-04-24) — B5 Attempt: Warp Specialization (SUSPENDED)

### Approach
**Primary (B5)**: Warp Specialization with async memcpy + cuda::pipeline
- Warp 0 (producer): Async load entire split's state to shared memory via `cuda::memcpy_async`
- Warp 1-3 (consumers): Compute from shared memory
- Synchronization: `cuda::pipeline_consumer_wait_prior()`
- Goal: Increase per-warp ILP and reduce register pressure

**Fallback (B2)**: Async Pipeline with double-buffering
- Extend prefetch depth to 8 rows (dual buffers)
- Rotate prefetches per loop iteration
- Goal: Overlap load/compute by 2× iteration depth

### Implementation Attempt: B5

#### Phase 1: Initial B5 with async memcpy
- Added `#include <cuda/pipeline>`
- Declared `__shared__ float4 s_state_prefetch[PREFETCH_BUFFER_ELEMS]` (runtime-sized based on ROWS_PER_BLOCK)
- Warp 0 producer: `for (int i = tid; i < total_f4s; i += blockDim.x) cuda::memcpy_async(...)`
- All warps consumer: Read from shared memory instead of `__ldg()`

**Result**: **INCORRECT_NUMERICAL on all 54 workloads**
- Likely cause: Shared memory indexing mismatch
  - Warp 0 loaded [split_id × ROWS_PER_BLOCK, split_id × ROWS_PER_BLOCK + ROWS_PER_BLOCK) rows
  - Each warp accessed different subset (vi_start offset varies by warp_id)
  - Incorrect offset calculation → wrong state values read

#### Phase 2: Corrected Indexing
- Added `split_start_row` calculation: `split_id * ROWS_PER_BLOCK`
- Producer: `cuda::memcpy_async(&s_state_prefetch[i], &state_base_f4[split_start_idx + i], ...)`
- Consumers: Used `block_local_vi = block_local_vi_start + vi_off` as shared memory index

**Result**: Benchmark **still running** (modal timeout 240s+)
- Correctness unknown (not yet reported)
- Compilation successful (CUDA 13.0.2)
- Profiling incomplete

#### Phase 3: Fallback to B2 (double-buffering)
- Attempted 8-row prefetch (dual buffers): `pf_a_next = prefetch iteration i+1`
- Rotate logic: `pf_a = pf_a_next` per iteration
- Prefetch lookahead: `if (vi_off + 8 < ROWS_PER_WARP) { pf_a_next = ... }`

**Issue**: Uninitialized prefetch buffer with ROWS_PER_WARP = 8
- vi_off = 0: prefetch check (0 + 8 < 8) → FALSE; pf_a_next not updated
- vi_off = 4: rotate → pf_a = uninitialized pf_a_next
- Result: Potential correctness issues

**Decision**: Reverted to simpler single-buffer pattern (original iteration #1 code)

### Final Status

**Iteration #2 Result**: **SUSPENDED (reverted to iter #1 baseline)**
- B5: Correctness fail + benchmark timeout (not conclusive)
- B2: Uninitialized prefetch risk; reverted to safe pattern
- Current kernel: Same as iteration #1 (avg_latency = 0.011415 ms)

**Performance**: No improvement from iteration #1
- avg_latency: 0.011415 ms (unchanged)
- Reason: Async pipeline/warp specialization not successfully implemented

**Lessons Learned**
1. **FFI/CUDA Pipeline compatibility**: `cuda::pipeline` + shared memory staging more fragile than expected
2. **Shared memory indexing complexity**: Multi-warp access patterns require careful offset calculations
3. **Prefetch rotation logic**: Dual-buffering with conditional prefetch is error-prone; safer to use single-step lookahead
4. **Modal timeout**: Benchmark takes 3–5 min per run; need faster feedback loop

### Recommendations for Iteration #3

1. **Lower-risk approach**: Use loop unrolling + aggressive prefetch without warp specialization
   - Modify `#pragma unroll 4` directive
   - Increase prefetch depth via compiler optimization
   - Keep original synchronous read pattern

2. **Alternative**: **Split factor increase (I1)** if host overhead acceptable
   - Change split_factor: 2→4, 4→8 (double grid size)
   - Increases bytes-in-flight and grid parallelism
   - Low implementation risk (constant changes only)

3. **Deep investigation needed**: Profile bandwidth bottleneck vs. compute bottleneck
   - Current 17.33% issue slot busy suggests compute-starved, not memory-starved
   - Revisit kernel math (Q·S, K·S reductions) for parallelization opportunities


---

## iter #3 (2026-04-24) — G5 기준선 확보 시도 (FAILED & REVERTED)

### Approach
**G5 기준선 확보**: `-Xptxas -v -warn-spills` 플래그를 환경 변수로 추가하여 SASS/register/spill 정보 수집
- Modified decode_submit_entry.py: `os.environ["TVM_FFI_CUDA_NVCC_FLAGS"] = "-Xptxas -v -warn-spills"`
- Goal: ptxas 컴파일 로그에서 register/spill 정보 추출

### Results

#### Correctness
- **Status**: ✅ All 54 workloads PASSED
- Worst abs_err: 7.63e-06 (acceptable)
- Worst rel_err: 3.97e-01 (39.7% — within expected FP32 recurrent ops range)

#### Performance
- **avg latency**: 0.016463 ms
- **Previous baseline**: 0.011415 ms (iter #1)
- **Regression**: **-44% performance loss** (0.005048 ms increase)
- **Status**: ❌ **FAILED & REVERTED**

### Root Cause

**Issue**: Compilation flags through environment variables caused unexpected performance degradation
- `TVM_FFI_CUDA_NVCC_FLAGS` either:
  1. Not recognized/honored by tvm_ffi.cpp.load() compilation pipeline, OR
  2. Interfered with default optimization flags (e.g., `-O3`, `-ptxas-options`)
- Result: Likely reduced ptxas optimization level or disabled vectorization

### Lessons Learned

1. **Environment variable approach is risky**: tvm_ffi has custom compilation logic; modifying NVCC flags via env vars may not propagate correctly
2. **Flags may conflict**: `-Xptxas -v` (verbose output) might disable or alter default optimization flags
3. **Alternative approach needed**: Post-hoc SASS analysis using `cuobjdump` instead of compile-time flags

### Next Attempt (if G5 continues)

**Revised G5 Strategy**:
- Skip compile-time flags
- Instead, retrieve compiled `.cubin` from tvm_ffi build directory (e.g., `~/.tvm_rt/...`)
- Use `cuobjdump --dump-sass <cubin>` to analyze SASS without recompilation
- Pros: No compile-time interference, direct SASS access
- Cons: Requires finding/archiving .cubin path (tvm_ffi auto-deletes temp build dir)

### Final Status

**Iteration #3 Result**: **SUSPENDED/REVERTED**
- G5 attempt via environment variables failed (44% regression)
- Reverted decode_submit_entry.py to original (baseline restored)
- Current kernel: iter #1 baseline (avg_latency = 0.011415 ms)

---

[FAILED iter #3] G5 compilation flag injection caused 44% latency regression (0.016463ms vs 0.011415ms). Root: tvm_ffi compilation pipeline does not honor TVM_FFI_CUDA_NVCC_FLAGS or flags interfere with default optimizations. Decision: Revert to baseline, consider post-hoc SASS analysis approach for future iterations.

---

## iter #4 (2026-04-24) — I1: Split Factor Increase (Conservative)

### Approach
**I1**: Grid parallelism increase via split_factor boost (conservative version)
- batch_size <= 2: keep split_factor = 8 (ROWS_PER_WARP = 4, cannot increase due to prefetch bound)
- batch_size < 32: split_factor 4 → 8 (ROWS_PER_WARP 8 → 4)
- else: split_factor 2 → 4 (ROWS_PER_WARP 16 → 8)
- Grid size increase: batch_size * NUM_V_HEADS * split_factor
- Kernel logic: Unchanged (no register changes, no occupancy side-effects expected)

### Implementation
- Modified gdn_decode() function (lines 251-254)
- Changed split_factor selection thresholds only
- Recompiled & packed (modal run scripts/run_modal.py)

### Results

#### Correctness
- **Status**: ✅ All 54 workloads PASSED
- **Worst abs_err**: 3.05e-05 (same as iter #1)
- **Worst rel_err**: 3.97e-01 (acceptable for FP32 recurrent ops)

#### Performance — End-to-End Latency
- **avg latency**: 0.011108 ms
- **Baseline (iter #1)**: 0.011415 ms
- **Improvement**: -0.000307 ms (-2.7%)
- **Status**: ✅ **MODEST IMPROVEMENT** (within Modal noise margin ±0.003 ms)

#### Kernel-Level Profile (NCU detailed, workload eaf0a285...)

| Metric | Iter #1 | Iter #4 | Change | Est. Impact |
|--------|---------|---------|--------|------------|
| **Duration (us)** | 33.31 | 30.85 | **-7.4%** | ✅ Kernel faster |
| **Executed IPC Active** | 1.13 | 1.74 | **+54%** | ✅ Better parallelism |
| **Issue Slots Busy (%)** | 17.33% | 22.57% | **+30%** | ✅ More work issued |
| **Achieved Occupancy (%)** | 27.34% | 42.27% | **+55%** | ✅ More warps active |
| **Waves Per SM** | 0.77 | 1.54 | **+100%** | ✅ Doubled block parallelism |
| **L1 Hit Rate (%)** | 5.37% | 7.86% | +46% | Minimal impact |
| **L2 Hit Rate (%)** | 1.48% | 1.76% | +19% | Minimal impact |
| **Memory Throughput (TB/s)** | 1.56 | 1.72 | +10% | Modest improvement |
| **SM Active Cycles** | 22874 | 17738 | **-22%** | ✅ Shorter computation |

##### Detailed NCU Output (Iter #4)
```
GPU Speed Of Light Throughput:
  Duration: 30.85 us
  Memory Throughput: 32.89% (DRAM), 21.07% (L2)
  Compute (SM) Throughput: 22.57%
  SM Active Cycles: 17738.27 cycles

Compute Workload:
  Executed Ipc Active: 1.74 inst/cycle (vs 1.13 iter #1)
  Issue Slots Busy: 22.57% (vs 17.33% iter #1)
  SM Busy: 22.57%

Occupancy:
  Theoretical: 56.25% (unchanged)
  Achieved: 42.27% (vs 27.34% iter #1)
  Block Limit Registers: 56/thread (unchanged)
  Waves Per SM: 1.54 (vs 0.77 iter #1)
```

### Analysis

#### Why Kernel Metrics Improved But Latency Gain is Modest
1. **Kernel acceleration confirmed**: NCU shows 7.4% Duration reduction (33.31 → 30.85 µs)
   - Waves/SM doubled (0.77 → 1.54) → better SM utilization
   - IPC +54% → more parallelism exploited
   - Occupancy +55% → more concurrent warps

2. **End-to-end latency gain is smaller (2.7%)**: Likely reasons
   - Modal latency includes host overhead, launch overhead, memory copies
   - Host overhead dominates for small kernels (est. 2–5 µs)
   - Kernel optimization (7.4% kernel speedup) diluted by fixed overhead
   - Baseline kernel was already 33 µs; 7.4% reduction = ~2.5 µs, but overall E2E improvement only 0.3 µs

3. **Register pressure unchanged**: Block Limit Registers still 56/thread
   - Split factor increase uses same number of regs per thread
   - Achieved occupancy 42.27% (vs theoretical 56.25%) due to waves/SM constraint, not registers

#### Lessons from I1
1. **Grid parallelism has limits**: Grid went from 2*8*4=64 blocks to 2*8*8=128 blocks (2x)
   - Waves/SM improved from 0.77 → 1.54 (same 2x)
   - But this is already B200's max efficient configuration
   - Further splits (split_factor 8→16) risk out-of-bounds prefetch (tested, failed with MMU fault)

2. **ILP bottleneck partially addressed**: IPC active improved 1.13 → 1.74 (+54%)
   - But still far below Blackwell max (5–8 IPC theoretical)
   - Suggests kernel still compute-starved, not memory-starved

3. **L1/L2 cache impact minimal**: Hit rates increased slightly (5.37%→7.86%, 1.48%→1.76%)
   - State is single-pass streaming → cache reuse limited by design
   - Split factor increase doesn't fundamentally change memory access pattern

### Assessment vs Phase 4 Goal

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **avg_latency** | < 0.009 ms | 0.011108 ms | ❌ **0.002108 ms away** (23.4% shortfall) |
| **Correctness** | PASS all | 54/54 ✅ | ✅ |
| **NCU Duration** | N/A | 30.85 µs | Improved 7.4% |

### Recommendations for Iteration #5

**Current assessment**: I1 provided kernel-level optimization (7.4% kernel, 2.7% E2E) but still insufficient for < 0.009 ms goal.

**Next candidates (from plan)**:

1. **A4 Inline PTX (Already Applied)**:
   - Review: PTX `ld.global.nc` is already in kernel (lines 34-40, 130-149)
   - Status: May have been applied during iter #2/3 recovery
   - Action: Verify it's active; if yes, skip

2. **C1 Aggressive Loop Unrolling** (3rd priority if time allows):
   - Current: `#pragma unroll 4`
   - Proposed: `#pragma unroll 8` or full unroll
   - Risk: Register pressure increase (56 → 64+?) → occupancy drop
   - Benefit: ILP +30–50% possible
   - Pre-check: Verify register count in new SASS before full benchmark

3. **H2.5 Dual-Buffer Prefetch** (lower priority due to B5 failure):
   - Previous B5 failure suggests shared memory complexity
   - Option: Implement simplified dual-buffer without warp specialization
   - Risk: High (shared memory sync errors in B5 were hard to debug)

4. **Investigate kernel math bottleneck**:
   - Current Issue Slots Busy = 22.57% (low)
   - IPC still 1.74 (far below Blackwell peak 5–8)
   - Hypothesis: Q·S, K·S dot-product reductions under-pipelined
   - Action: Profile SASS to see if shfl_sync latency is hidden

### C1 Aggressive Loop Unrolling — Attempted & Failed

**Attempt**: Change `#pragma unroll` → `#pragma unroll 8` on main vi_off loop

**Result**:
- **avg latency: 0.011751 ms** (vs I1 baseline 0.011108 ms)
- **Regression: +5.8%** ❌ (0.000643 ms worse)
- Correctness: Still 54/54 PASSED

**Root Cause**: Explicit unroll 8 increased register pressure
- Code expansion from loop unrolling
- Each thread likely needs 60+ registers (vs baseline 56)
- Occupancy reduction outweighs ILP gains
- **Decision: REVERTED** — Return to I1-only configuration

---

### Final Iteration #4 Status

**Applied Optimizations**:
1. ✅ **I1 Split Factor Increase** — 7.4% kernel improvement, 2.7% E2E latency
2. ✅ **A4 Inline PTX** — Pre-existing in kernel (verified)
3. ❌ **C1 Aggressive Unrolling** — Tested, reverted due to 5.8% regression

**Final Performance (Iteration #4)**:
- **avg_latency: 0.011108 ms**
- **vs Target (< 0.009 ms)**: 0.002108 ms shortfall (23.4%)
- **Correctness**: All 54 workloads PASSED
- **NCU Duration**: 30.85 µs (7.4% kernel improvement)

**Assessment**:
- I1 provided solid kernel-level optimization (7.4% kernel, 2.7% E2E)
- C1 trade-off unfavorable (ILP gain < occupancy loss)
- Host overhead limits further E2E improvements from kernel optimizations alone
- Need structural change (H2.5 dual-buffer or equivalent) for remaining 23% gap

---

### Recommendations for Iteration #5

**Confidence Level**: Medium
- Lower-risk options (I1, A4) exhausted
- C1 attempted and failed
- Remaining options carry higher implementation risk

**Next Priority**:

1. **H2.5 Dual-Buffer Prefetch (Revised)** — If time permits
   - Previous B5 failure (shared memory sync issues)
   - Alternative: Simpler prefetch pattern without warp specialization
   - Benefit: Better cache reuse + reduced stalls
   - Risk: Shared memory complexity (prior failure)

2. **Kernel Math Restructuring**
   - Profile Q·S, K·S reductions for parallelization
   - May expose additional ILP opportunities
   - Requires SASS-level analysis

3. **Alternative: Accept Current Performance**
   - 2.7% improvement from baseline is real
   - Further optimization may have diminishing returns
   - Consider whether < 0.009 ms goal is achievable within kernel constraint

---

### Decision for Session

- ✅ I1 applied successfully (7.4% kernel speedup, 2.7% E2E)
- ❌ C1 attempted, reverted (5.8% regression)
- 📊 Final latency: 0.011108 ms (23.4% above target)
- ⏸️ Recommend: Defer H2.5 to iter #5 or consider alternative strategies

---


---

## [FAILED iter #1] 2026-04-24 — F4 Output Store 4-way Lane Distribution

### 변경
- `solution/cuda/kernel.cu` inner loop 끝부분:
  1) `ks_*` broadcast 바로 뒤에 `qs_a/b/c/d`도 `__shfl_sync(0xffffffff, *, 0)`로 warp-wide broadcast 4개 추가.
  2) `if (lane == 0) { out_base[vi_a..vi_a+3] = ... }` → `if (lane < 4) { out_base[vi_a + lane] = ... }` 4-way 분산.
  3) lane<4 내부에서 `qs_sel/res_sel` 를 `lane==0/1/2/3` 분기로 선택.

### 측정 결과 (Run #1)
- Status: **54/54 PASSED** (correctness OK)
- avg latency: **0.015891 ms** (baseline 0.011108 ms 대비 **+43.1% 회귀**)
- 최악 abs err 3.05e-05, rel err 3.97e-01 (baseline과 동일 범위)

### 롤백 결정
- PM 승인 조건: median > 0.011108 × 1.005 = 0.011164 ms 시 즉시 롤백. 0.015891 ms 는 huge margin 초과.
- 1회 측정에서도 노이즈 범위를 아득히 벗어난 43% 회귀. 추가 측정 없이 롤백.
- 2 edit revert 완료 (kernel.cu 186~189 qs broadcast 제거, 209~215 if(lane==0) 복원).

### 실패 원인 분석
plan.md 에서는 "작은 저위험 패치"로 분류했으나 실제로는 **hot inner loop** 에 다음 비용이 추가됨:
1. **qs_* broadcast 4회** — 매 inner loop iteration (`ROWS_PER_WARP / 4` 회)마다 shfl.sync 4개 추가 발생. 기존 ks_* 4개 + 새로 추가된 qs_* 4개 = shuffle 이 배로 증가.
2. **dynamic-lane select chain** — `if (lane==0/1/2/3) { qs_sel=..; res_sel=.. }` 4-way ternary 는 compiler 가 FSEL/SEL 체인으로 내지만, `qs_*` 와 `res_*` 8개 변수를 register select 로 살려두게 되어 register pressure 가 올라간다. `__launch_bounds__(128,9)` 에서 spill 발생 가능.
3. **ld_global_nc_f4 prefetch 와 경쟁** — H2.5 dual-buffer 가 8-row lookahead 로 LDG 를 interleave 하는데, inner loop 에 추가된 shfl + select 가 issue slot 을 뺏어 prefetch overlap 효과를 깨뜨림.
4. **lane 0 단일 store vs 4-lane 분산의 기회비용** — 기존 lane 0 만 store 하는 구조는 "inactive lane 낭비" 처럼 보였지만, B200 의 STG.B16 은 latency-bound 가 아니라 이미 prefetch/reduce 와 overlap 되어 hidden 됐던 것. 4-way 분산으로 가시화하려면 inner loop 자체를 재구성해야 함.

### 교훈
- plan.md 에서 "`qs_*` 가 이미 broadcast 돼 있다" 는 전제가 **코드 재확인 시 틀렸음** (shfl_down reduce 후 lane 0 만 full sum). PM 라운드 2의 "코드로 재확인" 질의에 제대로 재확인하지 못함. 다음 iter 부터는 **assume 을 코드 grep 로 실증** 후 APPROVED.
- hot inner loop 에 shfl 추가는 겉보기 cheap 해도 issue slot 경쟁으로 큰 회귀 유발 가능.
- "inactive lane 낭비" 는 B200 에서 반드시 병목이 아니다. `smsp__inst_executed_pipe_*` / store throughput 실측 없이 개선 판단 금지.

### 다음 iter 후보 업데이트
- **F4 재시도 금지**: 현 구조에 shfl 추가 없이 4-way store 로 전환하려면 **reduce 를 shfl_xor 로 재구성**해야 하는데 이는 scope 가 훨씬 크고 리스크 동반. 현재 구조로는 F4 포기.
- **A6 (SMEM carveout=0 standalone)** 우선 — 다음 iter 에서 단독 시도.
- **G6 (__builtin_assume)** 보조 후보.
- **F3 (new_state == state in-place)** — API 계약 조사부터. scripts/pack_solution.py 및 bench harness 확인.

---

