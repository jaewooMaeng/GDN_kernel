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

