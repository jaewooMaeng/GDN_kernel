# GDN Decode Kernel 반복 최적화 워크플로우

> code agent 자율 실행 가이드. Phase 4 목표(median ≤ 0.009 ms) 달성까지 루프를 절대 멈추지 않는다.

---

## 0. 절대 규칙

1. Phase 4 목표 달성까지 루프 반복. 중간에 멈추지 않는다.
2. 성능 후퇴 또는 correctness 실패 시 즉시 롤백.
3. 한 번에 하나의 최적화만 적용 (원인 추적).
4. 매 반복마다 측정 → 로그 기록.
5. **Phase 3 이후는 5회 측정의 median으로 판정** (Modal 노이즈 ±0.003 ms 때문).
6. **외부 라이브러리 import 금지** — CuTe, CUTLASS, cuBLAS, cuDNN, Triton 등 직접 `#include`·링크 금지. 오직 CUDA Runtime, CUDA Driver API, inline PTX, libcu++ (`<cuda/...>`, CUDA Toolkit 동봉분), `<cuda_bf16.h>`·`<cuda_fp16.h>`, 기존 tvm ffi 바인딩만 허용. **라이브러리 구현/논문/공식 예제의 아이디어를 직접 수기 구현하는 것은 허용**. 참조 출처는 iteration 로그에 명시한다.
7. 구조적 변경(cluster, async pipeline, warp specialization) 전 반드시 현재 커널 **전체**를 백업.

---

## 1. 목표

| Phase | 목표 Latency | 판정 기준 | 상태 |
|-------|-------------|----------|------|
| Phase 1 | ≤ 0.012 ms | best of 3 runs | ✅ Iter 17 |
| Phase 2 | ≤ 0.010 ms | best of 3 runs | ✅ Iter 20 |
| Phase 3 | ≤ 0.010 ms | **median of 5 runs** | 진행 중 |
| Phase 4 | ≤ 0.009 ms | median of 5 runs | 진행 중 |

- 시작: 0.015 ms → 현재 best 0.010 ms / median 0.012 ms
- **Phase 3 핵심은 분산 감소** (median을 0.010 ms 이하로).
- **Phase 4는 구조적 개선**이 필요하다 (cluster, async pipeline, warp specialization, CUDA Graph 등 Blackwell 전용 기법).

---

## 2. 파일 위치

| 항목 | 경로 |
|------|------|
| 커널 | `solution/cuda/kernel.cu` |
| 패킹 | `scripts/pack_solution.py` |
| 벤치마크 | `scripts/run_modal.py` |
| 설정 | `config.toml` |

---

## 3. 성능 측정

```bash
python scripts/pack_solution.py
modal run scripts/run_modal.py
```

- 모든 workload `status = correct` 확인
- `Avg latency: X.XXX ms` 기록
- **Phase 3 이후**: 5회 반복 실행, 5개 값 모두 기록, **median**으로 판정

10회 이상 진전 없을 때 profiling:
```bash
ncu --set full --kernel-name gdn_decode_kernel -o profile.ncu-rep ...
cuobjdump --dump-sass kernel.cubin | grep -E "LDG|STG|SHFL|FMA|FFMA"
```
`smsp__inst_executed_pipe_*`, `smsp__warp_issue_stalled_*`, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate` 를 중점 확인.

---

## 4. 반복 루프

```
STEP 1 현재 상태 확인 → 목표 달성 시 다음 Phase / Phase 4 달성 시 종료
STEP 2 아래 후보에서 미시도 항목 선택 (Phase별 우선 카테고리 고려)
STEP 3 커널 수정 (수정 전 전체 코드 백업)
STEP 4 측정 실행 (Phase 3+ 는 5회)
STEP 5 판정 → 후퇴/incorrect 즉시 롤백, 개선 유지
```

**Phase별 우선 카테고리**
- Phase 3 (안정화 돌파): **A, G, I**
- Phase 4 (구조적 돌파): **B, D5 (CUDA Graph), H**

---

## 5. 타겟 하드웨어: NVIDIA B200 (Blackwell, sm_100a)

| 항목 | 수치 | Decode 관점 |
|------|------|------------|
| Compute Capability | 10.0 (sm_100 / sm_100a) | `sm_100a`로 컴파일해야 TMA bulk tensor, tcgen05, `__reduce_add_sync` 활용 |
| L2 캐시 | **126 MB** | 모든 batch에서 state 전체 상주 (B=1:512KB, B=16:8MB, B=64:32MB) → **L2 persistence 효과 극대** |
| Shared Memory/SM | 228 KB (블록당 최대 227 KB) | 현재 512B만 사용. 대용량 활용 여지 |
| Max Warps/SM | 64 | Occupancy 기준 |
| Register File/SM | 64K × 32-bit | 현재 (128, 9) 추정 ~56 reg/thread |
| HBM3e Bandwidth | ~8 TB/s | State read가 L2 hit이면 비병목 |
| Thread Block Clusters | 최대 16 (nonportable) | **V_PER_Q=2 → 2-CTA cluster로 q/k 공유 유력** |
| Distributed Shared Memory | 지원 | Cluster 내 q/k broadcast |
| TMA | 지원 | `cp.async.bulk`, 16B-aligned `cuda::memcpy_async` |
| TMEM / tcgen05 | 지원 | Decode(matrix-vector)엔 fit 애매 |
| **Kernel Launch Overhead** | **2~5 μs (일반), ~60 ns/node (CUDA Graph)** | **10μs 타겟의 22~55% 차지 → CUDA Graph 매우 유망** |

### State 크기 vs L2

```
State per (batch, v_head) = 128×128×4B = 64 KB
B=1  total = 512 KB  → L2의 0.4%
B=16 total = 8 MB    → L2의 6.3%
B=64 total = 32 MB   → L2의 25%
```
→ 모든 batch에서 state 완전 상주. **L2 bandwidth 최적화 > global bandwidth 최적화**.

### 9μs 타겟의 overhead 구조

```
목표 9,000 ns =
  Kernel launch (일반):   2,000~5,000 ns  (22~55%)
  Kernel launch (Graph):       ~60 ns      (<1%)
  L2 warm read per cacheline:  ~40 cycles
  SFU exp/log 1 op:            ~16 cycles   ← Iter 20에서 lane 0 집중으로 해결
```

**Phase 4 구간은 kernel 내부 + host-side launch + L2 hit rate + Modal I/O 안정성까지 총체적 관리 필요.**

---

## 6. 외부 라이브러리 정책 및 참고 구현

### 6.1. 허용 / 금지 요약

**허용**
- CUDA Runtime (`<cuda_runtime.h>`), CUDA Driver API (`<cuda.h>`)
- Inline PTX (`asm volatile(...)`) — Blackwell 전용 명령어 직접 사용 가능
- libcu++: `<cuda/std/...>`, `<cuda/pipeline>`, `<cuda/barrier>`, `<cuda/annotated_ptr>`, `<cuda/atomic>`
- `<cuda_bf16.h>`, `<cuda_fp16.h>`
- 기존 `tvm/ffi/...` 바인딩 (유지)
- CUB/Thrust device-side (CUDA Toolkit 동봉분에 한해)

**금지**
- `<cutlass/...>`, `<cute/...>` 직접 include
- cuBLAS/cuBLASLt/cuDNN 링크
- Triton JIT / FlashAttention / FLA 바인딩, 외부 pre-compiled `.so`/`.a`

**회색 지대 (허용, 명시 필요)**: 외부 프로젝트의 수학적 구조/레이아웃 패턴/PTX 시퀀스를 **읽고 직접 재작성**하는 것은 허용. 단순 복붙은 금지. iteration 로그에 "참조: CUTLASS `mma_sm100.h`" 등 출처 명시.

### 6.2. 참고할 수 있는 구현 (아이디어 추출용, import 금지)

| 프로젝트 | 파일 | 추출할 아이디어 |
|---------|------|--------------|
| CUTLASS (읽기만) | `include/cutlass/arch/mma_sm100.h`, `cute/atom/copy_traits_sm100*.h` | `cp.async.bulk.tensor` PTX 시퀀스, TMA descriptor 인자 구성 |
| CUTLASS | `include/cute/swizzle.hpp`, `swizzle_layout.hpp` | SMEM swizzle의 XOR 패턴 (수식을 함수로 직접 작성) |
| CUTLASS examples | `examples/70_blackwell_gemm` | Cluster launch + distributed SMEM + warp-specialized mainloop |
| FlashAttention v3 | `csrc/flash_attn_hopper/flash_fwd_kernel.h` | Async pipeline 상태기계, `mbarrier` 사용, producer/consumer 분리 |
| libcu++ 공식 예제 | `<cuda/pipeline>` 문서 | 3-stage `cuda::memcpy_async` + `cuda::aligned_size_t<16>` 사용법 |
| NVIDIA CUDA Programming Guide | "Thread Block Clusters", "L2 Persistence", "Distributed SMEM" | Cluster API, `cudaAccessPolicyWindow` 세부, `cluster.mapa.shared::cluster` |
| NVIDIA CUDA Graph 문서 | CUDA Graph API Reference | `cudaStreamBeginCapture`/`EndCapture`, graph exec 재사용 패턴 |

### 6.3. 자주 쓸 저수준 프리미티브 (라이브러리 없이)

```
# TMA / async copy
cp.async.bulk.tensor.1d.shared::cluster.global [smem], [desc], {coord};
cp.async.ca.shared.global [smem], [gmem], 16;
cuda::memcpy_async<cuda::aligned_size_t<16>>(...);

# Async barrier
mbarrier.init.shared.b64 [bar], count;
mbarrier.arrive.expect_tx.shared.b64 _, [bar], tx_count;
mbarrier.try_wait.parity.shared.b64 p, [bar], phase;

# Cluster
__cluster_dims__(2, 1, 1)
cudaLaunchKernelEx + cudaLaunchAttribute::clusterDim = {2,1,1}
cluster.mapa.shared::cluster [dst], [smem_local], rank;

# L2 persistence (host-side, 이미 적용됨)
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
cuda::annotated_ptr<float, cuda::access_property::persisting>

# Pipeline
cuda::pipeline<cuda::thread_scope_block, 3>

# Warp ops (SM100)
redux.sync.and/or/add.b32 / __reduce_add_sync()
setmaxnreg.inc.sync.aligned.u32 N / setmaxnreg.dec.sync.aligned.u32 N
```

---

## 7. 최적화 후보 목록

우선순위는 위에서부터. 시도한 것은 `[시도됨]`, 이미 적용된 것은 `[적용됨]`으로 표시. 각 카테고리의 **참조**는 아이디어 추출용이다 (직접 import 아님).

---

### A. 메모리 접근 최적화 (★ Phase 3 안정화 핵심 ★)

**참조**: NVIDIA CUDA Programming Guide "L2 Persistence" 섹션, CUDA SDK sample `graphMemoryFootprint`, libcu++ `<cuda/annotated_ptr>` 공식 예제.

- [x] **A1. L2 Persistence + hitRatio 공식 + max set-aside** [적용됨, host code]: `hitRatio = min(persistingL2CacheMaxSize / total_state_bytes, 1.0f)`, `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize)`, `hitProp=Persisting`, `missProp=Normal`. 현재 `setup_l2_persistence()`에서 (stream, ptr, bytes) 캐싱으로 중복 호출 방지.
- [ ] **A2. `cuda::annotated_ptr<float, cuda::access_property::persisting>`로 state 래핑**: 커널 내부에서 포인터 단위 hint. Stream attribute보다 정교. miss 경로는 `access_property::streaming` 으로 분리 가능. **[시도됨 2026-04-24 session iter #1, 후퇴]** `cuda::associate_access_property(..., persisting)` 기반의 축소 버전을 state base pointer에만 적용했지만 full benchmark avg가 `0.018830 ms`로 baseline `0.012920 ms`보다 크게 악화됐고, B=64 `eaf0a285`도 `0.029457 ms`로 후퇴했다. standalone cache-hint 단독안은 재시도하지 않고, 향후에는 `missProp=Streaming`, `__ldg/ld.global.nc`, 또는 B1 같은 구조적 중복 제거와 결합될 때만 검토.
- [ ] **A3. State read를 `__ldg()`로 명시**: read-only path 강제. `__restrict__` + nvcc 자동 생성 여부를 `cuobjdump --dump-sass`로 `LDG.E.128.CONSTANT` 확인 후 미생성이면 명시. `float4` 로드와 결합 시 `ld.global.nc.v4.f32`.
- [ ] **A4. `ld.global.nc.v4.f32` inline PTX**: A3이 자동 생성 안 될 때 non-coherent cache path 강제. 
  ```
  asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(x),"=f"(y),"=f"(z),"=f"(w) : "l"(ptr));
  ```
- [ ] **A5. `missProp=Streaming` 실험**: 현재 `missProp=Normal`. state miss 시 L2 오염 방지. Prefetch 리듬이 일정하면 차이 미미하나 B=64+ 대배치에서 state가 L2의 25% 차지 → miss 비중 의미 있을 수 있음. **[시도됨 2026-04-24 codex iter #6, 롤백]** `setup_l2_persistence()`의 `missProp`만 `cudaAccessPropertyNormal -> cudaAccessPropertyStreaming`으로 바꾸고 `hitRatio`/`hitProp`, kernel body, dispatch, launch policy는 그대로 뒀지만 full benchmark 1회차 avg가 `0.016806 ms`, B=64 `eaf0a285`가 `0.026108 ms`로 recent accepted band(`~0.021~0.024 ms`)를 명확히 벗어났다. approved veto에 따라 추가 4회 median 측정 없이 즉시 롤백했고, 같은 standalone soft L2 policy 안은 우선순위를 낮춘다. 이후 메모리 계열은 실제 load opcode 변화(A3/A4)나 offline codegen/SASS 기준선 확보 후의 `q/k` path 재정의처럼 더 직접적인 변경만 우선 검토한다.
- [ ] **A6. Shared memory carveout = 0 (L1 극대화)**: `cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0)`. 현재 s_v 512B만 사용 → L1 확대로 state L1 hit 증가 유도. **[시도됨 2026-04-24 codex iter #3, 롤백]** host launch 경로에 variant별 1회 `PreferredSharedMemoryCarveout=0`만 추가한 standalone 실험을 했지만, decision-gate B=64 workload `eaf0a285`가 `0.028602 ms`로 recent accepted band(`~0.021~0.024 ms`)보다 명확히 느려 full benchmark/NCU 전에 즉시 롤백했다. carveout 단독안은 재시도하지 않고, 이후에는 load opcode 변경(A3/A4)이나 q/k 중복 제거(B1)처럼 더 직접적인 구조 변화가 있을 때만 다시 본다.
- [ ] **A7. 반대 실험: carveout = MAX**: 대용량 SMEM으로 state 일부 bring-down 준비 (H 카테고리 도입 시 필수).
- [ ] **A8. s_v[] broadcast 효율 확인**: 모든 thread가 동일 `s_v[vi_a]` 접근 → hardware broadcast 기대. `cuobjdump` 로 `LDS.U.32.BROADCAST` 또는 `MOVM`의 broadcast 모드 확인. 불완전 시 `__shfl_sync(0xffffffff, reg_val, src_lane)`로 레지스터 broadcast 대체.
- [ ] **A9. bf16 vectorized load 폭 증가 시도**: 현재 k, q 각각 `uint2` (64-bit) × 1 → 128-bit 못 채움. 가능하면 레이아웃 조정으로 128-bit 통합 고려 (비현실적일 수 있음).
- [ ] **A10. Cold iter 진단**: Modal iter=50 중 초기 3~5 iter가 유독 느린지 개별 측정. 그렇다면 I1 (warmup kernel) 효과적.
- [x] **A11. Streaming store (`__stcs`) for new_state** [시도됨 Iter 6, 후퇴]. 재시도 금지.
- [ ] **A12. `ROWS_PER_WARP=4` dead prefetch 제거**: `gdn_decode_kernel<4>`에서 첫 4-row chunk만 쓰는데도 `next_a..d` 4개 row를 추가로 읽는 낭비를 제거한다. **[시도됨 2026-04-24 codex iter #4, 롤백]** `state_prefetch<4>` 특수화로 `RPW=4` 경로에서 `next_*` 초기 load/rotate를 없앴지만, representative subset 3-workload quick avg가 `0.028053 ms`, B=64 `eaf0a285` decision gate가 `0.029683 ms`로 recent accepted band(`~0.021~0.024 ms`)보다 명확히 느렸다. source-level로는 `RPW=8/16` 수학식을 안 건드렸어도 helperization이 codegen을 흔들었을 가능성을 배제할 수 없으므로, **large-batch path byte-for-byte 동일성이나 SASS 확인이 없는 같은 구현 형태는 보류**한다. **[재시도됨 2026-04-24 codex iter #5, 롤백]** 이번에는 `RPW=4`를 별도 커널로 물리 분리하고 case `4` dispatch만 교체해 `gdn_decode_kernel<8/16>` 본문을 보존했지만, B=64 `eaf0a285` decision gate가 `0.025462 ms`로 recent accepted band 상단(`~0.024 ms`)을 다시 넘었다. helperization 리스크를 제거해도 개선 근거가 없었으므로, 같은 방향은 separate translation unit/SASS 동일성 증명이 없으면 우선순위를 낮추고 kernel body 무변경 host-side 안(A5) 또는 실제 load opcode 검증(A3/A4)을 먼저 본다.

---

### B. B200 Blackwell 전용 구조적 최적화 (★ Phase 4 돌파구 ★)

**참조**: CUTLASS `sm90_mma_tma_gmma_ss_warpspecialized.hpp` (cluster_shape=(2,1,1) 구간), CUTLASS `examples/70_blackwell_gemm`, FlashAttention v3 mainloop, NVIDIA CUDA Programming Guide "Thread Block Clusters" + "Distributed Shared Memory".

- [ ] **B1. 2-CTA Cluster로 q/k 공유** ★ **가장 유력** ★: V_PER_Q=2 → 같은 qk_head를 쓰는 2개 v_head block을 cluster로 묶는다. **[시도됨 2026-04-24 session iter #2, 후퇴]** `batch_size >= 32` 경로만 compile-time `__cluster_dims__(2,1,1)` cluster kernel로 분기하고 rank 0 CTA가 q/k를 distributed shared memory에 적재해 pair-CTA가 재사용하도록 했지만 full benchmark avg가 `0.018499 ms`, B=64 `eaf0a285`가 `0.029220 ms`로 크게 악화됐다. minimal q/k-share-only 안은 재시도하지 않고, 향후에는 qk reduction 1회화 또는 `cuda::pipeline`/`cp.async`와 결합해 cluster sync cost를 숨길 수 있을 때만 다시 본다.
  - `__cluster_dims__(2, 1, 1)` 또는 host에서 `cudaLaunchKernelEx` + `cudaLaunchAttribute::clusterDim = {2,1,1}`.
  - Grid 재구성: `(batch × NUM_Q_HEADS × split_factor, 2, 1)`.
  - q, k를 distributed SMEM에 cluster 내 **1회만** 로드 후 `cluster.mapa.shared::cluster`로 상대 block 접근 → bf16 read **절반**.
  - 효과 예상: q/k load 오버헤드가 전체의 ~10%라 가정 시 5% (0.0005 ms) 수준. 단, cluster sync 오버헤드 상쇄 필요.
- [ ] **B2. `cuda::pipeline<thread_scope_block, 3>` + `cuda::memcpy_async` state row prefetch**: 현재 register-based 4-row pipeline (`pf_a,b,c,d`)을 SMEM 3-stage async pipeline으로 전환. **[시도됨 2026-04-24 codex iter #2, 후퇴]** current accepted `gdn_decode_kernel<8>` large-batch path만 2-stage `cp.async` shared double-buffer로 바꾼 축소안을 넣고 B=64 `eaf0a285` decision gate를 돌렸지만 `0.028864 ms` (`PASSED`)로 recent accepted band(`~0.021~0.024 ms`)보다 명확히 느렸다. `ROWS_PER_WARP=8` path는 4-row chunk가 두 번뿐이라 stage depth가 얕아 bytes-in-flight 증가보다 commit/wait + shared reload cost가 더 컸던 것으로 보이며, **같은 standalone async fetch 치환은 더 깊은 pipeline이나 q/k 중복 제거 결합 근거가 생길 때까지 보류**한다.
  ```
  stage 0: compute row i
  stage 1: arriving  row i+1 (2 iters ago 발행)
  stage 2: issuing   row i+2
  ```
  `cuda::aligned_size_t<16>` 사용 시 TMA path 자동 진입. bytes-in-flight 증가 → compute ↔ load overlap 강화.
- [ ] **B3. `cp.async.ca.shared.global` 저수준 PTX**: B2의 PTX 버전. 128B row를 async SMEM load. high-level API가 LDGSTS 생성 보장 약할 때 사용.
- [ ] **B4. TMA `cp.async.bulk.tensor.1d` for state row**: host에서 `cuTensorMapEncodeTiled`로 state `[B,8,128,128]` tensor map 1회 생성 (커널 인자로 전달, 재사용). 커널 내 `cp.async.bulk.tensor.1d.shared::cluster.global` 로 row bulk fetch. 주소 계산을 하드웨어가 offload → warp address compute 부담 제거.
- [ ] **B5. Warp Specialization (Producer/Consumer)**: 4 warps 중 1 warp은 TMA/async load 전담, 3 warps는 compute 전담. `mbarrier`로 동기화. Blackwell `setmaxnreg.inc`/`setmaxnreg.dec`로 warp role별 register 재분배 → producer warp은 적게, consumer warp은 많이.
- [ ] **B6. Distributed SMEM output aggregation**: split_factor 큰 경우(B=1, split=8) split별 partial output을 cluster 내 distributed SMEM에 모아 한 번에 global write. 현재는 split별 독립 write.
- [ ] **B7. Cluster size 4 / 8 / 16 확장**: `cudaFuncAttributeNonPortableClusterSizeAllowed` opt-in. 같은 batch의 여러 v_head를 더 큰 cluster로 묶어 q/k 공유 확대. Cluster sync overhead 증가 trade-off 주의.
- [ ] **B8. `mbarrier` 기반 warp-local sync로 `__syncthreads()` 대체**: 현재 s_v load 후 `__syncthreads()`. warp-local arrive/wait로 축소 가능한 구간 탐색.

---

### C. 연산 최적화

**참조**: CUDA Math API 문서 (`__fmaf_rn`, `__reduce_add_sync`), PTX ISA `redux.sync`, `wgmma.mma_async`.

- [x] **C1. Gate 계산 lane 0 집중 + shuffle broadcast** [적용됨 Iter 20]: SFU throughput 경쟁 해소, Phase 2 break-through.
- [ ] **C2. `__reduce_add_sync()` intrinsic으로 warp reduction 대체**: 현재 5-stage `__shfl_down_sync` butterfly. SM100 native `redux.sync.add.f32` 1-instruction. `-arch=sm_100a` 필수. PTX `redux.sync` 생성 확인. **[시도됨 iter #3, build blocked; 재시도됨 2026-04-24 session iter #1, inline PTX도 blocked]** 현재 Modal `tvm_ffi`/nvcc 경로에서는 `__reduce_add_sync(float)` overload가 보이지 않아 전 workload compile failure가 났고, inline PTX `redux.sync.add.f32`도 CUDA 13.0 `ptxas`에서 `Incorrect type '.f32' for operation '.add'`로 거부됐다. Modal toolchain에서 float warp-reduce add 지원이 실제로 확인되기 전에는 재시도하지 않는다.
- [ ] **C3. 명시적 FMA (`__fmaf_rn()`)**: 현재 dot product / state update가 nvcc FMA 자동 생성 의존. PTX 생성 물(FFMA vs FADD+FMUL) 검증 후 미생성 구간 명시적 호출. **[시도됨 2026-04-24 session iter #4, 후퇴]** `C4`와 묶어 `q/k`를 `float4`로 유지하고 `__fmaf_rn` helper, `g/beta/beta_g/qk_dot` block-scalar dedup를 기존 `s_v` barrier에 합쳐 live range를 줄이는 all-path 미세 최적화를 넣었지만 5회 avg가 `[0.012688, 0.013137, 0.013102, 0.012761, 0.018742] ms`, median `0.013102 ms`로 accepted baseline `0.012920 ms`보다 악화됐다. standalone FFMA/live-range cleanup만으로는 current low-issue / reg-limited occupancy 병목을 움직이지 못했고 long-tail variance도 커졌다. 같은 all-path 안은 재시도하지 않고, **SASS/ptxas 기준으로 register가 실제 `56 -> 52 이하`로 감소한다는 근거가 있거나 특정 batch path에만 격리해 검증할 수 있을 때만** 다시 본다.
- [ ] **C4. k_vals/q_vals를 `float4` 자료형으로 유지**: 현재 개별 float 배열. `float4 k4, q4;`로 보관 → nvcc FFMA4 자동 생성 유도, state `float4`와 layout 일치.
- [ ] **C5. Output 계산 fusion 추가**: `scale_g * qs_x + scale_qk * res_x` 이미 pre-computed 상수 활용. 추가 여지 낮음.
- [x] **C6. Fast math intrinsic (gate)** [시도됨 Iter 2, 후퇴]. 재시도 금지.

---

### D. 실행 구성 최적화

**참조**: NVIDIA CUDA Graph API Reference, CUTLASS `sm90_gemm_tma_warpspecialized_pingpong.hpp` (persistent pattern), NVIDIA blog "Reducing launch overhead with CUDA Graphs".

- [x] **D1. Split factor adaptive** [적용됨]: B≤2:8, B<32:4, B≥32:2.
- [x] **D2. Block size 128 통일** [적용됨 Iter 8].
- [x] **D3. `__launch_bounds__(128, 9)`** [적용됨 Iter 17].
- [ ] **D4. Split factor 추가 세분화**: B≤4:8, B≤16:4, B≥32:2 등 3~4단계로 확장. outlier batch(B=8, 16, 32)에서 효과 검증. 특히 B=32가 경계이므로 B=24, 48 등 중간 실험 가치. **[시도됨 iter #1 일부, 후퇴]** B≥32를 단순히 split=4로 올리는 실험은 median 0.012920 ms로 롤백. q/k/v/gate 중복과 per-block ILP 감소가 더 커 보이므로 같은 방식 재시도 금지.
- [ ] **D5. CUDA Graph capture (host-side)** ★ **Phase 4 최대 돌파구 후보** ★: `cudaStreamBeginCapture/EndCapture`로 decode call 그래프화. launch overhead 2~5 μs → ~60 ns/node. 10μs 타겟에서 **결정적 효과**. **[시도됨 iter #4, 후퇴]** `kernel.cu` host launch path에 수동 graph exec cache를 넣었지만 현재 flashinfer-bench isolated runner + TVM FFI 경로에서는 avg latency가 `0.014014 ms`, retry `0.017512 ms`로 baseline(`0.012920 ms` median)보다 명확히 악화됐다. workload별 graph instantiate/update 비용이 launch 절감 이득을 넘은 것으로 보이며, **같은 경로에서는 재시도 금지**.
  - 단계: (1) `scripts/run_modal.py`에서 bench 반복 루프 구조 확인 → graph capture/replay 허용 여부 판단. (2) 허용 시 host wrapper에 `cudaGraph_t`, `cudaGraphExec_t` 추가. (3) 첫 호출은 capture, 이후는 `cudaGraphLaunch`.
  - 제약: 입력 텐서 주소가 매번 동일해야 함 (또는 `cudaGraphExecUpdate` 사용).
  - 재시도 가능 조건: graph를 kernel wrapper 내부가 아니라 benchmark/harness 상위 반복 루프에서 한 번 캡처해 여러 decode call에 재사용할 수 있거나, stable buffer/process reuse가 보장되어 instantiate/update가 측정 구간 밖으로 빠질 때.
  - Framework 제약으로 불가한 경우 D6 대체.
- [ ] **D6. Persistent kernel + cooperative groups**: `cudaLaunchCooperativeKernel`. 단일 launch에서 모든 (batch, head, split) work를 grid-stride로 처리. Launch overhead 완전 제거. 구현 난이도 상.
- [ ] **D7. Stream priority 최고**: `cudaStreamCreateWithPriority`. Modal 환경에 concurrent work 있을 때 효과.
- [ ] **D8. Grid 크기를 B200 SM 수(≈148) 배수에 맞춤**: tail effect 감소. B=1, split=8 → 64 blocks (SM 43% util). Split 16으로 확대 시 128 blocks (87%). Split 확대가 RPW, 레지스터 압박에 미치는 영향 동시 확인. 단, iter #1에서 B≥32 grid만 늘리는 단순 split 확대는 benchmark 후퇴.

---

### E. 공유 메모리 / 레지스터 재분배

- [ ] **E1. q, k를 SMEM에 로드 후 block 내 broadcast**: 현재 128 lane이 독립 로드. Block 내 모든 warp가 동일 q/k 사용하므로 SMEM 1회 로드 + broadcast. Register 절약 → occupancy 향상 여지. SMEM bank broadcast는 conflict-free. **[시도됨 iter #2, 후퇴 / 재시도됨 2026-04-24 session iter #3, 후퇴]** 처음에는 q/k/v를 모두 shared에 staging하고 gate도 block당 1회만 계산하는 축소 버전을 시험했지만 avg latency가 `0.013278 ms`로 악화, B=64 workload `eaf0a285`가 `0.024691 ms`까지 상승했다. 이후 `warp0`만 q/k와 block scalar를 준비하고 기존 `s_v` barrier를 재활용하는 재변형도 full benchmark avg가 `0.012932 ms`로 same-day rollback baseline `0.012671 ms`를 넘지 못했다. shared q/k fan-out standalone 계열은 당분간 우선순위를 내리고, SASS로 register 감소나 issue-mix 개선 근거가 있을 때만 다시 본다.
- [x] **E2. s_v → warp-register + sync 제거** [시도됨 Iter 1, 후퇴].
- [ ] **E3. s_v를 warp 독립 register + `__shfl_sync` broadcast (sync 유지)**: E2와 구조 다름 — `__syncthreads()`는 유지하되 s_v shared 대신 warp 내 register + shuffle. 구조 차이로 재시도 가치 있음.
- [ ] **E4. k_vals/q_vals를 SMEM에 staging → ldmatrix.sync.aligned**: Tensor Core 도입(J1) 전 단계로도 유용. `ldmatrix.x4.m8n8.shared.b16` 로 16-bit 텐서 레지스터 분산 로드 실험.

---

### F. 수학적 재구성

- [x] **F1. ks/qs dot product 단일 패스** [적용됨].
- [x] **F2. Gate 계산 lane 0 집중** [적용됨, C1과 동일].
- [ ] **F3. State update in-place 확인**: `new_state == state` 허용되는지 API 확인 (`scripts/pack_solution.py` 시그니처, bench harness). 허용 시 store 트래픽 완전 제거 — 매 iter당 16 KB × 블록 수 절감.
- [ ] **F4. Output을 warp 내 4-lane 분산 쓰기**: 현재 lane 0만 4개 vi의 bf16(2B) × 4 = 8B store. lane 0~3이 각 1개 vi store → 4-way coalesced STG.B16, latency hiding.
- [ ] **F5. qs/ks reduction을 동일 연산으로 fuse**: `qk_local` (첫 dot) 과 `ks/qs` (state dot)가 구조 동일 — 템플릿 helper로 통합하여 unroll 효율 상승.

---

### G. 컴파일러 힌트 및 빌드 옵션 (Phase 3 우선)

**참조**: `nvcc --help` 공식 문서, NVIDIA PTX ISA 문서, CUDA Programming Guide "Maximize Utilization".

- [x] **G1. `#pragma unroll`** [적용됨].
- [x] **G2. `__restrict__`** [적용됨].
- [x] **G3. `__launch_bounds__(128, 9)`** [적용됨 Iter 17].
- [ ] **G4. `-arch=sm_100a` 명시 확인**: TMA bulk tensor, tcgen05, `__reduce_add_sync` (redux.sync), cluster 등 Blackwell 전용 활성화. 현재 `sm_100` 추정. `nvcc --verbose`로 실제 gencode 확인. `sm_100f` (family-specific) 대안.
- [ ] **G5. `-Xptxas -v -Xptxas -warn-spills` 로 register/spill 분석**: Iter 18 (128, 10) 실패 원인. 현재 (128, 9) 하에서 정확한 register/spill 수치 확인 후 G7 조합 기준선. **[시도됨 2026-04-24 codex iter #1, 롤백]** `decode_submit_entry.py` wrapper에 append-only ptxas flags를 넣고 benchmark 경로에서 5회 순차 측정을 다시 돌렸지만 `[0.011985, 0.016380, 0.011427, 0.016363, 0.015814] ms`, median `0.015814 ms`로 accepted baseline `0.012920 ms`보다 크게 악화됐다. `tvm_ffi` build path는 successful ptxas stdout를 surface하지 않아 codegen 기준선 고정 이득도 제한적이었으므로, **benchmark runtime path에서의 G5 flag 주입은 재시도하지 않고 이후에는 standalone build/objdump 같은 비측정 경로에서만 재확인**한다.
- [ ] **G6. `__builtin_assume()` 힌트**: 
  ```
  __builtin_assume(blockDim.x == BLOCK_SIZE);
  __builtin_assume(batch_size > 0);
  __builtin_assume((batch_size & (batch_size - 1)) == 0);  // power-of-2 가정 시
  ```
  분기/모듈로 연산 제거. 인덱스 계산 단순화.
- [ ] **G7. `-maxrregcount=48 or 56`를 launch_bounds와 조합**: (128, 10) 같이 공격적 설정 시 spill 방지하며 occupancy 확보.
- [ ] **G8. `__forceinline__` 누락 점검**: `softplus` 이미 적용. 신규 helper 추가 시 확인.
- [ ] **G9. `-use_fast_math` 전역 (주의)**: correctness 검증 필수. 국소 Iter 2 실패, 전역은 영향 범위 큼.

---

### H. 비동기 메모리 연산 (Phase 4 핵심 — B와 함께)

**참조**: libcu++ `<cuda/pipeline>` 공식 예제, CUTLASS `cute/atom/copy_traits_sm100*.h` (TMA descriptor), FlashAttention v3 mainloop pipeline 상태 기계, PTX ISA `cp.async.bulk` 섹션.

- [ ] **H1. `cp.async.ca.shared.global` 기반 state row prefetch**: register pipeline을 SMEM async pipeline으로. 16B-align 128B row → async SMEM. Compute ↔ load overlap.
- [ ] **H2. `cuda::pipeline<thread_scope_block, 3>` + `cuda::memcpy_async`** (B2 재차): H1의 high-level 래퍼. `cuda::aligned_size_t<16>`으로 TMA path. API 안정성 ○.
- [ ] **H3. TMA tensor map (host 1회 생성)**: `cuTensorMapEncodeTiled`로 state tensor map 생성 → 커널 내 `cp.async.bulk.tensor.1d`. 주소 계산 HW offload. Host wrapper 변경 필요.
- [ ] **H4. `__pipeline_memcpy_async` 저수준**: high-level API가 LDGSTS 생성 보장 약할 때 PTX 수준 primitive. **[시도됨 iter #5, 후퇴]** `ROWS_PER_WARP=16` 경로에 warp-private 2-stage 4-row staging을 넣고 `sm_100a` JIT flags까지 명시했지만 full benchmark avg가 `0.013124 ms`로 baseline `0.012920 ms`보다 악화, B=64 `eaf0a285`도 `0.024348 ms`로 baseline(`0.021~0.022 ms`)보다 느려졌다. 현재 형태의 per-thread async copy + shared reload은 재시도하지 않음.
- [ ] **H5. `mbarrier` 기반 async barrier**: `__syncthreads()` 대체 가능 구간.

---

### I. 안정성·분산 감소 (★ Phase 3 핵심 ★)

- [ ] **I1. L2 cache warmup kernel 선행 실행**: 벤치마크 측정 전 더미 kernel로 state L2 pre-load. Modal iter 초기 3~5개의 outlier 제거. A10에서 cold iter 확인 후 판단.
- [ ] **I2. `cudaCtxResetPersistingL2Cache()` 타이밍 최적화**: persistence 설정 후 reset 시점 조절. 측정 간섭 최소화.
- [ ] **I3. Grid 크기를 SM 배수에 맞춤** (D8과 중복, 안정성 관점): 작은 batch의 tail effect 감소가 median 개선에 기여.
- [ ] **I4. `cudaDeviceSynchronize()` 불필요 호출 제거**: Bench script 측 확인. 불필요한 sync가 측정값을 튀게 할 수 있음.
- [ ] **I5. `config.toml` warmup iter 증가**: 현재 10 → 20/30. trials도 증가. 최종 배포 전 되돌리되, Phase 3 디버깅 구간엔 유효.
- [ ] **I6. `__builtin_prefetch` 첫 read**: cold start state read 초기 지연 감소. 첫 block이 유독 느린 패턴 있을 때.
- [ ] **I7. Kernel launch ordering (stream priority + event)**: warmup/측정 분리 명확화.

---

### J. 구조적 재설계 (Phase 4 후순위)

- [ ] **J1. Tensor Core (wgmma / tcgen05) fit 검토**: state [128,128] × q/k [128] → matrix-vector. tcgen05은 MMA 전용 → q를 [16, 128] padding(batch 16 기준) matmul 변환 가능성 탐색. 메모리 이동/padding FLOPS 오버헤드가 이득 상쇄 가능성 높음. B=32+ 대배치 한정 실험 가치.
- [ ] **J2. State layout 변경 (k-last → v-last)**: `[B, H, V, K]` → `[B, H, K, V]`. Read 패턴 변화, FLA 표준과 일치. ldmatrix-friendly. API 호환성(state/new_state 입출력 형식) 확인 필수.
- [ ] **J3. 커널 2분할 (compute/write)**: 단일 커널 → 2 커널 + stream 병렬. CUDA Graph(D5)와 조합 시 효과적.
- [ ] **J4. Multi-kernel pipeline**: decode persistent + 다음 token 준비를 별도 stream overlap. 단일 커널 벤치 한정으로는 이득 제한적.

---

## 8. 현재 커널 구조 요약

```
Kernel: gdn_decode_kernel<ROWS_PER_WARP>
  Grid:  (B × 8 × split_factor, )
  Block: 128 threads (4 warps), __launch_bounds__(128, 9)
  Per block: (batch, v_head, split_id) 1개 처리
  Per warp:  ROWS_PER_WARP 개 V-rows 담당
  Pipeline:  4-row float4 state prefetch (pf_a, pf_b, pf_c, pf_d)
  Reduction: 5-stage butterfly __shfl_down_sync + shfl broadcast
  Gate:      lane 0만 softplus/exp 계산 후 3회 shfl broadcast (Iter 20)
  L2:        host-side persistence with hitRatio 공식 (setup_l2_persistence)

Split factor: B<32:8 (RPW=4), B≥32:4 (RPW=8)
State: [B, 8, 128, 128] fp32 k-last
I/O:   bf16 입력(q/k/v/a/b_gate), fp32 state/A_log/dt_bias, bf16 출력
```

### 리소스 사용 추정 (B200)

```
Reg/thread ≈ 56 (launch_bounds 9 가정): 9 × 128 × 56 = 64,512 < 65,536
Occupancy:  36 warps / 64 = 56.25%
SMEM:       512 B (s_v[128] × 4B) — 비병목
B200 SM ≈ 148:
  B=1,  split=8: grid=64   → 43% util (tail)
  B=2,  split=8: grid=128  → 86%
  B=4,  split=8: grid=256  → 1 partial wave
  B=16, split=8: grid=1024 → balanced
  B=32, split=4: grid=1024 → balanced
  B=64, split=4: grid=2048 → NCU상 1.54 waves/SM
```

### 현재 병목 (Phase 3/4 관점)

1. **Kernel launch overhead 2~5 μs = 10μs 타겟의 20~50%** → **D5 (CUDA Graph) 최우선**
2. 작은 batch(B=1~2) SM util 40% → **B1 (2-CTA cluster) 또는 split 확대**
3. Modal 노이즈 ±0.003 ms → **I1 (L2 warmup) + 측정 전략 안정화**
4. q/k bf16 read가 V_PER_Q=2 중복 → **B1 cluster로 반감 가능**
5. SFU 경쟁은 Iter 20에서 해소됨

---

## 9. 성능 로그

### 누적 히스토리

| Iter | 최적화 | Avg [ms] | 판정 |
|:----:|-------|:-------:|:----:|
| 1  | v register + sync 제거 | 0.015 | 롤백 |
| 2  | fast math gate | 0.017 | 롤백 |
| 3  | B=1 split=16 block=64 | INCORRECT | 롤백 |
| 4  | B>16 split=2 block=128 | 0.017 | 롤백 |
| 5  | `__launch_bounds__(256,4)` | 0.015 | 롤백 |
| 6  | `__stcs` new_state | 0.019 | 롤백 |
| 7 ✅ | ROWS_PER_WARP 템플릿화 | 0.014 | 유지 |
| 8 ✅ | 모든 B block=128, B>16 split=2 | 0.013 | 유지 |
| 9 ✅✅ | **B≥3 split=4 통일 (RPW=8)** | 0.011 | **Phase 1 달성** |
| 10–13 | split/s_v 변형 | ±0 | 롤백 |
| 14 | adaptive split + RPW 16/32 | 0.021 | 롤백 (reg 압박) |
| 15 | `__shfl_xor` + vec output | 0.015 | 롤백 |
| 16 ✅ | `__launch_bounds__(128,7)` | 0.013 | 유지 |
| 17 ✅✅ | **`__launch_bounds__(128,9)`** | 0.012 | **Phase 1 재달성** |
| 18 | `__launch_bounds__(128,10)` | 0.014 (spill) | 롤백 |
| 19 | B≤16 split=8 | 0.014 | 롤백 |
| 20 ✅✅✅ | **gate lane 0 + shfl broadcast** | **0.010 best / 0.012 median** | **Phase 2 달성** |
| R1 | B≥32 split=4로 grid 2배 확대 | 0.012920 median | 롤백 |
| R2 | q/k/v shared staging + gate block 1회 계산 | 0.013278 avg | 롤백 |
| R3 | split-local `s_v` staging + carveout=0 (`__reduce_add_sync(float)`는 build blocked) | 0.013244 median | 롤백 |
| R4 | `kernel.cu` host launch path 수동 CUDA Graph replay | 0.014014 avg / 0.017512 retry | 롤백 |
| R5 | RPW=16 warp-private `__pipeline_memcpy_async` + `sm_100a` JIT flags | 0.013124 avg | 롤백 |
| R6 | `B>=32` 전용 256-thread / 8-warp large-batch path | 0.012979 avg | 롤백 |
| R7 | inline PTX `redux.sync.add.f32` blocked, fallback state access-property persisting | 0.018830 avg | 롤백 |
| R8 | `B>=32` compile-time 2-CTA cluster q/k 공유 | 0.018499 avg | 롤백 |
| R9 | `warp0` q/k shared stage + CTA scalar dedup (all paths) | 0.012932 avg / rollback baseline 0.012671 avg | 롤백 |
| R10 | `float4` q/k + `__fmaf_rn` + existing-barrier block scalar dedup | 0.013102 median | 롤백 |
| R11 | G5-lite wrapper ptxas flags (`-Xptxas -v -warn-spills`) | 0.015814 median | 롤백 |
| R13 | A6 standalone `PreferredSharedMemoryCarveout=0` | `eaf0a285` decision gate 0.028602 avg | 롤백 |
| R14 | `ROWS_PER_WARP=4` dead-prefetch 제거 | subset 0.028053 avg / `eaf0a285` gate 0.029683 avg | 롤백 |
| R15 | `RPW=4` 물리 분리 + dead-prefetch 제거 재시도 | `eaf0a285` decision gate 0.025462 avg | 롤백 |
| R16 | A5 standalone `missProp=Streaming` | full benchmark 1회 0.016806 avg / `eaf0a285` 0.026108 | 롤백 |

**핵심 인사이트 (Iter 20)**: 32 lanes 동시 exp/log1p → SFU throughput 심각 경쟁. Lane 0 전담 + 3 shuffle로 SFU 경쟁 완전 제거 = Phase 2 break-through.

**R1 인사이트**: B=64의 `waves/SM` 부족을 단순 split 증가로 해결하려 했지만 B=64 latency가 기존 `0.021~0.022 ms`에서 `0.024~0.029 ms`로 악화. grid 증가만으로는 부족하고 q/k/v/gate 중복 제거 또는 launch overhead 제거가 필요.

**R2 인사이트**: block-invariant q/k/gate 중복 자체는 존재하지만, q/k/v를 shared에 올리는 축소형 E1은 현재 커널의 핵심 비용인 state row streaming을 줄이지 못했다. 반대로 shared scalar load와 block-wide barrier만 늘어나 B=64가 `0.024691 ms`로 악화됐다. 같은 계열은 cluster 공유나 qk reduction 1회화처럼 더 큰 중복 제거가 동반될 때만 재검토한다.

**R3 인사이트**: block이 실제로 쓰는 `v` row만 shared에 적재하고 `PreferredSharedMemoryCarveout=0`를 줘도 benchmark는 개선되지 않았다. 5회 avg가 `[0.012954, 0.013378, 0.013142, 0.013244, 0.017654] ms`로 튀었고 median이 `0.013244 ms`까지 악화됐다. `__reduce_add_sync(float)`도 현재 build path에서 바로 쓸 수 없었으므로, 다음에는 launch overhead 제거(D5)나 CTA 간 q/k 공유(B1)처럼 더 큰 구조적 중복 제거를 우선한다.

**R4 인사이트**: CUDA Graph 자체의 잠재 이득은 크지만, 현재 flashinfer-bench isolated runner + TVM FFI wrapper 경로에서는 graph instantiate/update 오버헤드가 초소형 decode launch savings를 상쇄했다. 첫 full benchmark avg가 `0.014014 ms`, update 생략 캐시를 넣은 retry도 `0.017512 ms`로 더 악화됐고, B=64 workload `eaf0a285`는 `0.030310 ms`, `0.031475 ms`까지 후퇴했다. 같은 경로에서는 D5를 접고, 다음에는 kernel body duration을 직접 줄일 수 있는 B1/H2 계열을 우선한다.

**R5 인사이트**: `ROWS_PER_WARP=16` large-batch 경로만 골라 warp-private `__pipeline_memcpy_async` 2-stage prefetch를 넣어도 benchmark는 개선되지 않았다. correctness는 유지됐지만 full benchmark avg가 `0.013124 ms`로 baseline `0.012920 ms`보다 느려졌고, B=64 `eaf0a285`도 `0.024348 ms`로 후퇴했다. per-thread async copy + shared reload + commit/wait overhead가 기존 register prefetch보다 비싸서 overlap 이득을 상쇄한 것으로 보인다. 다음 async 계열 재시도는 block/warp 단위 `cuda::pipeline` 또는 `cp.async`로 bytes-in-flight를 더 키우거나, B1처럼 중복 q/k work 제거와 결합할 때만 검토한다.

**R6 인사이트**: `batch_size >= 32` large-batch path만 256-thread / 8-warp로 키워 per-CTA parallelism을 늘려도 benchmark는 개선되지 않았다. 전체 avg가 `0.012979 ms`로 baseline `0.012920 ms`보다 소폭 나빠졌고, 핵심 타깃 B=64 `eaf0a285`는 `0.023968 ms`로 baseline(`0.021~0.022 ms`)보다 느려졌다. standalone 256-thread 확대만으로는 q/k load와 qk reduction의 warp 중복, `ROWS_PER_WARP=8`로 줄어든 per-warp ILP를 상쇄하지 못했다. 다음 large-batch 재시도는 q/k shared/cluster 공유처럼 warp 중복 제거와 결합될 때만 검토한다.

**R7 인사이트**: C2의 inline PTX 경로까지 시도했지만 현재 Modal CUDA 13.0 `ptxas`는 `redux.sync.add.f32`를 실제로 수용하지 않았다. 또한 A2 성격의 standalone state access-property persisting 힌트는 full benchmark avg를 `0.018830 ms`까지 악화시켰다. 현재 단계에서는 문서상 가능성만 있는 float warp-reduce/soft cache-hint보다, load opcode를 실제로 바꾸는 A3/A4 검증이나 q/k 중복 제거(B1)처럼 보다 구조적인 변화가 우선이다.

**R8 인사이트**: `batch_size >= 32` 경로만 compile-time 2-CTA cluster로 바꿔 rank 0 CTA의 distributed shared memory `q/k` staging을 pair-CTA가 재사용하게 해도 benchmark는 크게 악화됐다. full benchmark avg가 `0.018499 ms`로 baseline `0.012920 ms`보다 크게 느려졌고, 핵심 B=64 `eaf0a285`도 `0.029220 ms`까지 상승했다. minimal cluster q/k-share-only 안은 cluster barrier와 remote shared read overhead를 상쇄하지 못했으므로, 다음 cluster 재시도는 qk reduction 1회화 또는 async producer/consumer와 결합되는 더 강한 구조 변경일 때만 검토한다.

**R9 인사이트**: E1 계열을 더 좁혀 `warp0`만 q/k와 block-invariant scalar를 준비하고 기존 `s_v` barrier로 CTA 전체에 배포하는 변형까지 시도했지만, same-day baseline full benchmark를 넘지 못했다. full avg는 `0.012932 ms`였고 rollback 후 accepted baseline은 `0.012671 ms`였다. extra barrier를 제거해도 shared q/k fan-out cost가 남았고, standalone CTA-local q/k dedup만으로는 current kernel의 low-issue / small-grid 병목을 충분히 움직이지 못했다.

**R10 인사이트**: `q/k float4 + __fmaf_rn` helper와 기존 `__syncthreads()`를 재활용한 block-scalar dedup(`g/beta/beta_g/qk_dot`)을 묶어도 benchmark는 개선되지 않았다. 5회 avg가 `[0.012688, 0.013137, 0.013102, 0.012761, 0.018742] ms`였고 median은 `0.013102 ms`로 accepted baseline `0.012920 ms`보다 악화됐다. B=64 `eaf0a285`도 대부분 `0.024~0.025 ms`에 머물렀고 5회차는 `0.028653 ms`까지 튀었다. source-level FFMA/live-range cleanup만으로는 current low-issue / reg-limited occupancy 병목을 못 움직였으므로, 다음에는 실제 register 감소 근거가 있는 SASS/ptxas 기준선 확보나 특정 batch-path 격리 없이는 같은 계열을 재시도하지 않는다.

**R11 인사이트**: append-only `-Xptxas -v -warn-spills` 자체는 “compile/codegen 기준선 재고정” 의도였지만, benchmark runtime path에 wrapper flag를 직접 얹는 방식은 5회 순차 median이 `0.015814 ms`로 크게 후퇴해 채택할 수 없었다. 동시에 rollback 후 NCU는 현재 accepted large-batch 경로가 `gdn_decode_kernel<8>`, `Grid Size=2048`, `Duration=31.97 us`, `Issue Slots Busy=21.98%`, `Achieved Occupancy=40.11%`, `Registers/thread=56`, spill `0`임을 다시 보여 줬다. 다음 기준선 고정 작업은 benchmark path 밖의 standalone build/objdump로 옮기고, 실제 kernel-side 후보는 B2/H2 block-wide async pipeline이나 A5 `missProp=Streaming`처럼 Duration을 직접 건드리는 안으로 좁히는 편이 낫다.

**R12 인사이트**: current accepted `gdn_decode_kernel<8>` large-batch path에 2-stage `cp.async` shared double-buffer를 붙인 standalone async staging도 B=64 decision gate에서 `0.028864 ms`로 후퇴했다. `ROWS_PER_WARP=8` path는 현재 4-row stage가 두 번뿐이라 register pressure 완화보다 commit/wait + shared round-trip cost가 더 크게 작용했고, 같은 current-kernel<8> async fetch 치환은 더 깊은 pipeline이나 q/k 중복 제거와 결합되지 않는 한 우선순위를 낮춘다.

**R13 인사이트**: A6 standalone `PreferredSharedMemoryCarveout=0`도 B=64 decision gate에서 `0.028602 ms`로 recent accepted band(`~0.021~0.024 ms`)보다 명확히 느려졌다. 현재 커널은 shared usage가 작지만, carveout만으로는 low-issue / low-occupancy / poor-cache-hit 병목을 움직이지 못했고 host-side attribute 설정만으로 benchmark latency가 악화될 수 있음을 확인했다. 이후 메모리 계열은 carveout/L2 힌트 단독안보다 실제 load path(A3/A4) 변경이나 q/k 중복 제거(B1)처럼 더 직접적인 구조 변화에 집중한다.

**R14 인사이트**: `gdn_decode_kernel<4>` 전용 dead-prefetch 제거는 코드상으로는 맞는 낭비 제거였지만, helperization 형태로 넣자 representative subset과 B=64 guard가 모두 악화됐다. 즉 small-batch 전용 의도만으로는 충분하지 않고, 다음 `RPW=4` 계열은 `gdn_decode_kernel<8/16>` large-batch path가 실제로 byte-for-byte 동일하거나 SASS가 동일하다는 증거를 먼저 확보한 뒤에만 다시 시도한다.

**R15 인사이트**: `RPW=4`를 별도 커널로 물리 분리해 `gdn_decode_kernel<8/16>` 본문과 launch policy를 그대로 둔 상태에서도 B=64 guard가 `0.025462 ms`로 recent accepted band 상단을 넘었다. 즉 iter #4의 회귀를 helperization/codegen 흔들림만으로 설명하기 어렵고, dead-prefetch 제거 자체의 leverage가 작거나 small-batch 이득이 large-batch veto를 상쇄하지 못한다. 같은 `RPW=4` 계열은 우선순위를 낮추고, 다음에는 kernel body 무변경 A5 또는 실제 state load opcode 변화가 확인되는 A3/A4 쪽을 먼저 본다.

**R16 인사이트**: `setup_l2_persistence()`의 `missProp`만 `Streaming`으로 바꾸는 standalone soft L2 policy 안도 benchmark를 크게 악화시켰다. full benchmark avg가 `0.016806 ms`, 핵심 B=64 `eaf0a285`가 `0.026108 ms`로 recent accepted band를 명확히 넘었고, rollback 후 baseline NCU도 `gdn_decode_kernel<8>`, `Duration=31.46 us`, `Issue Slots Busy=21.73%`, `Achieved Occupancy=39.79%`, `Registers/thread=56`, `L1 hit=7.86%`, `L2 hit=1.76%`로 근본 병목이 그대로였다. host-side cache policy 단독안은 우선순위를 낮추고, 다음 메모리 계열은 실제 load opcode 변화(A3/A4)나 offline codegen/SASS 기준선 확보 후의 `q/k` path 검토처럼 codegen 사실이 보이는 방향으로만 좁힌다.

### Phase 3+ 기록 템플릿

```
### Iteration N
- 최적화: <카테고리.번호 + 설명 (예: "D5. CUDA Graph capture")>
- 변경 요약: <상세>
- 참조 구현: <있으면 명시 (예: "CUTLASS examples/70"), 없으면 "없음">
- 측정 5회: [X.XXX, X.XXX, X.XXX, X.XXX, X.XXX] ms
- Median:   X.XXX ms (이전 median: Y.YYY ms)
- 변화:     ±X.XXX ms
- Status:   correct / incorrect
- 판정:     유지 / 롤백
- 현재 Phase: 3 / 4
- 인사이트: <다음 iteration에 영향 줄 교훈>
```

---

## 10. 완료 조건

- [x] **Phase 1**: Avg ≤ 0.012 ms (Iter 17)
- [x] **Phase 2**: best of 3 ≤ 0.010 ms (Iter 20)
- [ ] **Phase 3**: **median of 5 ≤ 0.010 ms** (안정화)
- [ ] **Phase 4**: **median of 5 ≤ 0.009 ms** (구조적 개선)

**Phase 4 달성 시 워크플로우 종료 후 사용자에게 최종 보고. 미달성 시 절대 중단 없이 루프 계속.**

---

## 11. 롤백 정책 (간결)

- 수정 전 현재 커널 **전체** 코드 백업.
- correctness 실패 / latency 후퇴 / Phase 3+에서 median 후퇴 → **즉시 롤백**.
- 같은 최적화 2회 실패 → `[실패]` 처리, 넘어감.
- **Phase 3+**: best만 개선 + median/worst 악화 시에도 롤백 검토 (안정성 우선).
- 구조적 변경(cluster, async pipeline, warp specialization, Graph)은 partial 롤백 불가 → 전체 스냅샷 복원.
- 경계값(목표 ±0.003 ms) 결과는 1~2회 추가 측정으로 확정.

---

## 12. 전략 지침 (요약)

- **개선 < 0.001 ms**: 소규모 최적화 조합 고려. 단독으로는 판단 어려움.
- **누적**: 성공한 최적화 위에 다음을 쌓는다.
- **Phase 3 돌파 추천 순서**: G5 (spill 분석) → A2 (annotated_ptr) → A3/A4 (ldg/nc) → G4 (sm_100a 확인) → C2 (reduce_add_sync) → I1 (L2 warmup). 구현 난이도 낮은 것부터.
- **Phase 4 돌파 추천 순서**: B1 변형 (단, R8의 minimal q/k-share-only 안 제외) → B2/H2 (async pipeline) → B5 (warp specialization) → D5 (단, graph를 wrapper 바깥 반복 루프에서 재사용할 수 있을 때만). 현재 harness 경로에선 D5가 bad direction이었다.
- **10회 이상 답보**: `ncu --set full` + `cuobjdump --dump-sass` 로 병목 재확인 (`smsp__inst_executed_pipe_*`, `smsp__warp_issue_stalled_*`, SASS 내 FFMA/LDG/SHFL 비율). 사용자에게 결과 공유.
- `modal run` 컴파일 에러 즉시 해결. `-Xptxas -v` 출력은 매 변경 후 확인 (register/spill 추적).

---

## 13. 실패 메모 (iter #1 claude 2026-04-24)

- **F4 (output store 4-way lane 분산)**: ❌ +43.1% 회귀 (0.011108 → 0.015891 ms, 54/54 PASS). 현 inner loop 에 `qs_*` broadcast 4개 shfl 추가 + lane select 체인이 register pressure / issue slot 을 잠식. 다음 iter 재시도 금지. shfl_xor reduce 재구성 없이는 현 구조에서 4-way store 이득 없음.
- 다음 iter 우선순위: A6 (SMEM carveout=0 standalone) → G6 (__builtin_assume) → F3 (in-place state API 조사).
