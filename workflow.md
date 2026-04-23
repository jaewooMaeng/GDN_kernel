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
- [ ] **A2. `cuda::annotated_ptr<float, cuda::access_property::persisting>`로 state 래핑**: 커널 내부에서 포인터 단위 hint. Stream attribute보다 정교. miss 경로는 `access_property::streaming` 으로 분리 가능.
- [ ] **A3. State read를 `__ldg()`로 명시**: read-only path 강제. `__restrict__` + nvcc 자동 생성 여부를 `cuobjdump --dump-sass`로 `LDG.E.128.CONSTANT` 확인 후 미생성이면 명시. `float4` 로드와 결합 시 `ld.global.nc.v4.f32`.
- [ ] **A4. `ld.global.nc.v4.f32` inline PTX**: A3이 자동 생성 안 될 때 non-coherent cache path 강제. 
  ```
  asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(x),"=f"(y),"=f"(z),"=f"(w) : "l"(ptr));
  ```
- [ ] **A5. `missProp=Streaming` 실험**: 현재 `missProp=Normal`. state miss 시 L2 오염 방지. Prefetch 리듬이 일정하면 차이 미미하나 B=64+ 대배치에서 state가 L2의 25% 차지 → miss 비중 의미 있을 수 있음.
- [ ] **A6. Shared memory carveout = 0 (L1 극대화)**: `cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0)`. 현재 s_v 512B만 사용 → L1 확대로 state L1 hit 증가 유도.
- [ ] **A7. 반대 실험: carveout = MAX**: 대용량 SMEM으로 state 일부 bring-down 준비 (H 카테고리 도입 시 필수).
- [ ] **A8. s_v[] broadcast 효율 확인**: 모든 thread가 동일 `s_v[vi_a]` 접근 → hardware broadcast 기대. `cuobjdump` 로 `LDS.U.32.BROADCAST` 또는 `MOVM`의 broadcast 모드 확인. 불완전 시 `__shfl_sync(0xffffffff, reg_val, src_lane)`로 레지스터 broadcast 대체.
- [ ] **A9. bf16 vectorized load 폭 증가 시도**: 현재 k, q 각각 `uint2` (64-bit) × 1 → 128-bit 못 채움. 가능하면 레이아웃 조정으로 128-bit 통합 고려 (비현실적일 수 있음).
- [ ] **A10. Cold iter 진단**: Modal iter=50 중 초기 3~5 iter가 유독 느린지 개별 측정. 그렇다면 I1 (warmup kernel) 효과적.
- [x] **A11. Streaming store (`__stcs`) for new_state** [시도됨 Iter 6, 후퇴]. 재시도 금지.

---

### B. B200 Blackwell 전용 구조적 최적화 (★ Phase 4 돌파구 ★)

**참조**: CUTLASS `sm90_mma_tma_gmma_ss_warpspecialized.hpp` (cluster_shape=(2,1,1) 구간), CUTLASS `examples/70_blackwell_gemm`, FlashAttention v3 mainloop, NVIDIA CUDA Programming Guide "Thread Block Clusters" + "Distributed Shared Memory".

- [ ] **B1. 2-CTA Cluster로 q/k 공유** ★ **가장 유력** ★: V_PER_Q=2 → 같은 qk_head를 쓰는 2개 v_head block을 cluster로 묶는다.
  - `__cluster_dims__(2, 1, 1)` 또는 host에서 `cudaLaunchKernelEx` + `cudaLaunchAttribute::clusterDim = {2,1,1}`.
  - Grid 재구성: `(batch × NUM_Q_HEADS × split_factor, 2, 1)`.
  - q, k를 distributed SMEM에 cluster 내 **1회만** 로드 후 `cluster.mapa.shared::cluster`로 상대 block 접근 → bf16 read **절반**.
  - 효과 예상: q/k load 오버헤드가 전체의 ~10%라 가정 시 5% (0.0005 ms) 수준. 단, cluster sync 오버헤드 상쇄 필요.
- [ ] **B2. `cuda::pipeline<thread_scope_block, 3>` + `cuda::memcpy_async` state row prefetch**: 현재 register-based 4-row pipeline (`pf_a,b,c,d`)을 SMEM 3-stage async pipeline으로 전환.
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
- [ ] **C2. `__reduce_add_sync()` intrinsic으로 warp reduction 대체**: 현재 5-stage `__shfl_down_sync` butterfly. SM100 native `redux.sync.add.f32` 1-instruction. `-arch=sm_100a` 필수. PTX `redux.sync` 생성 확인.
- [ ] **C3. 명시적 FMA (`__fmaf_rn()`)**: 현재 dot product / state update가 nvcc FMA 자동 생성 의존. PTX 생성 물(FFMA vs FADD+FMUL) 검증 후 미생성 구간 명시적 호출.
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
- [ ] **D5. CUDA Graph capture (host-side)** ★ **Phase 4 최대 돌파구 후보** ★: `cudaStreamBeginCapture/EndCapture`로 decode call 그래프화. launch overhead 2~5 μs → ~60 ns/node. 10μs 타겟에서 **결정적 효과**.
  - 단계: (1) `scripts/run_modal.py`에서 bench 반복 루프 구조 확인 → graph capture/replay 허용 여부 판단. (2) 허용 시 host wrapper에 `cudaGraph_t`, `cudaGraphExec_t` 추가. (3) 첫 호출은 capture, 이후는 `cudaGraphLaunch`.
  - 제약: 입력 텐서 주소가 매번 동일해야 함 (또는 `cudaGraphExecUpdate` 사용).
  - Framework 제약으로 불가한 경우 D6 대체.
- [ ] **D6. Persistent kernel + cooperative groups**: `cudaLaunchCooperativeKernel`. 단일 launch에서 모든 (batch, head, split) work를 grid-stride로 처리. Launch overhead 완전 제거. 구현 난이도 상.
- [ ] **D7. Stream priority 최고**: `cudaStreamCreateWithPriority`. Modal 환경에 concurrent work 있을 때 효과.
- [ ] **D8. Grid 크기를 B200 SM 수(≈148) 배수에 맞춤**: tail effect 감소. B=1, split=8 → 64 blocks (SM 43% util). Split 16으로 확대 시 128 blocks (87%). Split 확대가 RPW, 레지스터 압박에 미치는 영향 동시 확인. 단, iter #1에서 B≥32 grid만 늘리는 단순 split 확대는 benchmark 후퇴.

---

### E. 공유 메모리 / 레지스터 재분배

- [ ] **E1. q, k를 SMEM에 로드 후 block 내 broadcast**: 현재 128 lane이 독립 로드. Block 내 모든 warp가 동일 q/k 사용하므로 SMEM 1회 로드 + broadcast. Register 절약 → occupancy 향상 여지. SMEM bank broadcast는 conflict-free.
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
- [ ] **G5. `-Xptxas -v -Xptxas -warn-spills` 로 register/spill 분석**: Iter 18 (128, 10) 실패 원인. 현재 (128, 9) 하에서 정확한 register/spill 수치 확인 후 G7 조합 기준선.
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
- [ ] **H4. `__pipeline_memcpy_async` 저수준**: high-level API가 LDGSTS 생성 보장 약할 때 PTX 수준 primitive.
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

Split factor: B≤2:8 (RPW=4), B<32:4 (RPW=8), B≥32:2 (RPW=16)
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
  B=4,  split=4: grid=128  → 86%
  B=16, split=4: grid=512  → balanced
  B=32, split=2: grid=512  → balanced
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

**핵심 인사이트 (Iter 20)**: 32 lanes 동시 exp/log1p → SFU throughput 심각 경쟁. Lane 0 전담 + 3 shuffle로 SFU 경쟁 완전 제거 = Phase 2 break-through.

**R1 인사이트**: B=64의 `waves/SM` 부족을 단순 split 증가로 해결하려 했지만 B=64 latency가 기존 `0.021~0.022 ms`에서 `0.024~0.029 ms`로 악화. grid 증가만으로는 부족하고 q/k/v/gate 중복 제거 또는 launch overhead 제거가 필요.

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
- **Phase 4 돌파 추천 순서**: D5 (CUDA Graph, **host만 수정**) → B1 (2-CTA cluster) → B2/H2 (async pipeline) → B5 (warp specialization). D5가 가장 레버리지 높음.
- **10회 이상 답보**: `ncu --set full` + `cuobjdump --dump-sass` 로 병목 재확인 (`smsp__inst_executed_pipe_*`, `smsp__warp_issue_stalled_*`, SASS 내 FFMA/LDG/SHFL 비율). 사용자에게 결과 공유.
- `modal run` 컴파일 에러 즉시 해결. `-Xptxas -v` 출력은 매 변경 후 확인 (register/spill 추적).
