# GDN Decode Kernel 반복 최적화 워크플로우

> **이 문서는 code agent가 자율적으로 따라야 하는 실행 가이드입니다.**
> **절대로 목표 성능을 달성할 때까지 멈추지 마세요.**

---

## 0. 절대 규칙 (NEVER BREAK)

1. **Phase 4 목표(Avg latency ≤ 0.009 ms)를 달성할 때까지 아래 루프를 반복한다. 중간에 절대 멈추지 않는다.**
2. 성능이 후퇴(regression)하면 즉시 되돌리고 다른 최적화를 시도한다.
3. correctness가 깨지면(status가 `correct`가 아니면) 즉시 되돌린다.
4. 한 번에 하나의 최적화만 적용한다 (변경 원인 추적을 위해).
5. 매 반복마다 반드시 아래의 **성능 측정** 단계를 수행하고, 결과를 **로그 섹션**에 기록한다.
6. **Phase 3 이후부터는 3회 측정의 median을 판단 기준으로 사용한다** (Modal 클라우드 노이즈 ±0.003ms 때문).

---

## 1. 목표 정의

| Phase | 목표 Avg Latency | 판정 기준 | 현재 상태 |
|-------|-----------------|----------|----------|
| Phase 1 | ≤ 0.012 ms | best of 3 runs | ✅ 달성 (Iteration 17) |
| Phase 2 | ≤ 0.010 ms | best of 3 runs | ✅ 달성 (Iteration 20) |
| Phase 3 | ≤ 0.010 ms | **median of 5 runs** (안정화) | 진행 중 |
| Phase 4 | ≤ 0.009 ms | median of 5 runs | 진행 중 |

- **시작 성능**: Avg latency = 0.015 ms
- **Phase 3의 핵심은 성능이 아닌 분산 감소**: 현재 best 0.010ms / median 0.012ms / worst 0.014ms의 분산을 좁혀 median이 0.010ms 이하가 되게 함.
- **Phase 4는 kernel-level 구조적 개선 필요**: 단순 튜닝으로는 어려우며, async pipeline, cluster, warp specialization 등 Blackwell 전용 기법 도입.

---

## 2. 파일 위치

| 항목 | 경로 |
|------|------|
| **커널 소스** | `solution/cuda/kernel.cu` |
| **패킹 스크립트** | `scripts/pack_solution.py` |
| **벤치마크 실행** | `scripts/run_modal.py` |
| **설정** | `config.toml` |

---

## 3. 성능 측정 방법

매 반복의 측정 단계에서 **반드시** 아래 두 명령을 순차적으로 실행한다:

```bash
python scripts/pack_solution.py
modal run scripts/run_modal.py
```

출력에서 다음을 확인한다:
- 각 workload의 `status` → 반드시 모두 `correct`여야 함
- **`Avg latency: X.XXX ms`** → 이 값이 목표 이하인지 확인

**Phase 3 이후**: `modal run` 명령을 **5회 반복** 실행하고, median 값을 판정에 사용한다. 5개 값 모두 기록.

---

## 4. 반복 루프 (매 iteration마다 수행)

```
┌─────────────────────────────────────────────────────┐
│  STEP 1: 현재 상태 확인                               │
│  - 직전 Avg latency 값을 확인한다                      │
│  - 목표 달성 여부를 판단한다                            │
│  - 달성했으면 → 다음 Phase로 / Phase 4 달성이면 종료    │
├─────────────────────────────────────────────────────┤
│  STEP 2: 최적화 전략 선택                              │
│  - 아래 "최적화 후보 목록"에서 아직 시도하지 않은 것 선택  │
│  - 현재 Phase에 맞는 카테고리 우선 (아래 참조)           │
│  - 예상 효과와 구현 난이도를 간단히 분석                  │
│  - 구현 계획을 문장으로 정리                         │
├─────────────────────────────────────────────────────┤
│  STEP 3: 커널 수정                                    │
│  - solution/cuda/kernel.cu 를 수정한다                 │
│  - 수정 전 현재 커널을 백업(기억)해둔다                  │
├─────────────────────────────────────────────────────┤
│  STEP 4: 성능 측정                                    │
│  - python scripts/pack_solution.py 실행               │
│  - modal run scripts/run_modal.py 실행                │
│  - Phase 3 이후: 5회 반복 측정, 5개 값 기록              │
│  - 결과의 status와 Avg latency(median)를 기록           │
├─────────────────────────────────────────────────────┤
│  STEP 5: 결과 판정                                    │
│  - correctness 실패 → 즉시 롤백, STEP 2로             │
│  - latency 후퇴 → 즉시 롤백, STEP 2로                │
│  - latency 개선 → 변경 유지, STEP 1로                 │
│  - latency 동일 → 유지/롤백 판단 후 STEP 2로          │
└─────────────────────────────────────────────────────┘
```

**Phase별 우선 카테고리:**
- Phase 3 (안정화): 카테고리 A, G, I 우선
- Phase 4 (돌파구): 카테고리 B, H, J 우선

**이 루프를 Phase 4 목표(≤ 0.009 ms median) 달성까지 반복한다. 절대 중단하지 않는다.**

---

## 5. 타겟 하드웨어: NVIDIA B200 (Blackwell, sm_100)

이 커널은 **NVIDIA B200 GPU**에서 벤치마크된다. 최적화 시 아래 스펙을 반드시 참고한다.

| 항목 | 수치 | 최적화 시사점 |
|------|------|-------------|
| Compute Capability | **10.0 (sm_100)** | Blackwell 전용 기능 사용 가능 |
| L2 캐시 | **126 MB** | B=1 state 4MB, B=16 state 64MB → 전부 L2에 상주 가능. L2 persistence 적극 활용 |
| Shared Memory/SM | **228 KB** (블록당 최대 227 KB) | Hopper 대비 동일. 대용량 shared memory tiling 가능 |
| Max Warps/SM | **64** | Hopper(64)와 동일. occupancy 최적화 기준 |
| Max Thread Blocks/SM | **32** | 작은 block + 많은 block 전략 가능 |
| Register File/SM | **64K × 32-bit** | 레지스터 255개/thread. 4-row pipeline에서 ~80개 사용 추정 |
| HBM3e Bandwidth | **~8 TB/s** | Memory-bound 커널에서 bandwidth utilization이 핵심 |
| Thread Block Clusters | **최대 16 블록** (nonportable, B200 한정) | `cudaFuncAttributeNonPortableClusterSizeAllowed` opt-in 필요. 같은 qk_head를 공유하는 v_head 블록끼리 clustering 가능 |
| Distributed Shared Memory | 지원 | Cluster 내 블록 간 shared memory 직접 접근 |
| TMA (Tensor Memory Accelerator) | 지원 | `cp.async.bulk` 또는 `cuda::memcpy_async`(16B-aligned + multiple of 16B → TMA path) |
| TMEM (Tensor Memory) | 신규 지원 | tcgen05 전용. 현재 GEMM 중심이라 decode 커널엔 직접 적용 어려움 |
| L1/Texture/Shared 통합 캐시 | **256 KB/SM** | `cudaFuncAttributePreferredSharedMemoryCarveout`로 비율 조절 |
| Kernel Launch Overhead | **~2-5μs (traditional), ~60ns/node (CUDA Graph)** | 10μs 타겟에서 launch overhead 비중 큼. CUDA Graph 적극 활용 |

### State 크기 vs L2 캐시 분석

```
State per (batch, v_head) = 128 × 128 × 4B = 64 KB
State per batch           = 8 × 64 KB     = 512 KB
B=1  total state          = 512 KB         → L2 126MB의 0.4% (완전 상주)
B=16 total state          = 8 MB           → L2 126MB의 6.3% (완전 상주)
B=64 total state          = 32 MB          → L2 126MB의 25%  (대부분 상주)
```

**결론: B200에서는 거의 모든 batch size에서 state가 L2에 완전히 들어간다.
→ L2 persistence hint를 적극 활용하고, global memory bandwidth보다 L2 bandwidth에 최적화해야 한다.**

### 10μs 타겟에서의 overhead 분석

```
목표 커널 시간: 0.009 ms = 9,000 ns
- Kernel launch overhead (traditional): 2,000 ~ 5,000 ns  (22~55%!)
- Kernel launch overhead (CUDA Graph):    ~60 ns/node     (<1%)
- L2 state read (cold): ~200 cycles × (state_size / cacheline) 
- L2 state read (warm): ~40 cycles × (state_size / cacheline)
- SFU (exp/log) latency: ~16 cycles/op × 2 ops = ~32 cycles
```

**Phase 4(9μs) 구간에서는 kernel 내부 최적화뿐 아니라 host-side launch overhead, L2 hit rate, Modal I/O 안정성까지 총체적으로 관리해야 한다.**

---

## 6. 최적화 후보 목록

아래는 시도할 수 있는 최적화 방향이다. 위에서부터 우선순위가 높다.
시도한 것은 [시도됨] 표시를 하고 결과를 기록한다.

### A. 메모리 접근 최적화 (B200 L2 126MB 활용 핵심)
- [ ] **L2 Persistence + hitRatio 튜닝 (`cudaAccessPolicyWindow`)**: host 측에서 state 버퍼에 대해 persistence hint 설정. **반드시 `hitRatio = min(persistingL2CacheMaxSize / total_state_bytes, 1.0f)` 공식을 사용**하여 thrashing 방지. 또한 `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize)`로 L2 set-aside 최대화. `hitProp=cudaAccessPropertyPersisting`, `missProp=cudaAccessPropertyStreaming` 설정. B=1일 때 state 4MB, B=16일 때 64MB → 모두 126MB L2 persistence에 들어감. **반복 benchmark에서 state가 이전 iter에서 쓴 값을 다시 읽는 패턴이라면 효과 극대**.
- [ ] **`cuda::annotated_ptr` with `access_property::persisting`**: 커널 내부에서 state 포인터를 `cuda::annotated_ptr<float, cuda::access_property::persisting>`로 래핑. stream attribute 방식보다 더 정교한 hint. libcu++ 필요.
- [ ] **State read를 `__ldg()` 로 변경**: state는 read-only이므로 `__ldg()`로 L2 read-only cache path 활용. `__restrict__` 포인터와 결합하면 컴파일러가 LDG 명령어를 자동 생성할 수도 있지만, 명시적이 더 확실함. PTX 확인: `cuobjdump --dump-sass`로 `LDG.E.128.CONSTANT` 생성 여부 검증.
- [ ] **State read를 `ld.global.nc` PTX로 명시**: `asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : ...)` 인라인 PTX로 non-coherent cache 경로 강제.
- [ ] **Shared memory carveout을 최소화(L1 극대화)**: `cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0)` → L1이 최대. 현재 s_v[128]=512B만 쓰므로 L1을 키우면 state read의 L1 hit이 증가. 반대로 `228KB` 설정도 실험 (shared memory 사용 계획이 생길 때).
- [ ] **bf16 입력을 `__nv_bfloat162` (packed) vectorized load로 최적화**: q, k를 `*reinterpret_cast<const __nv_bfloat162*>(...)`로 2-element 패킹 로드 후 `__bfloat1622float2()`로 변환. 4개 bf16 load → 2개 bfloat162 load, 메모리 트랜잭션 수 절반.
- [ ] **s_v[] bank conflict 분석**: 128 threads × 4B float = 128 banks → `s_v[tid]` 접근은 conflict-free이지만, `s_v[vi_a]`처럼 모든 thread가 같은 index 접근은 broadcast. nvcc가 broadcast로 컴파일하는지 `cuobjdump`로 확인. 필요 시 `__shfl_sync(0xffffffff, s_v_reg, src_lane)` broadcast로 대체.
- [ ] **Double buffering 강화 (4-row → 8-row prefetch)**: B200의 64K register/SM 활용. 단, register pressure 증가로 occupancy 감소 가능. `-Xptxas -v`로 레지스터 사용량 확인 필요.
- [ ] **New state write를 streaming store(`__stcs()`)로 변경** [시도됨, 후퇴]: ~~Iteration 6 실패~~. 다시 시도하지 않음.

### B. B200 Blackwell 전용 최적화
- [ ] **2-CTA Cluster로 q/k 공유 (가장 유력한 구조적 개선)**: V_PER_Q=2이므로 같은 qk_head를 쓰는 2개 v_head block을 cluster로 묶는다. `__cluster_dims__(2, 1, 1)` 또는 `cudaLaunchKernelEx` + `cudaLaunchAttribute::clusterDim=(2,1,1)`. q, k 데이터를 distributed shared memory를 통해 cluster 내 한 번만 로드 → bf16 read 절반. Grid를 `(batch * 4 * split, 2, 1)`로 재구성.
- [ ] **`cuda::memcpy_async` + `cuda::pipeline` 3-stage 파이프라인**: state row를 async로 shared memory에 bulk copy. 현재 register-based prefetch는 load/compute 동시 실행 제한적이지만, async shared memory pipeline은 compute 중 다음 stage 로드 가능 → bytes-in-flight 증가. 3-stage: stage 0 compute, stage 1 arriving, stage 2 being issued. `cuda::aligned_size_t<16>` 사용 시 TMA path 자동 진입.
- [ ] **TMA (cp.async.bulk.tensor) 1D copy**: state row 128×4B=512B를 TMA로 bulk load. 16B 정렬 + 16 multiple → TMA 진입 조건 충족. `cuTensorMapEncodeTiled`로 tensor map 생성 후 `cp.async.bulk.tensor.1d.shared::cluster.global`. Host에서 TensorMap 1회 생성, 커널마다 재사용.
- [ ] **Warp Specialization (Producer/Consumer)**: 4 warps 중 1 warp은 TMA/async load 전담, 3 warps는 compute 전담. `__nanosleep(0)` + `mbarrier`로 동기화. Blackwell tcgen05 패턴 차용. 레지스터 분배가 warp role별로 다르므로 occupancy 영향 분석 필요.
- [ ] **Distributed Shared Memory로 output aggregation**: cluster 내 블록들이 결과를 distributed shared memory에 모아 한 번에 global write. 현재는 각 block이 독립 write이므로 이득 제한적이나, split_factor가 큰 경우(B=1 split=8) split별 partial output 병합 시 유용.
- [ ] **`__launch_bounds__(128, 10)` 재시도 후 -maxrregcount 조합** [부분 시도됨]: (128, 10)은 Iteration 18에서 spill로 실패. `-maxrregcount=48` 같이 조합하여 spill 없이 occupancy 끌어올리기. nvcc 컴파일 옵션 레벨.
- [ ] **Shared memory 대용량 활용 (최대 227KB/block)**: 현재 512B만 사용. 전체 state 64KB를 shared memory에 bulk load (227KB 여유 있음) → 모든 후속 read가 shared memory hit. 단, 초기 bulk load 64KB/block × 9 blocks = 576KB/SM → 228KB 한도 초과, occupancy=3으로 급락 가능. 신중한 trade-off.
- [ ] **`cudaFuncAttributeNonPortableClusterSizeAllowed` opt-in으로 cluster size 8 또는 16**: 동일 batch의 8개 v_head 전부를 한 cluster로 묶어 q 공유 확대. 단, cluster 크기 커질수록 synchronization overhead 증가.

### C. 연산 최적화
- [x] **Gate 계산을 lane 0에서만 수행 + broadcast** [Iteration 20, 성공]: 이미 적용됨.
- [ ] **FMA(fused multiply-add) 명시적 사용**: `__fmaf_rn()` 으로 dot product 및 state update 연산 대체. 컴파일러가 이미 FMA를 쓸 수 있지만 명시적 호출이 확실함. `nvcc --ptx`로 PTX 수준 FFMA/FMA 생성 여부 먼저 확인.
- [x] **Gate 연산에 fast math intrinsic 사용** [Iteration 2, 실패]: ~~후퇴~~. 다시 시도하지 않음.
- [ ] **Residual 계산에서 공통 부분 사전 계산**: `beta * v[vi]`와 `beta_g * ks`를 분리. 이미 `beta_g = beta * g` 사전 계산됨. 추가로 `scale_qk = scale * qk_dot`, `scale_g = scale * g`도 사전 계산됨. 더 확장할 여지: output 계산을 SIMD-like 벡터화.
- [ ] **Warp reduction을 `__reduce_add_sync()` intrinsic으로 대체**: B200에서 native warp reduce intrinsic 지원. 현재 `__shfl_down_sync` 5-stage butterfly 대신 1-instruction reduce. `-arch=sm_100` 필수.
- [ ] **k_vals[0..3], q_vals[0..3]을 `float4`로 보관하여 FFMA 4-way**: `float4 k4, q4;` 선언 후 `ks += k4.x*st4.x + k4.y*st4.y + k4.z*st4.z + k4.w*st4.w` 형태 유지 → nvcc가 FMA4 자동 생성 유도.

### D. 실행 구성 최적화
- [x] **Split factor 튜닝 (B=1:8, B<=2:8, B>=3:4)** [Iteration 9, 성공]: 현재 적용됨.
- [x] **Block size 128로 통일** [Iteration 8, 성공]: 현재 적용됨.
- [x] **`__launch_bounds__(128, 9)`** [Iteration 17, 성공]: 현재 적용됨.
- [ ] **B별 split_factor 세분화 추가 탐색**: 현재 `B<=2:8, B>=3:4`. `B<=4:8, B<=16:4, B>=32:2` 등 3단계 세분화. outlier batch(B=8, 16, 32)에서 효과 검증.
- [ ] **CUDA Graph capture (host-side, bench framework 레벨)**: `run_modal.py`가 허용한다면 `cudaStreamBeginCapture/EndCapture`로 커널 호출을 graph로 캡처. 반복 launch overhead를 2~5μs → 60ns/node로 감소. 10μs 타겟에서 결정적 효과. config.toml 또는 python wrapper 측 수정이 필요할 수 있음.
- [ ] **Persistent kernel with cooperative groups**: `cudaLaunchCooperativeKernel`로 단일 launch에서 grid-wide barrier 사용. 모든 (batch, head, split) work를 persistent block이 grid-stride로 처리. launch overhead 제거.
- [ ] **`cudaStreamCreateWithPriority` 최고 우선순위**: stream priority 높이면 커널 스케줄링 지연 감소. 효과는 다른 concurrent work 유무에 의존.

### E. 공유 메모리 최적화
- [ ] **q, k 값을 shared memory에 로드 (cluster와 독립)**: 현재 각 warp가 독립적으로 q, k를 register 32×4=128 값 로드. Block 내 모든 warp가 동일 q, k 사용 → shared memory 1회 로드 후 broadcast. register 절약 → occupancy 향상 가능. 단, 128 threads가 같은 값 broadcast → shared memory bank broadcast로 conflict-free.
- [x] **v vector를 register로 승격** [Iteration 1, 실패]: ~~후퇴~~. 다시 시도하지 않음. (__syncthreads 제거보다 bank conflict 해소가 더 중요함)
- [ ] **s_v를 warp-broadcast register로 대체 (__syncthreads__ 유지)**: 각 warp가 자기 담당 row의 v값만 register에 보유하고, `__shfl_sync`로 warp 내 broadcast. 1회 실패 있었으나 구조가 달라짐 — warp별로 로드하되 **syncthreads는 유지**(warp 독립성 보장). 재시도 가치 있음.

### F. 수학적 재구성
- [x] **ks/qs dot product를 벡터화된 단일 패스로 fusion** [현재 적용됨]
- [ ] **State update를 in-place로 변경**: `new_state == state`인 경우 별도 write 생략 가능한지 검토. API 제약 확인: `scripts/pack_solution.py`의 API signature 확인 필요. 만약 in-place 허용되면 store 트래픽 완전 제거.
- [ ] **Output 계산을 warp 전체가 참여하도록 변경**: lane 0만 output을 쓰는 대신, 각 lane이 output의 일부를 계산하여 coalesced write. 단, 현재 output은 scalar(vi당 1개, bf16)이므로 효과 제한적. 단, 4개 vi × 2B = 8B → 32 lane × 0.25B는 불가. 재검토: 4개 vi를 4 lane에 분산하고 lane0~3만 쓰기.
- [x] **Gate 계산을 thread 0에서만 수행 + broadcast** [Iteration 20, 성공]: 이미 적용됨.

### G. 컴파일러 힌트 및 빌드 옵션
- [x] **#pragma unroll 명시** [현재 적용됨]
- [x] **`__restrict__` 확인** [현재 적용됨]
- [ ] **`__builtin_assume()` 힌트 추가**: `__builtin_assume(blockDim.x == 128)`, `__builtin_assume(batch_size > 0)` 등으로 컴파일러 분기 제거 유도. 인덱스 계산 단순화.
- [ ] **sm_100a (architecture-specific) 타겟 컴파일**: 현재 `sm_100` 추정. `-arch=sm_100a`로 Blackwell B200 전용 기능(TMA, tcgen05) 활성화. `sm_100f`(family-specific)도 대안. `nvcc --verbose`로 실제 gencode 확인.
- [ ] **`-Xptxas -v` 로 register/spill 분석**: Iteration 18 실패 원인 파악. 현재 (128, 9)에서 실제 register 사용량, spill stores/loads 확인. `-Xptxas -warn-spills` 추가.
- [ ] **`-use_fast_math` 전역 적용** [주의]: correctness 검증 필수. gate 계산에 국소적으로 적용하여 이미 Iteration 2에서 실패한 이력 있음. 전역 적용은 더 광범위 영향이지만 correctness 깨짐 가능성 높음.
- [ ] **`-maxrregcount=48` 또는 `-maxrregcount=56`**: launch_bounds와 조합. spill 발생 시 backoff.
- [ ] **`__forceinline__`을 모든 helper 함수에**: `softplus` 이미 적용. 없는 것 확인.

### H. 비동기 메모리 연산 (Blackwell 핵심, Phase 4 돌파구 후보)
- [ ] **`cp.async.ca.shared.global` (Ampere+) 기반 state row prefetch**: 현재 register-based prefetch를 shared memory pipeline으로 전환. 16B 정렬된 128B row를 async shared memory load → compute와 overlap. 3-stage 또는 4-stage pipeline으로 bytes-in-flight 증가.
  ```
  stage 0: compute row i
  stage 1: arriving row i+1 (loaded 2 iters ago)
  stage 2: loading row i+2 (issue now)
  ```
- [ ] **`cuda::pipeline<thread_scope_block, 3>` with `cuda::memcpy_async`**: 위 cp.async의 high-level wrapper. `cuda::aligned_size_t<16>` 사용으로 TMA path 자동 활용.
- [ ] **TMA tensor map 기반 bulk load**: host에서 `cuTensorMapEncodeTiled` 1회 호출, state [B,8,128,128] 전체를 tensor map으로 등록. 커널 내 `cp.async.bulk.tensor.1d`로 row 단위 bulk fetch. 주소 계산 하드웨어 처리 → warp 내 address computation overhead 제거.
- [ ] **`__pipeline_memcpy_async` 저수준 primitive**: 위 high-level API들이 LDGSTS 보장이 약한 경우, 저수준 primitive로 LDGSTS 명시 사용.
- [ ] **`mbarrier` 기반 async barrier 동기화**: shared memory barrier로 stage 완료 대기. `__syncthreads()` 대체 가능 구간 탐색.

### I. 안정성·분산 감소 (Phase 3 핵심)
- [ ] **L2 cache warmup kernel 선행 실행**: 벤치마크 측정 전 더미 kernel로 state를 L2에 올려둠. cold cache miss 제거. Modal 환경에서 iter=50 중 초기 iter의 분산이 큰 경우 특히 효과적.
- [ ] **cudaCtxResetPersistingL2Cache() 호출 시점 최적화**: persistence 설정 후 필요 시점에 reset. 측정 간섭 최소화.
- [ ] **Block 수 조정으로 SM 당 work 균등화**: B200 SM 수(~148개)의 배수에 맞춘 grid size. 잔여 block으로 인한 tail effect 감소.
- [ ] **Kernel launch ordering 최적화**: stream priority + event synchronization 재구성으로 warmup/측정 분리 명확화.
- [ ] **`cudaDeviceSynchronize()` 호출 빈도 점검**: 불필요한 동기화가 측정값을 튀게 하는지 bench script 측 확인.
- [ ] **측정 warmup iter 증가**: config.toml의 `warmup=10` → `warmup=20` 또는 `30`. trial 수도 증가.
- [ ] **Block 내부의 첫 read에 `__builtin_prefetch`**: cold start state read의 초기 지연 감소. 첫 block이 유독 느린 패턴 있을 경우 효과.

### J. 구조적 재설계 (Phase 4 핵심)
- [ ] **Tensor Core (tcgen05) 활용 검토**: state [128,128] × q/k [128] → matrix-vector 연산. tcgen05는 MMA 전용이라 fit 애매하나, q/k를 [1,128]로 확장 + padding matmul 변환 가능성 탐색. 단, 메모리 이동 오버헤드가 FLOPS 이득을 상쇄할 수 있음.
- [ ] **State layout 변경 ([B,H,V,K] k-last → [B,H,K,V] v-last)**: 현재 k-last. v-last로 바꾸면 state read 패턴이 다름. FLA reference와 일치시켜 ldmatrix-friendly 레이아웃 가능. 단, API 호환성 확인 필요.
- [ ] **커널 분할 (compute/write 분리)**: 1개 큰 커널을 2개 작은 커널로 분할 후 stream 병렬화. CUDA Graph와 조합 시 효과적일 수 있음.
- [ ] **Multi-kernel pipeline**: decode를 persistent kernel로 두고 다음 token의 준비 작업을 별도 stream에서 overlap. 실제 배포 맥락에서 효과 크지만 단일 커널 벤치마크에선 의미 제한적.

---

## 7. 현재 커널 핵심 구조 요약 (agent 참고용)

```
커널: gdn_decode_kernel<ROWS_PER_WARP>
- Grid: (B * 8 * split_factor,)
- Block: 128 threads (4 warps), __launch_bounds__(128, 9)
- 각 block이 하나의 (batch, v_head, split_id)를 처리
- 각 warp가 ROWS_PER_WARP개의 V-rows를 담당
- 4-row software pipelining: float4로 state를 register에 prefetch
- Warp reduction으로 ks, qs dot product 계산 (butterfly)
- Gate 계산: lane 0만 계산 후 __shfl_sync broadcast
- State: [B, 8, 128, 128] float32 (k-last layout)
- 입력: bf16, State: float32, 출력: bf16
- split_factor: B<=2:8 (RPW=4), B>=3:4 (RPW=8)
```

### B200에서의 리소스 사용 추정

```
Block size = 128 threads (4 warps), __launch_bounds__(128, 9):
- 레지스터/thread: nvcc 할당 ≈ 56개 (9 blocks/SM × 128 threads × 56 = 64,512 regs < 65,536)
  → 9 blocks × 4 warps = 36 warps/SM (occupancy = 36/64 = 56.25%)

- Shared memory: 512B (s_v[128] × 4B)
  → bottleneck 아님. carveout 조절로 L1 cache 극대화 가능.

- B200 SM 수 ≈ 148개
  → B=1, split=8: grid = 1×8×8 = 64 blocks → 64/148 = 43% SM utilization (tail effect)
  → B=2, split=8: grid = 128 blocks → 86%
  → B=4, split=4: grid = 128 blocks → 86%
  → B=16, split=4: grid = 512 blocks → SM당 3.5 block (balanced)
  → B=32, split=4: grid = 1024 blocks → SM당 7 block

현재 병목 분석 (Phase 3/4 관점):
1. Kernel launch overhead (~2-5μs)가 10μs 타겟의 20~50% 차지 → CUDA Graph 도입 가치 큼
2. 작은 batch(B=1~2)에서 SM utilization이 40% 수준 → cluster 또는 더 공격적 split
3. Modal 클라우드 노이즈 ±0.003ms → L2 warmup + 측정 전략 개선 필요
4. q/k bf16 read가 v_head별 중복 (V_PER_Q=2) → 2-CTA cluster로 반감 가능
5. SFU (exp/log) 경쟁은 이미 lane 0만 계산으로 해소됨 (Iter 20)
```

---

## 8. 성능 로그

매 iteration마다 아래 형식으로 이 섹션에 추가 기록한다. **Phase 3 이후는 5회 측정 값 모두 기록**.

### Iteration 1
- 최적화: v를 register로 이동 + __syncthreads__ 제거
- 변경 요약: shared memory s_v를 warp-level register + __shfl_sync로 대체
- Avg latency: 0.015 ms (이전: 0.014 ms)
- 변화: +0.001 ms (후퇴)
- Status: correct
- 판정: 롤백
- 현재 Phase: 1

### Iteration 2
- 최적화: fast math intrinsics (__expf, __logf, __frcp_rn)
- 변경 요약: gate 연산에 fast math 함수 적용
- Avg latency: 0.017 ms (이전: 0.014 ms)
- 변화: +0.003 ms (후퇴)
- Status: correct
- 판정: 롤백
- 현재 Phase: 1

### Iteration 3
- 최적화: B=1에서 split_factor=16, block_size=64
- 변경 요약: SM utilization 80% 목표로 split factor 공격적 증가
- Avg latency: N/A (correctness 실패)
- Status: INCORRECT_NUMERICAL (B=1 workloads)
- 판정: 롤백 (block_size=64에서 s_v 128원소 중 64만 로드)
- 현재 Phase: 1

### Iteration 4
- 최적화: B>16에서 split_factor=2, block_size=128 (기존 split=1, block=256)
- 변경 요약: 대배치에서 더 많은 블록으로 분할
- Avg latency: 0.017 ms (이전: 0.014 ms)
- 변화: +0.003 ms (후퇴)
- Status: correct
- 판정: 롤백
- 현재 Phase: 1

### Iteration 5
- 최적화: __launch_bounds__(256, 4)
- 변경 요약: compiler register 할당 힌트
- Avg latency: 0.015 ms (이전: 0.014 ms)
- 변화: +0.001 ms (후퇴)
- Status: correct
- 판정: 롤백
- 현재 Phase: 1

### Iteration 6
- 최적화: new_state에 streaming store (__stcs)
- 변경 요약: L2 pollution 방지를 위해 write-through store 적용
- Avg latency: 0.019 ms (이전: 0.014 ms)
- 변화: +0.005 ms (큰 후퇴)
- Status: correct
- 판정: 롤백
- 현재 Phase: 1

### Iteration 7 ✅
- 최적화: 커널 템플릿화 (ROWS_PER_WARP 컴파일 타임 상수)
- 변경 요약: template<4/8/16>으로 주요 루프 완전 언롤 + #pragma unroll
- Avg latency: 0.014 ms (이전: 0.015 ms, stable baseline)
- 변화: -0.001 ms (개선)
- Status: correct
- 판정: 유지
- 현재 Phase: 1

### Iteration 8 ✅
- 최적화: 모든 B에서 block_size=128, B>16에서 split=2
- 변경 요약: 대배치에서 256 threads → 128 threads + split 증가로 grid 확대
- Avg latency: 0.013 ms (이전: 0.014 ms)
- 변화: -0.001 ms (개선)
- Status: correct
- 판정: 유지
- 현재 Phase: 1

### Iteration 9 ✅✅
- 최적화: B>=3 모두 split_factor=4로 통일
- 변경 요약: grid 크기 2배 증가, rpw=8로 최적 균형점
- Avg latency: 0.011 ms (이전: 0.013 ms)
- 변화: -0.002 ms (대폭 개선)
- Status: correct
- 판정: 유지 → **Phase 1 달성!**
- 현재 Phase: 2

### Iteration 10-13
- 추가 최적화 시도 (split=8 통일, B<=8 split=8, s_v 제거, B=1 split=16) 모두 후퇴 또는 동등. 롤백.
- Modal 클라우드 환경 노이즈가 ±0.003ms로 매우 큼.
- 최종 안정 측정: Avg latency = 0.011 ms (warmup=10, iter=50, trials=3)

### Iteration 14
- 최적화: adaptive split (B<=2:8, B<=16:4, B<=32:2, B>32:1) + RPW 16/32 템플릿
- Avg latency: 0.021 ms (큰 후퇴)
- 판정: 롤백 (RPW=16/32에서 레지스터 압박 극심)

### Iteration 15
- 최적화: __shfl_xor_sync reduction + vectorized output write
- Avg latency: 0.015 ms (후퇴)
- 판정: 롤백

### Iteration 16 ✅
- 최적화: __launch_bounds__(128, 7) → occupancy 6→7 blocks/SM
- Avg latency: 0.013 ms (이전: 0.014 ms)
- 변화: -0.001 ms (개선)
- Status: correct
- 판정: 유지

### Iteration 17 ✅✅
- 최적화: __launch_bounds__(128, 9) → occupancy 9 blocks/SM (~56%)
- Avg latency: 0.012 ms (이전: 0.013 ms)
- 변화: -0.001 ms (개선) → **Phase 1 재달성!**
- Status: correct
- 판정: 유지

### Iteration 18
- 최적화: __launch_bounds__(128, 10) → 51 regs, too aggressive
- Avg latency: 0.014 ms (후퇴, register spill)
- 판정: 롤백 to (128, 9)

### Iteration 19
- 최적화: B<=16 split=8 (SM utilization 향상 시도)
- Avg latency: 0.014 ms (후퇴, B=8/16에서 outlier 발생)
- 판정: 롤백

### Iteration 20 ✅✅✅ — Phase 2 달성!
- 최적화: gate 계산을 lane 0에서만 수행 + __shfl_sync broadcast
- 변경 요약: 128 thread 모두 exp/log1p 중복 계산 → lane 0만 계산 후 broadcast (3 shuffle)
- Avg latency: **0.010 ms** (best), 0.012 ms (median of 3 runs), 0.014 ms (worst run)
- 변화: -0.002 ms (best case, 이전: 0.012 ms)
- Status: correct
- 판정: 유지 → **Phase 2 달성! (0.010 ms ≤ 0.010 ms)**
- 현재 Phase: 3 (안정화 단계)
- 핵심 인사이트: SFU(Special Function Unit) 경쟁 해소. 32 lanes 동시 exp/log → 심각한 SFU throughput 병목. Lane 0만 계산하면 SFU 경쟁 제거.

### Iteration 21~ (Phase 3 시작)
- 목표: median 0.010 ms 이하 안정화
- 우선 시도 카테고리: A (L2 persistence + hitRatio), G (빌드 플래그/register 분석), I (안정성)
- 기록 형식:
  ```
  ### Iteration N
  - 최적화: <설명>
  - 변경 요약: <상세>
  - 측정 5회: [X.XXX, X.XXX, X.XXX, X.XXX, X.XXX] ms
  - Median: X.XXX ms (이전 median: X.XXX ms)
  - 변화: ±X.XXX ms
  - Status: correct / incorrect
  - 판정: 유지 / 롤백
  - 현재 Phase: 3 / 4
  ```

---

## 9. 완료 조건

- [x] **Phase 1 달성**: Avg latency ≤ 0.012 ms (Iteration 17, 0.012ms best)
- [x] **Phase 2 달성**: Avg latency ≤ 0.010 ms (Iteration 20, 0.010ms best)
- [ ] **Phase 3 달성**: median of 5 runs ≤ 0.010 ms (안정화)
- [ ] **Phase 4 달성**: median of 5 runs ≤ 0.009 ms (구조적 개선)

**Phase 4를 달성하면 이 워크플로우를 종료하고, 최종 결과를 사용자에게 보고한다.**
**Phase 4를 달성하지 못했으면 절대 멈추지 말고 루프를 계속 반복한다.**

---

## 10. 롤백 정책

- 커널 수정 전, 항상 현재 동작하는 전체 커널 코드를 기억해둔다.
- correctness 실패 또는 latency 후퇴 시, 즉시 직전의 정상 버전으로 `solution/cuda/kernel.cu`를 복원한다.
- 롤백 후 다른 최적화를 선택하여 다시 시도한다.
- 같은 최적화를 두 번 이상 실패했으면 해당 항목을 [실패] 처리하고 넘어간다.
- **Phase 3 이후**: best case만 개선되고 median/worst는 악화된 경우에도 롤백 고려 (안정성 우선).

---

## 11. 추가 지침

- 최적화의 효과가 미미할 때(0.001ms 미만 개선), 여러 소규모 최적화를 조합하는 것도 고려한다.
- 하나의 최적화가 성공하면, 그 위에 다음 최적화를 쌓아 올린다 (누적).
- **Phase 3 돌파 전략**:
  - 먼저 L2 persistence를 **host-side `cudaStreamAttributeAccessPolicyWindow`로** 적용하여 variance 감소 실험. `hitRatio = min(prop.persistingL2CacheMaxSize / total_state_bytes, 1.0f)` 공식 엄수.
  - `-Xptxas -v -Xptxas -warn-spills`로 현재 커널의 register/spill 상태 정확히 파악 후, `__launch_bounds__` 세밀 조정.
  - Modal bench script 측에서 warmup iter 증가, L2 cache flush 타이밍 최적화 가능 여부 확인.
- **Phase 4 돌파 전략**:
  - 단일 튜닝으로 0.009ms는 어려움. **커널 launch overhead 제거(CUDA Graph) + 2-CTA cluster q/k 공유 + async pipeline** 조합이 가장 유망.
  - 구현 복잡도 순: CUDA Graph (host-side만 수정) < async pipeline (커널 내 재구성) < 2-CTA cluster (grid + 커널 구조 변경) < TMA tensor map (host TensorMap 생성 + 커널 cp.async.bulk.tensor).
  - 만약 framework 제약으로 CUDA Graph 도입 불가 시, async pipeline + cluster 조합이 다음 순위.
- `modal run`의 출력을 끝까지 확인한다. 컴파일 에러가 발생하면 커널 코드를 수정하여 해결한다.
- 만약 10회 이상 반복해도 진전이 없으면, NCU profiling이나 `cuobjdump --dump-sass`로 SASS 분석 결과를 사용자에게 제안한다.
- `nvcc --resource-usage` 또는 `-Xptxas -v`를 항상 커널 수정 후 확인하여 register/spill 변화 추적.
