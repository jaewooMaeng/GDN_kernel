# Iteration #2 (2026-04-24 session) 계획

## Step 1: 현재 상황 정리

### Iteration #1 결과 요약
- **이전 기준선**: 0.012920 ms (Phase 3 median of 5)
- **Iteration #1 개선**: avg latency 0.011415 ms (**+11.6% 개선**)
- **목표**: 0.009 ms (Phase 4)
- **남은 거리**: 0.002415 ms (**≈ 20% 추가 개선 필요**)

### Iteration #1 시도 결과 (A3: `__ldg()` 명시)
| 항목 | 수치 | 평가 |
|------|------|------|
| avg latency | 0.011415 ms | ✅ |
| NCU Kernel Duration | 33.31 us | 개선 미미 |
| Issue Slots Busy | 17.33% | ⚠️ 여전히 매우 낮음 |
| Achieved Occupancy | 27.34% | ⚠️ register pressure (56 reg/thread) 제한 |
| L1 Hit | 5.37% | ⚠️ cache 효과 미미 (single-pass streaming 특성) |

**교훈**: `__ldg()` 추가는 단편적 개선일 뿐, **근본 병목(low-issue-slots + register-pressure)을 해결하지 못함**.

---

### 현재 병목 분석 (NCU + kernel 특성 기반)

#### 병목 1: 극도로 낮은 Issue Slot 활용 (17.33%)
```
원인: Per-warp ILP 부족
- State read (global L2 hit, ~100 cycles latency)
- Q/K reduction (warp-wide reduction, ~50 cycles 대기)
- 이들이 순차적으로 발생하면서 warp은 대부분 stall 상태

해결 방향:
  a) Warp specialization: 1 warp load 전담 → 3 warp compute 병렬 → ILP ↑
  b) Double-buffering + async pipeline: 다음 iteration 데이터 미리 fetch → overlap
```

#### 병목 2: 낮은 Achieved Occupancy (27.34%)
```
원인: Register pressure (56 reg/thread)
- 매 loop iteration에서 q[], k[], s_v[128], state[128]을 유지
- Registers: 28+4 (for loop state) = ~56 reg/thread

해결 방향:
  a) Warp specialization + setmaxnreg: producer 64 reg, consumer 48 reg
  b) Shared memory staging: register spilling 대신 SMEM 사용
```

#### 병목 3: 작은 Grid (1024 blocks / 148 SMs = 6.9 blocks/SM)
```
원인: B=64 workload는 split_factor=2 (V_head 2개 split)
- 1024 blocks / 148 SMs = 정체 위험, bytes-in-flight 부족

해결 방향:
  a) Split factor 증가 (split=4, 8): blocks ↑ → grid parallelism ↑
  b) CUDA Graph로 host launch overhead 제거 (일부 보상)
```

#### 병목 4: L2 cache hit는 높지만 overall bandwidth 부족
```
근거: State read는 L2 warm 상태 (모든 batch에서 state ⊆ 126MB L2)
      하지만 bytes-in-flight는 여전히 부족
      
주요 이유:
  - L2-HBM 재주입이 느림 (한 번의 global read → ~100 cycles 대기)
  - Warp 4개가 동시에 다양한 위치에서 read → L2 port contention
```

---

## Step 2: 다음 후보 검토 및 선택

### 검토 대상: 4가지 방향

#### **후보 B5: Warp Specialization (producer/consumer + mbarrier)**

**구성**:
```cuda
// 4 warps 구성
// Warp 0: Async load 전담 (producer)
//   - State read, Q/K read를 cuda::memcpy_async로 dispatch
//   - mbarrier로 consumer 대기, setmaxnreg.inc로 register free
//
// Warp 1-3: Compute 전담 (consumer)
//   - mbarrier로 producer 대기
//   - setmaxnreg.dec로 더 많은 register 확보 (48~52/thread)
//   - Q/K reduction, S_V accumulation 수행
```

**근거**:
- **Issue Slot 병목 직접 해결**: Warp 0이 load를 독점하므로 warp 1-3은 compute에만 집중 → ILP ↑
- **Register pressure 해소**: setmaxnreg.dec로 warp 1-3이 48~52 reg/thread 사용 가능 → occupancy 30~35% ↑
- **Blackwell 지원**: sm_100a는 `__mbarrier_init_parity`, `setmaxnreg.inc/dec` PTX 지원 ✅
- **Bytes-in-flight 증가**: Async load가 독점 → HBM→L2 대역폭 포화 가능

**예상 결과**:
| 지표 | 이전 | 기대 |
|------|------|------|
| Median Latency | 0.011415 ms | 0.010~0.010.5 ms (성공 시) |
| Register/thread | 56 | 48~52 (consumer) / 32~40 (producer) |
| Achieved Occupancy | 27.34% | 32~38% |
| Issue Slots Busy | 17.33% | 25~30% |

**리스크** (높음):
- `mbarrier` 초기화에 grid-level register (`__grid_group`) 필요 → TVM FFI launcher 수정 필요 가능성
- Warp 0 (producer only)의 register spilling 위험 → PTX 검증 필수
- Producer/consumer 스케줄 미스매치 → 이득 상쇄 가능
- **Partial rollback 불가능** (구조 변경 → 전체 커널 재작성)

**실패 시 귀결**:
- `mbarrier` FFI 호환성 문제 → benchmark fail / compile error
- 성능 개선 미미 (producer/consumer 동기화 cost ≈ load 병렬화 이득)
- 평균 latency ≥ 0.012 ms

---

#### **후보 B2: Async Pipeline (3-stage double buffering)**

**구성**:
```cuda
// Stage 0: Preload (첫 iteration에만 q/k/state preload)
// Stage 1: Main loop
//   - memcpy_async로 다음 iteration 데이터 비동기 fetch
//   - 현재 데이터로 compute (reduction 포함)
//   - barrier 대기
// Stage 2: Epilogue
```

**근거**:
- **Overlap load/compute**: 현재 iteration compute 중에 다음 iteration load → ILP ↑
- **Bytes-in-flight 증가**: 최대 3개 iteration 데이터 동시 fetch 가능
- **구조 변경 최소**: 기존 loop를 async pipeline으로 감싸면 됨

**예상 결과**:
| 지표 | 이전 | 기대 |
|------|------|------|
| Median Latency | 0.011415 ms | 0.010.5~0.011 ms |
| Issue Slots Busy | 17.33% | 22~25% |
| Bytes-in-flight | 낮음 | 중간 (3-stage) |

**리스크** (중간):
- **Barrier overhead**: `__syncthreads()` vs `cuda::pipeline::commit()` 비용 비교 필요
- **Shared memory pressure**: 3-stage buffering에 SMEM 추가 사용 (현재 512B → ~1.5KB)
- **Loop unroll + async**: 컴파일러가 aggressive로 unroll → register pressure ↑ 가능
- **이전 실패 경험** (R5): "async pipeline per-thread" 실패 → 지금은 block-level로 수정

**실패 시 귀결**:
- Barrier cost > compute/load overlap 이득 → median ≥ 0.012 ms
- 추가 SMEM 사용 + register spilling → 오히려 성능 악화

---

#### **후보 I1: Split Factor 증가 (작은 grid → 큰 grid)**

**구성**:
```cuda
// 현재: split_factor = 2 (B=64: 8 v_heads × 2 = 16 split, 1024 blocks)
// 변경: split_factor = 4 (B=64: 8 v_heads × 4 = 32 split, 2048 blocks)
// 또는 split_factor = 8 (B=64: 8 v_heads × 8 = 64 split, 4096 blocks)

// 각 split은 128×128 state 중 일부만 accumulate
// host-side thread가 split 결과를 또 다시 reduce
```

**근거**:
- **Grid parallelism 증가**: 1024 → 4096 blocks → 6.9 → 27 blocks/SM → bytes-in-flight ↑
- **Low-issue 병목 부분 해결**: 더 많은 블록이 동시 run → L2 port contention 분산
- **구조 변경 최소**: split factor 상수만 변경

**예상 결과**:
| 지표 | 이전 | 기대 |
|------|------|------|
| Median Latency | 0.011415 ms | 0.011~0.011.3 ms (약간 악화 또는 무변화) |
| Blocks | 1024 | 4096 |
| Blocks/SM | 6.9 | 27.6 |
| Host-side reduce overhead | 낮음 | 중간 (4배 split → 4배 reduce kernel call) |

**리스크** (낮음):
- **Host-side overhead**: 더 많은 split → host reduce kernel 호출 증가 → host overhead ↑
  - 각 split reduce: ~100~200 ns
  - 4× split → 최악 ~800 ns 추가 overhead
  - Kernel 시간(9~10 µs) 대비 8~10% 부담 → 명백한 악화 우려
- **L2 cold miss 증가**: 더 많은 kernel → global state read 재반복 (하지만 state는 L2 warm이므로 미미)
- **Diminishing return**: 이미 낮은 issue slot이 더 나빠질 수 있음

**실패 시 귀결**:
- Host overhead > grid parallelism 이득 → median > 0.012 ms (악화)

---

#### **후보 G6: Memory Layout Optimization (SMEM prefetch pattern 강화)**

**구성**:
```cuda
// 현재: State read는 thread별로 global L2에서 직접 read
// 변경: 
//   - State를 tile 단위로 SMEM으로 prefetch
//   - 또는 warp-group 수준에서 coalesced global read → SMEM staging
//   - 그후 SMEM에서 필요한 데이터만 선택

// 리소스: Current SMEM 512B → 2KB~4KB (state tile 임시)
```

**근거**:
- **Memory coalescing 개선**: Warp-wide coalesced global read → 1~2 transactions vs 각각 128 transactions
- **Register pressure 완화**: State를 SMEM에 stage → loop 내 temp register 감소

**예상 결과**:
| 지표 | 이전 | 기대 |
|------|------|------|
| Global transactions | 높음 | 중간~낮음 |
| Register/thread | 56 | 50~54 |
| Achieved Occupancy | 27.34% | 28~30% |
| Latency | 0.011415 ms | 0.011.2~0.011.4 ms |

**리스크** (중간):
- **SMEM access latency**: Global→L2 (~100 cycles) vs SMEM→register (~30 cycles)는 이점
  - 하지만 SMEM write + read의 왕복 cost 고려 필요
- **Synchronization overhead**: `__syncthreads()` 추가 → 1~2 cycles × 3~5회 = ~10 cycles 부담
- **예상**: 개선 가능성은 낮음 (R5, R2 경험상)

**실패 시 귀결**:
- SMEM overhead > global L2 hit 이점 → latency 무변화 또는 악화

---

### 세 가지 전략 비교

| 후보 | 리스크 | 기대 효과 | 구현 난이도 | Rollback 비용 | 추천도 |
|------|--------|----------|-----------|-------------|--------|
| **B5** (warp spec) | 높음 | 매우 높음 (occupancy + ILP) | 높음 | 불가능 | ⭐⭐⭐ |
| **B2** (async pipeline) | 중간 | 높음 (bytes-in-flight + ILP) | 중간 | 불가능 | ⭐⭐⭐ |
| **I1** (split ↑) | 낮음 | 낮음 (host overhead 우려) | 낮음 | 쉬움 | ⭐ |
| **G6** (SMEM prefetch) | 중간 | 낮음 (cache effect 미미) | 중간 | 불가능 | ⭐ |

---

## Step 2: PM 검토 대화 및 결론

### 핵심 질문

#### Q1. "프로파일·근거" — 어떤 후보를 선택하고 왜인가?

**검토 관점**:
1. **Issue Slot Busy (17.33%) 해결 가능성**
   - B5 (warp spec): ✅ Directly → producer 독점 load → consumer ILP ↑
   - B2 (async pipeline): ⚠️ Partial → overlap로 부분 개선
   - I1 (split ↑): ❌ 근본 해결 안 함
   - G6 (SMEM): ❌ 근본 해결 안 함

2. **Occupancy 개선 (27.34% → 32%+ 목표)**
   - B5: ✅ setmaxnreg + register 감소 → occupancy 개선 확실
   - B2: ⚠️ Register spilling 위험 → occupancy 악화 가능
   - I1: ❌ 영향 없음
   - G6: ⚠️ SMEM overhead > 이득

3. **Blackwell 하드웨어 활용**
   - B5: ✅ mbarrier, setmaxnreg, distributed SMEM 활용
   - B2: ✅ memcpy_async + cuda::pipeline (libcu++지원)
   - I1: ⚠️ 일반적 기법
   - G6: ⚠️ 일반적 기법

**결론**: **B5 (Warp Specialization)가 가장 강한 근거를 가짐.**
- Issue slot 병목과 occupancy 병목을 동시에 해결
- Blackwell 하드웨어를 직접 활용
- 0.011 ms → 0.010.5 ms 이상 개선 기대 (20% 개선 목표 중 50% 이상)

---

#### Q2. "한 iteration에 과한가?" — 구현 일정과 리스크 규모

**B5 구현 범위**:
```
1. mbarrier init (grid-level) — TVM FFI 호환성 확인 필수
2. Warp 0: cuda::memcpy_async 기반 producer loop
3. Warp 1-3: setmaxnreg.dec + consumer loop
4. Synchronization: mbarrier.arrive/wait + cluster barrier (선택사항)
5. 전체 커널 재작성 (백업 필수)
6. PTX 검증 → SASS 확인
```

**일정 추정**:
- 구현: 3~4시간 (새로운 기법 학습 + PTX 디버깅)
- 빌드 + 벤치: 30분
- 총 4~5시간

**현실성**:
- 한 iteration으로 가능하나, **FFI 호환성이 blocker가 될 수 있음**
- 최악: TVM launcher가 `cudaLaunchCooperativeKernel` 미지원 → B5 불가능
- 이 경우 **B2로 fallback** 필요

---

#### Q3. "회귀 리스크" — B5 변경으로 다른 batch에서 문제?

**안전성 분석**:
1. **Mbarrier + setmaxnreg는 kernel-launch 레벨 설정**
   - 모든 kernel instance(각 batch)에 동일하게 적용
   - Register allocation은 컴파일 타임에 고정 → runtime 동작 예측 가능

2. **Warp specialization의 일관성**
   - Warp 0-3의 역할은 kernel 컴파일에 고정
   - Warp ID는 blockIdx.x, threadIdx.x로 결정 → 모든 batch에 동일

3. **이전 실패 경험 (R8: cluster)**
   - Cluster barrier가 overhead 추가 → median 악화
   - B5는 **mbarrier**, 즉 **hardware barrier** → 비용 더 낮음

**우려 사항**:
- Producer (warp 0)의 register spilling 가능성
  - 만약 async memcpy + register allocation이 40 regs 넘으면 spilling 발생
  - 이 경우 producer throughput ↓ → consumer stall ↑
- **대책**: PTX 검증에서 register/spill 수치 명시적 확인

---

### PM 최종 판정

#### 권장: **B5 (Warp Specialization) + B2 (Async Pipeline) 중 순선택**

**선택 기준**:
1. **B5 우선**: Issue slot + occupancy 병목을 동시 해결 → 0.010.5 ms 기대
   - FFI 호환성 확인 후 진행
   - 실패 시 → B2로 전환 (fallback plan)

2. **B2 대체안**: FFI 문제 발생 시 즉시 시작
   - 구현이 더 간단 (기존 loop를 async pipeline으로 감싸면 됨)
   - 예상 개선: 0.011 ms → 0.010.5~0.011 ms (B5보다 약하지만 확실)

3. **I1/G6 제외**: Risk (host overhead / SMEM overhead)가 return (grid parallelism / minor register 절감) 상회

---

#### **APPROVED** 방향

**최종 결정**: **Iteration #2에서 B5 (Warp Specialization) 시도**

**실행 계획**:
1. **Phase 1**: TVM FFI launcher 검토
   - `cudaLaunchCooperativeKernel` 지원 확인
   - mbarrier init을 grid-level에서 호출 가능한지 확인
   - 불가능 → B2로 즉시 전환

2. **Phase 2**: Kernel 구현 (B5)
   - Warp 0: `cuda::memcpy_async` producer
   - Warp 1-3: `setmaxnreg.dec` consumer
   - mbarrier + synchronization

3. **Phase 3**: PTX/SASS 검증
   - Register spilling 확인 (goal: producer ≤ 40, consumer ≤ 52)
   - Issue slot 변화 추적

4. **Phase 4**: Benchmark + NCU profiling
   - 5회 median 측정 (Phase 4 기준)
   - 목표: 0.010.5 ms 이상, correctness 100%

---

#### **대안 계획** (B5 FFI 호환성 실패 시)

**B2 (Async Pipeline) 즉시 전환**:
```
1. Loop 3-stage로 구조화 (preload, main, epilogue)
2. cuda::memcpy_async + cuda::pipeline + barrier
3. Double-buffering: state[2], q[2], k[2] SMEM (또는 register)
4. 컴파일 + 벤치 (4시간 내 완료)
5. 기대: median 0.010.5~0.011 ms (B5보다 약하지만 acceptable)
```

---

#### 성공 조건 및 다음 단계

**Iteration #2 성공 = Median ≤ 0.0105 ms (B5) 또는 ≤ 0.0110 ms (B2)**

**다음 iteration (Iteration #3)에서는**:
- B5 성공 시: **B2 적용** (async pipeline 추가) → 0.009.5 ms 도전
- B5/B2 모두 미달 시: **CUDA Graph** (host launch overhead 제거) + **Cluster communication** 재검토

---

## 최종 결론

| 항목 | 내용 |
|------|------|
| **선택 방향** | B5 (Warp Specialization: producer/consumer + mbarrier + setmaxnreg) |
| **기대 성능** | 0.010.5 ms (20% 추가 개선 목표) |
| **리스크 평가** | 중간 (FFI 호환성 확인 필수, fallback B2 준비) |
| **구현 범위** | 커널 전체 재구성, PTX 검증 필수 |
| **일정** | 4~5시간 (FFI 호환성 확인까지) |
| **Status** | **APPROVED** — Iteration #2 진행 승인 |

---

**본 계획은 다음을 근거로 수립되었습니다:**
- 병목 분석: Issue Slot (17.33%) + Occupancy (27.34%) + Register Pressure (56)
- 하드웨어 지원: Blackwell sm_100a의 mbarrier, setmaxnreg, distributed SMEM
- 이전 경험: R1~R10 반복에서 "단편적 개선" → 근본 구조 변경 필요 확인
- 시간 효율: Phase 4 목표 (0.009 ms)까지 2~3 iteration 예상, B5가 최적 경로

