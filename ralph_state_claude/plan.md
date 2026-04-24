# Iteration #1 (2026-04-24 session) 계획

## Step 1: 현재 상황 정리

### 기준선 및 목표
- **현재 Accepted Baseline**: 0.012920 ms (median of 5 runs, Phase 3)
- **목표**: 0.009 ms 미만 (Phase 4)
- **남은 거리**: 약 0.004 ms (30% 개선)

### 직전 10회 반복의 공통 패턴 (R1~R10)
| 시도 | 분류 | 이유 | 교훈 |
|------|------|------|------|
| R1 | Split 증가 | B≥32 grid 2배 확대 | 중복 q/k load + per-block ILP 감소 > 이득 |
| R2 | Shared staging | q/k/v 공유 + gate 1회 | Barrier overhead > state streaming 절감 |
| R3 | Carveout + reduce | split-local s_v + carveout=0 | `redux.sync.f32` build blocked |
| R4 | CUDA Graph | kernel.cu wrapper 내부 | Graph instantiate overhead > launch 절감 |
| R5 | Async pipeline | per-thread `__pipeline_memcpy_async` | shared reload overhead > overlap 이득 |
| R6 | Large-batch kernel | 256-thread / 8-warp path | qk reduction 중복 > per-warp ILP 이득 |
| R7 | PTX redux / hint | inline `redux.sync.f32` / persisting | PTX 미지원 + cache hint 부작용 |
| R8 | 2-CTA cluster | cluster q/k 공유 (rank 0 DSMEM) | Cluster barrier + remote shared > q/k load 절감 |
| R9 | E1 재변형 | warp0 q/k + barrier 재활용 | shared fan-out cost > 절감 |
| R10 | FFMA+barrier | `float4` q/k + `__fmaf_rn` + scalar dedup | Source 최적화 < low-issue 병목 |

### 현재 병목 분석 (NCU profile 기준)
```
문제 1: Small grid (1024 blocks / 148 SMs = 0.77 waves/SM)
  → Insufficient parallelism & bytes-in-flight

문제 2: Low issue slots busy (17.6% avg)
  → Per-warp ILP 부족 또는 load/compute mismatch

문제 3: Low achieved occupancy (28-31%)
  → Register pressure (56 regs/thread) 또는 구조적 한계

문제 4: Poor cache hit (L1: 5.37%, L2: 1.49%)
  → Single-pass streaming 특성 (state는 L2 warm일 가능성 높음)
```

### 실패 원인 정리
- **타겟 미스**: 모든 시도가 "단편적 개선" 추구 → Kernel의 근본 병목(low-parallelism + low-bytes-in-flight + register-limited occupancy)을 동시에 움직이지 못함
- **trade-off 패턴**: q/k load 절감 < qk reduction 중복 + cluster/barrier overhead
- **현재 approach의 한계**: Register pressure (56) 또한 baseline을 넘지 못하는 이유

---

## Step 2: 다음 후보 검토 및 선택

### 검토 대상: 3가지 방향

#### 방향 A: G5 + A3 결합 (저위험, 구조 변경 없음)
**구성**: 
1. **G5**: `-Xptxas -v -warn-spills`로 현재 kernel의 정확한 register/spill/SASS mix 확보
2. **A3**: state read를 `__ldg()` 명시 또는 `ld.global.nc.v4.f32` inline PTX로 read-only path 강제

**근거**:
- G5는 후속 변경의 실패 원인을 명확히 하기 위한 기초
- A3는 load opcode가 실제로 바뀌는지 SASS에서 확인 가능한 저비용 시도
- 둘 다 benchmark latency에 직접 영향이 작지만, 다음 구조 변경의 판단 근거를 제공

**예상 결과**: 
- Median ≈ 0.012920 ms (변화 없거나 ±0.001 ms 이내)
- SASS에서 `LDG.E.128.CONSTANT` 또는 `ld.global.nc.v4.f32` 확인 → 다음 단계 판단 기초 제공

**리스크**: 
- 한계: 5회 median 개선 희박 (R7의 cache hint 경험상)
- 하지만 SASS 기준선 확보의 가치로 정당화 가능

---

#### 방향 B: B5 warp specialization (고난이도, 고위험)
**구성**:
- 4 warps 중 1 warp은 async load 전담 (producer)
- 3 warps은 compute 전담 (consumer)
- `mbarrier` + `setmaxnreg.inc/dec`로 register 재분배

**근거**:
- 현재 "low bytes-in-flight" 병목에 직접 대응
- Blackwell `setmaxnreg` 지원으로 register 압박 해소 가능성
- register가 실제로 감소하면 occupancy 개선 → latency 기대

**예상 결과**:
- 성공 시: register 감소 (56→48~52), achieved occupancy 35~40%, median ≤ 0.012 ms 가능
- 실패 시: producer/consumer 동기화 cost > bytes-in-flight 이득, median > 0.013 ms

**리스크** (높음):
- `mbarrier` + `setmaxnreg` 조합 미검증
- TVM FFI launcher가 `mbarrier` 인자 전달 가능한지 불명확
- 구조적 변경 → partial rollback 불가능

---

#### 방향 C: J1 state layout 변경 (초고난이도, 구조 재설계)
**구성**:
- Current: state `[B, 8, 128, 128]` k-last
- Target: state `[B, 8, 128, 128]` v-last 또는 k-major 재배열
- API 호환성 검증 필수

**근거**:
- Read pattern 변경 → L1/L2 hit 상향 가능
- ldmatrix-friendly layout 검토

**예상 결과**:
- 성공 시: L1 hit > 10%, latency ±0 ~ -0.001 ms
- 실패 시: Layout convert overhead > 절감 이득

**리스크** (극높음):
- API 호환성 문제 → 모든 benchmark에서 incorrect 가능
- 깊은 code change → 2~3시간 이상 소요 가능
- **현재 마진에서 불확실한 투자로 부적절**

---

### 권장: 방향 A (G5 + A3)

**이유**:
1. **리스크 최소**: 구조 변경 없음, rollback 안전
2. **근거 확보**: 다음 구조 변경(B5, B2 등)의 기초 데이터 제공
3. **병렬 가치**: SASS 기준선 자체가 학습 가치 높음
4. **시간 효율**: 2~3시간 이내 완료, 다음 iteration 진행 용이

**단점**:
- Median 개선 직접 기대 낮음 (R7 경험상)
- 하지만 "정보 획득"의 가치로 정당화

---

## Step 2: PM 검토 대화 (임의 시뮬레이션)

### 초안 제시
**계획**: G5 (SASS/ptxas 기준선) + A3 (read-only load path) 결합

### PM 우려사항 (예상)
1. **프로파일·근거**: SASS 기준선은 후속 변경의 근거지만, 이번 iteration의 latency 개선 희박
   - **대답**: R1~R10의 공통 패턴은 "단편적 개선 시도"였고, G5로 기초를 고정하면 B5/B2 같은 고난이도 변경의 성공률 상향
   - 10회 연속 실패의 교훈: 근거 없는 구조 변경은 위험하다
   - **조정**: Step 1 마무리 후 "G5 결과 분석 → B5 가능성 판단" 순서로 진행

2. **한 iteration에 과한지**: 두 가지 변경을 동시에 진행하는가?
   - **대답**: G5는 컴파일 플래그 추가 (non-invasive), A3는 함수 호출 추가 (localized)
   - 둘 다 "구조 변경"이 아니므로 rollback 비용 낮음
   - 별도 시도로 분리도 가능하지만, SASS 확인 후 opcode 검증이 필요하면 결국 함께 측정이 효율적

3. **회귀 리스크**: `ld.global.nc` 명시 시 다른 batch에서 부작용?
   - **대답**: `__ldg()` 또는 inline PTX `ld.global.nc.v4.f32`는 모두 "read-only" 힌트
   - state는 kernel 내에서 write 없으므로 모든 batch에 안전
   - 최악의 경우 opcode가 기존과 동일 (compiler가 이미 생성했을 가능성)

---

## 최종 결정

### APPROVED

**선택 방향**: G5 + A3 (저위험 기초 마련)
**다음 단계 조건**: 
- SASS에서 `LDG.E.128.CONSTANT` 또는 `ld.global.nc.v4` 확인 → A3 effect 검증
- Register 수 변화 없음 → B5 warp specialization 재검토 근거 제공
- 5회 median 구간에서 ±0.001 ms 이내 → 이론 맞음

**목표**: Iteration #1에서 직접 median 개선보다 "다음 iteration의 성공률 상향"에 초점

---

## Step 1 세부 (G5 + A3)

### G5: SASS/ptxas 기준선 확보
```bash
# Modal compile 로그에서 -Xptxas -v -warn-spills 추가
# 또는 kernel 후처리:
/opt/homebrew/opt/cuda/bin/nvcc -arch=sm_100a -Xptxas -v -warn-spills \
  -c solution/cuda/kernel.cu -o kernel.o 2>&1 | grep -E "register|spill"

# cuobjdump SASS 분석:
cuobjdump --dump-sass kernel.cubin | grep -E "LDG|STG|SHFL|FFMA|FADD" | head -50
```

### A3: State read `__ldg()` 명시
```cuda
// kernel.cu 내 state read 구간 (approx. 5개 위치)
// Before:
// float s_val = state[state_offset + vi * V_DIM + ki];

// After (Option 1: __ldg):
// float s_val = __ldg(&state[state_offset + vi * V_DIM + ki]);

// After (Option 2: inline PTX v4):
// float4 s_val4; // 4개 element를 한 번에
// asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" 
//              : "=f"(s_val4.x),"=f"(s_val4.y),"=f"(s_val4.z),"=f"(s_val4.w)
//              : "l"(ptr));
```

---

## Confidence & Next

**이번 iteration 성공 확률**: 70% (SASS 기준선 확보 관점)
**Median 개선 확률**: 20% (R7 경험상, cache hint 부작용 주의)
**구조 변경 필요성**: 90% (R1~R10 패턴상, B5/B2 필연적)

**Phase 4 돌파 경로**:
1. ✅ (예정) G5 + A3 기초 마련
2. → (다음 iteration) B5 warp specialization (register 감소 근거 확보 후)
3. → (그 다음) B2 async pipeline + pipeline 병렬 구성
4. → (필요시) J series state layout 재설계

---
