# GDN Decode Kernel 반복 최적화 워크플로우 (Round 2)

> **이 문서는 code agent가 자율적으로 따라야 하는 실행 가이드입니다.**
> **절대로 목표 성능을 달성할 때까지 멈추지 마세요.**

---

## 0. 절대 규칙 (NEVER BREAK)

1. **Phase 3 목표(Avg latency ≤ 0.008 ms)를 달성할 때까지 아래 루프를 반복한다. 중간에 절대 멈추지 않는다.**
2. 성능이 후퇴(regression)하면 즉시 되돌리고 다른 최적화를 시도한다.
3. correctness가 깨지면(status가 `correct`가 아니면) 즉시 되돌린다.
4. **한 번에 여러 최적화를 동시 적용해도 좋다 (복합 최적화 권장).** 단, 후퇴 시 어떤 변경이 원인인지 분리할 수 있도록 변경 내용을 명확히 기록한다.
5. 매 반복마다 반드시 아래의 **성능 측정** 단계를 수행하고, 결과를 **로그 섹션**에 기록한다.
6. **Modal 클라우드 노이즈**: reference latency가 15~38ms로 변동하므로, 반드시 reference latency도 함께 기록하여 비교 기준으로 삼는다. reference latency가 20ms 이상이면 "나쁜" 인스턴스로 판정하고, 같은 코드를 한 번 더 측정하여 확인한다.

---

## 1. 목표 정의

| Phase | 목표 Avg Latency | 현재 상태 |
|-------|-----------------|----------|
| Phase 1 | ≤ 0.012 ms | ✅ 달성 (Round 1) |
| Phase 2 | ≤ 0.010 ms | 근접 (최고 0.011 ms) |
| Phase 3 | ≤ 0.008 ms | 미달성 |

- **시작 성능**: Avg latency = 0.011 ms (좋은 인스턴스), 0.014 ms (중간 인스턴스)
- **핵심**: 대배치(B=32: 0.014ms, B=64: 0.017ms)가 평균을 끌어올리고 있음.
- Phase 2 달성 후에도 멈추지 않고 Phase 3까지 계속한다.

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

매 반복의 측정 단계에서 **반드시** 아래 명령을 실행한다:

```bash
conda run -n fi-bench python scripts/pack_solution.py && conda run -n fi-bench modal run scripts/run_modal.py
```

출력에서 다음을 확인한다:
- 각 workload의 `status` → 반드시 모두 `correct`여야 함
- **`Avg latency: X.XXX ms`** → 이 값이 목표 이하인지 확인
- **`Avg reference latency: X.XXX ms`** → 인스턴스 품질 판정용

**벤치마크 설정** (`scripts/run_modal.py`):
- 현재: `BenchmarkConfig(warmup_runs=10, iterations=50, num_trials=3)`
- 충분한 warmup과 iteration으로 안정적인 측정 확보

---

## 4. 반복 루프 (매 iteration마다 수행)

```
┌─────────────────────────────────────────────────────┐
│  STEP 1: 현재 상태 확인                               │
│  - 직전 Avg latency / reference latency 확인          │
│  - 목표 달성 여부를 판단한다                            │
│  - 달성했으면 → 다음 Phase로 / Phase 3 달성이면 종료    │
├─────────────────────────────────────────────────────┤
│  STEP 2: 최적화 전략 선택 (복합 적용 가능!)             │
│  - 아래 "최적화 후보 목록"에서 미시도 항목 2~4개 선택    │
│  - 시너지가 있는 조합을 우선 선택                       │
│  - 예상 효과와 risk를 분석                             │
├─────────────────────────────────────────────────────┤
│  STEP 3: 커널 수정                                    │
│  - solution/cuda/kernel.cu 를 수정한다                 │
│  - 수정 전 현재 커널을 백업(기억)해둔다                  │
│  - 여러 최적화를 한 번에 반영                           │
├─────────────────────────────────────────────────────┤
│  STEP 4: 성능 측정                                    │
│  - 벤치마크 실행                                      │
│  - ref latency > 20ms면 "나쁜" 인스턴스 → 재측정       │
│  - 결과의 status, Avg latency, ref latency를 기록      │
├─────────────────────────────────────────────────────┤
│  STEP 5: 결과 판정                                    │
│  - correctness 실패 → 즉시 롤백, STEP 2로             │
│  - latency 후퇴 → 개별 최적화 분리 시도 또는 롤백       │
│  - latency 개선 → 변경 유지, STEP 1로                 │
└─────────────────────────────────────────────────────┘
```

**이 루프를 Phase 3 목표(≤ 0.008 ms) 달성까지 반복한다. 절대 중단하지 않는다.**

---

## 5. 타겟 하드웨어: NVIDIA B200 (Blackwell, sm_100)

| 항목 | 수치 | 최적화 시사점 |
|------|------|-------------|
| Compute Capability | **10.0 (sm_100)** | Blackwell 전용 기능 사용 가능 |
| L2 캐시 | **126 MB** | 모든 batch의 state가 L2에 상주 가능 |
| Shared Memory/SM | **228 KB** (블록당 최대 227 KB) | 대용량 shared memory tiling 가능 |
| Max Warps/SM | **64** | occupancy 최적화 기준 |
| Max Thread Blocks/SM | **32** | 작은 block + 많은 block 전략 가능 |
| Register File/SM | **64K × 32-bit** | 레지스터 255개/thread |
| HBM3e Bandwidth | **~8 TB/s** | Memory-bound 커널의 이론 한계 |
| SMs | **~160** | Grid size 설계 기준 |

---

## 6. Round 1 교훈 (반드시 참고)

### 성공한 최적화
1. **커널 템플릿화** (`template<ROWS_PER_WARP>`): 컴파일 타임 루프 언롤 → -0.001ms
2. **통일 block_size=128**: 4 warps for all cases → -0.001ms
3. **split_factor 최적화**: B≤2→split=8, B≥3→split=4 → -0.002ms (최대 개선)
4. **constexpr 블록 파라미터**: `SPLIT_FACTOR`, `HEADS_X_SPLIT` 등을 constexpr → 컴파일러 최적화

### 실패한 최적화 (재시도 금지 또는 다른 방식으로)
- [실패] v→register + __syncthreads__ 제거: shfl 오버헤드가 shared memory보다 큼
- [실패] fast math intrinsics 단독: __expf, __logf 등 단독 적용 시 후퇴
- [실패] streaming store __stcs: L2 writeback이 B200에서 더 효율적
- [실패] __launch_bounds__(256,4): 잘못된 힌트로 코드 품질 저하
- [실패] split_factor=8 통일: B=32/64에서 per-block overhead > SM utilization 이득
- [실패] B=1 split=16 + block=64: s_v 로딩 문제 (block < HEAD_DIM)

### 핵심 인사이트
- **B=32/64가 주요 병목**: 0.014~0.017ms (평균 끌어올림)
- **B=1~8은 이미 0.008~0.010ms**: 소배치는 충분히 빠름
- **per-block overhead가 큼**: gate 계산, q/k 로딩, qk_dot reduction, v 로딩, syncthreads가 작은 블록에서 비중 높음
- **B=64: 메모리 대역폭 ~30-50% 활용**: 이론 0.008ms vs 실측 0.017ms
- **Modal 노이즈 ±0.003ms**: reference latency로 인스턴스 품질 확인 필수

---

## 7. 현재 커널 핵심 구조

```
커널: gdn_decode_kernel<ROWS_PER_WARP>
- Grid: (B * 8 * split_factor,)
- Block: 128 threads (4 warps), __launch_bounds__(128)
- 템플릿: <4> for B≤2 (split=8), <8> for B≥3 (split=4)
- 모든 블록 파라미터가 constexpr (SPLIT_FACTOR, HEADS_X_SPLIT 등)
- 4-row software pipelining: float4 register prefetch
- Interleaved 8-way warp reduction (ks×4 + qs×4)
- beta_g, scale_g, scale_qk 사전 계산
- 입력: bf16, State: float32, 출력: bf16
```

---

## 8. 복합 최적화 후보 (Round 2)

> **핵심 전략**: 한 번에 2~4개의 시너지 있는 최적화를 동시 적용

### Combo A: 대배치 메모리 최적화 세트
아래를 한 번에 적용:
- [ ] **B>16에서 split=8 + block=64(2 warps) + loop-based s_v loading**: grid 4배 증가, per-block overhead는 2 warps로 절반
- [ ] **q/k를 shared memory로 1회 로드**: 블록 내 모든 warp가 공유 → register 절약, occupancy 향상
- [ ] **L1 cache carveout 최대화**: `cudaFuncAttributePreferredSharedMemoryCarveout(0%)` → 512B shared memory만 사용하므로 L1을 최대로

### Combo B: 커널 구조 재설계
- [ ] **Persistent kernel + grid-stride loop**: 고정 grid=160 (=SM 수), 각 블록이 atomicAdd로 작업 할당. launch overhead 제거, SM utilization 100%.
- [ ] **batch-adaptive를 커널 내부에서 처리**: split_factor를 커널 파라미터가 아닌 work-item 단위로 동적 결정

### Combo C: 연산 + 메모리 혼합
- [ ] **qs reduction 제거**: output = scale * dot(q, new_state_row)로 재구성. new_state 계산 후 q와 dot product로 output 계산. ks reduction만 남김 → shuffle 절반 감소.
- [ ] **8-row pipeline**: 4-row → 8-row 확장. 더 많은 ILP + prefetch 깊이 증가. register 소비 증가하지만 B200의 64K regs/SM에서 여유 있음.
- [ ] **__ldg() for state reads**: 텍스처 캐시 경로 활용으로 L1/L2 분리

### Combo D: B200 전용 기능
- [ ] **Thread Block Clusters (cluster_size=2)**: V_PER_Q=2이므로 같은 qk_head의 2개 v_head 블록을 클러스터로 묶음. Distributed Shared Memory로 q/k 공유.
- [ ] **TMA (Tensor Memory Accelerator)**: state row의 async bulk copy. 주소 계산 하드웨어 오프로드.
- [ ] **cp.async.bulk for state prefetch**: warp-level prefetch 대신 TMA-based bulk prefetch

### Combo E: 컴파일러 최적화 세트
- [ ] **__builtin_assume() 힌트 세트**: blockDim.x, warp_id 범위, lane 범위 등
- [ ] **명시적 FMA 사용**: __fmaf_rn()으로 dot product와 state update
- [ ] **sm_100 타겟 컴파일 확인**: -arch=sm_100 명시

---

## 9. 성능 로그

### Round 1 요약 (이전 세션)
- 시작: 0.015 ms → 최종: **0.011 ms** (좋은 인스턴스)
- Phase 1 (≤0.012ms): ✅ 달성
- Phase 2 (≤0.010ms): 근접 (0.001ms 부족)
- 총 13 iterations, 3개 성공, 10개 실패/롤백

### Round 2 시작
(여기부터 기록)

---

## 10. 완료 조건

- [x] Phase 1 달성: Avg latency ≤ 0.012 ms (Round 1)
- [ ] Phase 2 달성: Avg latency ≤ 0.010 ms
- [ ] Phase 3 달성: Avg latency ≤ 0.008 ms

**Phase 3를 달성하면 이 워크플로우를 종료하고, 최종 결과를 사용자에게 보고한다.**
**Phase 3를 달성하지 못했으면 절대 멈추지 말고 루프를 계속 반복한다.**

---

## 11. 롤백 정책

- 커널 수정 전, 항상 현재 동작하는 전체 커널 코드를 기억해둔다.
- correctness 실패 또는 latency 후퇴 시, 즉시 직전의 정상 버전으로 복원한다.
- 복합 최적화에서 후퇴 시, 개별 최적화를 분리하여 어떤 것이 원인인지 확인 시도.
- 같은 최적화 조합을 두 번 이상 실패했으면 [실패] 처리하고 넘어간다.

---

## 12. 추가 지침

- **한 번에 여러 최적화를 묶어서 적용한다** (Round 1의 1개씩 전략은 노이즈 대비 변화가 너무 작았음).
- 시너지 있는 조합을 우선 선택: 예) register 절약 + occupancy 향상 + memory prefetch 강화
- **커널 구조 자체를 재설계하는 것을 두려워하지 말 것**: persistent kernel, warp specialization, 완전히 다른 tiling 등.
- `modal run`의 reference latency를 반드시 확인. >20ms면 재측정.
- **B=32/64 개선이 핵심**: 이 workload들을 0.010ms 이하로 줄이면 Phase 2/3 달성 가능.
