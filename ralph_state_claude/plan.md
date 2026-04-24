# Iteration #2 (2026-04-24) — Step 1~2 계획 (PM 검토 포함)

## 현재 기준선 요약

| 항목 | 수치 | 출처 |
|------|------|------|
| **avg latency (baseline)** | **0.011108 ms** | `ralph_state_claude/latest_latency.txt` |
| **NCU Kernel Duration (baseline)** | **30.85 µs** | `ralph_state_claude/latest_ncu_duration_us.txt` |
| **Phase 4 Target** | < 0.009 ms | `workflow.md` |
| **남은 거리** | **–0.002108 ms (약 19%)** | 계산 |
| **Correctness** | ✅ 54/54 PASSED | iter #4 bench 유지 |
| **이미 적용된 기법** | `ld.global.nc` PTX, I1 split factor (≤2→8 / <32→8 / else→4), H2.5 dual-buffer 8-row prefetch, lane-0 output gate, L2 persistence, `__launch_bounds__(128,9)` | `solution/cuda/kernel.cu` 확인 |
| **NCU 병목 snapshot** | IPC 1.74 / Issue Slots Busy 22.57% / Occupancy 42.27% / L1 Hit 7.86% / L2 Hit 1.76% / Mem TP 1.72 TB/s | iter #4 NCU detailed |

### 최근 실패 이력 (재시도 금지 / 조건부)
- **[iter #1, 이번 루프] F4 lane<4 4-way store**: +43.1% 회귀 → 즉시 롤백. 원인: `qs_*`가 실제로는 warp-wide broadcast가 아니고 lane 0에만 완전값 존재(shfl_down reduce). 추가 shfl 4개 + dynamic-lane select가 hot inner loop의 issue slot을 침식. **현 구조에서 F4 재시도 금지**.
- **B5 warp specialization**: SMEM 인덱싱 오류 → SUSPENDED
- **C1 unroll 8**: +5.8% 회귀 (register 압박)
- **C2 `redux.sync.add.f32`**: 빌드/PTX 거부
- **A2 annotated_ptr**: +46% 회귀
- **B1 2-CTA cluster (minimal)**: +43% 회귀
- **D5 CUDA Graph (wrapper 내부)**: +8~35% 회귀
- **R2/R3 shared q/k staging**: 2회 회귀
- **R5 per-thread pipeline_memcpy_async**: +1.6% 회귀
- **R9/R10 shared q/k + FFMA dedup**: 회귀
- **G5 nvcc flag 환경변수**: –44% 회귀

### 이번 iter의 "절대 원칙"
1. **단일 변경 원칙**: 한 iter에 1개 변경만. (workflow.md)
2. **코드 재확인(grep 포함) 후 APPROVED**: iter #1 F4 실패의 직접 원인은 "plan이 코드 사실과 달랐음". 모든 가정은 실제 코드에서 재확인.
3. **Hot inner loop (`line 142~215`) 는 손대지 않는다**: shfl/select 추가 금지(iter #1 교훈). 변경은 **host-side 초기화, 템플릿 힌트, 커널 진입부** 등 inner-loop 외부에서만.
4. **회귀 tolerance**: median > 0.011108 × 1.005 ≈ 0.011164 ms → 즉시 롤백.

---

## Step 1: 후보군 도출 및 리스크 분석

### 후보 A (최상): **A6. SMEM carveout = 0 (L1 극대화) — standalone**

**상세:**
- Host 경로 1회 호출: `cudaFuncSetAttribute(gdn_decode_kernel<ROWS_PER_WARP>, cudaFuncAttributePreferredSharedMemoryCarveout, 0)` 을 템플릿 instantiation 별로 적용 (ROWS_PER_WARP = 4/8/16).
- `setup_l2_persistence`처럼 `static bool g_sm_carveout_setup` 가드로 프로세스당 1회만 실행.
- 커널 측 코드 변경 전혀 없음. 순수 host-side 런타임 힌트.

**코드 재확인 (grep 실증):**
- `solution/cuda/kernel.cu:117` 에서 `__shared__ float s_v[HEAD_DIM]` → 512 B (128 × 4 B) 정적 SMEM만 사용. dynamic SMEM 없음.
- `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` 미사용 → carveout 조정과 dynamic SMEM 충돌 없음.
- 결론: carveout=0 시 SMEM 512 B 는 L1 영역의 일부를 사용하게 되므로 spill/실패 경로 없음.

**NCU 근거 (간접):**
- L1 Hit Rate 7.86% (낮음) — L1 캐시 용량 늘리면 prefetch된 `state_base` float4 라인들이 L1에 잔류할 가능성 상승.
- 현 구조의 state footprint: `ROWS_PER_BLOCK × HEAD_DIM × 4 B` = 128 × 128 × 4 ≤ 64 KB / block. `split_factor=8, ROWS_PER_WARP=4` 시 block당 state read는 `4 × 4 × 128 × 4 B = 8 KB` 경로 (warp 당 16 rows × 128 × 4 = 8KB 아님 정정 — warp당 ROWS_PER_WARP×HEAD_DIM×4 = 4×128×4 = 2 KB, 4 warp 합 8 KB). Prefetch 후반 라인 재사용 없음이지만, **서로 다른 warp가 가까운 row 영역을 접근**하는 경우 L1 line 공유 가능(B200 L1 line = 128 B).
- Memory TP 1.72 TB/s = 32.89% DRAM 활용 → miss 경로 여유 有. L1 늘려서 miss를 조금이라도 줄이면 net 이득.

**기대 효과:** E2E –0 ~ –0.0003 ms (불확실, 다운사이드 ≈ 0).

**리스크: 낮음**
- SMEM 사용량이 물리 최소(512 B)라 carveout 변경이 spill 유발 불가능.
- 과거 R3 실패 조합 = "carveout=0 + split-local s_v staging". staging은 SMEM 사용량을 키워 L1을 역압박했을 가능성 — 지금은 staging 없으므로 **메커니즘 분리됨**.
- 실패 시 revert = 1줄 제거 + setup flag 제거.

**구현 난도:** 약 10분. host-side 4~6줄 추가.

**회귀 위험:** 매우 낮음. 다만 Blackwell B200 상에서 carveout=0이 **default setting과 다를 수 있음**(HW 기본 preferred carveout이 이미 "L1 max"일 가능성) — 이 경우 이득은 0이나 회귀도 0.

---

### 후보 B (중): **G6. `__builtin_assume()` 인덱스/범위 힌트 (inner-loop 외부)**

**상세:**
- 커널 진입부에 다음 `__builtin_assume` 추가 (hot inner loop 는 건드리지 않음):
  - `__builtin_assume(blockDim.x == BLOCK_SIZE);`
  - `__builtin_assume(split_id >= 0 && split_id < SPLIT_FACTOR);`
  - `__builtin_assume(lane >= 0 && lane < 32);`
  - `__builtin_assume(warp_id >= 0 && warp_id < NUM_WARPS);`
- nvcc/ptxas의 분기 제거 및 index 계산 simplify 유도.

**코드 재확인:**
- `kernel.cu:61~69` index 계산 영역 — template const로 이미 대부분 고정. assume 추가는 기존 constexpr/template 상수와 충돌 없음.
- 기존 `__launch_bounds__(128,9)` + `__restrict__` 이미 있음.

**NCU 근거:**
- Issue Slots Busy 22.57%의 일부는 주소 계산/분기. 다만 이미 template/constexpr 최적화가 많아 **ptxas가 이미 같은 추론을 내렸을 가능성 큼** → 이득 marginal.

**기대 효과:** –0 ~ –0.0001 ms. 부가 이득 수준.

**리스크: 매우 낮음**
- assume 이 틀리면 UB지만, 위 4개 조건은 **물리적으로 보장**.
- 구현 5분.

---

### 후보 C (조사만): **F3. state / new_state in-place 허용 여부 조사**

**상세:**
- 현재 커널: `state_base` 읽고 `new_state_base` 에 저장. 두 포인터가 같은 버퍼일 수 있다면 store 트래픽의 의미가 바뀜(다만 read 직후 write라 correctness는 보존됨 — 해당 row의 읽기가 끝난 뒤 쓰기 발생).
- 질문: `pack_solution.py` / bench harness 가 `state`와 `new_state`에 **동일 pointer** 를 줘도 되는가? 아니면 **별도 버퍼** 가정인가?
- 만약 후자라면 in-place 시도 자체 불가; 전자라면 `new_state` 파라미터를 받되 `new_state == state` 인 경우에도 정확성 유지되는지 확인.

**판정:** 이번 iter에서는 **조사만**, 코드 변경 없음. 결과에 따라 다음 iter 계획에 반영.

**리스크:** 조사 자체는 위험 없음 (read-only).

**구현 난도:** 조사 30분 (파이썬 harness grep).

---

### 후보 D (제외): **H2 `cuda::pipeline<3>` + memcpy_async / B5 warp specialization / D5 CUDA Graph / F4 재시도**
- **판정:** ⏸️ **이번 iter 범위 초과 또는 재시도 금지**. 전부 직전 이력상 회귀 또는 correctness fail.

---

### Step 1 후보 요약표

| 후보 | 우선도 | 기대효과 (E2E) | 리스크 | 구현 시간 | 판정 |
|------|--------|----------------|--------|-----------|------|
| **A6** SMEM carveout=0 (standalone, host-only) | ★★★ | –0 ~ –0.0003 ms | 낮음 | 10분 | **채택 후보 1** |
| **G6** `__builtin_assume` (inner-loop 외부만) | ★★ | –0 ~ –0.0001 ms | 매우 낮음 | 5분 | 차선(다음 iter) |
| **F3** state in-place 조사 | ☆ | (조사) | 없음 | 30분 | 병행 조사 |
| H2/B5/D5/F4 재시도 | ☆ | — | 매우 높음 / 재시도 금지 | — | 제외 |

---

## Step 2: PM 검토 대화

### 라운드 1

**PM (System Architect):**
> "iter #1에서 F4를 '저위험'으로 분류했다가 43% 회귀를 맞았지. 그 실패의 핵심은 **plan의 가정이 실제 코드와 달랐던 것**이다. 이번 A6·G6·F3를 보고 세 가지를 묻는다.
>
> (a) **프로파일·근거**: A6 이득이 L1 Hit 7.86%를 유의미하게 올릴 수 있다는 **직접 증거**가 있나? 아니면 iter #1처럼 간접 추론인가?
> (b) **한 iter에 과한가**: A6 + G6 + F3 조사를 한 번에 묶을 건가, 아니면 1개만 건드리고 나머지는 미룰 건가? iter #1 교훈대로 단일 변경 원칙을 지키나?
> (c) **회귀 리스크**: R3에서 carveout=0 조합이 한 번 실패했다. 'standalone은 메커니즘이 다르다'는 주장은 실증됐나, 아니면 추론인가?"

**Assistant (Tech Lead):**
> "정곡을 찔렀다. 순서대로.
>
> (a) **직접 증거는 없다**. `lts__t_bytes_op_read.sum` 중 L1 miss 비율의 직접 profiling이 필요한데 detailed set엔 rolled-up 지표만 있음. 다만 다음은 **사실관계로 확인**:
>    - `grep -n '__shared__' solution/cuda/kernel.cu` → `__shared__ float s_v[HEAD_DIM];` 1곳뿐, 512 B. dynamic SMEM 요청 없음.
>    - 즉, carveout=0 으로 바꿔도 kernel이 SMEM 부족으로 실패하는 경로가 **물리적으로 존재하지 않는다**. 다운사이드가 0에 가깝다는 점은 **실증**.
>    - 이득 여부는 '불확실' 로 명시한다. 0 일 수 있다.
>
> (b) **단일 변경 원칙 준수**. iter #2 스코프는 **A6 단독**. G6와 F3 조사는 **이번 iter에 혼합하지 않는다**.
>    - G6: 다음 iter에 단독 적용.
>    - F3: 이번 iter의 '관찰자 역할'에서 harness 코드 리뷰만 수행(본 계획서에서는 조사 액션만 기재, 커널 변경 없음).
>
> (c) **메커니즘 분리 실증**:
>    - R3 조합: carveout=0 **AND** split-local s_v staging. staging이 SMEM 사용량을 늘려 L1을 '지불하고 얻은 staging 공간'이 오히려 손해였을 수 있음(저자 코멘트: 'median 0.013244 ms 회귀').
>    - 현재: staging 제거됨(`grep 'staging\|s_v_staged' kernel.cu` = 0건), SMEM 사용량 512 B 고정.
>    - 메커니즘이 다르다는 것은 **코드 상 분리됨**으로 실증. 다만 PM 지적대로 **경험적 실증은 이번 측정으로 처음 얻는 것**임을 인정한다. 회귀가 난다면 즉시 롤백."

**PM:**
> "(a) 이득 불확실을 인정한 건 좋다. 문제는 '이득이 0일 수도 있다'면 **왜 이번 iter를 A6로 태우나?** 그 iter 예산을 F3 조사 + G6 적용 쪽으로 돌릴 생각은 없나?"

**Assistant:**
> "공정한 질문이다. 세 가지 이유:
>
> 1. **A6는 host-side 1 API call**. 구현 5~10분. iter 예산 거의 안 든다.
> 2. **다운사이드가 실증적으로 0**. 실패해도 median 변동은 Modal noise (±0.0003 ms) 이내일 가능성 높음.
> 3. **향후 Phase 4 (<0.009 ms) 진입 시 L1/L2 cache 경로 최적화가 누적 효과로 필요**. A6는 후속 L1 hit 최적화 (state layout J1, prefetch depth 조정)와 **독립적이어서 조합 가능**.
>
> 즉 A6는 '저위험·저이득 먼저 확보' 전략. 대안으로 PM이 G6 단독(이득 불확실·저리스크)을 더 선호한다면 바로 스위치 가능. 다만 **G6는 이미 template/constexpr로 ptxas가 대부분 알고 있을 가능성**이 커서 A6보다 '마이크로 이득' 성격이 더 강함."

**PM:**
> "알겠다. A6 단독, host-only 변경이라는 점은 수긍한다. 다만 추가 조건 3개 붙인다."

### 라운드 2 (조건 협상)

**PM:**
> "1. **측정 프로토콜**: bench 최소 3회, median of 5 권장. Modal noise ±0.0003 ms 고려.
> 2. **NCU 비교**: L1 Hit Rate (`l1tex__t_sector_hit_rate.pct`) 전후 수치를 반드시 기록해 A6 메커니즘을 **실증**하라. 이득이 0이어도 **데이터는 남긴다** (다음 L1 최적화 판단 근거).
> 3. **회귀 판정**: median > 0.011108 × 1.005 = **0.011164 ms** 즉시 롤백. correctness fail 도 즉시 롤백.
> 4. **Hot inner loop 불변**: iter #1 F4 실패 교훈. `kernel.cu:142~215` 는 한 글자도 건드리지 않는다.
> 5. **F3 조사는 별도 아티팩트**: 이번 iter의 plan.md / log.md 에 조사 결과만 기록, 커널 변경에 포함시키지 않는다. (단일 변경 원칙)
>
> 이 5개 동의하면 APPROVED."

**Assistant:**
> "모두 동의.
> - Scope: **A6 SMEM carveout=0 (standalone, host-only, ROWS_PER_WARP∈{4,8,16} 3개 템플릿 instantiation 전부)**.
> - Measurement: bench 5회 median + correctness 54/54 + NCU detailed 1회 (L1 Hit Rate 필수 기록).
> - Accept: median ≤ 0.011108 ms (baseline 유지 이상).
> - Reject (즉시 롤백): median > 0.011164 ms 또는 correctness FAIL.
> - Hot inner loop (`line 142~215`) 변경 금지.
> - F3 조사는 별도 조사 노트로 log.md 에 기록, 커널 변경 없음.
> - G6 는 다음 iter 단독 적용 후보."

### 라운드 3 (코드 재확인)

**PM:**
> "마지막. iter #1 실패의 직접 원인이 '가정이 코드와 달랐음'이었다. A6 적용 전 반드시 실증할 사실 3개를 코드로 재확인했나?"

**Assistant:**
> "실증 완료:
> 1. `solution/cuda/kernel.cu:117` `__shared__ float s_v[HEAD_DIM];` → 512 B 정적 SMEM, 유일.
> 2. `grep -n 'cudaFuncAttributeMaxDynamicSharedMemorySize\|cudaFuncSetAttribute' solution/cuda/kernel.cu` 결과 현재 해당 attribute 설정 없음 → 신규 추가 시 기존 설정과 충돌 없음.
> 3. `gdn_decode` host 함수 (`kernel.cu:250~303`) 에서 3개 템플릿 instantiation(ROWS_PER_WARP=4,8,16) 존재 → `cudaFuncSetAttribute`를 3개 템플릿 모두에 1회씩 적용해야 함. static bool 가드로 per-process once.
>
> 이 3개는 모두 **파일에서 직접 확인**. iter #1 F4의 'qs_* broadcast 있다고 가정' 같은 허위 가정 없음."

**PM:**
> "좋다. **APPROVED**."

---

## PM 최종 판정

### ✅ **APPROVED: A6 SMEM Carveout=0 (Standalone, Host-Only)**

**승인 사항:**
1. 변경 위치: `solution/cuda/kernel.cu`의 **host-side `gdn_decode` 진입부 또는 `setup_l2_persistence` 근처**에 `cudaFuncSetAttribute(gdn_decode_kernel<4/8/16>, cudaFuncAttributePreferredSharedMemoryCarveout, 0)` 를 static bool 가드와 함께 1회 적용.
2. **Hot inner loop (line 142~215) 절대 변경 금지.**
3. 측정: bench 5회 median + correctness 54/54 + NCU detailed 1회 (L1 Hit Rate 필수 기록).
4. Accept: median ≤ 0.011108 ms (baseline 유지 이상).
5. Reject(즉시 롤백): median > 0.011164 ms (baseline × 1.005) 또는 correctness FAIL.
6. 성공/중립/실패 무관하게 측정 데이터는 log.md에 남기고, G6/F3는 **다음 iter** 로 분리.

**현재 iter 스코프에서 제외:**
- G6 (`__builtin_assume`) — 다음 iter 단독 후보.
- F3 (state in-place) — 조사만, 커널 변경 없음. 결과는 log.md 부록으로 기록.
- H2/B5/D5/F4 재시도 / B1 cluster / A2 annotated_ptr / C1 unroll 8 / C2 redux — 전부 재시도 금지 (직전 실패).

**예상 결과:**
- E2E latency: 0.011108 ms → 0.01080~0.01111 ms (–0.3% ~ 0%, Modal noise 내).
- NCU Duration: 30.85 µs → 30.3~30.85 µs.
- L1 Hit Rate: 7.86% → 8~12% (다른 변경 없이 carveout 만의 효과 측정 가능).
- Phase 4 (<0.009 ms)까지 남은 거리는 여전히 –0.002 ms 수준이나, **L1 cache 여유 공간 확보**로 후속 J1/prefetch depth 최적화의 기반 마련.

**구현/측정 세션(다음 단계, 본 세션 범위 밖):**
- Step 3: A6 구현 (host-side `cudaFuncSetAttribute` 호출, static bool 가드).
- Step 4: pack → modal bench 5회 → NCU detailed 1회.
- Step 5: 판정 (accept/reject) → 유지/롤백 → log.md 기록.
- (병행) F3 조사: `scripts/pack_solution.py` 및 bench harness 에서 state/new_state 버퍼 동일 포인터 허용 여부 확인, log.md 부록 기록.

---

**APPROVED.**
