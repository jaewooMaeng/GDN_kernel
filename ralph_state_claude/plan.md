# Iteration #1 (2026-04-24) — Step 1~2 계획

## 현재 기준선 요약

| 항목 | 수치 | 출처 |
|------|------|------|
| **avg latency** | **0.011108 ms** | `ralph_state_claude/latest_latency.txt` |
| **NCU Kernel Duration** | **30.85 µs** | `ralph_state_claude/latest_ncu_duration_us.txt` |
| **Target (Phase 4)** | < 0.009 ms | `workflow.md` |
| **남은 거리** | **–0.002108 ms (약 19%)** | 계산 |
| **Correctness** | ✅ 54/54 PASSED | iter #4 bench |
| **Kernel 내 이미 적용된 기법** | `__ldg` (PTX `ld.global.nc`), I1 split factor, H2.5 dual-buffer 8-row prefetch, lane-0 gate, L2 persistence, `__launch_bounds__(128,9)` | `solution/cuda/kernel.cu` 확인 |
| **NCU 주요 병목** | IPC Active 1.74 / Issue Slots Busy 22.57% / Achieved Occupancy 42.27% / L1 Hit 7.86% / L2 Hit 1.76% / Memory TP 1.72 TB/s | iter #4 NCU detailed |

### 최근 실패 이력 (재시도 금지 또는 조건부)
- **B5 warp specialization**: iter #2 SUSPENDED — SMEM 인덱싱 오류
- **C1 unroll 8**: iter #4 +5.8% 회귀 → register 압박
- **C2 `redux.sync.add.f32`**: iter #3·R7 빌드/PTX 거부
- **A2 annotated_ptr**: R7 +46% 회귀
- **B1 2-CTA cluster (minimal)**: R8 +43% 회귀
- **D5 CUDA Graph (wrapper 내부)**: R4 +8~35% 회귀
- **R2/R3 shared q/k staging**: 2회 모두 회귀
- **R5 per-thread pipeline_memcpy_async**: +1.6% 회귀
- **R9/R10 shared q/k + FFMA dedup**: 회귀

---

## Step 1: 후보군 도출 및 리스크 분석

### 기본 방침
- 현재 커널에는 H2.5 dual-buffer, __ldg, I1, lane-0 gate, L2 persistence가 **이미 모두 적용**되어 있다. 그러나 E2E latency는 0.011108 ms에서 정체됐고 이번 iter의 남은 여지는 **구조적 대변경 없이 kernel body 내부를 손볼 수 있는 저위험 후보**에 있다.
- 구조적 변경 계열(B1/B2/B5/D5/H4)은 직전 5~6회 거듭 실패했다 → **이번 iter에서는 보수적 후보 위주**로 고른다.
- PM 관점에서 최소 요구: (a) NCU 근거가 있고, (b) 2~3시간 내 구현 가능하며, (c) 실패 시 즉시 롤백 가능한 **국소 패치**여야 한다.

---

### 후보 A: **F4. Output store를 lane 0~3으로 4-way 분산** (우선도: ★★★ 최상)

**상세:**
- 현재 `if (lane == 0) { out_base[vi_a..vi_a+3] = ... }` — lane 0 혼자 4회 bf16(2B) store.
- 변경: `if (lane < 4) out_base[vi_a + lane] = __float2bfloat16(...)`. lane 0~3이 각각 1 element 담당 → STG.B16 4-way coalesced.
- `res_a/b/c/d`, `qs_a/b/c/d`는 이미 `__shfl_sync(..., 0)`로 warp 전체에 broadcast돼 있으므로 lane 1~3에도 올바른 값이 있음(추가 shuffle 불필요).

**NCU 근거:**
- Issue Slots Busy 22.57% / IPC 1.74의 낮은 값 중 일부는 lane 0 직렬화 store에서 발생하는 idle lane과 관련.
- B200 STG.B16 throughput 기준 4-lane 분산은 lane 0 단일 대비 latency 숨김 가능.
- Store 명령 수는 동일(warp 당 4 store)하지만 lane 분산이 issue slot 점유를 단축.

**기대 효과:** NCU Duration –0.3~–0.8 µs (약 1~3%). E2E latency –0.0002~–0.0005 ms.

**리스크:** **낮음**
- `__shfl_sync(..., 0)`로 이미 lane 간 값이 일치 → correctness-safe.
- 부동소수점 정밀도 영향 **없음** (연산 순서 보존).
- 실패 시 2줄 revert.

**회귀 위험:** 매우 낮음. `if (lane < 4)` 분기는 B200 conditional-store에서 divergence 비용 미미.

**구현 난도:** 10분. `solution/cuda/kernel.cu` line 209–214 국소 편집.

---

### 후보 B: **A6. SMEM carveout = 0 (L1 극대화)** (우선도: ★★ 중상)

**상세:**
- `cudaFuncSetAttribute(gdn_decode_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0)`를 host 초기화 1회 적용.
- 현재 SMEM 사용량 512 B뿐 → carveout=0으로 L1 cache 용량 최대화(약 192 KB까지 가용).
- State read가 single-pass streaming이라 L1 hit 이득은 제한적이지만, prefetch 경로(`ld.global.nc`)의 L1 line 재사용 구간은 **block 내부 8 rows/warp × 4 warps = 32 rows = 16 KB** → L1에 완전히 들어감.

**NCU 근거:**
- L1 Hit Rate 7.86%는 **여전히 낮음** → carveout 확장으로 일부 개선 가능.
- Memory TP 1.72 TB/s는 32.89% DRAM 활용률로 여유 있음; miss 경로 여유 있으니 hint 조정의 net 이득은 +.
- 단, R3에서 `carveout=0` 단독 시도는 median 0.013244 ms 회귀 → 해당 실험은 **split-local s_v staging과 결합**해서 실패한 것이므로 **standalone carveout 조정은 미시도에 가까움**.

**기대 효과:** –0.0~–0.0003 ms (불확실하나 다운사이드는 거의 없음).

**리스크:** **낮음**
- Host 1줄 추가. SMEM 사용량이 512 B밖에 안되므로 carveout 변경이 spill을 유발할 가능성 없음.
- Fallback: 원복 1줄.

**회귀 위험:** 낮음. 단, R3 실패 조합과 분리해서 단독 측정해야 함.

**구현 난도:** 5분. `gdn_decode` 진입 1회 static initializer.

---

### 후보 C: **G6. `__builtin_assume()` 인덱스 힌트** (우선도: ★ 중)

**상세:**
- 커널 진입부에 `__builtin_assume(blockDim.x == BLOCK_SIZE)`, `__builtin_assume(split_id >= 0 && split_id < split_factor)` 힌트 추가.
- nvcc/ptxas의 분기 제거 및 인덱스 계산 단순화 유도.

**NCU 근거:**
- Issue Slots Busy 22.57%에서 주소 계산 overhead가 일부 포함됨.
- 이미 `__restrict__`, `#pragma unroll`, `__launch_bounds__`가 있으므로 추가 힌트 이득은 marginal하지만 **downside는 0**.

**기대 효과:** –0~–0.0001 ms. 부가 이득.

**리스크:** **매우 낮음**. assume 문이 틀렸을 경우 UB지만, 이번 힌트는 물리적으로 보장된 조건(블록 크기 고정).

**구현 난도:** 5분.

---

### 후보 D: **H2 `cuda::pipeline<thread_scope_block, 3>` + `memcpy_async`** (우선도: ☆ 후순위)

**상세:** 현재 register 기반 dual-buffer(curr/next)를 3-stage SMEM pipeline으로 재설계.

**리스크:** **매우 높음**
- B5/R5 2회 실패, async pipeline 계열은 harness 경로에서 setup cost가 이득 상쇄.
- SMEM 인덱싱 오류 재현 가능성 高.

**판정:** ⏸️ **이번 iter 범위 초과** — iter #2 이후 H2.5 포함 재설계 시점에 묶어서 검토.

---

### 후보 E: **F3. State in-place update (new_state == state 허용 여부)** (우선도: ☆ 조사 필요)

**상세:** API가 `new_state`에 `state`와 동일 포인터 전달을 허용하면 store 트래픽 완전 제거 (16 KB × block 수).

**리스크:** API 계약 확인 필요. `scripts/pack_solution.py`와 bench harness 상 `state/new_state`가 별도 tensor임을 가정하는지 조사 필요.

**판정:** ⏸️ **이번 iter에서는 조사만**. 다음 iter에서 조건부 진행.

---

### Step 1 후보 요약 표

| 후보 | 우선도 | 기대효과 (E2E) | 리스크 | 구현 시간 | 판정 |
|------|--------|----------------|--------|-----------|------|
| **F4** lane 0~3 4-way store | ★★★ | –0.0002~–0.0005 ms | 낮음 | 10분 | **채택 후보 1** |
| **A6** SMEM carveout=0 (standalone) | ★★ | –0~–0.0003 ms | 낮음 | 5분 | **채택 후보 2** |
| **G6** `__builtin_assume` | ★ | –0~–0.0001 ms | 매우 낮음 | 5분 | 채택 후보 3 (부가) |
| H2 block pipeline | ☆ | (불확실) | 매우 높음 | 3시간+ | 이번 iter 제외 |
| F3 state in-place | ☆ | –0.0003~–0.001 ms | 조사 필요 | 조사 30분 | 다음 iter |

---

## Step 2: PM 검토 대화

### 라운드 1

**PM(Product Manager / System Architect):**
> "지난 몇 iter 로그를 보니 구조적 변경 계열은 전부 실패했다(B1, B5, D5, H4, R2/R3/R5 등). 현 커널은 이미 H2.5, __ldg, I1, lane-0 gate, L2 persistence 가 다 들어가 있고 E2E latency는 0.011108 ms에 고정됐다. 이번 iter의 후보 F4/A6/G6는 '저위험 저이득' 조합인데 세 가지 궁금증이 있다.
>
> (a) **프로파일·근거**: F4가 실제 lane-0 store 직렬화가 병목이라는 NCU 근거가 있나? 'issue slots busy 22.57%'는 간접 증거 아닌가?
> (b) **한 iter에 과한가**: 세 후보를 동시에 적용하면 어떤 효과가 어디서 왔는지 구분이 안 된다. 한 번에 몇 개까지 쌓을 것인가?
> (c) **회귀 리스크**: R3에서 `carveout=0` 조합이 이미 실패했다. standalone 재시도라도 같은 결과가 나올 수 있지 않나?"

**Assistant(Technical Lead):**
> "좋은 지적이다. 순서대로 답한다.
>
> (a) F4의 직접 근거는 없다. `store pipe`의 utilization을 보려면 `lts__t_bytes_op_write` 또는 `smsp__inst_executed_pipe_adu`가 필요한데 현재 detailed set에는 안 들어간다. 다만 다음 두 간접 근거는 있다.
>    - SASS 관점: lane 0만 store하면 warp 한 번 issue 시 **inactive lane 31개** → issue slot 한 개가 낭비.
>    - 현 4 row × 1 store = 4 store/iter에 lane 0 직렬화. 4-way 분산 시 동일 4 store지만 **activation mask**가 `0xF` → issue가 packed되고 lane 0 waitfor가 없어진다. Blackwell STG.B16 throughput는 sub-warp 분산 친화적이다.
>    - 확신도: **70%** 정도. 틀려도 다운사이드는 거의 0 (정밀도/구조 불변).
>
> (b) 한 iter에 **단일 변경이 원칙** (workflow.md 절대 규칙 3). 따라서:
>    - **F4 단독 적용 → 측정** (벤치 5회 median, 다음 세션).
>    - 결과에 따라 A6·G6를 **별도 iter**로 순차 적용.
>    - 즉 이번 iter 구현 세션에서는 **F4 1개만** 건드린다.
>
> (c) R3 조합은 'split-local s_v staging + carveout=0'이었다. s_v staging이 extra SMEM을 실제로 사용하게 만들어 carveout=0가 반대로 성능을 끌어내렸을 수 있다. 지금은 s_v가 여전히 512 B뿐이라 carveout=0는 **L1을 키우는 방향**으로만 작용한다 → 메커니즘이 다르다. 다만 PM 지적대로 **단독 재시도도 downside 가능성**이 있으므로 **이번 iter에서는 A6/G6는 일단 제외**하고, F4 측정 결과 확인 후 다음 iter에 이어붙인다."

**PM:**
> "좋다. (b) 단일 변경 원칙과 (c) 조합 분리가 정리됐다. 그럼 이번 iter 스코프를 **F4 단독**으로 좁히자. 추가 조건:
>
> 1. 벤치는 최소 3회, Phase 3 이후이므로 median of 5 권장.
> 2. correctness는 전 54 workload 통과 필수.
> 3. NCU Duration 측정 병행해서 kernel-level 이득도 확인.
> 4. **만약 E2E가 ±0.0003 ms 이내(Modal 노이즈 범위)면 유지 vs 롤백 판단 보류**하고 A6/G6 조합은 별도 iter로 이어붙여 최종 판단.
> 5. 회귀(median 0.011108 + 0.5% 이상) 시 즉시 롤백.
>
> 이 조건에 동의하면 APPROVED 한다."

**Assistant:**
> "동의한다. 조건 요약:
> - Scope: **F4 4-way store 단독**
> - Measurement: 5회 bench + NCU detailed 1회
> - Accept: median ≤ 0.011108 ms 유지 + correctness PASS
> - Reject: median > 0.011108 × 1.005 ms 또는 correctness FAIL → 즉시 롤백
> - Fallback(다음 iter): A6 → G6 순차 적용"

### 라운드 2 (확인)

**PM:**
> "마지막 확인. F4 구현에서 lane < 4 분기가 B200에서 실제로 divergence 없이 predicated execution으로 처리되는지, 그리고 `res_a/b/c/d`·`qs_a/b/c/d`가 warp-wide로 broadcast된 값이 맞는지 코드로 재확인했나?"

**Assistant:**
> "확인했다. `solution/cuda/kernel.cu` line 175–189에서 `__shfl_down_sync`로 butterfly 후 line 186–189에서 `__shfl_sync(0xffffffff, *, 0)`로 **전 lane에 값 일치**. 따라서 `lane < 4` 조건은 **값 생성 후** 적용되므로 lane 1~3도 `res_*`, `qs_*` 값을 갖는다. B200에서 `if (lane < 4)` predicate는 SASS에서 `@P0 STG.E.U16` 형태로 변환되어 **branch 없는 predicated store**가 된다.
>
> 결론: correctness-safe, divergence-free. **APPROVED 수용 준비 완료**."

---

## PM 최종 판정

### ✅ **APPROVED: F4 Output Store 4-way Lane Distribution (단독)**

**승인 조건 요약:**
1. 변경은 `solution/cuda/kernel.cu` line 209–214 **1곳만** (lane 0 단일 if → lane < 4 분산).
2. 벤치 5회 median + 54-workload correctness + NCU detailed 1회 측정.
3. Accept: median ≤ 0.011108 ms.
4. Reject (즉시 롤백): median > 0.011108 × 1.005 ms 또는 correctness FAIL.
5. 성공/부분성공 무관하게 A6/G6는 **다음 iter**로 분리.

**현재 iter 스코프에서 제외:**
- A6 (SMEM carveout=0) — 다음 iter 후보
- G6 (`__builtin_assume`) — 다음 iter 후보
- H2/F3 — 조사 단계 후 추후 재검토
- B1/B5/D5 계열 — 재시도 금지(현 harness에서 실패)

**기대 결과:**
- E2E latency: 0.011108 ms → 0.0106~0.0110 ms (–0.2~–0.5%).
- NCU Duration: 30.85 µs → 30.0~30.5 µs.
- Phase 4 (0.009 ms)까지는 여전히 거리가 있으나, **lane idle 해소가 누적 기반**이 된다.

**구현/측정 세션(다음 단계):**
- Step 3: F4 구현 (line 209–214 편집).
- Step 4: pack → modal bench 5회 → NCU detailed 1회.
- Step 5: 판정 → 유지/롤백 결정 → log.md에 기록.

---

**APPROVED.**
