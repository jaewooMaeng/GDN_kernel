# Iteration #2 (2026-04-25, claude loop) — Step 1~2 계획 (PM 검토 포함)

> 본 문서는 iter #2 의 Step 1~2 전용이다. **kernel 수정 / pack / bench / NCU 실행 전부 본 세션 범위 밖**이다.
> 모든 수치·라인 번호는 read-only 로 `solution/cuda/kernel.cu`, `ralph_state_claude/iter1_bench_run1.stdout`, `ralph_state_claude/iter1_bench_run2.stdout`, `ralph_state_claude/iter1_precheck_diff.stdout`, `ralph_state_claude/iter_metrics.tsv`, `ralph_state_claude/latest_latency.txt`, `ralph_state_claude/latest_ncu_duration_us.txt`, `ralph_state_claude/last_kernel_sha.txt`, `ralph_logs_claude/log.md`, `git log --oneline`, `git diff HEAD -- solution/cuda/kernel.cu` 로 직접 실증 확인됨.

---

## 0. 직전 iter #1 결과 요약 (2026-04-25, read-only 실증)

### iter #1 에서 시도한 내용 (recap)

- **단일 델타 (APPROVED)**: working tree 에 이미 존재하던 G6 (`kernel.cu:61~64` __builtin_assume 4 줄) + G10 (`kernel.cu:76, 83, 86` beta_g broadcast → per-lane recompute, 3 hunk).
- 신규 kernel edit 0 줄. pre-check diff 는 `ralph_state_claude/iter1_precheck_diff.stdout` 에 저장됨 (plan 의 기술과 byte-identical 일치).

### iter #1 실측 latency (Modal 5-trial avg, 54/54 PASSED)

| run | avg_latency | baseline 대비 | 판정 |
|-----|-------------|---------------|------|
| Run1 | **0.011600 ms** | **+4.42%** | > reject cut 0.011164 ms |
| Run2 | **0.011622 ms** | **+4.63%** | > reject cut 0.011164 ms |
| Run3 | (stdout 38 줄 only — 벤치 미완) | — | abandoned |

출처: `ralph_state_claude/iter1_bench_run1.stdout:47`, `iter1_bench_run2.stdout:47`, `iter1_bench_run3.stdout` (38 줄에서 끊김).

### iter #1 판정: **REJECTED (확정)**

- 2 회 독립 측정 모두 > 0.011164 ms (즉시 롤백 cut) 이자 > 0.011408 ms (모호 영역 상한).
- **+4.4~4.6% regression → Modal noise 범위 아님**. 추가 3-run 재측정조차 불필요한 margin 초과 (iter #1 plan §5 Round 2 '이중 기준' 에 따라 즉시 롤백 대상).
- Run3 완주 안 했지만 **판정에 필요한 샘플 2 개가 이미 reject cut 밖** 이므로 추가 측정 없이 REJECTED 확정.
- Correctness 자체는 둘 다 54/54 PASS → 회귀는 **수치가 아니라 성능** 측면.

### iter #1 이 남긴 전제 파괴

iter #1 plan §2 Round 2 에서 합의된 "Modal noise 범위 ±0.0003 ms (≈2.7%)" 가정이 깨졌다. 실측 분산이 **적어도 ±0.0005 ms 또는 그 이상** 이거나, G6+G10 이 진짜 회귀 (둘 중 하나, 또는 둘 다).

세부 원인 추정 (read-only, SASS 미관측):
1. **G10 부작용**: `float beta_g = beta * g;` 를 warp-wide 로 옮기면 32 lane 모두 FMUL 1 개씩 수행 (entry 영역 한정, hot loop 밖) + `beta_g` live-range 가 hot loop 시작 직전까지만 유지. 원안은 lane 0 1 FMUL + 1 SHFL broadcast 로 "hot loop 내 live" 만 필요. register allocator 관점에서 둘은 실질 동등 — regression 원인이라기엔 희박.
2. **G6 부작용**: `__builtin_assume(threadIdx.x < BLOCK_SIZE)` 가 ptxas 로 내려가면 일부 branch predicate 를 제거하지만, entry 영역의 lane==0 조건 등 다른 predicate 의 스케줄링이 변동하여 hot loop 시작 레지스터 상태가 달라질 가능성. `__launch_bounds__(128,9)` 의 "9 block / SM" 빡빡한 register budget (~56 reg / thread) 과 맞물려 마이너한 변동이 occupancy 경계를 넘는 현상 가능.
3. **Modal variance 자체가 ±0.0005 ms 수준일 가능성**: trial 수 (5) 가 적고, B200 풀 점유 여부·shared tenant 상황 등 외부 변동 영향 가능.

세 가설을 **구분** 하려면 HEAD 재확인 bench (새 5-run + 가급적 NCU) 가 필수.

### HEAD 정보 재확인 (`git log --oneline -1`)

```
4250b14 ralph-claude iter 0001 — avg_latency=0.011108ms
```

- HEAD kernel sha256: `ralph_state_claude/last_kernel_sha.txt` 는 현재 working-tree(=G6+G10 포함) SHA (`e2152fb5...`). HEAD 복원 시 sha 가 바뀔 것이며, 복원 후 재계산 필요.
- HEAD 의 accepted metric: avg 0.011108 ms, NCU Duration 30.85 µs (`ralph_state_claude/latest_latency.txt`, `latest_ncu_duration_us.txt`).

### 누적 상태 (`ralph_state_claude/iter_metrics.tsv`)

- 직전 8 회 연속 plan/impl_session_fail 기록 + 2026-04-25T08:30:05 iter 1 impl_session_fail 1 개 더 (실제로는 본 계획이 커버하는 iter #1 attempt 에 대응).
- **G6+G10 delta 가 working tree 에 여전히 잔존** (`git status -s` 에 `M solution/cuda/kernel.cu` 존재).
- **loop 건강 상태**: 여전히 "정체된 delta + 최근 attempt regression" 이중 부담.

---

## Step 1. 후보군 도출 및 리스크 분석

### 본 iter 의 "절대 원칙" (iter #1 의 교훈 반영)

1. **단일 변경 원칙**: kernel 변경 0 줄이거나 단일 델타.
2. **Hot inner loop 불변**: `kernel.cu:146~219` 한 글자도 건드리지 않는다 (F4, iter #1 공통 교훈).
3. **iter #1 regression 근본 원인 분리 우선**: 새 델타를 쌓기 전에, iter #1 의 regression 이 (a) G6+G10 의 진짜 회귀인지, (b) Modal variance 폭발인지 **먼저 분리**. 분리 없이 또 다른 델타를 올리면 다음 accept/reject 도 같은 함정에 빠진다.
4. **노이즈 모델 재설정**: 기존 ±0.0003 ms 가정이 깨졌으므로 새 데이터로 tolerance 재산정. 그 전까지는 **모든 accept 는 더 큰 margin 으로 요구**.
5. **재시도 금지 테이블 확장**: G6+G10 (본 iter 명목 이름 **N0-reject**) = 재시도 금지 등재. 추가 재평가는 신규 SASS 실증 + 새 assume 설계 동반 시에만.

### 후보 N1 (primary 채택 후보): **rollback-only + HEAD 재확인 (kernel edit 0 줄, 정체된 regression delta 청산)**

**동기**
- 현재 working tree 에 G6+G10 delta 가 남아 있고 iter #1 이 REJECTED 로 종결. 남겨두면 다음 iter 의 Step 3 precheck 에서 "HEAD 와 working tree 의 delta 가 실측 0 이 아님" 이라는 교란 요인이 매번 재발.
- iter #1 이 깨뜨린 노이즈 가정 (±0.0003 ms) 을 재설정하지 않으면 iter #3 이후의 accept/reject 기준 자체가 부실.

**내용 (Step 3~5 에서 수행할 작업, 본 세션에선 계획만)**
- **Step 3 (구현 = 0 줄, 복원만)**:
  - precheck: `git status -s` → `M solution/cuda/kernel.cu` 확인.
  - `git diff HEAD -- solution/cuda/kernel.cu > ralph_state_claude/iter2_precheck_diff.stdout` (rollback 전 상태 증거 저장).
  - `git checkout -- solution/cuda/kernel.cu` 로 HEAD (4250b14) 복원.
  - rollback 후 `git diff HEAD -- solution/cuda/kernel.cu` 가 empty 임을 확인.
  - `shasum -a 256 solution/cuda/kernel.cu` → HEAD 의 sha256 산출. `ralph_state_claude/last_kernel_sha.txt` 에 기록.
- **Step 4 (측정)**:
  - `/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/python scripts/pack_solution.py` → `ralph_state_claude/iter2_pack.stdout`.
  - Modal full bench **5 회** (`iter2_bench_run1~5.stdout`). 각 run 은 5-trial avg (Modal default).
  - 5-run avg-of-avg + median 계산. 기존 iter #1 의 G6+G10 2 run 데이터 (0.011600, 0.011622) 와 함께 **HEAD 는 어느 분포에 속하는지** 분리.
  - NCU detailed 1 회 (`--workload-uuid eaf0a285-447c-4432-8e68-d287acc3cb08 --ncu-set detailed`) → `iter2_ncu.stdout`. HEAD 의 Duration 을 재측정하여 기존 30.85 µs 와 비교.
- **Step 5 (판정 + 노이즈 모델 재산정)**:
  - HEAD 5-run median 이 0.01095 ~ 0.01125 ms 범위 이내 → HEAD baseline = 확정 (iter_metrics.tsv 에 `iter 1 ok` 신규 등재, latest_latency.txt 갱신).
  - HEAD 5-run median 이 0.01130 ms 초과 → "Modal variance 폭발" 가설 강화. G6+G10 regression 판정 보류.
  - HEAD 5-run std 산출 → iter #3 이후의 accept/reject cut 을 **median±3σ** 기반으로 재정의.
  - G6+G10 은 **재시도 금지 (N0-reject)** 등재.
- **commit**: `ralph iter 0002 (claude) — rollback-only, HEAD recalibration avg=<median> ms`.

**근거 — 왜 한 iter 를 rollback-only 에 태우는가**
- (i) iter #1 이 0.011600 / 0.011622 로 **clean reject** 를 이미 보여준 상태. 본 iter 의 primary 산출물은 "새 delta 의 accept" 가 아니라 "**regression delta 청산 + 노이즈 모델 재산정**".
- (ii) 현 상태에서 새 델타 (예: N2 bf16 packed STG.64) 를 바로 태우면 (a) G6+G10 은 여전히 working tree 에 잔존 → 두 변경이 섞임 (F4 패턴 재발), (b) 노이즈 모델이 깨진 상태라 accept 판정이 또 실패할 위험.
- (iii) "rollback 은 변경이 아니라 반환" — kernel edit 0 줄, correctness 리스크 0, 롤백 리스크 0. 단일 변경 원칙에 완벽히 부합.
- (iv) 병행으로 N2 / N4 / N7 은 read-only 조사만 진행해 **iter #3 후보 자료** 를 선제 확보.

**기대 결과 분포**
- HEAD 5-run median: **0.01095 ~ 0.01125 ms** (iter #1 accepted baseline 주변). 이 분포 폭이 곧 Modal noise.
- HEAD NCU Duration: **30.6 ~ 31.1 µs**.
- Correctness: 54/54 (HEAD 는 이미 accepted).
- Phase 4 (< 0.009 ms) 까지 남은 거리: 변함없이 ≈ –0.002 ms.

**리스크 평가**
- **correctness**: 0 (HEAD 로 복원).
- **성능 회귀**: 0 (HEAD 로 복원).
- **측정 실패**: 5 회 중 ≥ 3 회 완주하면 median 산출 가능. Run3 같은 타임아웃/중단 대비 여유 run 2 개 확보.
- **iter 소모**: 1 iter. 상대적 비용은 크나, 다음 iter 부터의 accept 판정 정확도가 크게 올라감 → **투자 회수**.
- **롤백 용이도**: N/A (본 iter 자체가 rollback).

### 후보 N2 (조사 트랙, 코드 변경 0 줄): **lane-0 bf16 out 4× store → packed STG 2× 압축 검토**

**대상 (`kernel.cu:213~218` 직접 확인)**
```cuda
if (lane == 0) {
    out_base[vi_a]   = __float2bfloat16(scale_g * qs_a + scale_qk * res_a);
    out_base[vi_a+1] = __float2bfloat16(scale_g * qs_b + scale_qk * res_b);
    out_base[vi_a+2] = __float2bfloat16(scale_g * qs_c + scale_qk * res_c);
    out_base[vi_a+3] = __float2bfloat16(scale_g * qs_d + scale_qk * res_d);
}
```

**목표**
- 4× `STG.B16` serial 를 2× `STG.B32` (두 bf162 로 패킹) 또는 1× `STG.64` 로 축소.
- 레퍼런스: `__floats2bfloat162_rn(float, float) → __nv_bfloat162`.
- 주소 정렬: out_base = output + batch·NUM_V_HEADS·HEAD_DIM + vh·HEAD_DIM. vi_a = vi_start + vi_off. vi_start = split_id·ROWS_PER_BLOCK + warp_id·ROWS_PER_WARP. ROWS_PER_BLOCK = ROWS_PER_WARP·4. ROWS_PER_WARP ∈ {4, 8, 16} (I1 분기). **vi_a 는 항상 4 의 배수** → `out_base + vi_a` 는 `bf16 × 4 = 8 bytes` 경계 정렬 → STG.64 적용 가능.

**본 iter 판정: 조사만**
- 이유 1: hot inner loop 내부 변경. **단일 변경 원칙 + N1 과 동시 진행 불가**.
- 이유 2: iter #1 교훈 — 겉보기 "clean 한 bf16 패킹" 도 hot loop 의 issue slot 경쟁을 바꿔 regression 가능 (F4 패턴). 별도 iter 에서 accept/reject 판정 필요.
- 본 iter 산출물: `ralph_logs_claude/log.md` iter #2 부록 섹션에 (a) packing SASS 예상 (b) alignment 증명 (c) 기대 효과 (d) 리스크 을 텍스트로만 기록.

**리스크 (본 iter)**: 0 (코드 수정 없음).

### 후보 N3 (조사 트랙, 코드 변경 0 줄): **state==new_state in-place 재평가**

- `solution/cuda/decode_submit_entry.py` + flashinfer-bench harness 의 tensor alloc 패턴 read-only 조사.
- 목적: harness 가 `state` 와 `new_state` 를 같은 pointer 로 allocate 하는 경우가 있는가? 있다면 kernel 내부에서 aliasing 을 활용할 여지가 있는가?
- 본 iter 는 읽기만. 결과는 log.md 부록.

### 후보 N4 (조사 트랙, 코드 변경 0 줄): **출력 bf16 `__stcs` 또는 streaming store 수식어**

- hot loop 끝의 `new_state` 와 `out_base` write 에 `__stcs` / `.cg` 등의 store modifier 를 고려.
- B200 의 sm_100a 에서 `STG.E.CS` 등의 의미와 L1/L2 trade-off 조사.
- read-only 검토. iter #3+ 후보.

### 후보 N5 (배제): **G6+G10 재시도 (N0-reject)** — iter #1 REJECTED

- iter #1 에서 +4.4~4.6% regression (2 회 독립 측정).
- **재시도 금지**. 재평가 조건: (i) SASS-level 로 regression 원인 실증, (ii) 원인 회피형 대체 설계, (iii) 단독 iter 에서 accept 증명. 이 세 조건 모두 충족되기 전엔 봉인.

### 후보 N6 (배제): **shfl_xor butterfly, A5 / A6, B1 / B5, C1~C3, D5, F4, R5~R17, G7, G11, I1+ (split=16 / RPW=2)**

- iter #1 plan §1 workflow.md §9 누적 회귀 이력, log.md F4 실패 사후 분석 (log.md:432~470) 에 의거 **재시도 금지 또는 선결 조건 미충족**.
- 본 iter 제외.

### Step 1 후보 요약표

| 후보 | 우선도 | 기대효과 (E2E) | 리스크 | 구현 시간 | 판정 |
|------|--------|----------------|--------|-----------|------|
| **N1** rollback-only + HEAD 5-run 재확인 (kernel edit 0 줄) | ★★★ | latency 개선 0 (목적은 regression 청산 + 노이즈 모델 재산정) | 최저 (correctness 0, 회귀 0, 롤백 단순) | Step 3: 복원 1 분. Step 4: bench 5 회 + NCU 1 회. | **채택 — iter #2 단일 액션** |
| N2 lane-0 bf16 out packed STG.64 SASS 검토 | ★★ | (조사) | 없음 (코드 변경 없음) | 30~60 분 | **병행 read-only, iter #3 주요 후보** |
| N3 state/new_state in-place 재평가 | ★ | (조사) | 없음 | 30 분 | **병행 read-only, iter #3 부후보** |
| N4 output store modifier (`__stcs`) 조사 | ★ | (조사) | 없음 | 30 분 | **병행 read-only, iter #3 부후보** |
| N5 G6+G10 재시도 (N0-reject) | ☆ | +4.4~4.6% 회귀 확정 | 확정 회귀 | — | **재시도 금지 등재** |
| N6 shfl_xor / A5~A6 / B1~B5 / C1~C3 / D5 / F4 / R5~R17 / G7 / G11 / I1+ | ☆ | — | 높음 / 재시도 금지 | — | **본 iter 제외** |

---

## Step 2. PM 검토 대화

### Round 1 — PM 의 세 가지 기본 질문

**PM (System Architect):**
> "iter #1 이 +4.4~4.6% regression 으로 REJECTED 됐다. 본 iter #2 의 primary action 이 '새 델타' 가 아니라 'rollback + HEAD 재확인' 이라는 주장이다. 세 가지 관점으로 답해라.
>
> (a) **프로파일·근거**: 이 rollback-only 액션이 latency 를 '개선' 하는 것도 아닌데, 왜 iter 를 쓰나? 곧바로 N2 (bf16 packed STG.64) 를 태우지 않는 이유는?
> (b) **한 iter 에 과한가**: 단순 `git checkout --` + 5-run bench + NCU 1 회다. 이건 오히려 **너무 약한 iter** 아닌가? Phase 4 (< 0.009 ms) 목표에 진전이 없다.
> (c) **회귀 리스크**: rollback 자체는 HEAD 복원이라 risk 0 이라는 주장은 이해하나, iter #1 에서 이미 Run3 가 중간에 끊겨버렸다. 5-run 측정 자체가 또 불안정하면 어떻게 할 것인가?"

**Assistant (Tech Lead):**
> "(a) **iter #1 은 '노이즈 모델' 가정을 깨뜨렸다**. iter #1 plan §2 Round 2 에서 합의된 '±0.0003 ms 노이즈' 는 실측으로 부정됐다 (G6+G10 이 +0.0005 ms = +4.4% regression). 이 상태에서 N2 를 곧바로 태우면 두 문제가 겹친다:
>   1. working tree 에 **G6+G10 delta 가 여전히 잔존**. N2 edit 을 추가하면 두 변경이 섞여 F4 패턴 재발.
>   2. 노이즈 모델이 깨진 채 N2 의 accept/reject 판정 → 또 다른 오판 가능.
>
>   rollback-only 로 **청산** 하고 **HEAD 의 진짜 분산** 을 5-run + NCU 로 측정해야 iter #3 부터의 accept/reject 가 의미를 가진다. 본 iter 는 latency 개선이 아니라 **loop 판정 인프라의 재보정**.
>
>   또 — '곧바로 N2' 를 택했을 때의 기회비용: N2 가 accept 되면 좋은 결과지만, reject 되면 (1) G6+G10 여파와 겹쳐 원인 분석 불가, (2) iter #3 이 또 청산 iter 가 돼 오히려 2 iter 낭비. 본 iter 의 '청산' 은 iter #3 의 성공 확률을 크게 올린다.
>
> (b) **'약한 iter' 라는 지적은 일부 맞다**. 그러나 8 회 연속 session_fail + iter #1 regression 의 compound risk 상태에서 한 iter 를 '깨끗한 판정 기준점 확보' 에 쓰는 것이 Phase 4 로의 전체 경로를 단축한다. 증거:
>   - iter #1 plan §0 에 이미 '본 iter 의 진정한 레버리지는 다음 iter 가 존재할 수 있게 만드는 것' 으로 기록됨. iter #1 은 그 목표도 실패 (regression).
>   - 본 iter 는 (i) regression delta 청산 (ii) 노이즈 모델 재산정 (iii) N2/N3/N4 조사 산출물 3 개를 동시에 낸다. 최소 1 iter 에 투입할 만한 산출물.
>
>   Phase 4 진전은 iter #3~+ 에서 회수. 본 iter 없이 iter #3 에서 N2 를 하면 '정체된 G6+G10 잔존 + 노이즈 모델 무시' 로 iter #2 와 동일한 실패 패턴 재발 위험이 높다.
>
> (c) **측정 안정성** 문제에 대해:
>   - iter #1 에서 Run3 가 Modal timeout 으로 끊긴 것은 환경 문제이지 본 kernel 문제 아님.
>   - 본 iter 는 **5 회 시도 중 최소 3 회 완주 → median 산출** 을 accept 조건으로 둔다. 3 회 미만 완주 시 + 2 회 추가 (최대 7 회). 여전히 미달 시 iter #3 로 이월하며 `ralph_state_claude/iter_metrics.tsv` 에 `measurement_unstable` 상태로 등재.
>   - 각 run 은 독립 Modal invocation 이므로 앞 run 의 timeout 이 뒤 run 에 전파되지 않음.
>   - HEAD 의 kernel 코드 자체는 이미 4250b14 로 accepted baseline 이므로 correctness 관점에서 새 리스크 없음."

### Round 2 — PM 의 후속 질문

**PM:**
> "(a) iter #1 regression 이 정말 G6+G10 의 진짜 회귀라는 증거가 있나? Modal variance 가 ±0.0005 ms 인 거라면 HEAD 재확인 bench 에서도 같은 분산이 나올 수 있다. 그 경우 G6+G10 판정은 결국 '확증 불가' 로 끝나지 않나?"

**Assistant:**
> "두 케이스로 나뉘며 **두 결과 모두 본 iter 의 산출물로 가치가 있다**.
>
>   1. **케이스 A — HEAD 5-run median 이 0.01095 ~ 0.01125 ms 범위에 오며 std < 0.0003 ms**:
>      - HEAD 는 여전히 0.01108 ms 근방, 분산 기존 가정 내.
>      - G6+G10 의 0.01160, 0.01162 는 **확정 회귀** (std 3σ 밖).
>      - G6+G10 재시도 금지 확정, 노이즈 모델 ±0.0003 ms 유지.
>
>   2. **케이스 B — HEAD 5-run std ≥ 0.0004 ms 또는 median 이 0.0113+**:
>      - Modal variance 가 커졌거나 환경 변동 존재.
>      - G6+G10 regression 은 'variance 내 회귀 가능' 으로 보류.
>      - 노이즈 모델 재산정 (예: ±0.0005~0.0007 ms). iter #3 이후 모든 accept/reject 기준이 이 새 tolerance 로 업데이트.
>      - 이 경우에도 G6+G10 은 '확정 개선 증거 없음' 이므로 여전히 **보류 = 재시도 금지** (재시도는 변동 가라앉힌 후 별도 iter).
>
>   즉 두 경로 모두 iter #3 이후에 사용할 **정량 데이터** 산출. '확증 불가' 엔딩이 아니라 '노이즈 모델 업데이트' 엔딩.
>
>   추가: NCU Duration 은 Modal wall-clock latency 보다 variance 가 훨씬 작다 (kernel-only, external jitter 무관). HEAD 의 NCU Duration 이 30.85 µs 근처 (±0.3 µs) 인지 재확인하는 것이 G6+G10 의 regression 원인 분리에 결정적.  30.85 µs ± 0.3 인데 G6+G10 의 NCU 를 iter #3 이후에 다시 측정해 비교하는 경로가 남아 있다."

**PM:**
> "(b) N2 / N3 / N4 조사를 본 iter 에 병행하겠다는데, '조사' 가 코드 edit 으로 번질 위험은 어떻게 거나? iter #1 plan Round 2 에서도 같은 우려가 있었다."

**Assistant:**
> "**Scope lock** 으로 박는다.
>   - N2·N3·N4 는 **read-only 전용 트랙**. 본 iter Step 3~5 동안 `solution/` 이하 어떤 파일도 *직접 편집* 하지 않는다.
>   - N2 SASS 조사는 (i) 현 kernel.cu 의 lane 0 store 부분 Read, (ii) 가상 packed 코드 의 mental-SASS, (iii) bf162 alignment 증명, (iv) 리스크 정리. 모두 텍스트.
>   - 조사 결과는 **`ralph_logs_claude/log.md` iter #2 부록** 섹션 1 곳에만 기록. `solution/` 디렉토리 미접촉.
>   - 감사 경로: Step 5 완료 후 `git diff --stat HEAD` 에 `solution/cuda/kernel.cu` 의 rollback (즉 HEAD 와 동일 = no diff) + 메타 파일들 (`ralph_state_claude/*.stdout`, `iter_metrics.tsv`, `last_kernel_sha.txt`, `latest_latency.txt`, `latest_ncu_duration_us.txt`, `plan.md`, `ralph_logs_claude/log.md`) 만 나타나야 함. 그 외 경로 modification 은 즉시 reject.
>
>   iter #1 에서 이 점을 동일하게 합의했고 실제로 `solution/cuda/kernel.cu` 외 `solution/` 편집은 없었다. 본 iter 도 동일 원칙 유지."

**PM:**
> "(c) '단일 변경 원칙' 을 말하면서 동시에 'rollback = 변경 0 줄' 을 primary action 이라 한다. 그러면 이것은 **변경이 0 개인 iter** 인가? 변경 0 개짜리 iter 는 loop 워크플로우상 이상한 상태 아닌가? 다음 iter 가 HEAD 에서 시작할 때 본 iter 가 만든 무엇을 쓰는가?"

**Assistant:**
> "kernel edit 0 줄이 맞다. 그러나 iter 의 산출물은 코드 diff 만이 아니다:
>
>   1. **regression delta 청산**: working tree 의 G6+G10 이 제거돼 iter #3 의 precheck 가 깨끗한 상태에서 시작. iter #1 이후 '현재 file = HEAD' 의 불일치로 발생한 plan-code 괴리가 해소됨.
>   2. **노이즈 모델 5-run 데이터**: `iter2_bench_run[1-5].stdout` 5 개 파일이 iter #3 이후의 accept/reject tolerance 기준 근거가 됨.
>   3. **HEAD NCU Duration 재확인값**: `iter2_ncu.stdout` 이 30.85 µs ±? 의 실측 증거 재확인.
>   4. **N2/N3/N4 조사 산출물**: iter #3 후보 선정 근거 텍스트 (log.md 부록).
>   5. **iter_metrics.tsv 정상 진입 1 행**: 8+ 연속 session_fail 을 끊고 `ok` 상태로 한 행 올림.
>   6. **재시도 금지 테이블에 G6+G10 등재**: 미래 plan 세션들이 동일 실수를 반복하지 않게 막음.
>
>   다음 iter 는 (i) 깨끗한 HEAD 기준점, (ii) 업데이트된 노이즈 모델, (iii) N2 등 구체적 후보 자료 를 가지고 N2 를 단일 델타로 시도할 수 있다."

### Round 3 — 조건 협상

**PM:**
> "좋다. 조건 10 개 붙인다. 전부 동의하면 APPROVED.
>
> 1. **Scope lock**: 본 iter 의 kernel 코드 편집 신규 edit 0 줄. working tree 의 G6+G10 delta 는 `git checkout -- solution/cuda/kernel.cu` 로 **완전 복원**. 복원 후 `git diff HEAD -- solution/cuda/kernel.cu` 가 empty 여야 함.
> 2. **Hot inner loop 불변 원칙 유지**: 본 iter 에서도 `kernel.cu:146~219` 는 전혀 건드리지 않음 (rollback 도 hot loop 를 HEAD 그대로 두므로 자연 충족).
> 3. **Pre-rollback 증거**: Step 3 진입 즉시 `git diff HEAD -- solution/cuda/kernel.cu` 를 `ralph_state_claude/iter2_precheck_diff.stdout` 에 저장 (rollback 전 상태 고증). `shasum -a 256 solution/cuda/kernel.cu` pre/post 값을 둘 다 log.md 에 기록.
> 4. **측정 프로토콜**: pack → Modal full bench 5 회 + NCU detailed 1 회. 각 run stdout 을 `ralph_state_claude/iter2_bench_run[1-5].stdout`, `iter2_ncu.stdout` 로 저장. 5-run avg/median/std 를 `ralph_logs_claude/log.md` iter #2 섹션에 기록.
> 5. **측정 완주 규칙**: 5 회 중 ≥ 3 회 완주하면 median 산출 유효. 2 회 이하면 + 2 회 추가 (최대 7 회). 그래도 3 회 완주 미달 시 iter_metrics.tsv 에 `measurement_unstable` 등재 후 iter #3 이월.
> 6. **HEAD 분포 판정 기준**:
>    - 케이스 A: 5-run median ∈ [0.01095, 0.01125] ms **and** std < 0.0003 ms → HEAD baseline 재확인. G6+G10 재시도 금지 확정, 노이즈 모델 ±0.0003 ms 유지.
>    - 케이스 B: std ≥ 0.0004 ms 또는 median > 0.01130 ms → 노이즈 모델 ±median·3σ/√5 로 재산정하여 tolerance 업데이트. G6+G10 은 여전히 재시도 금지 (증명 불충분).
>    - 케이스 C: median < 0.01090 ms → HEAD 가 실제로 더 빠른 것. latest_latency.txt 를 median 값으로 업데이트.
> 7. **commit 메시지**: `ralph iter 0002 (claude) — rollback-only, HEAD recalibration avg=<median> ms`. median 은 5-run (또는 유효 N-run) median 값.
> 8. **재시도 금지 테이블 등재**: G6+G10 을 log.md 의 "재시도 금지" 섹션 + workflow.md §9 에 **N0-reject (G6+G10)** 으로 등재. 재평가 조건 3 개 명기 (SASS-level 원인 실증 / 대체 설계 / 단독 iter 재증명).
> 9. **N2 / N3 / N4 read-only 격리**: 세 트랙은 코드 수정 금지. 결과는 log.md iter #2 부록으로만 기록. Step 5 완료 후 `git diff --stat HEAD` 에 `solution/cuda/kernel.cu` 수정이 없어야 함 (rollback 후라 HEAD == working tree).
> 10. **롤백 경로**: rollback-only iter 이므로 '회귀' 가 원리상 불가능. 만약 측정 과정에서 3 회 완주 미달 → iter #3 이월 (kernel 상태는 이미 HEAD 라 추가 조치 없음).
>
> 10 개 전부 동의하면 **APPROVED**."

**Assistant:**
> "10 개 모두 동의.
>
> - **Kernel diff scope**: 본 iter 의 kernel.cu 신규 edit 0 줄. `git checkout --` 으로 HEAD 복원. rollback 후 `git diff HEAD -- solution/cuda/kernel.cu` empty 확인.
> - **Hot loop 불변**: `kernel.cu:146~219` 건드리지 않음 (rollback 이므로 HEAD 내용 그대로 유지).
> - **Pre-rollback 증거**: `git diff HEAD -- solution/cuda/kernel.cu > ralph_state_claude/iter2_precheck_diff.stdout`. pre/post sha256 둘 다 log.md 기록.
> - **측정 프로토콜**: pack → 5-run bench + NCU detailed 1 회. stdout 모두 `ralph_state_claude/iter2_*.stdout` 에 저장.
> - **측정 완주 규칙**: 3 회 완주 미만 시 +2 회 추가 (최대 7 회). 여전히 미달 시 iter #3 이월.
> - **HEAD 분포 판정**: 3-케이스 분기 (A=재확인, B=variance 업데이트, C=median 하향 갱신). 각 케이스별 대응 정의.
> - **Commit 메시지**: `ralph iter 0002 (claude) — rollback-only, HEAD recalibration avg=<median> ms`.
> - **재시도 금지 등재**: G6+G10 → N0-reject 로 log.md + workflow.md §9 에 기록. 재평가 조건 3 개 명기.
> - **N2 / N3 / N4 read-only 격리**: 코드 수정 금지, 결과는 log.md iter #2 부록. git diff --stat 으로 감사.
> - **롤백 경로**: rollback-only 이므로 회귀 없음. 측정 실패 시 iter #3 이월.
>
> 다음 iter 후보 (본 iter 결과 기반 분기):
>   - 케이스 A → iter #3 primary = **N2 (lane-0 bf16 packed STG)**, 단일 델타. 백업 후보 N4 (output store modifier).
>   - 케이스 B → iter #3 primary = **노이즈 모델 업데이트 + N2 시도** (새 tolerance 기준으로). 혹은 외부 변동이 심하면 더 안전한 entry-only 후보 (예: G10 단독 재시도를 별도 iter 에서 고려) 로 대체.
>   - 케이스 C → latest_latency.txt 하향 갱신. iter #3 primary = **N2**.
>
> 제외 확인: A5 / A6 / B1 / B5 / C1~C3 / D5 / F4 / R5~R17 / G7 / G11 / I1+ / shfl_xor butterfly / G6+G10 재시도 — 본 iter 에서 전부 제외."

### Round 4 — 코드 재확인 (최종)

**PM:**
> "마지막. 본 iter 실행 전 코드로 재확인해야 할 3 가지 사실을 다시 읊어라."

**Assistant:**
> "세 가지.
>
> 1. `git log --oneline -1 HEAD` → `4250b14 ralph-claude iter 0001 — avg_latency=0.011108ms`. HEAD 는 A4 + H2.5 + I1 + L2-persist accepted baseline. 본 iter 는 이 HEAD 로 working tree 를 **복원**.
> 2. `git diff HEAD -- solution/cuda/kernel.cu` → G6 (`kernel.cu:61~64` __builtin_assume 4 줄) + G10 (`kernel.cu:76, 83, 86` 3 hunk) 만 출력. 본 iter Step 3 `git checkout --` 후 이 diff 가 empty 가 되어야 함.
> 3. `ralph_state_claude/iter1_bench_run1.stdout:47` 및 `iter1_bench_run2.stdout:47` → 각각 `avg latency: 11.600 µs (0.011600 ms)`, `avg latency: 11.622 µs (0.011622 ms)`. 모두 54/54 PASSED 이지만 **reject cut 0.011164 ms 를 초과** → iter #1 REJECTED 확정 근거.
>
> 세 사실 모두 본 세션에서 `git log` / `Read iter1_bench_run*.stdout` / `git diff HEAD --` 로 직접 재확인 완료. iter #1 의 '현재 file 에 보인다 = HEAD 에 있다' 식 착각 없음 ('HEAD 기준 + working tree 의 G6+G10 delta 는 rollback 대상' 으로 명시)."

**PM:**
> "좋다. **APPROVED**."

---

## PM 최종 판정

### ✅ **APPROVED — N1: rollback-only + HEAD 재확인 bench (kernel 신규 edit 0 줄)**

**승인 사항:**
1. **변경 범위**: `git checkout -- solution/cuda/kernel.cu` 로 HEAD (4250b14) 복원. kernel 신규 edit 0 줄. working tree 의 G6+G10 delta 는 완전 제거.
2. 그 외 어떤 `solution/` 파일도 편집 금지 (`solution/cuda/decode_submit_entry.py`, wrapper 포함 전부 불변).
3. **Hot inner loop `kernel.cu:146~219` 불변** (rollback 이 HEAD 유지).
4. Step 3 진입 즉시 pre-rollback 증거 저장: `git diff HEAD -- solution/cuda/kernel.cu > ralph_state_claude/iter2_precheck_diff.stdout`. pre/post `shasum -a 256 solution/cuda/kernel.cu` 둘 다 log.md iter #2 섹션에 기록.
5. **측정**: pack → Modal full bench 5 회 (`iter2_bench_run[1-5].stdout`) + NCU detailed 1 회 (`--workload-uuid eaf0a285-447c-4432-8e68-d287acc3cb08 --ncu-set detailed`, `iter2_ncu.stdout`). 5-run avg/median/std + NCU Duration 을 `ralph_logs_claude/log.md` iter #2 섹션에 기록.
6. **완주 규칙**: 5 회 중 ≥ 3 회 완주 → median 산출 유효. 미만 시 + 2 회 추가 (총 ≤ 7 회). 여전히 미달 시 `measurement_unstable` 등재, iter #3 이월.
7. **HEAD 분포 판정 (3-케이스)**:
   - **A**: median ∈ [0.01095, 0.01125] ms **and** std < 0.0003 ms → HEAD baseline 확정, 노이즈 ±0.0003 ms 유지, G6+G10 재시도 금지 확정.
   - **B**: std ≥ 0.0004 ms 또는 median > 0.01130 ms → 노이즈 모델 재산정, G6+G10 여전히 재시도 금지 (증명 불충분).
   - **C**: median < 0.01090 ms → latest_latency.txt 하향 갱신.
8. **commit 메시지**: `ralph iter 0002 (claude) — rollback-only, HEAD recalibration avg=<median> ms`.
9. **재시도 금지 테이블 등재**: **N0-reject (G6+G10)** → `ralph_logs_claude/log.md` "재시도 금지" 섹션 + `ralph_state_claude/workflow.md §9` (있다면) 에 기록. 재평가 조건 3 개 (SASS 원인 실증 / 대체 설계 / 단독 iter 재증명) 명기.
10. **N2 / N3 / N4 read-only 격리**: 세 트랙 모두 코드 수정 금지, 결과는 log.md iter #2 부록으로만 기록. Step 5 완료 후 `git diff --stat HEAD` 에 `solution/cuda/kernel.cu` 수정이 없어야 함 (rollback 이므로 HEAD == working tree).

**이번 iter 스코프에서 제외 (재시도 금지 또는 선결 조건 미충족):**
- **N0-reject (G6+G10)** ← iter #1 REJECTED 판정 (+4.4~4.6%). 본 iter 의 주요 청산 대상.
- A5 missProp=Streaming, A6 SMEM carveout=0, B1 2-CTA cluster, B5 warp spec, C1 pragma unroll 8, C2 redux.sync PTX, C3 FFMA 수정, D5 CUDA Graph, F4 4-way lane store, R5 RPW=16 async, R6 256-thread, R8~R17, G7 maxrregcount, G11 pack broadcast, I1+ split=16 (RPW=2), shfl_xor butterfly.

**예상 결과:**
- HEAD median latency: 0.01095 ~ 0.01125 ms (기존 0.011108 ms 근방) 또는 케이스 B/C 분기.
- HEAD NCU Duration: 30.6 ~ 31.1 µs (기존 30.85 µs 근방).
- Correctness: 54/54 유지 (HEAD = accepted baseline).
- Phase 4 (< 0.009 ms) 까지 남은 거리: 변함없이 ≈ –0.002 ms (본 iter 는 latency 개선 iter 가 아님).
- **본 iter 의 진정한 레버리지**:
  1. iter #1 의 정체된 G6+G10 delta 청산 + regression 판정 확정.
  2. Modal variance 모델 재산정 → iter #3 이후의 accept/reject 정확도 개선.
  3. N2 / N3 / N4 조사 산출물 → iter #3 후보 선정 근거 확보.
  4. iter_metrics.tsv 의 session_fail 정체 해소 (ok 진입 1 행).
  5. **재시도 금지 테이블 업데이트** 로 미래 plan 세션들의 G6+G10 재시도 오류 방지.

**다음 단계 (본 세션 범위 밖):**
- Step 3: pre-rollback 증거 저장 → `git checkout -- solution/cuda/kernel.cu` → post 상태 확인 (`git diff HEAD`, `shasum`).
- Step 4: `/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/python scripts/pack_solution.py` → Modal full bench 5 회 + NCU detailed 1 회. 모든 stdout 저장.
- Step 5: 3-케이스 분기 판정 → `git commit` (rollback-only iter 로 kernel diff 0, 메타 파일 갱신만 포함). `ralph_state_claude/iter_metrics.tsv` / `last_kernel_sha.txt` / `latest_latency.txt` / `latest_ncu_duration_us.txt` 동기화. log.md 갱신 (G6+G10 reject 기록 + N2/N3/N4 조사 부록 + 재시도 금지 테이블 업데이트).
- (병행 read-only) N2: `solution/cuda/kernel.cu:213~218` lane-0 bf16 store 의 packed STG.64 SASS 예상 + alignment 증명. N3: `scripts/run_modal.py` + harness 의 state/new_state alloc 패턴 확인. N4: `__stcs` / `.cg` store modifier 후보 조사. 세 결과 모두 log.md iter #2 부록 섹션으로 기록.

---

**APPROVED.**
