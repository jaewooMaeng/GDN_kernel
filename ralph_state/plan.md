# Iteration #7 Step 1~2 계획

## 현재 기준선

- accepted benchmark 기준선은 `log.md` 기준 median `0.012920 ms`다. `ralph_state/latest_latency.txt`의 `0.016806 ms`는 iter #6 rejected run 값이지 accepted baseline이 아니다.
- current accepted large-batch shape는 `solution/cuda/kernel.cu:265-300` 기준 `batch_size >= 32 -> split_factor = 4 -> gdn_decode_kernel<8>`이다.
- current accepted profile은 `log.md`의 rollback 후 NCU 요약 기준 `Duration 31.46 us`, `Issue Slots Busy 21.73%`, `Achieved Occupancy 39.79%`, `Registers/thread 56`, local spill `0`, `L1 hit 7.86%`, `L2 hit 1.76%`다. 즉, 병목은 여전히 low-issue / reg-limited occupancy / poor-cache-hit 성격이다.
- 현재 코드 사실관계:
  - state row load는 이미 `ld.global.nc.v4.f32`다. 근거: `solution/cuda/kernel.cu:33-39`, `132-140`.
  - `q/k` load는 아직 plain `uint2` global load 후 bf16 unpack 경로다. 근거: `solution/cuda/kernel.cu:84-106`.
  - host-side L2 persistence는 rollback 후 `hitProp=Persisting`, `missProp=Normal` 상태다. 근거: `solution/cuda/kernel.cu:240-247`.
- 최근 반복된 실패 패턴:
  - host-side soft hint A2/A5/A6는 모두 회귀했다.
  - `RPW=4` dead-prefetch 계열은 helperization과 physical split 둘 다 회귀했다.
  - minimal cluster q/k-share-only, standalone async staging, standalone 256-thread large-batch도 각각 회귀했다.
- 따라서 이번 iteration은 launch shape, split policy, barrier topology, host-side cache policy를 건드리지 않는 좁은 kernel-body 변경만 우선 검토하는 편이 맞다.

## Step 1 정리

### 후보 1. A3 `q/k` only `__ldg` read-only load 검증 + 적용 (1순위)

- 요지:
  - `solution/cuda/kernel.cu:84-106`의 `q/k` packed `uint2` load만 read-only helper로 치환한다.
  - state loop, `s_v` barrier, split policy, `launch_bounds`, L2 persistence는 유지한다.
- 근거:
  - 현재 state read path는 이미 A4 상태라서, 같은 메모리 축에서 source-level로 남아 있는 직접 레버는 `q/k` plain load 쪽뿐이다.
  - NCU는 low-issue / reg-limited occupancy / poor-cache-hit를 보여주지만, 최근 더 큰 구조 변경은 전부 회귀했다. 이번엔 actual load opcode를 건드릴 수 있는 가장 좁은 delta가 필요하다.
  - `q/k`는 footprint가 작고 같은 `qkh` 기준으로 반복 사용되므로, generic global load보다 read-only path가 유리하다면 B64 `Duration(us)`를 소폭이라도 줄일 여지가 있다.
- 실행 전제:
  - benchmark path에 flag를 넣지 않고, standalone `cuobjdump`/SASS 확인으로 `q/k` load opcode가 실제로 바뀌는지 먼저 본다.
  - SASS상 opcode 변화가 없거나 register가 `56`보다 늘면 runtime 측정 없이 즉시 중단한다.
- 리스크:
  - `q/k` traffic 비중이 작아서 leverage가 제한적일 수 있다.
  - helperization만으로도 live range가 바뀌어 register pressure가 늘 수 있다.
  - `__ldg`가 Blackwell codegen에서 이미 generic load와 동일하게 내려가면 사실상 no-op일 수 있다.

### 후보 2. A4 `q/k` only inline PTX `ld.global.nc` (후보 1 실패 시 다음 순번)

- 요지:
  - 후보 1에서 `__ldg`가 opcode를 못 바꾸는 경우에만, `q/k`의 8-byte packed load를 inline PTX read-only/non-coherent 경로로 강제한다.
- 근거:
  - state 쪽에서는 이미 `ld.global.nc.v4.f32`를 쓰고 있으므로, 같은 계열의 explicit opcode 제어를 `q/k`에 한정해서 확인할 수 있다.
  - host-side policy 실험과 달리 actual load instruction을 바꾸므로 falsifiable하다.
- 리스크:
  - `__ldg`보다 구현 리스크가 높고, one-iteration scope가 쉽게 커질 수 있다.
  - q/k에 `nc`를 넣는 것이 read-only cache residency에 꼭 유리하다고 장담할 수 없다.
  - 후보 1과 같은 iteration에 섞으면 원인 분리가 안 된다.

### 후보 3. block-wide async pipeline / cluster 재도전 (이번 iteration 보류)

- 요지:
  - low-issue와 bytes-in-flight를 정면으로 건드리는 방향이긴 하지만, 이번 iteration에는 넣지 않는다.
- 근거:
  - standalone async staging, minimal cluster q/k-share-only, standalone 256-thread가 이미 각각 회귀했다.
  - 남은 구조 개선은 `q/k` reduction 1회화, producer-consumer pipeline, cluster sync hiding 같은 결합형 안인데, 이는 한 iteration 범위를 넘기 쉽다.
- 리스크:
  - regression 폭이 클 가능성이 높다.
  - source churn이 커져 원인 추적이 어려워진다.

### Step 1 결론

- 이번 iteration #7의 1순위는 `q/k` plain global load만 read-only path로 재정의하는 A3 좁은 안이다.
- 단, 이것은 `offline SASS gate`를 통과할 때만 benchmark 단계로 넘긴다.
- 후보 2인 inline PTX A4는 이번 iteration에 섞지 않는다. A3가 no-op일 때의 다음 순번으로만 둔다.
- host-side soft hint, `RPW=4` dead-prefetch, standalone cluster/async/256-thread 계열은 이번 iteration 후보에서 제외한다.

## Step 2 PM 검토 대화

### 1차 검토

**PM**

> (a) 프로파일 근거를 보면 주병목은 state streaming을 동반한 low-issue / reg-limited occupancy다. `q/k` load만 건드려서 `Duration(us)`가 의미 있게 줄까?
> (b) 한 iteration에 과한지는 아닌데, upside가 너무 작은 실험이면 그냥 노이즈일 수 있다.
> (c) 최근 실패 대부분이 "작아 보이는 수정이 codegen을 흔들어 회귀"한 케이스였다. 이번에도 그 위험이 있지 않나?

**엔지니어**

> 맞다. 이번 안은 high-upside card가 아니라 "남아 있는 가장 좁은 opcode-level card"다.
> 현재 state row load는 이미 `ld.global.nc.v4.f32`이고, host-side L2 hint 계열도 A2/A5/A6로 세 번 확인했는데 전부 회귀했다. cluster/async/256-thread도 각각 독립적으로 regress했다.
> 그래서 iteration #7은 state path나 launch shape를 또 흔드는 대신, 아직 plain load인 `q/k`만 겨냥해 source delta를 최소화하는 편이 더 낫다.

**PM**

> 그래도 `A3 + offline SASS + 필요하면 A4`까지 적어두면 한 iteration 범위가 벌어진다. 원인 분리도 흐려진다.

**엔지니어**

> 그 지적이 맞다. 계획을 더 좁히겠다.
> 이번 iteration의 실제 변경안은 **A3 `q/k` only `__ldg` helper** 1개로 고정하겠다.
> offline SASS 확인은 변경 전후 opcode와 register를 보는 gate일 뿐, 두 번째 최적화로 세지 않겠다.
> `A4 inline PTX`는 이번 iteration에서 구현하지 않고, A3가 no-op일 때 다음 iteration 후보로 넘기겠다.

### PM 수정 요구

**PM**

> 좋다. 그러면 다음 조건을 계획에 명시해라.
>
> 1. `solution/cuda/kernel.cu:84-106`의 `q/k` load 외에는 건드리지 마라.
> 2. state load helper(`ld_global_nc_f4`), hot loop 산술, `s_v` barrier, split policy, `launch_bounds`, L2 persistence는 그대로 둬라.
> 3. standalone SASS gate에서 `q/k` load opcode 변화가 없거나 register가 `56`을 넘으면 pack/bench 자체를 하지 말고 중단해라.
> 4. runtime 측정으로 넘어가더라도 B64 decision gate를 먼저 보고, 최종 채택 판정은 full benchmark median + correctness로만 해라.

**엔지니어**

> 수용한다.
> 수정된 실행안은 다음과 같다.
>
> 1. 변경 범위는 `q/k` packed load helper 단일 지점으로 한정한다.
> 2. `state`/`v`/gate/load policy/dispatch는 보존한다.
> 3. SASS gate 실패 시 runtime 실험을 생략한다.
> 4. runtime 실험을 하더라도 B64 gate는 veto일 뿐, 최종 판정은 full benchmark median과 correctness만 사용한다.

### 2차 검토

**PM**

> (a) 프로파일·근거 측면에서는 동의한다. 지금 남은 사실 기반 low-risk 카드는 "state는 이미 A4, `q/k`는 아직 plain load"라는 비대칭을 직접 건드리는 안뿐이다.
> (b) 한 iteration에 과한지도 아니다. kernel-body delta를 `q/k` load helper 하나로 묶었고, 나머지는 검증 gate일 뿐이다.
> (c) 회귀 리스크는 여전히 있지만, `register > 56` 차단과 B64 veto를 걸면 통제 가능한 수준이다.

**PM**

> 최종 승인 조건은 다음 두 줄로 요약된다.
>
> 1. iteration #7의 실행안은 **A3 `q/k` only `__ldg` read-only load` 단독안**이다.
> 2. `A4 inline PTX`, cluster, async pipeline, host-side memory policy 재실험은 이번 iteration에 섞지 않는다.

**엔지니어**

> 동의한다.
> iteration #7은 `q/k` plain load를 read-only path로 바꾸는 좁은 안 하나만 실험 대상으로 삼겠다.
> opcode 변화와 register 유지가 확인되지 않으면 runtime 단계로 넘기지 않는다.

APPROVED
