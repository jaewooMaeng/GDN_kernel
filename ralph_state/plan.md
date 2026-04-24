# Iteration #5 Step 1~2 계획

## 현재 기준선

- accepted 기준선은 `0.012920 ms` median 이다. 이 값이 다음 iteration 판정 기준이다.
- accepted large-batch 기준 NCU는 `gdn_decode_kernel<8>`, `Grid Size=2048`, `Duration=31.97 us`, `Registers/thread=56`, `Achieved Occupancy=40.11%`, `Issue Slots Busy=21.98%`다. 즉 현재 병목은 single-pipeline 포화가 아니라 low-issue / reg-limited occupancy / tail effect 쪽이다.
- 현재 코드 사실관계:
  - `batch_size >= 32 -> split_factor = 4 -> gdn_decode_kernel<8>` 경로가 active다. 근거: `solution/cuda/kernel.cu:265-300`.
  - state read는 이미 `ld.global.nc.v4.f32` inline PTX를 쓰고 있다. 근거: `solution/cuda/kernel.cu:33-40`, `132-162`.
  - `missProp`은 아직 `cudaAccessPropertyNormal`이다. 근거: `solution/cuda/kernel.cu:224-247`.
- `ralph_state/latest_latency.txt`의 `0.029683 ms`는 accepted baseline이 아니라, 직전 rejected probe(A12 helperization gate) 값이다. 다음 iteration의 기준선으로 사용하면 안 된다.

## Step 1 정리

### 후보 1. `RPW=4` 경로 물리 분리 후 dead prefetch 제거 재시도

- 요지:
  - `ROWS_PER_WARP=4`에서만 dead prefetch를 제거하되, `gdn_decode_kernel<8/16>` codegen을 건드리지 않도록 아예 물리적으로 분리한다.
  - 구체적으로는 `rows_per_warp == 4`일 때만 별도 커널/별도 특수화 경로로 보내고, large-batch `gdn_decode_kernel<8>` 본문은 byte-level로 보존하는 방향이다.
- 근거:
  - 현재 코드에서 `solution/cuda/kernel.cu:137-140`는 `next_a..d`를 무조건 preload한다.
  - 하지만 `solution/cuda/kernel.cu:143-163`의 루프는 `vi_off += 4`이고, `ROWS_PER_WARP=4`면 정확히 1회만 돈다.
  - 따라서 `RPW=4`에서는 `next_*` 4개 row가 실제 계산에 소비되지 않는 dead prefetch다.
  - avg/median 최종 판정 기준상 small/medium batch 다수가 섞이는 전체 54 workload에서 이 낭비를 줄이는 레버리지가 있다.
  - 직전 A12 실패 원인은 아이디어 자체보다 helperization이 `RPW=8/16` codegen까지 흔들었을 가능성이 더 크다.
- 기대 효과:
  - small-batch path의 불필요한 state read를 직접 줄여 avg/median 개선을 노릴 수 있다.
  - large-batch path를 물리적으로 보존하면 auxiliary metric인 B64 `Duration(us)` 회귀도 방지하기 쉽다.
- 리스크:
  - 잘못 구현하면 dispatch/templating만 바꿔도 `gdn_decode_kernel<8>` codegen이 다시 흔들릴 수 있다.
  - 본질적으로 small-batch 중심 안이라 B64 `Duration(us)`를 직접 낮추는 안은 아니다.
  - 커널 수가 늘면 host launch 관리가 약간 복잡해진다.

### 후보 2. `missProp=Streaming` 단독 host-side 실험

- 요지:
  - `setup_l2_persistence()`의 `missProp`만 `Normal -> Streaming`으로 바꾸는 매우 좁은 host-side 실험이다.
- 근거:
  - 구현이 작고 correctness 리스크가 낮다.
  - current kernel shape와 register pressure를 그대로 둔 채 memory policy만 조정할 수 있다.
- 리스크:
  - 현재 `hitRatio`가 높고 state load도 이미 `ld.global.nc.v4.f32`라 leverage가 작다.
  - 남은 격차(`0.012920 -> <0.009 ms`) 대비 개선 폭이 너무 작아 Modal 노이즈에 묻힐 가능성이 높다.
  - auxiliary metric/B64 duration을 유의미하게 움직일 근거가 약하다.

### 후보 3. large-batch 전용 producer/consumer async pipeline

- 요지:
  - `gdn_decode_kernel<8>` large-batch path에 한해 producer warp + consumer warps 구조로 state row prefetch를 숨기는 더 강한 구조 변경이다.
- 근거:
  - accepted NCU가 `Issue Slots Busy=21.98%`, `Registers/thread=56`, `Achieved Occupancy=40.11%`라 bytes-in-flight와 scheduler utilization을 동시에 끌어올릴 필요는 있다.
  - B64 `Duration(us)`를 직접 겨냥하는 후보라는 점은 분명하다.
- 리스크:
  - 이번 세션에서 이미 standalone cluster, standalone `cp.async` shared staging, 256-thread 확대가 모두 후퇴했다.
  - producer/consumer + barrier 설계는 한 iteration 치고 범위가 크고 회귀 위험이 높다.
  - correctness와 tail-latency 분산까지 흔들 가능성이 크다.

### Step 1 결론

- 이번 iteration #5의 1순위 후보는 **후보 1: `RPW=4` 경로 물리 분리 후 dead prefetch 제거 재시도**로 둔다.
- 후보 2는 fallback 후보로만 남긴다.
- 후보 3은 Phase 4용 구조적 카드로 보류한다.
- 추가 정리:
  - 문서상 후보로 남아 있는 `A3/A4 read-only load path 검증`은 현재 코드에 이미 반영돼 있으므로 이번 iteration 후보에서 제외한다.
  - 이번 iteration의 핵심은 “새 최적화”보다 “large-batch codegen을 건드리지 않는 격리 방식”이다.

## Step 2 PM 검토 대화

### 1차 검토

**PM**

> (a) 보조 목표가 B64 NCU `Duration(us)` 감소인데, 왜 large-batch 구조안이 아니라 small-batch `RPW=4`를 1순위로 잡았나?  
> (b) 한 iteration에 과하지 않은가? 커널 분리까지 가면 생각보다 범위가 커질 수 있다.  
> (c) 직전 A12가 이미 실패했는데, 같은 아이디어 재시도로 보인다. 회귀 리스크를 어떻게 다르게 통제할 건가?

**엔지니어**

> (a) 맞다. 다만 final judge는 여전히 full benchmark avg/median + correctness다.  
> 현재 accepted NCU는 B64에서 `31.97 us`이지만, 같은 accepted 코드 안에 `RPW=4` dead prefetch라는 코드상 확정 낭비가 남아 있다. `solution/cuda/kernel.cu:137-140`의 `next_*` preload는 `RPW=4`에서 소비되지 않는다.  
> 반면 large-batch 직접 개선안은 최근 시도한 standalone cluster, standalone async staging, 256-thread 확대가 모두 후퇴했다.

**PM**

> 그래도 A12는 이미 실패했다. “helperization이 codegen을 흔들었을 수 있다”는 건 추정 아닌가?

**엔지니어**

> 추정이 맞다. 그래서 같은 형태로는 재시도하지 않는다.  
> 이번 승인안은 “dead prefetch 제거” 자체보다 “`RPW=4` path를 물리적으로 분리해 `gdn_decode_kernel<8>`를 보존한다”가 핵심이다.  
> 즉 재시도 포인트는 최적화 아이디어가 아니라 격리 방식이다.

### PM 수정 요구

**PM**

> 좋다. 그런데 두 가지를 더 못 박아라.
>
> 1. `gdn_decode_kernel<8/16>` 본문과 launch policy는 이번 iteration에서 수정 금지다.  
> 2. `A5` fallback을 같은 iteration에 섞지 마라. 이번에는 단일 변경만 보겠다.

**엔지니어**

> 수용한다.
>
> 수정된 계획:
> 1. 변경 범위는 `rows_per_warp == 4` dispatch와 그 전용 커널/전용 특수화로 한정한다.
> 2. `solution/cuda/kernel.cu:265-300`의 large-batch dispatch 정책(`batch_size >= 32 -> split_factor = 4 -> gdn_decode_kernel<8>`)은 유지한다.
> 3. `gdn_decode_kernel<8/16>` 수학식, prefetch depth, state load opcode, host memory policy는 이번 iteration에서 건드리지 않는다.
> 4. `A5 missProp=Streaming`은 이번 iteration 구현에 포함하지 않는다.

### 2차 검토

**PM**

> (a) 프로파일·근거: accepted B64 NCU 병목은 large-batch에 있지만, 코드상 dead work가 명확한 지점은 `RPW=4`다. final judge가 avg/median인 점까지 포함하면 우선순위로 이해할 수 있다.  
> (b) 한 iteration에 과한지: large-batch 경로를 보존하고 `RPW=4` 전용 분리만 하면 과하지 않다.  
> (c) 회귀 리스크: “직전 실패 아이디어 재시도”가 아니라 “large-batch codegen 무변경을 전제로 한 격리 재설계”로 정의하면 통제 가능하다.

**PM**

> 단, 실행 조건은 아래와 같이 제한한다.
>
> 1. Step 3에서는 먼저 `RPW=4` 전용 경로만 분리하고, `gdn_decode_kernel<8>` 경로는 의도적으로 untouched 상태를 유지한다.  
> 2. 측정 단계에서는 `eaf0a285-447c-4432-8e68-d287acc3cb08` B64 guard를 veto로 쓴다. B64가 recent accepted band(`~0.021~0.024 ms`)를 벗어나면 avg가 소폭 좋아 보여도 즉시 롤백한다.  
> 3. full benchmark 최종 판정은 5회 median + correctness로만 한다.  
> 4. 이번 iteration에서는 `A5`, cluster, async pipeline, launch policy 변경을 절대 섞지 않는다.

**엔지니어**

> 동의한다.  
> iteration #5 실행안은 **`RPW=4` 경로 물리 분리 + dead prefetch 제거 단일안**으로 고정한다.  
> auxiliary metric 우선순위를 반영해 B64 guard를 veto 조건으로 두고, large-batch path는 보존한다.

## 최종 결론

- iteration #5 Step 1 결론:
  - 1순위 후보: **`RPW=4` 경로 물리 분리 후 dead prefetch 제거 재시도**
  - fallback: `A5 missProp=Streaming` 단독안
  - 보류: large-batch producer/consumer async pipeline
- iteration #5 Step 2 결론:
  - PM 요구에 따라 이번 iteration 범위를 `RPW=4` 전용 경로로 축소했다.
  - `gdn_decode_kernel<8/16>` 본문, load opcode, split policy, host memory policy는 유지한다.
  - B64 guard workload를 veto로 사용하는 조건부 실행안으로 확정했다.

APPROVED
