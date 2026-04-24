# Iteration #6 Step 1~2 계획

## 현재 기준선

- accepted 기준선은 full benchmark `median 0.012920 ms`다. `ralph_state/latest_latency.txt`의 `0.025462 ms`는 iter #5의 rejected B64 gate 값이지 기준선이 아니다.
- 현재 accepted large-batch 기준 NCU는 `gdn_decode_kernel<8>`, `Duration 31.97 us`, `Grid Size 2048`, `Registers/thread 56`, `Achieved Occupancy 40.11%`, `Issue Slots Busy 21.98%`다. 즉 single-pipeline 포화보다 low-issue / reg-limited occupancy / cache hit 저조가 더 큰 문제다.
- 현재 코드 사실관계:
  - state row load는 이미 `ld.global.nc.v4.f32`다. 근거: `solution/cuda/kernel.cu:33-39`, `132-140`.
  - `q/k`는 여전히 plain `uint2` global load 후 bf16 unpack 경로다. 근거: `solution/cuda/kernel.cu:84-106`.
  - host-side L2 persistence는 `hitProp=Persisting`, `missProp=Normal`이다. 근거: `solution/cuda/kernel.cu:240-247`.
  - large-batch dispatch는 여전히 `batch_size >= 32 -> split_factor = 4 -> gdn_decode_kernel<8>`이다. 근거: `solution/cuda/kernel.cu:265-300`.
- 직전 iter #5의 `RPW=4` 물리 분리 + dead prefetch 제거는 B64 gate가 `0.025462 ms`까지 악화되어 롤백됐다. 같은 계열은 우선순위를 내린다.

## Step 1 정리

### 후보 1. A5 standalone `missProp=Streaming`

- 요지:
  - `setup_l2_persistence()`에서 `attr.accessPolicyWindow.missProp`만 `cudaAccessPropertyNormal -> cudaAccessPropertyStreaming`으로 바꾸는 host-side 단일 변경이다.
- 근거:
  - current kernel body와 launch/codegen을 건드리지 않으면서 메모리 정책만 바꿀 수 있다.
  - state load는 이미 `ld.global.nc.v4.f32`이므로, 같은 “state read path” 계열에서 아직 남은 좁은 레버는 `missProp`뿐이다.
  - accepted NCU에서 DRAM/L2가 포화가 아닌데도 cache hit가 매우 낮다. 이 상황에서 `missProp=Streaming`은 state miss가 다른 캐시 라인을 덜 오염시키는지 확인할 가치가 있다.
  - 최근 실패들이 대부분 kernel body 또는 launch shape 변경에서 나왔으므로, 이번엔 `gdn_decode_kernel<8>` codegen을 보존하는 쪽이 더 안전하다.
- 기대 효과:
  - B64 `Duration(us)`를 직접 크게 줄인다고 장담할 수는 없지만, codegen 리스크 없이 memory behavior만 좁게 확인할 수 있다.
  - full benchmark avg/median 판정 전 B64 gate로 빠르게 veto 하기 쉽다.
- 리스크:
  - A2 standalone cache hint, A6 carveout 같은 host-side 힌트도 이미 회귀했다. 즉 “host-side라서 안전하다”는 뜻이지 “성능상 유리하다”는 뜻은 아니다.
  - 개선 폭이 작으면 Modal 노이즈에 묻힐 수 있다.
  - B64 duration이 소폭 좋아져도 full 54-workload avg/median이 나빠지면 최종적으로 버려야 한다.

### 후보 2. `q/k` read path 명시적 read-only 경로 검증

- 요지:
  - 현재 state는 이미 `ld.global.nc.v4.f32`이므로, A3/A4를 다시 본다면 state 재적용이 아니라 `q/k` 또는 소형 scalar load 쪽을 명시적으로 read-only path로 바꾸는 방향이어야 한다.
- 근거:
  - `solution/cuda/kernel.cu:84-106`의 `q/k` load는 plain global load다.
  - 최근 로그와 workflow가 A3/A4를 계속 후보로 남겼지만, 현 코드 기준으로 “state load A4 재적용”은 신규성이 없다.
- 리스크:
  - `q/k` footprint는 state보다 훨씬 작아서 B64 `Duration(us)` leverage가 제한적일 수 있다.
  - 실제 load opcode 변화가 확인되지 않으면 source churn만 늘어난다.
  - SASS/objdump 확인 없이 바로 넣으면 또 “아이디어는 맞는데 codegen 사실은 불명확”한 반복이 된다.

### 후보 3. benchmark path 밖 codegen 기준선 재확인

- 요지:
  - benchmark runtime path에 flag를 얹지 않고, standalone build/objdump로 accepted `gdn_decode_kernel<8>`의 `reg=56`, spill `0`, load opcode 사실관계를 먼저 고정한다.
- 근거:
  - iter #1의 G5-lite는 benchmark path에 ptxas flag를 넣는 방식이라 성능을 망쳤다.
  - 이후 구조 변경을 다시 할 때는 “정말 register를 깎았는지”, “opcode가 바뀌었는지”를 benchmark 밖에서 먼저 보는 편이 낫다.
- 리스크:
  - 이 자체는 latency를 줄이지 않는다.
  - 한 iteration의 유일한 실험 카드로 쓰기에는 너무 진단용이다.

### Step 1 결론

- 이번 iteration #6의 1순위 후보는 **후보 1: A5 standalone `missProp=Streaming`** 으로 둔다.
- 후보 2는 즉시 구현 후보가 아니라, **후보 3으로 opcode 변화 가능성을 먼저 확인할 때만** 다음 순서로 올린다.
- 후보 3은 “이번 iteration의 성능 실험”이 아니라 “다음 구조 변경 리스크를 줄이는 보조 작업”으로 분리한다.
- 추가 정리:
  - `RPW=4` dead-prefetch 계열은 iter #4, #5 연속 회귀로 우선순위를 낮춘다.
  - 이미 state load가 `ld.global.nc.v4.f32`인 만큼, 문서상 A3/A4를 그대로 읽고 state 쪽에 다시 적용하는 계획은 제외한다.

## Step 2 PM 검토 대화

### 1차 검토

**PM**

> (a) 프로파일·근거 관점에서 보면 현재 병목은 low-issue / occupancy / cache hit 저조다.  
> 그런데 왜 A5 같은 작은 host-side policy 변경을 1순위로 두나?  
> (b) 한 iteration에 과한지는 아니더라도, 너무 작은 안이라 의미 없는 노이즈 실험이 될 수도 있다.  
> (c) A2, A6 같은 host-side 힌트가 이미 회귀했는데, 이번에도 비슷하게 실패할 가능성이 높지 않나?

**엔지니어**

> 맞다. upside가 큰 카드는 아니다.  
> 다만 지금은 “큰 카드”가 아니라 “`gdn_decode_kernel<8>` codegen을 안 흔들고 B64 duration을 직접 확인할 수 있는 가장 좁은 카드”가 필요하다.  
> 현재 state load는 이미 `ld.global.nc.v4.f32`이고, `RPW=4` 계열도 두 번 연속 실패했다. cluster, async staging, 256-thread 확대도 모두 회귀했다. 이 상태에서 다음 iteration을 또 구조 변경으로 가면 회귀 확률이 더 높다.

**PM**

> 그래도 A5는 효과 근거가 약하다. “missProp=Streaming이면 좋아질 수 있다” 정도면 너무 추상적이다.

**엔지니어**

> 그래서 A5를 “고확률 개선안”이 아니라 “single-change falsifiable probe”로 정의하겠다.  
> current kernel body는 유지하고, `solution/cuda/kernel.cu:246`의 `missProp` 한 필드만 바꾼다.  
> 이 안이 B64 gate에서 바로 나빠지면 memory-policy 단독안의 여지는 접고, 다음엔 offline SASS 기준선 확보 후 `q/k` load path 쪽으로 넘어가면 된다.

### PM 수정 요구

**PM**

> 좋다. 대신 계획을 더 좁혀라.
>
> 1. 이번 iteration에서는 `hitRatio`, `hitProp`, kernel body, dispatch, launch bounds, split policy를 절대 건드리지 마라.  
> 2. `A5`와 `A3/A4`, 또는 `A5`와 offline G5 작업을 같은 benchmark iteration에 섞지 마라.  
> 3. B64 guard를 veto로 쓰되, 최근 accepted band(`~0.021~0.024 ms`)를 다시 넘으면 full benchmark 5회 median까지 가지 말고 즉시 롤백해라.

**엔지니어**

> 수용한다.
>
> 수정된 실행안:
> 1. 변경 범위는 `setup_l2_persistence()`의 `missProp` 단일 필드로 한정한다.
> 2. `solution/cuda/kernel.cu:240-247` 외에는 손대지 않는다.
> 3. `A3/A4` 재검토나 standalone build/objdump는 별도 작업으로 분리하고, 같은 성능 iteration에 섞지 않는다.
> 4. B64 gate가 recent accepted band를 벗어나면 avg가 소폭 좋아 보여도 즉시 롤백한다.

### 2차 검토

**PM**

> (a) 프로파일·근거: 현재 커널은 state load opcode가 이미 A4 상태라, 당장 남은 저위험 memory-policy 레버가 `missProp`이라는 점은 납득된다. B64 duration을 보조 지표로 먼저 확인하겠다는 순서도 맞다.  
> (b) 한 iteration에 과한지: 아니다. 오히려 의도적으로 좁혀서 codegen 리스크를 차단한 실험이다.  
> (c) 회귀 리스크: host-side 힌트의 회귀 전례는 있지만, 이번엔 단일 필드 변경 + B64 veto + full median 판정이라는 통제 장치가 있다.

**PM**

> 단, 다음을 최종 조건으로 건다.
>
> 1. 이번 iteration의 실제 구현안은 **A5 standalone `missProp=Streaming` 단독안** 하나뿐이다.  
> 2. `A3/A4`는 state 재적용이 아니라 `q/k` path 재정의가 필요하며, 그 전에는 standalone codegen/SASS 확인이 먼저다.  
> 3. full benchmark 최종 판정은 5회 median + correctness로만 한다. B64 duration은 우선순위 높은 veto 지표일 뿐 최종 판정 기준을 대체하지 않는다.

**엔지니어**

> 동의한다.  
> iteration #6 실행안은 **A5 standalone `missProp=Streaming`** 으로 고정한다.  
> 다음 순번 후보는 “offline codegen 기준선 확보 후 `q/k` read path 재검토”로 남기되, 이번 iteration에는 섞지 않는다.

## 최종 결론

- iteration #6 Step 1 결론:
  - 1순위 후보: **A5 standalone `missProp=Streaming`**
  - 차순위 후보: offline codegen/SASS 기준선 확보 후의 `q/k` read-only load path 재검토
  - 제외/보류: `RPW=4` dead-prefetch 계열, cluster 단독안, async staging 단독안, benchmark path 내부 G5-lite 재주입
- iteration #6 Step 2 결론:
  - PM 요구에 따라 이번 iteration 범위를 `missProp` 단일 필드 변경으로 축소했다.
  - kernel body, state load opcode, `q/k` load 코드, split policy, launch bounds는 유지한다.
  - B64 guard를 veto로 사용하되, 최종 채택 판단은 full benchmark 5회 median + correctness로 한다.

APPROVED
