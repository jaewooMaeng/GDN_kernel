# Iteration #4 Step 1~2 계획

## 현재 기준선

- 현재 accepted 코드 기준 benchmark avg latency는 `0.011108 ms`다.
- 현재 accepted NCU 기준 `kernel Duration`은 `30.85 us`다.
- 현재 dispatch는 `batch_size < 32 -> split_factor = 8 -> gdn_decode_kernel<4>`, `batch_size >= 32 -> split_factor = 4 -> gdn_decode_kernel<8>`이다. 근거: `solution/cuda/kernel.cu:265-300`.
- 현재 커널은 이미 `ld.global.nc.v4.f32` state load, `__launch_bounds__(128, 9)`, lane-0 gate, H2.5 dual-buffer prefetch를 사용 중이다.
- 최근 실패로 제외해야 하는 안:
  - `A6` standalone `PreferredSharedMemoryCarveout=0`
  - `B1` minimal 2-CTA cluster q/k 공유
  - `B2/H2` current `gdn_decode_kernel<8>` 경로의 standalone async shared staging
  - `D5` wrapper 내부 CUDA Graph
  - `G5` benchmark runtime path에서의 ptxas flag 주입
  - `F4` output 4-lane 분산 store

## Step 1 정리

### 후보 1. `ROWS_PER_WARP=4` 전용 prefetch depth 축소

- 요지:
  - 현재 H2.5 prefetch는 모든 템플릿에서 `curr_a..d`와 `next_a..d`를 무조건 먼저 읽는다.
  - 그런데 `ROWS_PER_WARP=4`에서는 메인 루프가 정확히 1회만 돈다.
  - 따라서 `next_a..d`는 load만 되고 소비되지 않는 dead prefetch다.
- 코드 근거:
  - `solution/cuda/kernel.cu:132-140`에서 `next_a..d`를 무조건 preload한다.
  - `solution/cuda/kernel.cu:142-158`에서 루프는 `vi_off += 4`로 진행한다.
  - `ROWS_PER_WARP=4`이면 `vi_off=0` 한 번만 실행되므로, `next_*`는 `curr_* = next_*`로 회전만 되고 실제 `st4_*`로 쓰이지 않는다.
  - 즉 `gdn_decode_kernel<4>`에서는 warp당 4-row useful load + 4-row dead load가 공존한다. state read 기준으로 dead load 비율이 50%다.
  - host dispatch상 `batch_size < 32`는 전부 `rows_per_warp = 4`를 탄다. 근거: `solution/cuda/kernel.cu:265-269`.
- 기대 효과:
  - B<32 경로의 kernel Duration을 직접 줄일 가능성이 가장 높다.
  - benchmark avg/median은 여러 batch가 섞이므로, B64 전용 구조 변경보다 avg 개선에 더 직접적일 수 있다.
  - 변경 범위를 `gdn_decode_kernel<4>`로만 제한하면 current `gdn_decode_kernel<8>` large-batch path를 그대로 보존할 수 있다.
- 리스크:
  - 이번 후보는 현재 보유한 NCU baseline workload(`eaf0a285`, B=64)의 Duration을 직접 줄이는 안은 아니다.
  - 템플릿 분기 방식이 거칠면 `RPW=8/16`까지 codegen이 흔들릴 수 있다.
  - `RPW=4` 경로만 만져도 register allocation이 달라질 수 있으므로 SASS/benchmark 확인 전까지는 가정 금지.

### 후보 2. `A5` standalone `missProp=Streaming`

- 요지:
  - `setup_l2_persistence()`의 `missProp=Normal`만 `Streaming`으로 바꾸는 host-side 실험.
- 근거:
  - 구현/롤백이 가장 쉽고 correctness 리스크가 낮다.
- 리스크:
  - 현재 `state_bytes` 대비 persisting window가 충분히 커 `hitRatio`가 사실상 `1.0`에 가깝다.
  - current bottleneck이 low-issue / occupancy / dead work 쪽이라 leverage가 약하다.
  - kernel Duration 개선 폭이 너무 작아 Modal 노이즈에 묻힐 가능성이 높다.

### 후보 3. `F3` state/new_state alias 가능 여부 조사

- 요지:
  - `new_state == state` alias 허용 여부를 bench/API 관점에서 확인하는 read-only 조사.
- 근거:
  - `decode_submit_entry.py`는 현재 `kernel(q, ..., state, ..., output, new_state)` 형태로 별도 포인터를 그대로 전달한다.
  - `scripts/pack_solution.py`는 packing만 담당하므로 aliasing 계약을 제공하지 않는다.
  - 향후 in-place 계열을 보려면 harness/API 계약부터 확인해야 한다.
- 리스크:
  - 이번 iteration의 kernel Duration을 직접 줄이는 안은 아니다.
  - 조사 결과가 곧바로 next iteration 코드 변경으로 이어진다는 보장이 없다.

### 이번 Step 1 판단

- 후보 2는 너무 안전하지만 남은 격차(`0.011108 -> <0.009`) 대비 레버리지가 약하다.
- 후보 3은 필요하지만 “조사”에 가깝고 이번 iteration의 주 변경안으로 쓰기엔 목표와 거리가 있다.
- 후보 1은 B64 NCU auxiliary metric과 완전히 일치하지는 않지만, 현재 코드에서 **실제로 확인된 dead work**를 줄이는 유일한 중간 규모 후보다.
- 따라서 iteration #4의 1순위 후보는 **`ROWS_PER_WARP=4` 전용 prefetch depth 축소**로 둔다.
- 후보 2(`A5`)는 fallback, 후보 3(`F3`)는 별도 read-only 조사로 분리한다.

## Step 2 PM 검토 대화

### 1차 검토

**PM**

> (a) 보조 목표가 NCU `Duration(us)` 감소인데, 왜 B=64 profile 경로가 아닌 `gdn_decode_kernel<4>`를 1순위로 잡았나?  
> (b) 한 iteration에 과하지 않은가? prefetch 템플릿 분기까지 건드리면 생각보다 넓어질 수 있다.  
> (c) 회귀 리스크를 어떻게 자를 건가?

**엔지니어**

> (a) 지적은 맞다. 다만 이번 후보는 “추정상 좋아 보이는 안”이 아니라, 현재 코드에서 **읽기만 하고 절대 쓰지 않는 load가 존재한다**는 사실에 기반한다.  
> `solution/cuda/kernel.cu:132-140`의 `next_a..d`는 `ROWS_PER_WARP=4`에서는 dead prefetch다.  
> current accepted dispatch에서는 `batch_size < 32`가 전부 이 경로를 타므로, avg latency 관점에서는 직접적인 낭비 제거다.

**PM**

> 그래도 auxiliary metric은 B64 NCU다. 이 안은 그 수치를 못 내릴 수도 있다. user 요청 우선순위와 어긋나는 것 아닌가?

**엔지니어**

> 인정한다. 그래서 이 안을 “B64 Duration 최적화”가 아니라 “benchmark avg/median 최적화 우선, B64 path는 보존”으로 위치시킨다.  
> current `gdn_decode_kernel<8>` large-batch path는 `30.85 us`, `Waves/SM 1.54`, `Achieved Occupancy 42.27%`까지 이미 끌어올린 상태다.  
> 반면 `gdn_decode_kernel<4>` 쪽은 코드만 봐도 dead load가 확정적이므로, 한 iteration 안에서 가장 근거가 강한 kernel-side 절감 포인트는 이쪽이다.

### PM 불만과 수정 요구

**PM**

> 그러면 범위를 더 줄여라.  
> `RPW=4`만 손대고, `RPW=8/16` path는 byte-for-byte 유지한다는 수준으로 못 박아라.  
> 그리고 fallback 후보를 같은 iteration에 섞지 마라.

**엔지니어**

> 수용한다.
>
> 수정된 계획:
> 1. 다음 Step 3의 코드 변경 범위는 `gdn_decode_kernel<4>` prefetch init/rotate 부분만 제한한다.
> 2. `gdn_decode_kernel<8>`와 `gdn_decode_kernel<16>`의 수학식, prefetch depth, store path는 건드리지 않는다.
> 3. `A5`와 `F3`는 이번 iteration 코드 변경에 포함하지 않는다.
> 4. 구현 방식도 `if constexpr (ROWS_PER_WARP == 4)` 또는 `RPW=4` 전용 helper로 한정해 다른 템플릿 codegen 흔들림을 최소화한다.

### 2차 검토

**PM**

> (a) 프로파일·근거: B64 NCU와 직접 맞지는 않지만, `RPW=4` dead prefetch는 코드상 사실이다. 근거는 충분하다.  
> (b) 한 iteration에 과한지: `RPW=4` prefetch path 한정이면 과하지 않다.  
> (c) 회귀 리스크: `RPW=8/16`를 건드리지 않고, `A5/F3`를 섞지 않으면 통제 가능하다.

**PM**

> 단, 다음 측정 단계에서는 반드시 아래 게이트를 지켜라.
>
> 1. 대표 small/medium workload와 B64 guard workload(`eaf0a285-447c-4432-8e68-d287acc3cb08`)로 먼저 스크리닝한다.
> 2. 그 뒤 full benchmark 5회 median과 correctness로 최종 판정한다.
> 3. B64가 악화되면 avg가 약간 좋아 보여도 채택하지 않는다.
> 4. 이번 iteration에서는 `A5`, `F3`, launch policy, split policy를 절대 섞지 않는다.

**엔지니어**

> 동의한다.  
> 이번 iteration #4의 실행 후보는 **`ROWS_PER_WARP=4` 전용 dead prefetch 제거** 단일안으로 고정한다.  
> `A5`는 fallback, `F3`는 별도 read-only 조사로 남겨둔다.

## 최종 결론

- iteration #4 Step 1 결론:
  - 1순위 후보: **`gdn_decode_kernel<4>` 전용 prefetch depth 축소**
  - fallback: `A5` standalone `missProp=Streaming`
  - 별도 조사: `F3` state/new_state alias 계약 확인
- iteration #4 Step 2 결론:
  - PM 요구에 따라 범위를 `RPW=4` path로만 줄였고, 다른 후보와 혼합하지 않기로 수정했다.
  - 최종 승인된 실행안은 **`RPW=4` dead prefetch 제거 단일안**이다.
  - 본 세션에서는 계획만 확정하며, pack/bench/NCU와 커널 수정은 수행하지 않는다.

**APPROVED**
