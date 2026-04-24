# Iteration #2 Step 1~2 계획

## 현재 기준선

- benchmark accepted 기준선: `0.012920 ms` median
- 최근 accepted kernel NCU 기준: `gdn_decode_kernel<8>`, `Grid Size=2048`, `Block Size=128`, `Duration=31.97 us`
- 같은 NCU에서 확인된 핵심 수치: `Issue Slots Busy=21.98%`, `Achieved Occupancy=40.11%`, `Registers/thread=56`, spill `0`, `L1 hit=7.85%`, `L2 hit=1.77%`
- 현재 작업트리 기준 large-batch dispatch는 `batch_size >= 32 -> split_factor = 4 -> ROWS_PER_WARP = 8`이다.
- 현재 커널에는 이미 `ld.global.nc.v4.f32` 기반 state load가 들어가 있으므로, `workflow.md`의 일부 “미시도 후보” 표기는 현 코드와 완전히 일치하지 않는다.

## Step 1 정리

### 후보 A. B2/H2 축소안: large-batch 전용 block-wide async pipeline

- 내용:
  - `batch_size >= 32` 경로만 대상으로 한다.
  - 현재 `split_factor=4`, `BLOCK_SIZE=128`, `__launch_bounds__(128, 9)`, output/state 수학식은 그대로 둔다.
  - 바꾸는 것은 state row fetch 경로 하나뿐이다.
  - 현재 register lookahead 기반 prefetch를 block/warp 단위 `cuda::pipeline` 또는 동등한 `cp.async` 계열 shared double-buffer로 바꿔 bytes-in-flight와 issue overlap을 늘린다.
- 근거:
  - 최신 accepted NCU의 병목은 여전히 single-pipeline saturation이 아니라 `low issue + reg-limited occupancy + bytes-in-flight 부족` 쪽이다.
  - `Duration=31.97 us`, `Issue Slots Busy=21.98%`, `Occupancy=40.11%`, spill `0` 조합은 “연산식 미세 정리”보다 load/issue overlap을 직접 건드리는 편이 맞다는 뜻이다.
  - 최근 실패한 안들 중 graph, cluster, wrapper flag 주입은 각각 harness overhead, sync cost, compile-path 교란이 주원인이었고, 이 후보는 그 세 방향을 피한다.
- 리스크:
  - prior async 계열 실패 이력이 있어 correctness와 latency 회귀 가능성이 높다.
  - shared stage와 wait/commit가 늘면 오히려 register pressure나 barrier cost가 커질 수 있다.
  - 범위를 넓히면 한 iteration 치고 과해진다.

### 후보 B. A6 standalone: shared-memory carveout=0만 단독 적용

- 내용:
  - 커널 body는 유지하고 host launch 쪽에서 `cudaFuncAttributePreferredSharedMemoryCarveout=0`만 건다.
- 근거:
  - shared 사용량이 매우 작고, 최근 accepted NCU에서 `L1/TEX Throughput=59.74%`라 L1 여유 확대 자체는 설명 가능한 방향이다.
  - host-side 단독 변경이라 rollback이 쉽다.
- 리스크:
  - state 접근이 단발성 streaming 성격이 강해 실효가 없을 수 있다.
  - R3에서 `split-local s_v + carveout=0` 조합이 이미 median 후퇴를 냈다. standalone carveout만 남겨도 upside는 작을 가능성이 높다.
  - kernel Duration을 의미 있게 줄일 근거는 후보 A보다 약하다.

### 후보 C. benchmark path 밖의 codegen 기준선 고정

- 내용:
  - benchmark runtime path는 건드리지 않고 standalone build/objdump 경로로 register, spill, 실제 load opcode만 재확인한다.
- 근거:
  - 직전 codex iter #1에서 wrapper flag 주입은 `0.015814 ms` median으로 크게 후퇴했다.
  - 현재 코드/문서 드리프트는 분명히 존재한다.
- 리스크:
  - 이번 iteration의 kernel Duration이나 benchmark latency를 직접 줄이는 변경은 아니다.
  - Step 3 이후의 실제 성능 iteration을 한 번 더 뒤로 미루는 셈이다.

### 이번 Step 1 판단

- 후보 B와 C는 리스크는 낮지만, 현재 목표가 `0.009 ms` 미만이고 recent accepted kernel Duration이 아직 `31.97 us`인 점을 보면 레버리지가 너무 작다.
- 후보 A가 가장 위험하지만, 현재 프로파일의 핵심 지표와 직접 연결되는 유일한 후보이기도 하다.
- 다만 “all-path async pipeline”은 범위가 너무 크므로 그대로는 승인하지 않는다.
- 따라서 Step 2에서는 후보 A를 **large-batch 전용, fetch 경로만 교체하는 축소안**으로 줄여 PM 재검토를 받는다.

### 이번 iteration에서 제외하는 방향

- `A5 missProp=Streaming` 단독:
  - B=64 state footprint는 약 `32 MB`이고 최근 로그의 persisting L2 set-aside는 약 `82.9 MB`다.
  - 현재 `hitRatio`가 사실상 `1.0`으로 잡힐 가능성이 높아, `missProp` 변경만으로는 large-batch 핵심 병목에 미치는 영향이 약하다고 본다.
- cluster / graph / wrapper flag 재주입:
  - 최근 직접적인 회귀 근거가 이미 충분하다.

## Step 2 PM 검토 대화

### 1차 검토

**PM**

> 세 가지만 본다.  
> (a) 프로파일과 코드 근거가 충분한가?  
> (b) 한 iteration에 과하지 않은가?  
> (c) 회귀 리스크를 실제로 통제할 수 있는가?

**엔지니어**

> 초기안은 후보 A를 1순위로 본다. 이유는 분명하다.  
> 최신 accepted kernel은 `gdn_decode_kernel<8>`이고 `Duration=31.97 us`, `Issue Slots Busy=21.98%`, `Occupancy=40.11%`, `Registers/thread=56`, spill `0`이다.  
> 이 조합이면 instruction-form 미세조정보다 bytes-in-flight와 load/issue overlap을 직접 올리는 쪽이 더 맞다.

**PM**

> 근거는 이해했다. 하지만 “async pipeline”은 최근 실패한 계열과 닮아 있다.  
> 현재 형태 그대로면 한 iteration 치고 과하다.

**엔지니어**

> 맞다. 그래서 all-path 변경은 접겠다.  
> 범위를 `batch_size >= 32` current large-batch path로만 자르고, `split_factor=4`, `BLOCK_SIZE=128`, math body, output/new_state store, q/k/v/gate 처리, launch shape는 모두 유지하겠다.  
> 즉 이번 iteration에서 건드리는 것은 state fetch/stage 경로 하나뿐이다.

### PM 불만과 수정 요구

**PM**

> 아직도 넓다.  
> 최근 실패한 것은 대부분 “좋아 보이는 구조 변경을 한 번에 너무 많이 얹은 것”이었다.  
> 다음 세 조건으로 다시 줄여라.

1. small-batch path는 byte-for-byte 동일하게 남길 것
2. cluster, graph, compile flag, register forcing은 금지할 것
3. 성공 판정 전에 회귀 차단 게이트를 명확히 둘 것

### 수정안

**엔지니어**

> 수정안은 다음과 같다.

1. 변경 범위:
   - `batch_size >= 32` path만 수정
   - current `gdn_decode_kernel<8>` 계열만 대상
   - q/k dot, gate scalar, output/new_state 수학식은 그대로 유지
   - state fetch를 shared double-buffer 기반 block-wide async staging으로만 교체
2. 명시적 비대상:
   - small-batch path 미수정
   - cluster 미사용
   - CUDA Graph 미사용
   - wrapper compile flag 미주입
   - `__launch_bounds__`, split 정책, block size 미변경
3. 회귀 차단 게이트:
   - correctness 실패 즉시 전체 롤백
   - ptxas/SASS 기준에서 spill이 생기거나 register가 `56`을 넘으면 전체 롤백
   - quick screen에서 핵심 B=64 workload가 baseline보다 명확히 느리면 full 5회까지 가지 않고 롤백
   - full benchmark는 avg/median 둘 다 기준으로 보되, 최종 채택은 benchmark latency와 correctness가 우선이다
   - NCU Duration은 보조 지표로만 쓰고, `31.97 us`보다 악화되면 채택하지 않는다

### 2차 검토

**PM**

> 다시 세 가지를 본다.  
> (a) 프로파일·근거: low-issue, reg-limited occupancy, spill 0이라는 현재 증거와 후보 A의 연결은 성립한다.  
> (b) 한 iteration에 과한지: 수정안은 large-batch path와 fetch 경로 하나로만 좁혀져 수용 가능하다.  
> (c) 회귀 리스크: small-batch untouched, no cluster/graph/flags, early rollback gate가 명확해졌다.

**PM**

> 후보 B와 C는 왜 승인하지 않는가?

**엔지니어**

> 후보 B는 안전하지만 레버리지가 너무 작고, 후보 C는 이번 iteration의 성능을 직접 움직이지 못한다.  
> 반면 수정된 후보 A는 위험을 제한한 상태에서 kernel Duration을 실제로 건드릴 수 있는 유일한 안이다.

**PM**

> 좋다. iteration #2의 Step 1~2 결론으로 **APPROVED** 한다.

## 최종 결론

- iteration #2의 실행 후보는 후보 A의 축소안 하나로 고정한다.
- 즉, 다음 실제 구현 단계에서는 `batch_size >= 32` current `gdn_decode_kernel<8>` 경로에 한해 state fetch만 block-wide async staging으로 교체한다.
- small-batch path, cluster/graph, compile flag, launch shape, split policy는 건드리지 않는다.
- 최종 판정 기준은 benchmark avg/median latency와 correctness이며, NCU `Duration(us)` 감소는 보조 지표로만 사용한다.
