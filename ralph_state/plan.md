# Iteration #1 Step 1~2 계획

## 현재 기준선

- benchmark 기준선: avg latency `0.012671 ms`
- NCU 기준선: kernel Duration `32.58 us`
- 최종 목표: avg latency `< 0.009 ms`
- 보조 목표: 다음 iteration에서는 NCU kernel Duration 감소 가능성이 높은 방향을 우선 보되, 최종 판정은 benchmark latency와 correctness로 한다.

## 현재 코드/기록 불일치

- 현재 `solution/cuda/kernel.cu`에는 이미 `ld.global.nc.v4.f32` 기반 `ld_global_nc_f4()`가 들어가 있다.
- 현재 `solution/cuda/decode_submit_entry.py`는 이미 `TVM_FFI_CUDA_ARCH_LIST=10.0a`를 설정한다.
- 현재 `solution/cuda/kernel.cu`의 split dispatch는 `batch_size <= 2: 8`, `batch_size < 32: 8`, 그 외 `4`다.
- 즉, `workflow.md`의 후보 우선순위와 “미시도” 상태 일부는 현재 작업트리와 완전히 일치하지 않는다.

위 불일치 때문에, 다음 iteration에서 구조 변경을 바로 넣기 전에 “지금 작업트리 기준으로 무엇이 이미 들어가 있고 무엇이 진짜 미시도인지”를 먼저 고정해야 한다.

## Step 1 정리

### 후보 A. H2/B2 계열 축소안

- 내용: `batch_size >= 32` 경로만 대상으로 block-scope `cuda::pipeline`/`cuda::memcpy_async` 기반 state-row prefetch를 넣어 current per-thread lookahead를 대체한다.
- 근거:
  - 최신 accepted 기준 NCU Duration이 `32.58 us`로 여전히 높다.
  - 최근 기록상 `Issue Slots Busy`가 약 `17.6%`, `Achieved Occupancy`가 약 `30.6%` 수준이라 kernel body의 load/issue overlap 개선 여지가 남아 있다.
  - 단순 split 확대, minimal cluster, per-thread async copy는 모두 실패했다. 남은 방향은 “더 큰 bytes-in-flight” 또는 “중복 제거” 쪽이다.
- 리스크:
  - 한 iteration 치고 수정 범위가 크다.
  - shared memory stage, commit/wait, barrier 재배치까지 들어가 correctness 리스크가 높다.
  - 현재 코드에는 이미 8-row lookahead 성격의 prefetch가 있어, 잘못 건드리면 오히려 register pressure와 sync cost만 늘 수 있다.
- 판정: 방향성은 맞지만, 이번 iteration #1의 단일 실행안으로는 과하다.

### 후보 B. G5-lite: 현재 accepted 코드의 ptxas/SASS 기준선 재고정

- 내용: 다음 iteration에서는 커널 수학/레이아웃은 건드리지 않고, host-side compile flag 주입을 최소 범위로 추가해 현재 accepted 코드의 register/spill/codegen 사실을 다시 고정한다.
- 근거:
  - 현재 코드와 `workflow.md` 사이에 드리프트가 있다.
  - `ld.global.nc`가 이미 들어가 있는데 workflow상 A3/A4가 아직 후보처럼 보이는 점만 봐도, 지금 기준선 없이 다음 구조 변경을 고르면 잘못된 문제를 최적화할 위험이 크다.
  - 현재 정말 `56 regs/thread`, spill `0`, 기대한 load opcode가 맞는지부터 확정해야 다음 구조 변경의 리스크를 줄일 수 있다.
- 리스크:
  - kernel Duration을 직접 줄이는 변경은 아니다.
  - compile flag 주입을 잘못하면 기존 build flag를 덮어써서 성능이 흔들릴 수 있다.
  - 따라서 “append only”, “기존 arch 설정 유지”, “kernel body 무변경”이 전제다.
- 판정: 이번 iteration #1의 1순위 후보.

### 후보 C. A5 standalone

- 내용: `setup_l2_persistence()`의 `missProp`를 `Normal`에서 `Streaming`으로 바꾸는 host-side 단독 실험.
- 근거:
  - 현재 코드상 `missProp=Normal`이다.
  - kernel body를 안 건드리면서 메모리 경로를 조정할 수 있다.
- 리스크:
  - B200 L2가 크고 state footprint도 상대적으로 작아서 효과가 작거나 0일 수 있다.
  - 잘못하면 persisting 이득보다 cache policy 부작용이 더 클 수 있다.
- 판정: G5-lite 이후 차선 후보.

### 후보 D. A6 standalone

- 내용: `cudaFuncAttributePreferredSharedMemoryCarveout=0`로 L1 쪽 여유를 늘리는 host-side 단독 실험.
- 근거:
  - 현재 shared 사용량이 매우 작다.
  - host-only라 rollback이 쉽다.
- 리스크:
  - 현재 state load가 `ld.global.nc` 경로이고 재사용성이 낮아서, L1 여유 확대가 곧바로 Duration 감소로 이어질 근거는 약하다.
  - 효과가 있어도 미세할 가능성이 높다.
- 판정: A5와 비슷한 저위험 후보이지만, 현재 프로파일 근거는 더 약하다.

### Step 1 결론

- 바로 성능을 건드리는 후보 중에서는 후보 A가 가장 “Duration 지향적”이다.
- 하지만 현재 작업트리와 문서 상태가 어긋나 있어, iteration #1에서 후보 A를 바로 구현하면 “틀린 기준선 위에 큰 구조 변경”을 올리는 셈이 된다.
- 따라서 이번 iteration #1의 실질 우선순위는 다음과 같이 잡는다.

1. `G5-lite`로 현재 accepted 코드의 compile/codegen 기준선 재고정
2. 그 결과를 바탕으로 다음 iteration에서 `H2/B2 축소안` 또는 `A5 standalone` 중 하나만 단일 변경으로 선택

## Step 2 PM 검토 대화

### 1차 검토

**PM**

> 세 가지를 본다.  
> (a) 프로파일과 코드 근거가 충분한가?  
> (b) 한 iteration에 과하지 않은가?  
> (c) 회귀 리스크를 실제로 낮추는 계획인가?

**엔지니어**

> 초안은 후보 A를 1순위로 두고 싶었다. 이유는 분명하다. 지금 숫자상으로는 kernel body Duration을 줄일 수 있는 쪽이 `H2/B2`류뿐이기 때문이다.  
> 다만 현재 작업트리를 직접 보니 `workflow.md` 기준 후보 상태가 이미 일부 낡았다. 현재 커널에는 `ld.global.nc.v4.f32`가 이미 들어가 있고, arch list도 `10.0a`가 이미 설정돼 있다. split dispatch도 문서와 다르다.  
> 이 상태에서 구조 변경을 먼저 넣으면 “실제 baseline이 무엇인지 모른 채 큰 수술”을 하는 셈이라 리스크가 높다.

**PM**

> 즉, 후보 A의 방향은 맞지만 지금 바로 실행안으로 승인하기에는 근거가 부족하다는 말인가?

**엔지니어**

> 맞다.  
> 후보 A는 Duration 지향성은 가장 강하지만, 이번 iteration #1에 넣기엔 범위가 크다. async pipeline만 넣어도 shared stage 설계, wait/commit, 경로 분기, correctness 검증이 같이 따라온다. minimal cluster와 per-thread async copy가 이미 실패한 상태라 더더욱 현재 codegen 기준선 없이 들어가면 실패 비용이 크다.

**PM**

> 그러면 후보 B는 어떤가? G5-lite는 성능 개선이 아니라 계측 준비 아닌가? 이번 iteration 예산을 진짜 개선이 아니라 재정렬에 쓰겠다는 건데, 그게 과도하게 소극적인 것 아닌가?

**엔지니어**

> 그 우려는 맞다. 그래서 초안이 충분히 공격적이지 않다는 점은 인정한다.  
> 하지만 지금은 “문서상 미시도 후보”와 “실제 코드상 이미 들어간 변경”이 섞여 있다. 이 상태에서 다음 iteration을 A5나 A6 같은 micro-tuning으로 보내도, 왜 좋아졌는지/나빠졌는지 설명력이 약하다.  
> iteration #1을 G5-lite로 쓰면 직접적인 latency 개선 확률은 낮지만, iteration #2부터는 적어도 잘못된 후보를 다시 고르는 실수를 줄일 수 있다.

### PM 불만 및 수정 요구

**PM**

> 불만이 있다.  
> “안전한 정리 작업”만 하고 끝내면 Duration 목표에 너무 멀다.  
> 단, 후보 A를 지금 승인하기도 어렵다. 한 iteration 치고 너무 크고, 최근 실패 패턴과도 닮았다.  
> 계획을 수정해라. 조건은 세 가지다.

1. 이번 iteration은 단일 변경이어야 한다.
2. 그 변경은 다음 큰 구조 변경의 실패 확률을 실제로 낮춰야 한다.
3. 회귀 시 즉시 롤백 기준이 명확해야 한다.

### 수정안

**엔지니어**

> 수정안은 다음과 같다.

1. 이번 iteration #1의 승인안은 `G5-lite` 단독으로 제한한다.
2. 범위는 host-side compile flag 주입과 codegen 사실 확인까지만 둔다.
3. kernel body, split policy, memory policy는 이번 iteration에서 건드리지 않는다.
4. 다음 iteration 후보는 미리 예약하지 않고, `G5-lite` 결과를 본 뒤 아래 게이트로 고른다.

- register/spill/codegen이 현재 로그 가정과 다르면:
  - 먼저 기준선 문서부터 갱신하고, 그 뒤에 micro-change를 다시 고른다.
- register가 여전히 높고 current prefetch가 병목으로 보이면:
  - 다음 iteration은 `H2/B2 축소안` 단독
- codegen은 기대와 맞고 cache policy 쪽만 손볼 가치가 남아 있으면:
  - 다음 iteration은 `A5 standalone` 단독

### 2차 검토

**PM**

> 좋다. 이제 세 항목을 다시 본다.  
> (a) 프로파일·근거: 현재 코드/문서 드리프트가 실제로 확인됐고, 그래서 기준선 재고정이 필요하다는 논리는 성립한다.  
> (b) 한 iteration에 과한지: `G5-lite` 단독이면 과하지 않다.  
> (c) 회귀 리스크: compile flag 주입이 build 동작을 흔들 수 있으니 이 부분을 더 명확히 적어라.

**엔지니어**

> 회귀 리스크 통제는 이렇게 적겠다.

- flag는 “replace”가 아니라 반드시 “append only”로 넣는다.
- `TVM_FFI_CUDA_ARCH_LIST=10.0a`는 유지한다.
- kernel source는 건드리지 않는다.
- benchmark avg/median 또는 correctness가 흔들리면, host-side flag 주입만 즉시 롤백한다.
- 이번 iteration의 목적은 성능 채택이 아니라 기준선 확정이다. 따라서 성능 변화가 있더라도 그 자체를 개선으로 채택하지 않는다.

**PM**

> 좋다. 이번 iteration #1 Step 1~2의 최종 계획으로 승인한다. **APPROVED**.

## 최종 결론

- 이번 세션에서는 코드 변경/실행 없이 Step 1~2만 정리했다.
- 다음 iteration #1의 승인안은 `G5-lite` 단독이다.
- 이유는 현재 코드와 문서의 드리프트가 확인됐고, 이 상태에서 구조 변경을 먼저 넣는 것이 더 큰 회귀 리스크이기 때문이다.
- `H2/B2 축소안`은 가장 유력한 “Duration 지향” 후보지만, 이번 iteration #1에서는 범위 초과로 보류한다.

## 다음 iteration 실행안 요약

1. `G5-lite` 단독 적용
2. kernel body 무변경
3. compile/codegen 기준선 재고정
4. 결과에 따라 다음 iteration에서 `H2/B2 축소안` 또는 `A5 standalone` 중 하나만 선택

