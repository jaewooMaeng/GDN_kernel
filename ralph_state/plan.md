# Iteration #3 Step 1~2 계획

## 현재 기준선

- 현재 코드 기준 large-batch dispatch는 `batch_size >= 32 -> split_factor = 4 -> gdn_decode_kernel<8>` 이다.
- 최근 accepted 기준으로 확인된 핵심 수치는 `Duration = 31.97 us`, `Grid Size = 2048`, `Block Size = 128`, `Issue Slots Busy = 21.98%`, `Achieved Occupancy = 40.11%`, `Registers/thread = 56`, spill `0` 이다.
- 현재 커널은 이미 `ld.global.nc.v4.f32` 기반 state load를 쓰고 있고, `__shared__ float s_v[HEAD_DIM]`만 사용하므로 shared memory 사용량은 매우 작다.
- host 쪽 `setup_l2_persistence()`는 현재 `hitProp=Persisting`, `missProp=Normal` 이다.
- B=64 기준 state footprint는 약 `32 MiB`이고 persisting L2 set-aside는 최근 기록상 약 `82.9 MiB`라서, 현 설정에서는 `hitRatio`가 사실상 `1.0`으로 잡힐 가능성이 높다.

## Step 1 정리

### 후보 A. A6 standalone: `cudaFuncAttributePreferredSharedMemoryCarveout = 0`

- 내용:
  - 커널 body, dispatch, split policy, `__launch_bounds__`, state/q/k/v 수학식은 건드리지 않는다.
  - host launch 경로에서 carveout만 `0`으로 줘서 L1/TEX를 최대화하는 단독 실험으로 제한한다.
- 근거:
  - 현재 커널은 shared memory를 사실상 `s_v[128] = 512B`만 쓰므로 carveout을 shared 쪽에 남겨둘 이유가 약하다.
  - NCU에서 `L1/TEX Throughput`가 상대적으로 높고, current kernel의 state load가 `ld.global.nc.v4.f32`라 read-only cache 경로 쪽 민감도가 있을 수 있다.
  - kernel body를 건드리지 않으면서 `Duration(us)`를 직접 건드릴 수 있는 남은 저위험 후보 중 하나다.
- 리스크:
  - current bottleneck이 본질적으로 low-issue / reg-limited occupancy 쪽이라 효과가 거의 없을 수 있다.
  - 과거 R3에서 `split-local s_v staging + carveout=0` 조합이 후퇴했으므로, carveout 자체도 upside가 작을 가능성은 있다.
  - host attribute 하나로 끝나지만, 개선 폭이 작아 Modal 노이즈에 묻힐 수 있다.

### 후보 B. A5 standalone: `missProp = Streaming`

- 내용:
  - 현재 `setup_l2_persistence()`의 `missProp=Normal`만 `Streaming`으로 바꾸는 host-side 단독 실험.
- 근거:
  - low L2 hit 상황에서 비-persisting miss가 L2를 오염시키는지 확인할 수 있다.
  - 구현과 롤백이 매우 쉽다.
- 리스크:
  - 현재 코드/워크로드에서는 `hitRatio`가 사실상 `1.0`이라 leverage가 작다.
  - B=64 current state footprint가 persisting window보다 작아, 핵심 path에 미치는 영향이 약할 가능성이 높다.
  - Duration 개선 근거는 후보 A보다 더 약하다.

### 후보 C. G6 standalone: `__builtin_assume()`

- 내용:
  - `blockDim.x == 128` 같은 상수성 힌트를 추가해 index math를 단순화하려는 미세 최적화.
- 근거:
  - 구현량은 작다.
- 리스크:
  - profile 근거가 약하다.
  - `batch_size` 관련 가정은 잘못 넣으면 UB를 만들 수 있다.
  - 최근 병목이 issue mix보다 occupancy / memory-side 성격이라 우선순위가 낮다.

### 이번 Step 1 판단

- 후보 B는 너무 안전하지만 레버리지가 작다. 현재 코드와 프로파일을 같이 보면 `missProp` 변경만으로 `Duration(us)`를 움직일 가능성이 낮다.
- 후보 C는 한 iteration 크기로는 작지만, profile 근거가 약하고 잘못 쓰면 correctness/UB 리스크가 있다.
- 후보 A가 남은 저위험 안 중에서는 가장 직접적으로 `Duration(us)`를 건드릴 수 있다.
- 다만 과거 R3 회귀가 있으므로, **반드시 carveout 단독 실험**으로 제한해야 한다. 다른 미세 변경과 묶지 않는다.

## Step 2 PM 검토 대화

### 1차 검토

**PM**

> (a) 프로파일과 코드 근거가 충분한가?  
> (b) 한 iteration에 과하지 않은가?  
> (c) 회귀 리스크를 실제로 통제할 수 있는가?

**엔지니어**

> 후보 A를 1순위로 본다. 현재 커널은 `ld.global.nc.v4.f32` 기반 read-only load를 쓰고, shared memory는 `s_v[128]`만 써서 carveout을 shared 쪽에 둘 이유가 약하다.  
> accepted 기준선도 `Duration=31.97 us`, `Issue Slots Busy=21.98%`, `Occupancy=40.11%`, regs `56`, spill `0`이라, kernel body를 흔들지 않는 host-side 메모리 partition 조정이 한 iteration 크기로 가장 맞다.

**PM**

> 근거는 알겠다. 그런데 R3에서 carveout=0이 이미 들어간 조합이 망했다.  
> 이번에도 다른 요소와 섞이면 같은 실수를 반복하는 것 아닌가?

**엔지니어**

> 맞다. 그래서 R3의 교훈을 그대로 반영해서, 이번엔 `split-local s_v staging` 같은 본체 변경을 전부 금지하고 carveout만 단독으로 본다.  
> 즉, kernel body와 dispatch는 byte-for-byte 유지하고 launch attribute 하나만 바꾼다.

### PM 불만과 수정 요구

**PM**

> 아직도 두 가지를 더 확인해야 한다.  
> 첫째, 왜 후보 B가 아니라 후보 A인가?  
> 둘째, 회귀 차단 게이트를 더 명확히 써라.

### 수정 답변

**엔지니어**

> 후보 B는 현재 코드에서 leverage가 더 약하다. `setup_l2_persistence()`가 이미 `hitRatio = min(max_bytes / state_bytes, 1.0)`로 계산되고, B=64 기준 state가 persisting window보다 작아서 `hitRatio`가 사실상 `1.0`이 된다.  
> 즉 `missProp=Streaming`은 핵심 state path에 영향을 거의 못 줄 가능성이 높다. 반면 후보 A는 L1/TEX partition 자체를 바꾸므로, 현재 `ld.global.nc.v4.f32` 경로와 `q/k/v` read path 전체에 영향을 줄 수 있다.

**엔지니어**

> 회귀 차단 게이트는 다음처럼 둔다.

1. correctness 실패 시 즉시 롤백
2. quick screen에서 핵심 B=64 workload가 recent accepted band보다 명확히 느리면 full benchmark 전에 롤백
3. full benchmark 판정은 avg/median latency와 correctness를 최종 기준으로 삼고, NCU `Duration(us)`는 보조 지표로만 본다
4. `Duration(us)`가 줄더라도 benchmark avg/median이 후퇴하면 채택하지 않는다
5. 이번 iteration에서는 carveout 외 다른 host/kernel 변경을 추가하지 않는다

### 2차 검토

**PM**

> (a) 프로파일·근거: 현재 코드의 tiny shared usage와 `ld.global.nc.v4.f32` 경로, 그리고 `hitRatio` 계산을 근거로 후보 A 우선순위를 설명한 점은 납득된다.  
> (b) 한 iteration에 과한지: launch attribute 하나만 바꾸는 단독 실험이면 과하지 않다.  
> (c) 회귀 리스크: R3와 같은 혼합 변경을 금지하고 rollback gate도 명확하다.

**PM**

> 결론은 후보 A 단독 실험으로 간다. **APPROVED**

## 최종 결론

- iteration #3의 실행 후보는 **후보 A 단독**, 즉 `cudaFuncAttributePreferredSharedMemoryCarveout = 0` standalone 실험으로 고정한다.
- 후보 B(`missProp=Streaming`)와 후보 C(`__builtin_assume`)는 이번 iteration에서는 보류한다.
- 최종 판정 기준은 benchmark avg/median latency와 correctness이며, NCU `Duration(us)` 감소는 보조 지표로만 사용한다.
