# PRD: Gated DeltaNet Decode CUDA Kernel

## 1. Overview

**프로젝트:** FlashInfer Bench @ MLSys 2026 — `gated_delta_net` 트랙  
**목표:** Gated DeltaNet의 decode step (T=1)에 대한 고성능 CUDA 커널 작성  
**Definition ID:** `gdn_decode_qk4_v8_d128_k_last`  
**타겟 하드웨어:** NVIDIA B200 (Blackwell)  
**평가 환경:** Docker `flashinfer/flashinfer-ci-cu132:latest`, GPU clocks locked (`nvidia-smi -ac 3996,1965`)

---

## 2. 산출물

다음 두 파일을 완성해야 한다:

| 파일 | 역할 |
|------|------|
| `solution/cuda/kernel.cu` | CUDA 커널 구현체 |
| `solution/cuda/binding.py` | Python → CUDA 바인딩 (TVM FFI 또는 torch) |

`config.toml`도 아래와 같이 수정:

```toml
[solution]
name = "gdn-decode-cuda-v1"
definition = "gdn_decode_qk4_v8_d128_k_last"
author = "team-name"

[build]
language = "cuda"
entry_point = "kernel"
```

---

## 3. 수학적 정의 (Gated Delta Rule — Decode Step)

논문 Eq. 10 기반. 단일 토큰(T=1) decode step의 연산:

### 3.1 Gate & Beta 계산

```
g = exp(-exp(A_log) * softplus(a + dt_bias))    // per-head scalar, shape [B, H_v]
beta = sigmoid(b)                                // per-head scalar, shape [B, H_v]
```

여기서:
- `A_log`: 학습된 파라미터 (per-head)
- `a`: 입력 dependent 파라미터, shape `[B, 1, H_v]`
- `dt_bias`: 학습된 bias, shape `[H_v]`
- `b`: 입력 dependent 파라미터, shape `[B, 1, H_v]`

### 3.2 State Update (핵심 연산)

State `S`의 layout은 `[B, H_v, V, K]` (K가 마지막 — "k-last").

각 batch, 각 head에 대해 (K×V 행렬 연산으로 이해하면 편하지만, 실제 저장은 `[V, K]`):

```
# old_state에 gating 적용
old_state = g * S                          // [V, K] elementwise broadcast

# 현재 key로 기존 value 검색
old_v = k @ old_state                      // [K] @ [K, V]^T → 실제로 k @ old_state^T → [V]
                                           // 코드상: k @ old_state (old_state를 [K,V]로 전치해서 사용)

# delta rule: 새 value = beta * v + (1-beta) * old_v
new_v = beta * v + (1 - beta) * old_v      // [V]

# state에서 old association 제거, new association 추가
S_new = old_state - outer(k, old_v) + outer(k, new_v)   // [K, V]
      = old_state + outer(k, beta * (v - old_v))         // 간소화 가능

# output 계산
output = scale * q @ S_new                 // [K] @ [K, V] → [V]
```

### 3.3 간소화된 형태

위 연산을 정리하면:

```
old_state = g * S                                    // gate 적용
old_v = k @ old_state                                // retrieve
delta_v = beta * (v - old_v)                         // delta
S_new = old_state + outer(k, delta_v)                // update
output = scale * (q @ S_new)                         // query
```

이것은 논문의 gated delta rule `S_t = S_{t-1} (α_t(I - β_t k_t k_t^T)) + β_t v_t k_t^T`을 decode step으로 전개한 것이다.

---

## 4. 텐서 사양

### 4.1 입력 텐서

| 텐서 | Shape | Dtype | 설명 |
|------|-------|-------|------|
| `q` | `[B, 1, H_qk, K]` | bf16 | Query, T=1 |
| `k` | `[B, 1, H_qk, K]` | bf16 | Key, T=1 |
| `v` | `[B, 1, H_v, V]` | bf16 | Value, T=1 |
| `state` | `[B, H_v, V, K]` | float32 | RNN hidden state (in-place 업데이트) |
| `A_log` | `[H_v]` | float32 | Gate 파라미터 (log 공간) |
| `a` | `[B, 1, H_v]` | bf16 | Gate 입력 |
| `dt_bias` | `[H_v]` | float32 | Gate bias |
| `b` | `[B, 1, H_v]` | bf16 | Beta 입력 |
| `scale` | scalar (float) | float32 | Attention scale, 기본값 `1/sqrt(K)` |

### 4.2 출력 텐서

| 텐서 | Shape | Dtype | 설명 |
|------|-------|-------|------|
| `output` | `[B, 1, H_v, V]` | bf16 | Decode 출력 |
| `state` | `[B, H_v, V, K]` | float32 | 업데이트된 hidden state (in-place) |

### 4.3 고정 차원 (Definition에서 파싱)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `H_qk` (num_q_heads = num_k_heads) | **4** | Query/Key head 수 |
| `H_v` (num_v_heads) | **8** | Value head 수 |
| `K` (head_dim) | **128** | Key/Query 차원 |
| `V` (head_dim) | **128** | Value 차원 |
| `T` | **1** | Decode이므로 고정 |
| GQA ratio | **2** | `H_v / H_qk = 8/4 = 2` (2개의 v-head가 1개의 qk-head 공유) |

### 4.4 Batch Size 범위

워크로드에 따라 다양. 일반적으로 B=1~256 범위를 커버해야 한다. FlashInfer-Bench에서 제공하는 workload 데이터셋에 정의된 B 값들을 모두 통과해야 한다.

---

## 5. GQA (Grouped Query Attention) 처리

Q/K head 4개가 V head 8개에 매핑된다. 즉 `v_head_idx // 2`로 대응되는 `qk_head_idx`를 결정:

```
qk_head_idx = v_head_idx / (H_v / H_qk)   // = v_head_idx / 2
```

- v_head 0,1 → qk_head 0
- v_head 2,3 → qk_head 1
- v_head 4,5 → qk_head 2
- v_head 6,7 → qk_head 3

각 v_head는 독립적인 state `[V, K]`를 유지하며, gate/beta도 per-v-head이다.  
단, query/key는 해당 qk_head의 것을 공유한다.

---

## 6. Reference Python 구현

정확성 검증의 기준이 되는 reference 코드 (전문):

```python
import math
import torch
import torch.nn.functional as F

def matmul(a, b):
    return a.float() @ b.float()

@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert K == 128 and V == 128
    assert T == 1

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Gate and beta
    x = a.float() + dt_bias.float()              # [B, 1, H_v]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, H_v]
    beta = torch.sigmoid(b.float())              # [B, 1, H_v]

    q_f32 = q.squeeze(1).float()     # [B, H_qk, K]
    k_f32 = k.squeeze(1).float()     # [B, H_qk, K]
    v_f32 = v.squeeze(1).float()     # [B, H_v, V]
    g_f32 = g.squeeze(1).float()     # [B, H_v]
    beta_f32 = beta.squeeze(1).float()

    if state is not None:
        state_f32 = state.float()    # [B, H_v, V, K]
    else:
        state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)

    # GQA expansion: repeat q/k for each v-head group
    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [B, H_v, K]
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)  # [B, H_v, K]

    new_state = torch.zeros_like(state_f32)
    output = torch.zeros(B, num_heads, V, dtype=torch.float32, device=device)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            q_h = q_exp[b_idx, h_idx]         # [K]
            k_h = k_exp[b_idx, h_idx]         # [K]
            v_h = v_f32[b_idx, h_idx]         # [V]
            # state: [V, K] → transpose to [K, V] for matmul
            h_state = state_f32[b_idx, h_idx].clone().transpose(-1, -2)  # [K, V]
            g_val = g_f32[b_idx, h_idx]       # scalar
            beta_val = beta_f32[b_idx, h_idx] # scalar

            old_state = g_val * h_state                         # [K, V]
            old_v = k_h @ old_state                             # [V]
            new_v = beta_val * v_h + (1 - beta_val) * old_v    # [V]
            state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)  # [K,1]@[1,V] → [K,V]
            state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)  # [K,V]
            h_state = old_state - state_remove + state_update      # [K,V]

            output[b_idx, h_idx] = scale * (q_h @ h_state)     # [V]
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2) # [V, K]

    output = output.unsqueeze(1).to(torch.bfloat16)  # [B, 1, H_v, V]
    return output, new_state
```

---

## 7. CUDA 커널 설계 지침

### 7.1 병렬화 전략

핵심 연산은 **각 (batch, v_head) 쌍**이 독립적이다. 따라서:

- **Grid 차원:** `(B, H_v)` = `(B, 8)` — 각 블록이 하나의 (batch, head) 처리
- **Block 내부:** K=128, V=128 차원의 행렬-벡터 곱을 warp 수준으로 분할

### 7.2 주요 연산 분석 (per block)

하나의 (batch, head)에 대해:

1. **Gate/Beta 계산:** 스칼라 연산 (softplus, exp, sigmoid) — 1개 스레드 또는 warp leader
2. **State scaling:** `old_state = g * state` — `V×K = 128×128 = 16384` 원소 elementwise 곱 (float32)
3. **old_v = k @ old_state:** `[K]` dot `[K, V]` (열 기준) → `[V]` — K=128 reduction, V번
4. **new_v 계산:** elementwise `[V]`
5. **State update:** `old_state + outer(k, delta_v)` — `K×V` 원소 업데이트
6. **output = q @ new_state:** `[K]` dot `[K, V]` → `[V]` — K=128 reduction, V번

### 7.3 메모리 접근 패턴

- **State `[V, K]` (k-last):** 저장 형식이 `[V, K]`이므로, K가 contiguous 차원
  - `k @ state^T` (= `k @ state_transposed`)는 state를 `[K, V]`로 접근해야 하므로, 실제 메모리에서는 strided access
  - **핵심 최적화 포인트:** Shared memory에 state 타일을 로드하여 bank conflict 없이 접근
- **q, k:** `[K]` 벡터 (128 float) — warp 단위 로드 가능
- **v:** `[V]` 벡터 (128 float)
- State는 **float32**로 유지 (누적 정밀도), q/k/v는 **bf16** 입력 → float32로 변환하여 연산

### 7.4 Shared Memory 사용 전략

State 크기 = `128 × 128 × 4 bytes = 64 KB` per (batch, head).  
B200의 shared memory 한도가 충분한지 확인 필요 (Blackwell은 최대 228KB shared memory 지원).

**전략 옵션:**

- **Option A (State를 shared memory에 전부 로드):** 64KB → 가능하면 가장 빠름. State를 smem에 로드 → 모든 연산을 smem에서 수행 → global memory로 write-back
- **Option B (타일 기반):** V 차원을 타일로 분할하여 여러 패스로 처리. Shared memory 부족 시 사용

### 7.5 Warp 수준 최적화

- K=128 → 4 warps (각 warp 32 스레드)가 K 차원 커버, 또는 1 warp가 K를 4번 순회
- `__shfl_xor_sync` 등 warp-level reduction 활용하여 dot product 계산
- Vectorized load: `float4` (16바이트) 단위로 global memory 접근

### 7.6 연산 순서 최적화

간소화된 형태를 사용하면 outer product가 1회로 줄어든다:

```
old_state = g * state
old_v = k @ old_state          // matvec
delta_v = beta * (v - old_v)   // elementwise
state_new = old_state + outer(k, delta_v)   // rank-1 update
output = scale * q @ state_new              // matvec
```

이는 2번의 matvec (K×V) + 1번의 rank-1 update + elementwise 연산이다.

### 7.7 수치 안정성

- 모든 state 연산은 **float32**로 수행
- `exp(-exp(A_log) * softplus(x))` 에서 overflow 주의:  
  - `softplus(x) = log(1 + exp(x))` → x가 매우 크면 x로 근사
  - `exp(A_log)`가 크고 softplus도 크면, `-exp(A_log)*softplus(x)`가 매우 큰 음수 → `g ≈ 0` (안전)
  - `exp(A_log)`가 0에 가까우면 `g ≈ 1` (안전)
- bf16 입력은 커널 진입 시 float32로 변환, 출력 시 bf16으로 변환

---

## 8. `kernel.cu` 구현 요구사항

### 8.1 커널 시그니처 (DPS 스타일)

FlashInfer-Bench는 **Destination Passing Style (DPS)** 을 사용한다. 출력 텐서가 미리 할당되어 파라미터로 전달된다:

```cuda
__global__ void gdn_decode_kernel(
    // 입력
    const __nv_bfloat16* q,      // [B, 1, H_qk, K]
    const __nv_bfloat16* k,      // [B, 1, H_qk, K]
    const __nv_bfloat16* v,      // [B, 1, H_v, V]
    const float* A_log,          // [H_v]
    const __nv_bfloat16* a,      // [B, 1, H_v]
    const float* dt_bias,        // [H_v]
    const __nv_bfloat16* b,      // [B, 1, H_v]
    float scale,
    // 입출력 (in-place)
    float* state,                // [B, H_v, V, K] — read & write
    // 출력
    __nv_bfloat16* output        // [B, 1, H_v, V]
);
```

> **주의:** 정확한 시그니처는 FlashInfer-Bench 데이터셋의 definition JSON을 확인하여 맞춰야 한다. 위는 reference code 기반 추정이다. `flashinfer-bench`를 설치하고 definition을 직접 확인할 것.

### 8.2 필수 포함 헤더

```cuda
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
```

### 8.3 성능 목표

- FlashInfer baseline (`flashinfer_wrapper_9b7f1e`) 대비 동등 이상의 throughput
- 모든 workload에서 정확성 통과 (기본 tolerance 사용)
- Memory-bound 커널이므로 global memory 접근 최소화가 핵심

---

## 9. `binding.py` 구현 요구사항

### 9.1 바인딩 방식 선택

**Option A: TVM FFI (기본)**

```python
import ctypes
from tvm.ffi import register_func

@register_func("flashinfer.kernel")
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output):
    # 1. 텐서에서 data_ptr 추출
    # 2. grid/block 차원 계산
    # 3. CUDA 커널 런치
    pass
```

**Option B: PyTorch custom op**

`config.toml`에서 `binding = "torch"` 설정 시:

```python
import torch
from torch.utils.cpp_extension import load_inline

# kernel.cu를 컴파일하고 torch에서 호출
```

### 9.2 바인딩에서 처리할 사항

1. 입력 텐서들의 shape에서 B (batch_size) 동적 추출
2. Grid 차원: `(B, H_v)` 또는 `(B * H_v, 1)` — 커널 설계에 맞게
3. Block 차원: K/V 크기와 shared memory 사용량에 따라 결정 (예: 128 or 256 threads)
4. Shared memory 크기 동적 설정 (필요 시 `cudaFuncSetAttribute`로 max shared memory 설정)
5. CUDA stream 처리 (TVM FFI 사용 시 현재 stream 획득)

---

## 10. 검증

### 10.1 로컬 테스트

```bash
# config.toml 수정 후
python scripts/pack_solution.py
python scripts/run_local.py
```

### 10.2 정확성 기준

- Reference 구현 대비 element-wise 비교
- Default tolerance (atol/rtol은 EVALUATION.md에 명시되지 않았으므로 기본값 사용)
- `--required-matched-ratio 0.9` 이상 필요할 수 있음

### 10.3 성능 프로파일링

```bash
# NCU 프로파일링
ncu --set detailed ./your_test_binary

# 또는 FlashInfer-Bench의 NCU 도구 사용
```

확인할 메트릭:
- Global memory throughput (GB/s) — B200의 메모리 대역폭 대비 활용률
- Occupancy
- Shared memory bank conflict 여부
- Warp stall 원인

---

## 11. 최적화 우선순위 (권장 개발 순서)

### Phase 1: 정확성 확보 (Naive 커널)
1. 단일 스레드 per block으로 reference 로직 그대로 포팅
2. 모든 workload에서 정확성 통과 확인
3. 바인딩 코드 완성 및 `flashinfer-bench run` 통과

### Phase 2: 기본 병렬화
1. Grid = `(B, H_v)`, Block = 적절한 스레드 수
2. K 차원 reduction을 warp-level로 구현
3. State를 shared memory에 로드

### Phase 3: 메모리 접근 최적화
1. Vectorized load (`float4`) 사용
2. State read/write coalescing
3. Bank conflict 제거

### Phase 4: 고급 최적화
1. State의 [V,K] layout 특성을 고려한 tiling
2. `__ldg` (read-only cache) 활용 for q, k, v
3. Register pressure 최적화
4. Persistent kernel 고려 (small B일 때)
5. Occupancy tuning

---

## 12. 주의사항 및 흔한 실수

1. **State layout 혼동:** State는 `[B, H_v, V, K]`로 저장 (V행 K열). Reference code에서는 `[K, V]`로 transpose하여 연산한 후 다시 `[V, K]`로 돌려놓는다. 커널에서도 이 변환을 정확히 처리해야 한다.

2. **GQA 인덱싱 오류:** v_head `h`에 대응하는 qk_head는 `h / 2` (정수 나눗셈). q, k 텐서 접근 시 이 매핑을 정확히 적용해야 한다.

3. **In-place state update:** state는 읽은 후 업데이트해야 한다. 같은 메모리를 다른 블록이 동시에 접근하지 않는지 확인 (각 블록이 고유한 (batch, head)를 처리하므로 문제없음).

4. **bf16 ↔ float32 변환:** 입력은 bf16, 연산은 float32, 출력은 bf16. CUDA의 `__bfloat162float()` / `__float2bfloat16()` 사용.

5. **scale 기본값:** `scale`이 0이거나 None이면 `1/sqrt(128) ≈ 0.0884`로 설정.

6. **DPS 준수:** output 텐서는 미리 할당되어 전달됨. 커널 내에서 새로 할당하지 말 것.

---

## 13. 참고 자료

- **논문:** "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025), arXiv:2412.06464v3
  - 핵심 수식: Eq. 10 (gated delta rule), Section 3.3 (chunkwise algorithm)
  - Table 1 (online learning objectives 비교)
- **공식 코드:** https://github.com/NVlabs/GatedDeltaNet
- **FlashInfer-Bench:** https://github.com/flashinfer-ai/flashinfer-bench
- **경진대회 레포:** https://github.com/jaewooMaeng/GDN_kernel (forked from Bammuri/mlsys26)
- **FlashInfer Trace 데이터셋:** https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
  - Definition JSON에서 정확한 함수 시그니처, 텐서 shape/dtype 규격 확인 필수

---

## 부록 A: 연산량 분석

Per (batch, head) 기준:

| 연산 | FLOPs | Memory (bytes) |
|------|-------|----------------|
| State scaling (g * state) | 16,384 mul | 64KB read + 64KB write (if in-place) |
| old_v = k @ state | 2 × 128 × 128 = 32,768 | 512B (k) + 64KB (state) read |
| delta_v 계산 | ~384 | negligible |
| Rank-1 update | 2 × 128 × 128 = 32,768 | 512B + 512B read, 64KB write |
| output = q @ state | 2 × 128 × 128 = 32,768 | 512B + 64KB read |

총 ~98K FLOPs vs ~256KB memory access → **Memory-bound** 연산.  
따라서 메모리 접근 패턴 최적화가 성능의 핵심이다.

## 부록 B: FlashInfer-Bench 로컬 실행 Quick Reference

```bash
# 1. 환경 설정
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal

# 2. 데이터셋 다운로드
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
export FIB_DATASET_PATH=/path/to/mlsys26-contest

# 3. 솔루션 패키징 & 로컬 실행
python scripts/pack_solution.py
python scripts/run_local.py

# 4. Modal 클라우드 (B200) 실행
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/mlsys26-contest
modal run scripts/run_modal.py
```