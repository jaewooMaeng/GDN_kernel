# PRD: Gated DeltaNet Prefill CUDA Kernel

## 1. Overview

**프로젝트:** FlashInfer Bench @ MLSys 2026 — `gated_delta_net` 트랙  
**목표:** Gated DeltaNet의 prefill step (T > 1)에 대한 고성능 CUDA 커널 작성  
**Definition ID:** `gdn_prefill_qk4_v8_d128_k_last`  
**타겟 하드웨어:** NVIDIA B200 (Blackwell)  
**평가 환경:** Docker `flashinfer/flashinfer-ci-cu132:latest`, GPU clocks locked (`nvidia-smi -ac 3996,1965`)  
**Baseline:** `flashinfer_wrapper_123ca6`

---

## 2. 산출물

다음 두 파일을 완성해야 한다:

| 파일 | 역할 |
|------|------|
| `solution/cuda/kernel.cu` | CUDA 커널 구현체 |
| `solution/cuda/binding.py` | Python → CUDA 바인딩 (TVM FFI 또는 torch) |

`config.toml` 수정:

```toml
[solution]
name = "gdn-prefill-cuda-v1"
definition = "gdn_prefill_qk4_v8_d128_k_last"
author = "team-name"

[build]
language = "cuda"
entry_point = "kernel"
```

평가 명령 (EVALUATION.md 기준):

```bash
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions gdn_prefill_qk4_v8_d128_k_last \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 \
  --warmup-runs 1 --iterations 5 --num-trials 3
```

---

## 3. Decode vs. Prefill 핵심 차이

| 구분 | Decode | Prefill |
|------|--------|---------|
| Sequence length T | 1 (고정) | 가변 (수백~수천) |
| 핵심 연산 | 1-step state update + matvec | 전체 시퀀스의 output 계산 + 최종 state |
| 병목 | Memory-bound (state read/write) | Compute-bound (matmul 다수) |
| 적합한 알고리즘 | 순차 1-step | **Chunkwise Parallel Form** |
| 텐서 코어 활용 | 불가 (벡터 연산) | **필수** (chunk 내 matmul) |

Prefill에서는 논문 Section 3.3의 chunkwise parallel algorithm을 구현해야 한다. 단순 sequential step-by-step은 GPU 활용률이 극히 낮아 성능 경쟁력이 없다.

---

## 4. 수학적 정의

### 4.1 Gated Delta Rule (논문 Eq. 10, 기본 recurrence)

단일 step의 recurrence:

```
S_t = S_{t-1} · (α_t (I − β_t k_t k_t^T)) + β_t v_t k_t^T
o_t = S_t q_t
```

여기서:
- `S_t ∈ R^{d_v × d_k}`: hidden state (행렬)
- `α_t ∈ (0, 1)`: data-dependent gating (scalar per head)
- `β_t ∈ (0, 1)`: writing strength (scalar per head)
- `k_t ∈ R^{d_k}`: key (L2-normalized)
- `v_t ∈ R^{d_v}`: value
- `q_t ∈ R^{d_k}`: query

이 recurrence를 T step 전부 순차적으로 돌리면 O(T · d_v · d_k) 이지만, GPU parallelism을 전혀 활용하지 못한다.

### 4.2 Gate 계산

```
g_t = exp(-exp(A_log) · softplus(a_t + dt_bias))     // = α_t
β_t = sigmoid(b_t)
```

Cumulative decay product (chunk 내):
```
γ_j = Π_{i=1}^{j} α_i       (chunk 시작부터 j번째 position까지의 누적 곱)
```

### 4.3 Chunkwise Parallel Form (논문 Section 3.3 — 핵심 알고리즘)

시퀀스를 chunk size `C`로 분할한다. Chunk index `[t]`, chunk 내 position index `r` (1-based).

#### 4.3.1 Chunk 내 State 전개

Recurrence를 부분 전개하면 (논문 Section 3.3 첫 수식):

```
S^r_{[t]} = S_{[t]} · F^r_{[t]} + G^r_{[t]}
```

여기서:
- `F^r_{[t]} = Π_{i=1}^{r} α^i_{[t]} (I − β^i_{[t]} k^i_{[t]} k^{iT}_{[t]})` — 누적 transition
- `G^r_{[t]}` — chunk 내 contribution (새로 쓰인 정보)

핵심 관찰: `F^r_{[t]} = γ^r_{[t]} · P^r_{[t]}` 여기서 `P^r_{[t]}`는 DeltaNet의 Householder 누적곱이고, `γ^r_{[t]}`는 gating의 누적곱.

#### 4.3.2 WY Representation (논문 Eq. 4-7의 gated 확장)

DeltaNet의 WY representation을 gating을 포함하여 확장:

**T matrix (UT Transform):**
```
T̃_{[t]} = [I + strictLower(diag(β_{[t]}) · (Γ_{[t]} ⊙ K_{[t]} K^T_{[t]}))]^{-1} · diag(β_{[t]})
```

여기서 `Γ_{[t]}`는 chunk 내 decay mask: `(Γ_{[t]})_{ij} = γ^i_{[t]} / γ^j_{[t]}` (i ≥ j일 때).

이 행렬은 **C × C** 크기의 하삼각(lower triangular) 행렬이며, `strictLower`는 대각 아래만 취한다.

**Ũ_g (Gated U matrix):**
```
Ũ_g[t] = T̃_{[t]} · V_{[t]}    ∈ R^{C × d_v}
```

**W matrix:**
```
W_{[t]} = T_{[t]} · K_{[t]}    ∈ R^{C × d_k}
```

여기서 `T_{[t]}`는 gating 없는 원래 DeltaNet의 T matrix (Eq. 6):
```
T_{[t]} = [I + strictLower(diag(β_{[t]}) · K_{[t]} K^T_{[t]})]^{-1} · diag(β_{[t]})
```

> **주의:** Ũ_g와 W는 서로 다른 T matrix를 사용한다. Ũ_g는 decay가 포함된 T̃를, W는 원래 T를 사용.

#### 4.3.3 Decay 표기 (← →)

```
←q^r_{[t]} = γ^r_{[t]} · q^r_{[t]}           // decay to first position
←w^r_{[t]} = γ^r_{[t]} · w^r_{[t]}           // decay to first position
→k^r_{[t]} = (γ^C_{[t]} / γ^r_{[t]}) · k^r_{[t]}   // decay to last position
→S_{[t]}   = γ^C_{[t]} · S_{[t]}             // decay state over entire chunk
```

#### 4.3.4 최종 Chunkwise 수식

**Output 계산 (per chunk):**
```
O_{[t]} = ←Q_{[t]} · S^T_{[t]} + (Q_{[t]} K^T_{[t]} ⊙ Γ_{[t]}) · (Ũ_g[t] − ←W_{[t]} · S^T_{[t]})
```

- 첫 번째 항: **inter-chunk** — 이전 chunk까지 누적된 state로부터의 기여 (matmul)
- 두 번째 항: **intra-chunk** — 현재 chunk 내 local attention (masked matmul)

**State 업데이트 (per chunk):**
```
S_{[t+1]} = →S_{[t]} + (Ũ_g[t] − ←W_{[t]} · S^T_{[t]})^T · →K_{[t]}
```

#### 4.3.5 알고리즘 복잡도

| 연산 | 복잡도 |
|------|--------|
| T̃ 계산 (C×C triangular solve) | O(C²) |
| Ũ_g = T̃ · V | O(C² · d_v) |
| W = T · K | O(C² · d_k) |
| Inter-chunk: ←Q · S^T | O(C · d_k · d_v) |
| Intra-chunk QK^T | O(C² · d_k) |
| Intra-chunk apply mask & matmul | O(C² · d_v) |
| State update | O(C · d_k · d_v) |

총: O(L/C · (C² · d + C · d²)) where d = max(d_k, d_v).
C = O(d)이면 O(L · d²)로 linear in sequence length.

---

## 5. 텐서 사양

### 5.1 입력 텐서

| 텐서 | Shape | Dtype | 설명 |
|------|-------|-------|------|
| `q` | `[B, T, H_qk, K]` | bf16 | Query |
| `k` | `[B, T, H_qk, K]` | bf16 | Key (L2-normalized) |
| `v` | `[B, T, H_v, V]` | bf16 | Value |
| `state` | `[B, H_v, V, K]` | float32 | 초기 hidden state |
| `A_log` | `[H_v]` | float32 | Gate 파라미터 |
| `a` | `[B, T, H_v]` | bf16 | Gate 입력 |
| `dt_bias` | `[H_v]` | float32 | Gate bias |
| `b` | `[B, T, H_v]` | bf16 | Beta 입력 |
| `scale` | scalar | float32 | `1/sqrt(K)` |

### 5.2 출력 텐서

| 텐서 | Shape | Dtype | 설명 |
|------|-------|-------|------|
| `output` | `[B, T, H_v, V]` | bf16 | 전체 시퀀스 출력 |
| `state` | `[B, H_v, V, K]` | float32 | 최종 hidden state (in-place) |

### 5.3 고정 차원

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `H_qk` | **4** | Query/Key head 수 |
| `H_v` | **8** | Value head 수 |
| `K` | **128** | Key/Query head dim |
| `V` | **128** | Value head dim |
| GQA ratio | **2** | H_v / H_qk |

### 5.4 가변 차원

| 파라미터 | 범위 | 설명 |
|---------|------|------|
| `B` | 1 ~ 수십 | Batch size |
| `T` | 수백 ~ 수천+ | Sequence length |
| `C` (chunk size) | 설계 선택 (64, 128, 256 등) | Chunk size (tunable) |

---

## 6. GQA (Grouped Query Attention) 처리

Decode와 동일:
- v_head 0,1 → qk_head 0
- v_head 2,3 → qk_head 1
- v_head 4,5 → qk_head 2
- v_head 6,7 → qk_head 3

Q, K 텐서는 `[B, T, H_qk=4, K]`이고, 각 v_head에 대해 대응하는 qk_head의 Q, K를 사용한다.

---

## 7. Chunkwise Parallel Algorithm — CUDA 구현 상세

### 7.1 알고리즘 전체 흐름 (의사코드)

```
for each (batch b, v_head h) in parallel:    // Grid 차원
    qk_h = h / 2                              // GQA mapping
    S = initial_state[b, h]                   // [V, K] (k-last)
    
    for chunk_idx = 0 to (T / C - 1):        // chunk 순차 처리 (inter-chunk dependency)
        // 1. 현재 chunk의 Q, K, V, α, β 슬라이스
        Q_c = q[b, chunk_idx*C : (chunk_idx+1)*C, qk_h, :]   // [C, K]
        K_c = k[b, chunk_idx*C : (chunk_idx+1)*C, qk_h, :]   // [C, K]
        V_c = v[b, chunk_idx*C : (chunk_idx+1)*C, h, :]       // [C, V]
        α_c, β_c = compute_gates(...)                          // [C] each
        
        // 2. Cumulative decay γ 계산
        γ[j] = Π_{i=1}^{j} α_c[i]   for j = 1..C             // [C]
        
        // 3. Decay mask Γ 구축 (C × C 하삼각)
        Γ[i][j] = γ[i] / γ[j]   if i >= j, else 0            // [C, C]
        
        // 4. T̃ matrix 계산 (gated, for Ũ_g)
        //    [I + strictLower(diag(β) · (Γ ⊙ KK^T))]^{-1} · diag(β)
        //    → C×C lower triangular system solve
        
        // 5. T matrix 계산 (ungated, for W)
        //    [I + strictLower(diag(β) · KK^T)]^{-1} · diag(β)
        
        // 6. Ũ_g = T̃ · V_c    [C, V]
        // 7. W = T · K_c        [C, K]
        
        // 8. Decay 적용
        ←Q = diag(γ) · Q_c                                    // [C, K]
        ←W = diag(γ) · W                                      // [C, K]
        →K = diag(γ_C / γ) · K_c                              // [C, K]
        
        // 9. Inter-chunk output: ←Q · S^T                    // [C, K] × [K, V] → [C, V]
        O_inter = ←Q @ S^T                                    // matmul!
        
        // 10. Correction term: ←W · S^T                      // [C, K] × [K, V] → [C, V]
        correction = ←W @ S^T                                  // matmul!
        
        // 11. Intra-chunk: (QK^T ⊙ Γ) · (Ũ_g − correction)
        QK = Q_c @ K_c^T                                      // [C, C] matmul!
        QK_masked = QK ⊙ Γ                                    // elementwise [C, C]
        O_intra = QK_masked @ (Ũ_g − correction)              // [C, C] × [C, V] → [C, V] matmul!
        
        // 12. 최종 output
        O_c = scale * (O_inter + O_intra)                     // [C, V]
        output[b, chunk_idx*C : (chunk_idx+1)*C, h, :] = O_c
        
        // 13. State 업데이트
        →S = γ_C · S                                           // [V, K] scaling
        delta = (Ũ_g − correction)^T @ →K                     // [V, C] × [C, K] → [V, K] matmul!
        S = →S + delta
    
    final_state[b, h] = S
```

### 7.2 핵심 Matmul 목록 (per chunk, per head)

| # | 연산 | 크기 | FLOP 개수 |
|---|------|------|----------|
| 1 | `KK^T` (for T, T̃ 계산) | [C,K] × [K,C] → [C,C] | 2·C²·K |
| 2 | `T̃ · V` (Ũ_g) | [C,C] × [C,V] → [C,V] | 2·C²·V |
| 3 | `T · K` (W) | [C,C] × [C,K] → [C,K] | 2·C²·K |
| 4 | `←Q · S^T` (inter-chunk) | [C,K] × [K,V] → [C,V] | 2·C·K·V |
| 5 | `←W · S^T` (correction) | [C,K] × [K,V] → [C,V] | 2·C·K·V |
| 6 | `Q · K^T` (intra-chunk QK) | [C,K] × [K,C] → [C,C] | 2·C²·K |
| 7 | `QK_masked · (Ũ_g−corr)` | [C,C] × [C,V] → [C,V] | 2·C²·V |
| 8 | `delta^T · →K` (state update) | [V,C] × [C,K] → [V,K] | 2·C·K·V |

총 per chunk: `6·C²·K + 2·C²·V + 6·C·K·V` FLOPs  
K=V=128, C=128 기준: `6·128²·128 + 2·128²·128 + 6·128·128·128 = 8·128³ + 6·128³ = 14·128³ ≈ 29.4M` FLOPs per chunk per head.

### 7.3 T Matrix 계산 (Lower Triangular Solve)

T̃ 계산은 `[I + strictLower(M)]^{-1} · diag(β)` 형태이다. 여기서 `M = diag(β) · (Γ ⊙ KK^T)`.

이는 **forward substitution**으로 풀 수 있다:

```
// A = I + strictLower(diag(β) · (Γ ⊙ KK^T))
// Solve A · X = diag(β) for X (= T̃)
// Since A is unit lower triangular, forward substitution:

for i = 0 to C-1:
    T̃[i, :] = β[i] · e_i         // 초기화 (단위 벡터 × β)
    for j = 0 to i-1:
        T̃[i, :] -= A[i, j] · T̃[j, :]
```

또는 행 단위로:
```
t̃_i = β_i · (e_i − Σ_{j<i} A[i,j] · t̃_j)
```

이 연산은 **sequential dependency**가 있어 C 내에서 완전 병렬화가 어렵다. 그러나 C가 작으면 (64~128) thread block 내에서 처리 가능하다.

**실용적 접근:**
- C ≤ 128이면 warp 내에서 iterative하게 계산
- 각 row의 계산이 이전 row에 의존하므로, row 순서대로 처리하되, 각 row 내의 dot product는 병렬화
- 또는 재귀적으로: `t̃_i`를 구할 때 `A[i, 0..i-1] · T̃[0..i-1, :]`은 이미 계산된 row들과의 matvec

### 7.4 Chunk Size 선택 가이드

| C | 장점 | 단점 |
|---|------|------|
| 64 | Shared mem 부담 적음, T matrix 빠름 | Matmul 크기 작아 tensor core 효율 낮음 |
| 128 | K=V=128과 맞아 자연스러운 tiling | State [V,K]=64KB + 추가 버퍼 필요 |
| 256 | Matmul 크기 커서 tensor core 효율 높음 | T matrix 계산 O(C²) 부담, smem 부족 가능 |

**권장:** C=64 또는 C=128. B200의 shared memory가 228KB이므로 C=128도 충분히 가능.

---

## 8. CUDA 커널 아키텍처 설계

### 8.1 병렬화 전략

```
Grid:  (num_chunks_or_batches, B * H_v)  또는 (B, H_v)
Block: (num_threads)  — chunk 내 연산을 cooperative하게 처리
```

**핵심 제약:** Chunk 간에는 state dependency가 있으므로, 같은 (batch, head)의 chunk들은 순차 처리해야 한다.

**Option A: 1 block per (batch, head)**
- Block이 chunk들을 순차 반복
- Block 내 스레드들이 chunk 내 matmul 병렬 수행
- 단순하지만 B·H_v가 작으면 GPU 활용 부족

**Option B: Multi-block per (batch, head) with chunk pipelining**
- 각 chunk를 별도 block으로 처리하되, 이전 chunk의 state를 global memory 통해 전달
- Block 간 동기화 필요 (atomic flag 등)
- 복잡하지만 large T일 때 유리

**Option C: Fused kernel + cuBLAS 호출**
- T matrix 등 작은 sequential 부분은 커스텀 커널
- 큰 matmul들은 cuBLAS batch GEMM 호출
- Kernel launch overhead가 반복됨

**권장:** Phase 1에서는 Option A로 시작. GPU occupancy가 부족하면 Option B로 전환.

### 8.2 Block 내부 연산 분할

C=128, K=V=128 기준으로 한 block이 처리할 연산:

```
Block (예: 256 threads = 8 warps)

Step 1: Load Q_c, K_c, V_c, α_c, β_c into shared memory
Step 2: Compute γ (prefix product of α) — 1 warp, sequential scan
Step 3: Compute KK^T — [128, 128] × [128, 128]^T → [128, 128]
         → 각 warp가 KK^T의 일부 행 담당
Step 4: Build A matrix, forward substitution for T̃ and T
Step 5: Ũ_g = T̃ · V, W = T · K — matmul
Step 6: Decay 적용 (elementwise)
Step 7: Inter-chunk: ←Q · S^T — matmul
Step 8: Correction: ←W · S^T — matmul  
Step 9: QK^T 계산 및 mask 적용
Step 10: Intra-chunk matmul
Step 11: Output 합산 및 저장
Step 12: State 업데이트
Step 13: 다음 chunk로 이동 (Step 1로)
```

### 8.3 Shared Memory 사용 계획

C=128, K=V=128, float32 기준:

| 버퍼 | Size | Bytes |
|------|------|-------|
| State S | V × K | 64 KB |
| Q_c | C × K | 64 KB |
| K_c | C × K | 64 KB |
| V_c | C × V | 64 KB |
| T̃ or T matrix | C × C | 64 KB |
| KK^T / QK^T | C × C | 64 KB |
| Ũ_g, W | C × max(K,V) | 64 KB |
| γ, β, etc. | C | 0.5 KB |

전부 동시에 올리면 ~448 KB → B200의 228KB 초과.

**해결 방안:**
1. **버퍼 재사용 (time-multiplexing):** 모든 버퍼를 동시에 필요로 하지 않음. 예를 들어 T matrix 계산이 끝나면 해당 smem을 Ũ_g로 재활용
2. **State를 register file에 분산 보관:** 각 스레드가 state의 일부 행을 register에 보유
3. **Chunk size 축소:** C=64이면 대부분의 버퍼가 반으로 줄어듦
4. **일부 중간 결과는 global memory 임시 저장**

### 8.4 Tensor Core 활용 (WMMA / MMA)

B200은 4세대 Tensor Core를 지원한다. Matmul 집약적인 연산에 적극 활용:

- `mma.sync` 또는 `wmma` API 사용
- bf16 입력 → float32 누적 모드 (`wmma::accumulator` in float32)
- [C, K] × [K, V] matmul은 C=128, K=128, V=128 → 표준 GEMM tile 크기와 호환

**적용 가능한 연산:**
- Step 5: Ũ_g = T̃ · V (C×C × C×V)
- Step 7: ←Q · S^T (C×K × K×V)
- Step 8: ←W · S^T (C×K × K×V)
- Step 9: Q · K^T (C×K × K×C)
- Step 10: QK_masked · delta (C×C × C×V)
- Step 12: delta^T · →K (V×C × C×K)

---

## 9. 순차 Recurrent 구현 (Baseline / Phase 1)

Chunkwise 알고리즘이 복잡하므로, 먼저 정확성 검증을 위한 naive sequential 구현을 권장:

```cuda
// 각 (batch, head) 블록에서:
for (int t = 0; t < T; t++) {
    float g = compute_gate(A_log[h], a[b,t,h], dt_bias[h]);
    float beta = sigmoid(b[b,t,h]);
    
    // Load q, k, v for this timestep
    float q_vec[K], k_vec[K], v_vec[V];
    load_qkv(q, k, v, b, t, h, q_vec, k_vec, v_vec);
    
    // State update: S = g*S*(I - beta*kk^T) + beta*v*k^T
    // Equivalently:
    // old_v = k @ S^T  (matvec)
    // delta = beta * (v - old_v)
    // S += outer(delta, k)   (after scaling S by g)
    
    scale_state(S, g);                    // S *= g
    float old_v[V];
    matvec(S, k_vec, old_v);              // old_v = k @ S (treating S as [K,V])
    float delta[V];
    for (int i = 0; i < V; i++)
        delta[i] = beta * (v_vec[i] - old_v[i]);
    rank1_update(S, k_vec, delta);        // S += outer(k, delta)
    
    // Output: o = scale * q @ S
    float out[V];
    matvec(S, q_vec, out);               // out = q @ S
    store_output(output, b, t, h, out, scale);
}
```

이 구현은 O(T · K · V) per (batch, head)이고, 정확성 확인용이다.

---

## 10. Reference Python 구현 (Prefill용 확장)

Decode reference를 T>1로 확장한 sequential version:

```python
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def run_prefill(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Gates: [B, T, H_v]
    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, T, H_v]
    beta = torch.sigmoid(b.float())

    q_f32 = q.float()         # [B, T, H_qk, K]
    k_f32 = k.float()
    v_f32 = v.float()         # [B, T, H_v, V]

    state_f32 = state.float() if state is not None else \
        torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=q.device)

    # GQA expansion
    q_exp = q_f32.repeat_interleave(num_heads // num_q_heads, dim=2)  # [B,T,H_v,K]
    k_exp = k_f32.repeat_interleave(num_heads // num_k_heads, dim=2)

    output = torch.zeros(B, T, num_heads, V, dtype=torch.float32, device=q.device)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            S = state_f32[b_idx, h_idx].clone().transpose(-1, -2)  # [K, V]
            
            for t in range(T):
                q_h = q_exp[b_idx, t, h_idx]    # [K]
                k_h = k_exp[b_idx, t, h_idx]    # [K]
                v_h = v_f32[b_idx, t, h_idx]    # [V]
                g_val = g[b_idx, t, h_idx]      # scalar
                beta_val = beta[b_idx, t, h_idx]

                old_state = g_val * S
                old_v = k_h @ old_state           # [V]
                new_v = beta_val * v_h + (1 - beta_val) * old_v
                S = old_state - k_h.unsqueeze(1) @ old_v.unsqueeze(0) \
                              + k_h.unsqueeze(1) @ new_v.unsqueeze(0)
                
                output[b_idx, t, h_idx] = scale * (q_h @ S)
            
            state_f32[b_idx, h_idx] = S.transpose(-1, -2)

    return output.to(torch.bfloat16), state_f32
```

---

## 11. `binding.py` 구현 요구사항

### 11.1 TVM FFI 바인딩

```python
import ctypes
from tvm.ffi import register_func

@register_func("flashinfer.kernel")
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output):
    """
    Args (DPS style — output이 마지막 파라미터):
        q:      [B, T, H_qk, K]  bf16
        k:      [B, T, H_qk, K]  bf16
        v:      [B, T, H_v, V]   bf16
        state:  [B, H_v, V, K]   float32 (in-place update)
        A_log:  [H_v]            float32
        a:      [B, T, H_v]      bf16
        dt_bias:[H_v]            float32
        b:      [B, T, H_v]      bf16
        scale:  float scalar
        output: [B, T, H_v, V]   bf16 (pre-allocated)
    """
    # 1. Shape 추출
    B, T = q.shape[0], q.shape[1]
    
    # 2. Grid/Block 결정
    grid = (B, 8)  # 8 = H_v
    block = (256,)  # or 128, depends on kernel design
    smem_size = ...  # shared memory 크기
    
    # 3. 커널 런치
    # (TVM FFI 또는 ctypes로 .so 로드하여 호출)
    pass
```

### 11.2 PyTorch 바인딩 (alternative)

```python
import torch

def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output):
    # torch.utils.cpp_extension.load_inline 또는 미리 컴파일된 .so 사용
    pass
```

> **정확한 시그니처:** FlashInfer-Bench 데이터셋의 `gdn_prefill_qk4_v8_d128_k_last` definition JSON을 반드시 확인하여 파라미터 순서와 이름을 맞출 것.

---

## 12. 검증 & 평가

### 12.1 정확성

- Reference sequential 구현 대비 element-wise 비교
- Prefill은 누적 오차가 발생할 수 있으므로, chunk 단위로 중간 state도 비교 권장

### 12.2 성능

- Baseline `flashinfer_wrapper_123ca6` 대비 latency 비교
- `--warmup-runs 1 --iterations 5 --num-trials 3` 으로 평가

### 12.3 로컬 테스트

```bash
python scripts/pack_solution.py
python scripts/run_local.py
```

---

## 13. 개발 로드맵

### Phase 1: Naive Sequential (정확성 확보)
- Step-by-step recurrence를 CUDA로 포팅
- 1 block per (batch, head), 내부 sequential loop
- K/V 차원만 thread 병렬화
- **목표:** 모든 workload 정확성 통과

### Phase 2: Chunkwise Parallel (기본)
- Chunk size C=64 또는 128 선택
- T matrix forward substitution 구현
- Chunk 내 matmul들을 shared memory 기반으로 구현
- State를 register 또는 shared memory에 유지
- **목표:** Baseline 대비 2-5× 가속

### Phase 3: Tensor Core 활용
- WMMA 또는 MMA intrinsic으로 핵심 matmul 교체
- bf16 matmul with float32 accumulation
- Tile 크기를 tensor core에 맞게 조정 (16×16×16 등)
- **목표:** Baseline 대비 5-10× 이상

### Phase 4: 고급 최적화
- Double buffering: 현재 chunk 연산 중 다음 chunk 데이터 prefetch
- Pipeline: state update와 output 계산 overlap
- Shared memory bank conflict 제거
- Register pressure 최적화
- Multi-block per head (large T 대응)
- Chunk size auto-tuning
- **목표:** B200 peak에 근접

---

## 14. 주의사항

1. **State layout `[V, K]` (k-last):** Reference code에서 `transpose(-1, -2)`로 `[K, V]`로 변환하여 연산한다. 커널에서도 이 변환을 정확히 처리하거나, 처음부터 `[K, V]`로 다루고 마지막에 write-back 시 전환.

2. **Chunk boundary 처리:** T가 C로 나누어 떨어지지 않을 수 있다. 마지막 chunk는 실제 크기가 C보다 작을 수 있으므로 padding 또는 masking 처리 필요.

3. **T̃ vs T:** Ũ_g 계산에는 decay가 포함된 T̃를, W 계산에는 원래 T를 사용한다. 두 matrix를 혼동하면 정확성이 깨진다.

4. **Cumulative decay product γ의 수치 안정성:** α가 매우 작으면 γ가 underflow할 수 있다. Log-space에서 계산하는 것을 고려.

5. **Forward substitution 정밀도:** T matrix 계산 시 float32로 수행. C가 커지면 누적 오차 주의.

6. **Inter-chunk state 전달:** 같은 (batch, head)의 chunk들은 반드시 순서대로 처리. State를 shared memory 또는 register에 유지하여 global memory round-trip 최소화.

7. **GQA 인덱싱:** Q, K는 `H_qk=4` head, V/α/β/state는 `H_v=8` head. 혼동 금지.

8. **DPS 준수:** output 텐서는 미리 할당되어 전달됨. `state`도 in-place update.

---

## 15. 참고 자료

- **논문:** "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025)
  - Section 2.2: DeltaNet chunkwise (Eq. 3-9) — 기반 알고리즘
  - Section 3.3: Gated DeltaNet chunkwise (핵심) — decay term 확장
  - Appendix A: Extended WY representation 증명
- **공식 코드:** https://github.com/NVlabs/GatedDeltaNet
- **Flash Linear Attention:** https://github.com/fla-org/flash-linear-attention
  - `fla/ops/generalized_delta_rule` 디렉토리에 Triton 기반 참고 구현 존재
- **FlashInfer-Bench:** https://github.com/flashinfer-ai/flashinfer-bench
- **경진대회 레포:** https://github.com/jaewooMaeng/GDN_kernel
- **데이터셋:** https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
- **WY Representation:** Bischof & Van Loan, 1985 — Householder matrix 누적곱의 효율적 표현
- **UT Transform:** Joffrain et al., 2006 — W, U의 행렬 형태 도출

---

## 부록 C: Flash Linear Attention (fla) 참조 구현

`fla-org/flash-linear-attention` 레포의 Triton 기반 Gated DeltaNet 구현이다.  
CUDA 커널 작성 시 **알고리즘 흐름, 분해 구조, 수치 처리 패턴**을 이해하는 데 핵심 참조 자료로 활용할 것.

설치: `pip install fla-core`

### C.1 naive.py — 순차 Recurrent & Chunk Reference (정확성 기준)

소스 위치: `fla/ops/gated_delta_rule/naive.py`

```python
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F
from einops import rearrange


def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of recurrent gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]          ← NOTE: g is in LOG SPACE (not raw α)
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]   # ← gate in log space → exp
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)    # ← old_v retrieval
        b_v = b_v * b_beta[..., None]                       # ← delta_v = beta * (v - old_v)
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)  # ← rank-1 update
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


def naive_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of chunk gated delta rule.
    
    이 코드가 논문 Section 3.3의 chunkwise parallel form을 직접 구현한 것이다.
    CUDA 커널의 알고리즘 정확성 검증에 가장 중요한 참조.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H]          ← LOG SPACE
        beta: [B, T, H]
        chunk_size: int
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool
    """
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))

    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]       # ← v_beta = beta * v
    k_beta = k * beta[..., None]  # ← k_beta = beta * k
    assert l % chunk_size == 0

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        [q, k, v, k_beta, decay.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)           # ← cumulative log decay γ (in log space)
    decay_exp = decay.exp()[..., None]
    # L_mask: Γ[i,j] = exp(γ_i - γ_j) for i >= j, 0 otherwise
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()

    # ===== T̃ matrix 계산 (forward substitution) =====
    # A = -strictLower(diag(β) · (Γ ⊙ KK^T))
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    # Forward substitution: solve (I + strictLower(M))^{-1}
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + \
            (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    # attn은 이제 T̃ (또는 논문의 T matrix의 gated 버전에 해당)

    # ===== Ũ_g = T̃ @ (beta * v), W처리도 포함 =====
    k_cumsum = attn @ v             # ← 이것이 Ũ_g에 해당 (실제로는 u = T̃ @ v_beta)
    k_cumdecay = attn @ (k_beta * decay_exp)  # ← 이것이 ←W에 해당 (decay 적용된 W)
    v = k_cumsum                    # v를 u(= Ũ_g)로 교체

    S = k.new_zeros(b, h, d_k, d_v)   # State: [B, H, K, V]
    if initial_state is not None:
        S = initial_state.to(torch.float32)

    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)

    # ===== Chunk 순회 =====
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        
        # Intra-chunk attention: QK^T with decay mask
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        
        # Correction: ←W @ S^T (= k_cumdecay @ S)
        v_prime = (k_cumdecay[:, :, i]) @ S
        
        # v_new = Ũ_g - correction
        v_new = v_i - v_prime
        
        # Inter-chunk: ←Q @ S^T (with decay applied to Q)
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        
        # Output = inter + intra @ v_new
        o[:, :, i] = o_inter + attn @ v_new
        
        # State update: →S + (v_new)^T @ →K
        S = S * decay[:, :, i, -1, None, None].exp() + \
            (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()
             [..., None]).transpose(-1, -2) @ v_new

    if not output_final_state:
        S = None

    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S
```

### C.2 chunk.py — 최적화된 Triton 기반 Chunk 구현 (Forward 흐름)

소스 위치: `fla/ops/gated_delta_rule/chunk.py`

이 파일은 여러 Triton sub-kernel들을 조합한 high-level orchestration이다.  
Forward의 핵심 흐름:

```python
def chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, output_final_state, ...):
    # Step 1: Cumulative local sum of g (log-space gate)
    g = chunk_local_cumsum(g, chunk_size=64, ...)
    
    # Step 2: Compute scaled dot(k, k^T) with gating → A matrix의 하삼각 부분
    A = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, ...)
    
    # Step 3: Triangular solve → T̃ matrix (= A^{-1})
    A = solve_tril(A=A, ...)
    
    # Step 4: WY representation → w (←W에 해당), u (Ũ_g에 해당)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, ...)
    
    # Step 5: Inter-chunk state propagation → h (chunk별 시작 state), v_new (Ũ_g - correction)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=..., ...)
    
    # Step 6: Chunk output 계산 (inter + intra)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, ...)
    
    return g, o, A, final_state, initial_state
```

핵심 sub-kernel 목록:

| Sub-kernel | 위치 | 역할 |
|-----------|------|------|
| `chunk_local_cumsum` | `fla.ops.utils` | chunk 내 g의 cumulative sum (log space) |
| `chunk_scaled_dot_kkt_fwd` | `fla.ops.common.chunk_scaled_dot_kkt` | `diag(β) · (Γ ⊙ KK^T)` 하삼각 계산 |
| `solve_tril` | `fla.ops.utils` | `(I + strictLower(M))^{-1}` forward substitution |
| `recompute_w_u_fwd` | `fla.ops.gated_delta_rule.wy_fast` | T̃ @ (β·V) → u, T̃ @ (β·g·K) → w |
| `chunk_gated_delta_rule_fwd_h` | `fla.ops.common.chunk_delta_h` | Inter-chunk state 전파 + correction |
| `chunk_fwd_o` | `fla.ops.common.chunk_o` | 최종 output 계산 (inter + intra) |

### C.3 wy_fast.py — WY Representation 계산 Triton 커널 (핵심 연산)

소스 위치: `fla/ops/gated_delta_rule/wy_fast.py`

`recompute_w_u_fwd_kernel`의 핵심 로직 (Triton):

```python
# Grid: (num_chunks, B*H) — 각 program이 1개 chunk의 1개 head 처리
# 입력: k[BT,K], v[BT,V], beta[BT], A[BT,BT] (T̃ matrix), g[BT] (cumsum된 log gate)

# u = T̃ @ (beta * v) — Ũ_g 계산
b_vb = (b_v * b_b[:, None])      # v * beta
b_u = tl.dot(b_A, b_vb)          # T̃ @ v_beta

# w = T̃ @ (beta * exp(g) * k) — ←W 계산 (decay 포함)
b_kb = b_k * b_b[:, None]        # k * beta
b_kb *= b_g[:, None]             # k * beta * exp(g)  ← decay 적용!
b_w = tl.dot(b_A, b_kb)          # T̃ @ k_beta_g
```

### C.4 FLA 구현에서 얻을 수 있는 핵심 인사이트

1. **g는 LOG SPACE로 전달된다.** Reference code에서의 `exp(-exp(A_log) * softplus(a + dt_bias))`가 이미 계산되어 log space(`log(α)`)로 전달됨. 따라서 FlashInfer 벤치마크의 커널 시그니처에서 g가 어떤 형태인지 반드시 확인할 것.

2. **Chunk size는 64로 고정되어 있다** (`BT=64`). FLA에서 이 값을 선택한 이유는 T matrix의 forward substitution이 O(C²)이고, shared memory 제약 때문.

3. **State layout은 `[B, H, K, V]`이다** (K가 먼저). FlashInfer 대회의 definition은 `k_last` = `[B, H, V, K]`이므로 전치가 필요할 수 있다.

4. **분해 구조:** FLA는 연산을 6개의 독립 sub-kernel로 분해했다. CUDA 커널에서는 이들을 **하나의 fused kernel로 합치거나**, 또는 **유사한 분해 구조를 따르되 kernel launch 오버헤드를 줄이는 방식**을 고려해야 한다.

5. **Forward substitution (`solve_tril`)은 sequential하다.** C=64의 64×64 lower triangular solve. 이 연산은 병렬화 불가능한 bottleneck이며, FLA에서도 별도 커널로 분리했다.

6. **Blackwell 대응:** `wy_fast.py`에서 `IS_NVIDIA_BLACKWELL` 분기로 `safe_dot` workaround가 있다. Triton compiler 버그 관련이며, CUDA 커널에서는 해당 없지만 참고할 것.

7. **Autotune:** FLA는 `num_warps ∈ {2,4,8}`, `num_stages ∈ {2,3,4}`로 autotune한다. CUDA에서도 유사한 파라미터 탐색이 필요하다.