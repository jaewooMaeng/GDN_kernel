#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

static constexpr int K_DIM = 128;
static constexpr int V_DIM = 128;
static constexpr int NUM_Q_HEADS = 4;
static constexpr int NUM_V_HEADS = 8;
static constexpr int GQA_RATIO = NUM_V_HEADS / NUM_Q_HEADS;
static constexpr int CHUNK = 64;
static constexpr int NTHREADS = 128;
static constexpr int NWARPS = NTHREADS / 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Shared memory layout (total ~100KB, within B200's 228KB):
//   T_mat    [C*C]       16 KB  - T̃ matrix (lower triangular)
//   QK_mat   [C*C]       16 KB  - QK^T masked
//   K_c      [C*K]       32 KB  - chunk keys
//   buf_D    [C*K]       32 KB  - Q_c (temp), then w_decay
//   gamma_s  [C]         256 B  - cumulative log decay
//   beta_s   [C]         256 B  - beta values
//   v_beta_buf[C]        256 B  - beta*v per vi
//   inter_buf[C]         256 B  - inter results
//   corr_buf [C]         256 B  - correction results
//   v_new_buf[C]         256 B  - u - correction
//   xwarp_buf[4*C*2]     2 KB   - cross-warp reduction buffer

static constexpr int SMEM_FLOATS = 2*CHUNK*CHUNK + 2*CHUNK*K_DIM + 6*CHUNK + 4*CHUNK*2;
static constexpr int SMEM_BYTES  = SMEM_FLOATS * sizeof(float);

__global__ __launch_bounds__(NTHREADS)
void gdn_prefill_chunk_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ init_state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_in,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_in,
    const int64_t* __restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane_id  = tid & 31;
    const int qk_head  = head_idx / GQA_RATIO;

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end   = static_cast<int>(cu_seqlens[seq_idx + 1]);
    const int seq_len   = seq_end - seq_start;
    const int state_off = (seq_idx * NUM_V_HEADS + head_idx) * V_DIM * K_DIM;

    if (seq_len <= 0) {
        for (int vi = 0; vi < V_DIM; vi++)
            new_state[state_off + vi * K_DIM + tid] =
                init_state[state_off + vi * K_DIM + tid];
        return;
    }

    // State in registers: thread tid holds column tid of S[V][K]
    float S[V_DIM];
    for (int vi = 0; vi < V_DIM; vi++)
        S[vi] = __ldg(&init_state[state_off + vi * K_DIM + tid]);

    const float A_val  = __ldg(&A_log[head_idx]);
    const float dt_val = __ldg(&dt_bias[head_idx]);

    extern __shared__ float smem[];
    float* T_mat      = smem;
    float* QK_mat     = T_mat + CHUNK * CHUNK;
    float* K_c        = QK_mat + CHUNK * CHUNK;
    float* buf_D      = K_c + CHUNK * K_DIM;
    float* gamma_s    = buf_D + CHUNK * K_DIM;
    float* beta_s     = gamma_s + CHUNK;
    float* v_beta_buf = beta_s + CHUNK;
    float* inter_buf  = v_beta_buf + CHUNK;
    float* corr_buf   = inter_buf + CHUNK;
    float* v_new_buf  = corr_buf + CHUNK;
    float* xwarp_buf  = v_new_buf + CHUNK;

    const int num_chunks = (seq_len + CHUNK - 1) / CHUNK;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int cstart = seq_start + chunk * CHUNK;
        const int clen   = min(CHUNK, seq_end - cstart);

        // ============ 1. Gates: log(alpha), beta ============
        if (tid < CHUNK) {
            if (tid < clen) {
                int pos = cstart + tid;
                float a_v = __bfloat162float(__ldg(&a_in[pos * NUM_V_HEADS + head_idx]));
                float b_v = __bfloat162float(__ldg(&b_in[pos * NUM_V_HEADS + head_idx]));
                float x = a_v + dt_val;
                float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
                gamma_s[tid] = -expf(A_val) * sp;
                beta_s[tid]  = 1.0f / (1.0f + expf(-b_v));
            } else {
                gamma_s[tid] = 0.0f;
                beta_s[tid]  = 0.0f;
            }
        }
        __syncthreads();

        // ============ 2. Prefix sum of log-decay ============
        if (tid == 0)
            for (int i = 1; i < CHUNK; i++)
                gamma_s[i] += gamma_s[i - 1];
        __syncthreads();

        // ============ 3. Load K_c[C][K] ============
        for (int c = 0; c < CHUNK; c++) {
            float kv = 0.0f;
            if (c < clen)
                kv = __bfloat162float(__ldg(&k[(cstart + c) * NUM_Q_HEADS * K_DIM + qk_head * K_DIM + tid]));
            K_c[c * K_DIM + tid] = kv;
        }
        __syncthreads();

        // ============ 4. A matrix = -strictLower(k_beta @ k^T * L_mask) ============
        // Parallelized: each thread handles ~32 of the 4096 elements
        for (int idx = tid; idx < CHUNK * CHUNK; idx += NTHREADS) {
            int i = idx / CHUNK, j = idx % CHUNK;
            float val = 0.0f;
            if (i > j && i < clen && j < clen) {
                float dot = 0.0f;
                float bi = beta_s[i];
                for (int kk = 0; kk < K_DIM; kk++)
                    dot += K_c[i * K_DIM + kk] * bi * K_c[j * K_DIM + kk];
                val = -dot * expf(gamma_s[i] - gamma_s[j]);
            }
            T_mat[idx] = val;
        }
        __syncthreads();

        // ============ 5. Forward substitution: T̃ = (I + strictLower(A))^{-1} ============
        for (int i = 1; i < clen; i++) {
            float new_val = 0.0f;
            bool active = (tid < i);
            if (active) {
                int j = tid;
                float sum = 0.0f;
                for (int m = j + 1; m < i; m++)
                    sum += T_mat[i * CHUNK + m] * T_mat[m * CHUNK + j];
                new_val = T_mat[i * CHUNK + j] + sum;
            }
            __syncthreads();
            if (active)
                T_mat[i * CHUNK + tid] = new_val;
            __syncthreads();
        }
        // Add identity
        if (tid < CHUNK)
            T_mat[tid * CHUNK + tid] = 1.0f;
        __syncthreads();

        // ============ 6. Load Q_c into buf_D, save Q_local ============
        float Q_local[CHUNK];
        for (int c = 0; c < CHUNK; c++) {
            float qv = 0.0f;
            if (c < clen)
                qv = __bfloat162float(__ldg(&q[(cstart + c) * NUM_Q_HEADS * K_DIM + qk_head * K_DIM + tid]));
            Q_local[c] = qv;
            buf_D[c * K_DIM + tid] = qv;
        }
        __syncthreads();

        // ============ 7. QK_masked = (Q @ K^T) * L_mask, lower triangular ============
        for (int idx = tid; idx < CHUNK * CHUNK; idx += NTHREADS) {
            int i = idx / CHUNK, j = idx % CHUNK;
            float val = 0.0f;
            if (j <= i && i < clen && j < clen) {
                float dot = 0.0f;
                for (int kk = 0; kk < K_DIM; kk++)
                    dot += buf_D[i * K_DIM + kk] * K_c[j * K_DIM + kk];
                val = dot * expf(gamma_s[i] - gamma_s[j]);
            }
            QK_mat[idx] = val;
        }
        __syncthreads();

        // ============ 8. w_decay = T̃ @ (k * beta * exp(gamma)), overwrite buf_D ============
        for (int c = 0; c < CHUNK; c++) {
            float acc = 0.0f;
            int lim = min(c + 1, clen);
            for (int j = 0; j < lim; j++)
                acc += T_mat[c * CHUNK + j] * K_c[j * K_DIM + tid] * beta_s[j] * expf(gamma_s[j]);
            buf_D[c * K_DIM + tid] = acc;
        }
        __syncthreads();

        // ============ 9. Per-vi loop: inter, corr, u, intra, output, state ============
        float gamma_last = gamma_s[clen - 1];

        for (int vi = 0; vi < V_DIM; vi++) {
            float s_val = S[vi];

            // 9a. Load v_beta for this vi
            if (tid < clen) {
                int pos = cstart + tid;
                float vv = __bfloat162float(__ldg(&v[pos * NUM_V_HEADS * V_DIM + head_idx * V_DIM + vi]));
                v_beta_buf[tid] = beta_s[tid] * vv;
            }
            __syncthreads();

            // 9b. Batched cross-warp reduction for inter and corr
            for (int c = 0; c < clen; c++) {
                float q_eg  = Q_local[c] * expf(gamma_s[c]);
                float w_c   = buf_D[c * K_DIM + tid];
                float p_int = warp_reduce_sum(q_eg * s_val);
                float p_cor = warp_reduce_sum(w_c * s_val);
                if (lane_id == 0) {
                    xwarp_buf[warp_id * CHUNK * 2 + c * 2]     = p_int;
                    xwarp_buf[warp_id * CHUNK * 2 + c * 2 + 1] = p_cor;
                }
            }
            __syncthreads();

            // Combine warp sums
            if (tid < clen) {
                float si = 0.0f, sc = 0.0f;
                for (int w = 0; w < NWARPS; w++) {
                    si += xwarp_buf[w * CHUNK * 2 + tid * 2];
                    sc += xwarp_buf[w * CHUNK * 2 + tid * 2 + 1];
                }
                inter_buf[tid] = si;
                corr_buf[tid]  = sc;
            }
            __syncthreads();

            // 9c. u = T̃ @ v_beta, v_new = u - corr
            if (tid < clen) {
                float u_c = 0.0f;
                for (int j = 0; j <= tid; j++)
                    u_c += T_mat[tid * CHUNK + j] * v_beta_buf[j];
                v_new_buf[tid] = u_c - corr_buf[tid];
            }
            __syncthreads();

            // 9d. intra = QK_masked @ v_new, output = scale*(inter + intra)
            if (tid < clen) {
                float intra = 0.0f;
                for (int j = 0; j <= tid; j++)
                    intra += QK_mat[tid * CHUNK + j] * v_new_buf[j];
                int pos = cstart + tid;
                int oidx = pos * NUM_V_HEADS * V_DIM + head_idx * V_DIM + vi;
                output[oidx] = __float2bfloat16(scale * (inter_buf[tid] + intra));
            }

            // 9e. State update: S[vi] = s_val * exp(gamma_last) + sum_c v_new[c]*K_c[c][tid]*exp(gamma_last-gamma[c])
            float delta = 0.0f;
            for (int c = 0; c < clen; c++)
                delta += v_new_buf[c] * K_c[c * K_DIM + tid] * expf(gamma_last - gamma_s[c]);
            S[vi] = s_val * expf(gamma_last) + delta;
            __syncthreads();
        }
    }

    // Write final state
    for (int vi = 0; vi < V_DIM; vi++)
        new_state[state_off + vi * K_DIM + tid] = S[vi];
}

void GDNPrefillKernel(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView A_log,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView cu_seqlens,
    double scale,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView new_state
) {
    const int num_seqs = static_cast<int>(cu_seqlens.size(0)) - 1;

    float scale_f = static_cast<float>(scale);
    if (scale_f == 0.0f)
        scale_f = 1.0f / sqrtf(static_cast<float>(K_DIM));

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(k.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(v.data_ptr());
    const float* state_ptr = static_cast<const float*>(state.data_ptr());
    const float* A_log_ptr = static_cast<const float*>(A_log.data_ptr());
    const __nv_bfloat16* a_ptr = static_cast<const __nv_bfloat16*>(a.data_ptr());
    const float* dt_bias_ptr = static_cast<const float*>(dt_bias.data_ptr());
    const __nv_bfloat16* b_ptr = static_cast<const __nv_bfloat16*>(b.data_ptr());
    const int64_t* cu_seqlens_ptr = static_cast<const int64_t*>(cu_seqlens.data_ptr());
    __nv_bfloat16* output_ptr = static_cast<__nv_bfloat16*>(output.data_ptr());
    float* new_state_ptr = static_cast<float*>(new_state.data_ptr());

    if (state_ptr == nullptr) {
        cudaMemsetAsync(new_state_ptr, 0,
                        num_seqs * NUM_V_HEADS * V_DIM * K_DIM * sizeof(float), stream);
        state_ptr = new_state_ptr;
    }

    dim3 grid(num_seqs, NUM_V_HEADS);
    dim3 block(NTHREADS);

    cudaFuncSetAttribute(gdn_prefill_chunk_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_BYTES);

    gdn_prefill_chunk_kernel<<<grid, block, SMEM_BYTES, stream>>>(
        q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
        cu_seqlens_ptr, scale_f, output_ptr, new_state_ptr
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, GDNPrefillKernel);
