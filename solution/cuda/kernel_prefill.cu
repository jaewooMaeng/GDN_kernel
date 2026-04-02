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

// Shared memory: T_mat[C*C] + QK[C*C] + K_c[C*K] + buf_D[C*K] + gamma[C] + beta[C]
//              + v_new[C] + exp_gamma[C] + xwarp[4*C*2] + inter[C] + corr[C]
static constexpr int SMEM_FLOATS =
    2*CHUNK*CHUNK + 2*CHUNK*K_DIM + CHUNK*4 + 4*CHUNK*2 + 2*CHUNK;
static constexpr int SMEM_BYTES = SMEM_FLOATS * sizeof(float);

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
    float* v_new_buf  = beta_s + CHUNK;
    float* exp_gamma  = v_new_buf + CHUNK;
    float* xwarp_buf  = exp_gamma + CHUNK;
    float* inter_buf  = xwarp_buf + 4 * CHUNK * 2;
    float* corr_buf   = inter_buf + CHUNK;

    const int num_chunks = (seq_len + CHUNK - 1) / CHUNK;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int cstart = seq_start + chunk * CHUNK;
        const int clen   = min(CHUNK, seq_end - cstart);

        // ===== 1. Gates =====
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

        // ===== 2. Prefix sum gamma =====
        if (tid == 0)
            for (int i = 1; i < CHUNK; i++) gamma_s[i] += gamma_s[i - 1];
        __syncthreads();

        // ===== 2b. Precompute exp(gamma[c]) =====
        if (tid < CHUNK)
            exp_gamma[tid] = expf(gamma_s[tid]);
        __syncthreads();

        // ===== 3. Load K_c =====
        for (int c = 0; c < CHUNK; c++) {
            float kv = 0.0f;
            if (c < clen)
                kv = __bfloat162float(__ldg(&k[(cstart + c) * NUM_Q_HEADS * K_DIM + qk_head * K_DIM + tid]));
            K_c[c * K_DIM + tid] = kv;
        }
        __syncthreads();

        // ===== 4. A matrix (parallel dot products) =====
        for (int idx = tid; idx < CHUNK * CHUNK; idx += NTHREADS) {
            int i = idx / CHUNK, j = idx % CHUNK;
            float val = 0.0f;
            if (i > j && i < clen && j < clen) {
                float dot = 0.0f, bi = beta_s[i];
                for (int kk = 0; kk < K_DIM; kk++)
                    dot += K_c[i * K_DIM + kk] * bi * K_c[j * K_DIM + kk];
                val = -dot * expf(gamma_s[i] - gamma_s[j]);
            }
            T_mat[idx] = val;
        }
        __syncthreads();

        // ===== 5. Forward substitution =====
        for (int i = 1; i < clen; i++) {
            float nv = 0.0f;
            bool act = tid < i;
            if (act) {
                float s = 0.0f;
                for (int m = tid + 1; m < i; m++)
                    s += T_mat[i * CHUNK + m] * T_mat[m * CHUNK + tid];
                nv = T_mat[i * CHUNK + tid] + s;
            }
            __syncthreads();
            if (act) T_mat[i * CHUNK + tid] = nv;
            __syncthreads();
        }
        if (tid < CHUNK) T_mat[tid * CHUNK + tid] = 1.0f;
        __syncthreads();

        // ===== 6. Load Q and precompute Q_exp =====
        float Q_local[CHUNK];
        float Q_exp[CHUNK];
        for (int c = 0; c < CHUNK; c++) {
            float qv = 0.0f;
            if (c < clen)
                qv = __bfloat162float(__ldg(&q[(cstart + c) * NUM_Q_HEADS * K_DIM + qk_head * K_DIM + tid]));
            Q_local[c] = qv;
            buf_D[c * K_DIM + tid] = qv;
            Q_exp[c] = qv * exp_gamma[c];
        }
        __syncthreads();

        // ===== 7. QK_masked =====
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

        // ===== 8. w_decay (using precomputed exp_gamma) =====
        for (int c = 0; c < CHUNK; c++) {
            float acc = 0.0f;
            int lim = min(c + 1, clen);
            for (int j = 0; j < lim; j++)
                acc += T_mat[c * CHUNK + j] * K_c[j * K_DIM + tid] * beta_s[j] * exp_gamma[j];
            buf_D[c * K_DIM + tid] = acc;
        }
        __syncthreads();

        // ===== 9. Precompute K_decay and decay_total =====
        float gamma_last = gamma_s[clen - 1];
        float decay_total = expf(gamma_last);
        float K_decay[CHUNK];
        for (int c = 0; c < CHUNK; c++)
            K_decay[c] = K_c[c * K_DIM + tid] * expf(gamma_last - gamma_s[c]);

        // ===== 10. Per-vi loop =====
        for (int vi = 0; vi < V_DIM; vi++) {
            float s_val = S[vi];

            // Load v_beta
            if (tid < clen) {
                int pos = cstart + tid;
                float vv = __bfloat162float(__ldg(&v[pos * NUM_V_HEADS * V_DIM + head_idx * V_DIM + vi]));
                v_new_buf[tid] = beta_s[tid] * vv;
            }
            __syncthreads();

            // Batched cross-warp reduction for inter and corr
            for (int c = 0; c < clen; c++) {
                float p_int = warp_reduce_sum(Q_exp[c] * s_val);
                float p_cor = warp_reduce_sum(buf_D[c * K_DIM + tid] * s_val);
                if (lane_id == 0) {
                    xwarp_buf[warp_id * CHUNK * 2 + c * 2]     = p_int;
                    xwarp_buf[warp_id * CHUNK * 2 + c * 2 + 1] = p_cor;
                }
            }
            __syncthreads();

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

            // u = T̃ @ v_beta (compute into register, then write v_new)
            float u_val = 0.0f;
            if (tid < clen) {
                for (int j = 0; j <= tid; j++)
                    u_val += T_mat[tid * CHUNK + j] * v_new_buf[j];
            }
            __syncthreads();

            if (tid < clen)
                v_new_buf[tid] = u_val - corr_buf[tid];
            __syncthreads();

            // intra = QK @ v_new, write output
            if (tid < clen) {
                float intra = 0.0f;
                for (int j = 0; j <= tid; j++)
                    intra += QK_mat[tid * CHUNK + j] * v_new_buf[j];
                int pos = cstart + tid;
                int oidx = pos * NUM_V_HEADS * V_DIM + head_idx * V_DIM + vi;
                output[oidx] = __float2bfloat16(scale * (inter_buf[tid] + intra));
            }

            // State delta (using precomputed K_decay)
            float delta = 0.0f;
            for (int c = 0; c < clen; c++)
                delta += v_new_buf[c] * K_decay[c];
            S[vi] = s_val * decay_total + delta;
            __syncthreads();
        }
    }

    for (int vi = 0; vi < V_DIM; vi++)
        new_state[state_off + vi * K_DIM + tid] = S[vi];
}

void GDNPrefillKernel(
    tvm::ffi::TensorView q, tvm::ffi::TensorView k, tvm::ffi::TensorView v,
    tvm::ffi::TensorView state, tvm::ffi::TensorView A_log, tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias, tvm::ffi::TensorView b,
    tvm::ffi::TensorView cu_seqlens, double scale,
    tvm::ffi::TensorView output, tvm::ffi::TensorView new_state
) {
    const int num_seqs = static_cast<int>(cu_seqlens.size(0)) - 1;
    float scale_f = static_cast<float>(scale);
    if (scale_f == 0.0f) scale_f = 1.0f / sqrtf(float(K_DIM));

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    auto q_p  = static_cast<const __nv_bfloat16*>(q.data_ptr());
    auto k_p  = static_cast<const __nv_bfloat16*>(k.data_ptr());
    auto v_p  = static_cast<const __nv_bfloat16*>(v.data_ptr());
    auto st_p = static_cast<const float*>(state.data_ptr());
    auto al_p = static_cast<const float*>(A_log.data_ptr());
    auto a_p  = static_cast<const __nv_bfloat16*>(a.data_ptr());
    auto dt_p = static_cast<const float*>(dt_bias.data_ptr());
    auto b_p  = static_cast<const __nv_bfloat16*>(b.data_ptr());
    auto cu_p = static_cast<const int64_t*>(cu_seqlens.data_ptr());
    auto o_p  = static_cast<__nv_bfloat16*>(output.data_ptr());
    auto ns_p = static_cast<float*>(new_state.data_ptr());

    if (st_p == nullptr) {
        cudaMemsetAsync(ns_p, 0, num_seqs * NUM_V_HEADS * V_DIM * K_DIM * sizeof(float), stream);
        st_p = ns_p;
    }

    cudaFuncSetAttribute(gdn_prefill_chunk_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);

    gdn_prefill_chunk_kernel<<<dim3(num_seqs, NUM_V_HEADS), NTHREADS, SMEM_BYTES, stream>>>(
        q_p, k_p, v_p, st_p, al_p, a_p, dt_p, b_p, cu_p, scale_f, o_p, ns_p);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, GDNPrefillKernel);
