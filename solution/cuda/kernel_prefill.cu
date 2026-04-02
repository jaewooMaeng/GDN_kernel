#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

static constexpr int K_DIM = 128;
static constexpr int V_DIM = 128;
static constexpr int K_PAD = K_DIM + 1; // 129, for bank-conflict-free S reads
static constexpr int NUM_Q_HEADS = 4;
static constexpr int NUM_V_HEADS = 8;
static constexpr int GQA_RATIO = NUM_V_HEADS / NUM_Q_HEADS;
static constexpr int CHUNK = 64;
static constexpr int NTHREADS = 128;

// Shared memory layout (~225KB total, within B200's 228KB)
// Fixed: T_mat[C*C] + QK[C*C] + gamma[C] + beta[C] + exp_gamma[C]
// Variable (time-multiplexed):
//   Phase1: K_c[C*K] + buf_D[C*K]
//   Phase2: S_smem[V*K_PAD] + w_decay[C*K] + Q_exp[C*K] + inter[C*V] + corr[C*V]
static constexpr int OFF_TMAT = 0;
static constexpr int OFF_QKMAT = CHUNK * CHUNK * 4;
static constexpr int OFF_GAMMA = OFF_QKMAT + CHUNK * CHUNK * 4;
static constexpr int OFF_BETA  = OFF_GAMMA + CHUNK * 4;
static constexpr int OFF_EXPG  = OFF_BETA + CHUNK * 4;
static constexpr int OFF_VAR   = OFF_EXPG + CHUNK * 4;
// Phase 2 variable offsets
static constexpr int OFF_SSMEM  = OFF_VAR;
static constexpr int OFF_WDEC2  = OFF_SSMEM + V_DIM * K_PAD * 4;
static constexpr int OFF_QEXP   = OFF_WDEC2 + CHUNK * K_DIM * 4;
static constexpr int OFF_INTER  = OFF_QEXP + CHUNK * K_DIM * 4;
static constexpr int OFF_CORR   = OFF_INTER + CHUNK * V_DIM * 4;
static constexpr int SMEM_BYTES = OFF_CORR + CHUNK * V_DIM * 4;

__global__ __launch_bounds__(NTHREADS)
void gdn_prefill_phase4_kernel(
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

    extern __shared__ char smem[];
    float* T_mat     = reinterpret_cast<float*>(smem + OFF_TMAT);
    float* QK_mat    = reinterpret_cast<float*>(smem + OFF_QKMAT);
    float* gamma_s   = reinterpret_cast<float*>(smem + OFF_GAMMA);
    float* beta_s    = reinterpret_cast<float*>(smem + OFF_BETA);
    float* exp_gamma = reinterpret_cast<float*>(smem + OFF_EXPG);

    // Phase 1 pointers (variable region)
    float* K_c   = reinterpret_cast<float*>(smem + OFF_VAR);
    float* buf_D = reinterpret_cast<float*>(smem + OFF_VAR + CHUNK * K_DIM * 4);

    // Phase 2 pointers
    float* S_smem    = reinterpret_cast<float*>(smem + OFF_SSMEM);
    float* w_decay2  = reinterpret_cast<float*>(smem + OFF_WDEC2);
    float* Q_exp_s   = reinterpret_cast<float*>(smem + OFF_QEXP);
    float* inter_mat = reinterpret_cast<float*>(smem + OFF_INTER);
    float* corr_mat  = reinterpret_cast<float*>(smem + OFF_CORR);

    const int num_chunks = (seq_len + CHUNK - 1) / CHUNK;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int cstart = seq_start + chunk * CHUNK;
        const int clen   = min(CHUNK, seq_end - cstart);

        // ========== PHASE 1: A matrix, forward sub, QK, w_decay ==========

        // 1. Gates
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

        // 2. Prefix sum gamma + exp_gamma
        if (tid == 0)
            for (int i = 1; i < CHUNK; i++) gamma_s[i] += gamma_s[i - 1];
        __syncthreads();
        if (tid < CHUNK) exp_gamma[tid] = expf(gamma_s[tid]);
        __syncthreads();

        // 3. Load K_c
        for (int c = 0; c < CHUNK; c++) {
            float kv = 0.0f;
            if (c < clen)
                kv = __bfloat162float(__ldg(&k[(cstart + c) * NUM_Q_HEADS * K_DIM + qk_head * K_DIM + tid]));
            K_c[c * K_DIM + tid] = kv;
        }
        __syncthreads();

        // 4. A matrix
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

        // 5. Forward substitution
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

        // 6. Load Q to buf_D, compute QK
        for (int c = 0; c < CHUNK; c++) {
            float qv = 0.0f;
            if (c < clen)
                qv = __bfloat162float(__ldg(&q[(cstart + c) * NUM_Q_HEADS * K_DIM + qk_head * K_DIM + tid]));
            buf_D[c * K_DIM + tid] = qv;
        }
        __syncthreads();

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

        // 7. w_decay = T̃ @ (K*beta*exp_gamma) into buf_D
        for (int c = 0; c < CHUNK; c++) {
            float acc = 0.0f;
            int lim = min(c + 1, clen);
            for (int j = 0; j < lim; j++)
                acc += T_mat[c * CHUNK + j] * K_c[j * K_DIM + tid] * beta_s[j] * exp_gamma[j];
            buf_D[c * K_DIM + tid] = acc;
        }
        __syncthreads();

        // 8. Precompute K_decay in registers
        float gamma_last = gamma_s[clen - 1];
        float decay_total = expf(gamma_last);
        float K_decay[CHUNK];
        for (int c = 0; c < CHUNK; c++)
            K_decay[c] = K_c[c * K_DIM + tid] * expf(gamma_last - gamma_s[c]);

        // 9. Copy w_decay to safe location before S_smem overwrites buf_D
        for (int c = 0; c < CHUNK; c++)
            w_decay2[c * K_DIM + tid] = buf_D[c * K_DIM + tid];
        __syncthreads();

        // ========== PHASE 2: Batch inter/corr, u, intra, output, state ==========

        // 10. Write S to S_smem (padded stride K_PAD=129)
        for (int vi = 0; vi < V_DIM; vi++)
            S_smem[vi * K_PAD + tid] = S[vi];
        __syncthreads();

        // 11. Compute Q_exp_smem = Q * exp(gamma), reload Q from global
        for (int c = 0; c < CHUNK; c++) {
            float qv = 0.0f;
            if (c < clen)
                qv = __bfloat162float(__ldg(&q[(cstart + c) * NUM_Q_HEADS * K_DIM + qk_head * K_DIM + tid]));
            Q_exp_s[c * K_DIM + tid] = qv * exp_gamma[c];
        }
        __syncthreads();

        // 12. Compute inter[C][V] = Q_exp @ S^T (fp32 dot products, all threads)
        for (int idx = tid; idx < clen * V_DIM; idx += NTHREADS) {
            int c  = idx / V_DIM;
            int vi = idx % V_DIM;
            float val = 0.0f;
            for (int kk = 0; kk < K_DIM; kk++)
                val += Q_exp_s[c * K_DIM + kk] * S_smem[vi * K_PAD + kk];
            inter_mat[c * V_DIM + vi] = val;
        }
        __syncthreads();

        // 13. Compute corr[C][V] = w_decay @ S^T
        for (int idx = tid; idx < clen * V_DIM; idx += NTHREADS) {
            int c  = idx / V_DIM;
            int vi = idx % V_DIM;
            float val = 0.0f;
            for (int kk = 0; kk < K_DIM; kk++)
                val += w_decay2[c * K_DIM + kk] * S_smem[vi * K_PAD + kk];
            corr_mat[c * V_DIM + vi] = val;
        }
        __syncthreads();

        // Phase 2c: v_beta, u, v_new, intra+output, state
        // Reuse S_smem space for v_beta (only first C*V floats needed, fits in 32KB < 66KB)
        float* v_beta = reinterpret_cast<float*>(smem + OFF_SSMEM);

        // 14. Load v_beta[C][V] from global
        for (int idx = tid; idx < clen * V_DIM; idx += NTHREADS) {
            int c  = idx / V_DIM;
            int vi = idx % V_DIM;
            int pos = cstart + c;
            float vv = __bfloat162float(__ldg(&v[pos * NUM_V_HEADS * V_DIM + head_idx * V_DIM + vi]));
            v_beta[c * V_DIM + vi] = beta_s[c] * vv;
        }
        __syncthreads();

        // 15. Compute v_new = u - corr (u = T̃ @ v_beta, overwrite corr in-place)
        float* v_new = corr_mat;
        for (int idx = tid; idx < clen * V_DIM; idx += NTHREADS) {
            int c  = idx / V_DIM;
            int vi = idx % V_DIM;
            float u_val = 0.0f;
            for (int j = 0; j <= c; j++)
                u_val += T_mat[c * CHUNK + j] * v_beta[j * V_DIM + vi];
            v_new[c * V_DIM + vi] = u_val - corr_mat[c * V_DIM + vi];
        }
        __syncthreads();

        // 16. Compute intra + output (fused), intra = QK @ v_new
        for (int idx = tid; idx < clen * V_DIM; idx += NTHREADS) {
            int c  = idx / V_DIM;
            int vi = idx % V_DIM;
            float intra = 0.0f;
            for (int j = 0; j <= c; j++)
                intra += QK_mat[c * CHUNK + j] * v_new[j * V_DIM + vi];
            int pos = cstart + c;
            int oidx = pos * NUM_V_HEADS * V_DIM + head_idx * V_DIM + vi;
            output[oidx] = __float2bfloat16(scale * (inter_mat[c * V_DIM + vi] + intra));
        }
        __syncthreads();

        // 17. State update: S[vi] = S[vi] * decay + sum_c v_new[c][vi] * K_decay[c]
        for (int vi = 0; vi < V_DIM; vi++) {
            float delta = 0.0f;
            for (int c = 0; c < clen; c++)
                delta += v_new[c * V_DIM + vi] * K_decay[c];
            S[vi] = S[vi] * decay_total + delta;
        }
        __syncthreads();
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

    cudaFuncSetAttribute(gdn_prefill_phase4_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);

    gdn_prefill_phase4_kernel<<<dim3(num_seqs, NUM_V_HEADS), NTHREADS, SMEM_BYTES, stream>>>(
        q_p, k_p, v_p, st_p, al_p, a_p, dt_p, b_p, cu_p, scale_f, o_p, ns_p);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, GDNPrefillKernel);
