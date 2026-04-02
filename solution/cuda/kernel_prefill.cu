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

__device__ __forceinline__ float prefill_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ __launch_bounds__(128)
void gdn_prefill_cuda_kernel(
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

    const int qk_head_idx = head_idx / GQA_RATIO;

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end   = static_cast<int>(cu_seqlens[seq_idx + 1]);
    const int seq_len   = seq_end - seq_start;

    const int state_base = (seq_idx * NUM_V_HEADS + head_idx) * V_DIM * K_DIM;

    if (seq_len <= 0) {
        #pragma unroll 4
        for (int vi = 0; vi < V_DIM; vi++)
            new_state[state_base + vi * K_DIM + tid] =
                init_state[state_base + vi * K_DIM + tid];
        return;
    }

    float S[V_DIM];
    #pragma unroll 4
    for (int vi = 0; vi < V_DIM; vi++)
        S[vi] = __ldg(&init_state[state_base + vi * K_DIM + tid]);

    const float A_val  = __ldg(&A_log[head_idx]);
    const float dt_val = __ldg(&dt_bias[head_idx]);

    __shared__ float smem[8];

    for (int t = 0; t < seq_len; t++) {
        const int pos = seq_start + t;

        float a_val = __bfloat162float(__ldg(&a_in[pos * NUM_V_HEADS + head_idx]));
        float b_val = __bfloat162float(__ldg(&b_in[pos * NUM_V_HEADS + head_idx]));

        float x  = a_val + dt_val;
        float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
        float g  = expf(-expf(A_val) * sp);
        float beta = 1.0f / (1.0f + expf(-b_val));

        const int qk_off = pos * NUM_Q_HEADS * K_DIM + qk_head_idx * K_DIM + tid;
        float q_val = __bfloat162float(__ldg(&q[qk_off]));
        float k_val = __bfloat162float(__ldg(&k[qk_off]));

        #pragma unroll 4
        for (int vi = 0; vi < V_DIM; vi++)
            S[vi] *= g;

        float qk_w = prefill_warp_reduce_sum(q_val * k_val);
        if (lane_id == 0) smem[warp_id] = qk_w;
        __syncthreads();
        float q_dot_k = smem[0] + smem[1] + smem[2] + smem[3];
        __syncthreads();

        const int ov_base = pos * NUM_V_HEADS * V_DIM + head_idx * V_DIM;

        for (int vi = 0; vi < V_DIM; vi++) {
            float s_val = S[vi];

            float kv_w = prefill_warp_reduce_sum(k_val * s_val);
            float qv_w = prefill_warp_reduce_sum(q_val * s_val);

            if (lane_id == 0) {
                smem[warp_id * 2]     = kv_w;
                smem[warp_id * 2 + 1] = qv_w;
            }
            __syncthreads();

            float old_v   = smem[0] + smem[2] + smem[4] + smem[6];
            float q_dot_s = smem[1] + smem[3] + smem[5] + smem[7];

            float v_elem = __bfloat162float(__ldg(&v[ov_base + vi]));
            float delta  = beta * (v_elem - old_v);

            S[vi] = s_val + k_val * delta;

            if (tid == 0)
                output[ov_base + vi] =
                    __float2bfloat16(scale * (q_dot_s + delta * q_dot_k));

            __syncthreads();
        }
    }

    #pragma unroll 4
    for (int vi = 0; vi < V_DIM; vi++)
        new_state[state_base + vi * K_DIM + tid] = S[vi];
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
    dim3 block(K_DIM);

    gdn_prefill_cuda_kernel<<<grid, block, 0, stream>>>(
        q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
        cu_seqlens_ptr, scale_f, output_ptr, new_state_ptr
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, GDNPrefillKernel);
