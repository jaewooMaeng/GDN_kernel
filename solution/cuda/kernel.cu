#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int K_DIM = 128;
constexpr int V_DIM = 128;
constexpr int NUM_Q_HEADS = 4;
constexpr int NUM_V_HEADS = 8;
constexpr int GQA_RATIO = NUM_V_HEADS / NUM_Q_HEADS;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ void gdn_decode_cuda_kernel(
    const __nv_bfloat16* __restrict__ q,        // [B, H_qk, K]  (squeezed T=1)
    const __nv_bfloat16* __restrict__ k,        // [B, H_qk, K]
    const __nv_bfloat16* __restrict__ v,        // [B, H_v, V]
    const float* __restrict__ state,            // [B, H_v, V, K]
    const float* __restrict__ A_log,            // [H_v]
    const __nv_bfloat16* __restrict__ a_in,     // [B, H_v]
    const float* __restrict__ dt_bias,          // [H_v]
    const __nv_bfloat16* __restrict__ b_in,     // [B, H_v]
    float scale,
    __nv_bfloat16* __restrict__ output,         // [B, H_v, V]
    float* __restrict__ new_state               // [B, H_v, V, K]
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __shared__ float smem[8];

    const int qk_head_idx = head_idx / GQA_RATIO;

    // g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
    float a_val = __bfloat162float(__ldg(&a_in[batch_idx * NUM_V_HEADS + head_idx]));
    float dt_val = __ldg(&dt_bias[head_idx]);
    float A_val = __ldg(&A_log[head_idx]);
    float b_val = __bfloat162float(__ldg(&b_in[batch_idx * NUM_V_HEADS + head_idx]));

    float x = a_val + dt_val;
    float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    float g = expf(-expf(A_val) * sp);
    float beta = 1.0f / (1.0f + expf(-b_val));

    const int qk_offset = batch_idx * NUM_Q_HEADS * K_DIM + qk_head_idx * K_DIM + tid;
    float q_val = __bfloat162float(__ldg(&q[qk_offset]));
    float k_val = __bfloat162float(__ldg(&k[qk_offset]));

    // Precompute dot(q, k) to avoid a second reduction per V iteration:
    // output[v] = scale * (dot(q, old_state[v,:]) + delta[v] * dot(q, k))
    float qk_prod = warp_reduce_sum(q_val * k_val);
    if (lane_id == 0) smem[warp_id] = qk_prod;
    __syncthreads();
    float q_dot_k = smem[0] + smem[1] + smem[2] + smem[3];
    __syncthreads();

    const int state_base = (batch_idx * NUM_V_HEADS + head_idx) * V_DIM * K_DIM;
    const int v_base = batch_idx * NUM_V_HEADS * V_DIM + head_idx * V_DIM;

    for (int vi = 0; vi < V_DIM; vi++) {
        float old_s = g * __ldg(&state[state_base + vi * K_DIM + tid]);

        // Dual reduction: dot(k, old_state[v,:]) and dot(q, old_state[v,:])
        float kv = warp_reduce_sum(k_val * old_s);
        float qv = warp_reduce_sum(q_val * old_s);

        if (lane_id == 0) {
            smem[warp_id * 2] = kv;
            smem[warp_id * 2 + 1] = qv;
        }
        __syncthreads();

        float old_v_val = smem[0] + smem[2] + smem[4] + smem[6];
        float q_dot_old_s = smem[1] + smem[3] + smem[5] + smem[7];

        float v_elem = __bfloat162float(__ldg(&v[v_base + vi]));
        float delta = beta * (v_elem - old_v_val);

        new_state[state_base + vi * K_DIM + tid] = old_s + k_val * delta;

        if (tid == 0) {
            output[v_base + vi] = __float2bfloat16(scale * (q_dot_old_s + delta * q_dot_k));
        }

        __syncthreads();
    }
}

// TVM FFI host function — receives DLPack TensorViews directly from the framework.
// Tensor shapes include the T=1 dimension but memory layout is equivalent to squeezed.
void GDNDecodeKernel(
    tvm::ffi::TensorView q,            // [B, 1, H_qk, K]
    tvm::ffi::TensorView k,            // [B, 1, H_qk, K]
    tvm::ffi::TensorView v,            // [B, 1, H_v, V]
    tvm::ffi::TensorView state,        // [B, H_v, V, K]
    tvm::ffi::TensorView A_log,        // [H_v]
    tvm::ffi::TensorView a,            // [B, 1, H_v]
    tvm::ffi::TensorView dt_bias,      // [H_v]
    tvm::ffi::TensorView b,            // [B, 1, H_v]
    double scale,
    tvm::ffi::TensorView output,       // [B, 1, H_v, V]
    tvm::ffi::TensorView new_state     // [B, H_v, V, K]
) {
    int64_t B = q.size(0);
    float scale_f = static_cast<float>(scale);
    if (scale_f == 0.0f) {
        scale_f = 1.0f / sqrtf(static_cast<float>(K_DIM));
    }

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
    __nv_bfloat16* output_ptr = static_cast<__nv_bfloat16*>(output.data_ptr());
    float* new_state_ptr = static_cast<float*>(new_state.data_ptr());

    if (state_ptr == nullptr) {
        cudaMemsetAsync(new_state_ptr, 0,
                        B * NUM_V_HEADS * V_DIM * K_DIM * sizeof(float), stream);
        state_ptr = new_state_ptr;
    }

    dim3 grid(static_cast<int>(B), NUM_V_HEADS);
    dim3 block(K_DIM);

    gdn_decode_cuda_kernel<<<grid, block, 0, stream>>>(
        q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
        scale_f, output_ptr, new_state_ptr
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, GDNDecodeKernel);
