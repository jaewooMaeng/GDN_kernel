/*
 * GDN Decode Kernel: gdn_decode_qk4_v8_d128_k_last
 *
 * Single-token decode with recurrent state update.
 * State layout: [B, H=8, V=128, K=128] float32 (k-last)
 * GVA: q_heads=4, k_heads=4, v_heads=8 (2 v_heads per q/k head)
 *
 * Optimization: Warp-parallel V-rows with loop fusion and float4 vectorized loads.
 * Each warp independently handles 32 vi rows. Simultaneous ks/qs reductions
 * avoid a second global memory read of state. Algebraic reformulation:
 *   qk_dot   = sum_k(q[k] * k[k])
 *   ks_sum   = sum_k(k[k] * state[vi,k])
 *   qs_sum   = sum_k(q[k] * state[vi,k])
 *   residual = beta * (v[vi] - g * ks_sum)
 *   output[vi]          = scale * (g * qs_sum + qk_dot * residual)
 *   new_state[vi,k]     = g * state[vi,k] + k[k] * residual
 *
 * Register-based 4-row software pipelining: state rows are loaded directly into
 * registers via float4 loads, processing 4 V-rows per iteration with prefetching.
 * Next 4 rows are prefetched into registers while current 4 rows are computed.
 * No shared memory needed for state. Default writeback caching for L2 residency.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>

using bf16 = __nv_bfloat16;

constexpr int NUM_Q_HEADS = 4;
constexpr int NUM_K_HEADS = 4;
constexpr int NUM_V_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int V_PER_Q = NUM_V_HEADS / NUM_Q_HEADS;  // 2

__device__ __forceinline__ float softplus(float x) {
    return log1pf(expf(x));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/*
 * V-split blocks: each (batch, v_head) can be split across multiple blocks.
 * Grid: (B * NUM_V_HEADS * split_factor,)
 * Block: 128 threads (4 warps) for B<=16, 256 threads (8 warps) for B>16.
 * Each warp handles rows_per_warp vi rows.
 */
__global__ void gdn_decode_kernel(
    const bf16* __restrict__ q,         // [B, 1, 4, 128]
    const bf16* __restrict__ k,         // [B, 1, 4, 128]
    const bf16* __restrict__ v,         // [B, 1, 8, 128]
    const float* __restrict__ state,    // [B, 8, 128, 128] k-last [H,V,K]
    const float* __restrict__ A_log,    // [8]
    const bf16* __restrict__ a,         // [B, 1, 8]
    const float* __restrict__ dt_bias,  // [8]
    const bf16* __restrict__ b_gate,    // [B, 1, 8]
    const float scale,
    bf16* __restrict__ output,          // [B, 1, 8, 128]
    float* __restrict__ new_state,      // [B, 8, 128, 128]
    int batch_size,
    int split_factor
) {
    const int idx = blockIdx.x;
    const int heads_x_split = NUM_V_HEADS * split_factor;
    const int batch = idx / heads_x_split;
    const int remainder = idx % heads_x_split;
    const int vh = remainder / split_factor;
    const int split_id = remainder % split_factor;
    const int qkh = vh / V_PER_Q;
    const int tid = threadIdx.x;

    const int warp_id = tid / 32;     // 0..3 or 0..7
    const int lane = tid % 32;        // 0..31

    const int rows_per_block = HEAD_DIM / split_factor;  // 128/split
    const int num_warps = blockDim.x / 32;
    const int rows_per_warp = rows_per_block / num_warps;  // per warp

    // Compute gates (uniform across all threads in block)
    float a_val = __bfloat162float(a[batch * NUM_V_HEADS + vh]);
    float dt_val = dt_bias[vh];
    float A_val = A_log[vh];
    float g = expf(-expf(A_val) * softplus(a_val + dt_val));
    float beta = 1.0f / (1.0f + expf(-__bfloat162float(b_gate[batch * NUM_V_HEADS + vh])));

    __shared__ float s_q[HEAD_DIM];
    __shared__ float s_k[HEAD_DIM];

    const int k_base_offset = batch * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM;
    const int q_base_offset = batch * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM;

    // Precompute qk_dot = sum_k(q[k] * k[k]) via warp reduction
    // Load v vector into shared memory (all warps cooperate)
    __shared__ float s_v[HEAD_DIM];
    {
        // Only first HEAD_DIM threads load (handles 256-thread blocks)
        if (tid < HEAD_DIM) {
            s_v[tid] = __bfloat162float(v[batch * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + tid]);
            if (batch_size == 1) {
                s_q[tid] = __bfloat162float(q[q_base_offset + tid]);
                s_k[tid] = __bfloat162float(k[k_base_offset + tid]);
            }
        }
    }
    __syncthreads();

    float k_vals[4];
    float q_vals[4];
    {
        int base = lane * 4;
        if (batch_size == 1) {
            k_vals[0] = s_k[base + 0];
            k_vals[1] = s_k[base + 1];
            k_vals[2] = s_k[base + 2];
            k_vals[3] = s_k[base + 3];
            q_vals[0] = s_q[base + 0];
            q_vals[1] = s_q[base + 1];
            q_vals[2] = s_q[base + 2];
            q_vals[3] = s_q[base + 3];
        } else {
            k_vals[0] = __bfloat162float(k[k_base_offset + base + 0]);
            k_vals[1] = __bfloat162float(k[k_base_offset + base + 1]);
            k_vals[2] = __bfloat162float(k[k_base_offset + base + 2]);
            k_vals[3] = __bfloat162float(k[k_base_offset + base + 3]);
            q_vals[0] = __bfloat162float(q[q_base_offset + base + 0]);
            q_vals[1] = __bfloat162float(q[q_base_offset + base + 1]);
            q_vals[2] = __bfloat162float(q[q_base_offset + base + 2]);
            q_vals[3] = __bfloat162float(q[q_base_offset + base + 3]);
        }
    }

    float qk_local = q_vals[0] * k_vals[0] + q_vals[1] * k_vals[1]
                   + q_vals[2] * k_vals[2] + q_vals[3] * k_vals[3];
    float qk_dot = warp_reduce_sum(qk_local);
    qk_dot = __shfl_sync(0xffffffff, qk_dot, 0);

    // State base pointers
    const float* state_base = state + (batch * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    float* new_state_base = new_state + (batch * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;

    // Output base pointer
    bf16* out_base = output + batch * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM;

    // Each warp handles rows_per_warp vi rows within this block's split
    const int vi_start = split_id * rows_per_block + warp_id * rows_per_warp;

    // Prefetch first 4 rows into registers
    float4 pf_a = *reinterpret_cast<const float4*>(state_base + vi_start * HEAD_DIM + lane * 4);
    float4 pf_b = *reinterpret_cast<const float4*>(state_base + (vi_start + 1) * HEAD_DIM + lane * 4);
    float4 pf_c = *reinterpret_cast<const float4*>(state_base + (vi_start + 2) * HEAD_DIM + lane * 4);
    float4 pf_d = *reinterpret_cast<const float4*>(state_base + (vi_start + 3) * HEAD_DIM + lane * 4);

    for (int vi_off = 0; vi_off < rows_per_warp; vi_off += 4) {
        const int vi_a = vi_start + vi_off;
        const int vi_b = vi_a + 1;
        const int vi_c = vi_a + 2;
        const int vi_d = vi_a + 3;

        // Current rows from prefetched registers
        float4 st4_a = pf_a;
        float4 st4_b = pf_b;
        float4 st4_c = pf_c;
        float4 st4_d = pf_d;

        // Prefetch next 4 rows (if exist)
        if (vi_off + 4 < rows_per_warp) {
            pf_a = *reinterpret_cast<const float4*>(state_base + (vi_a + 4) * HEAD_DIM + lane * 4);
            pf_b = *reinterpret_cast<const float4*>(state_base + (vi_a + 5) * HEAD_DIM + lane * 4);
            pf_c = *reinterpret_cast<const float4*>(state_base + (vi_a + 6) * HEAD_DIM + lane * 4);
            pf_d = *reinterpret_cast<const float4*>(state_base + (vi_a + 7) * HEAD_DIM + lane * 4);
        }

        // Compute dot products for all 4 rows simultaneously
        float ks_a = k_vals[0]*st4_a.x + k_vals[1]*st4_a.y + k_vals[2]*st4_a.z + k_vals[3]*st4_a.w;
        float qs_a = q_vals[0]*st4_a.x + q_vals[1]*st4_a.y + q_vals[2]*st4_a.z + q_vals[3]*st4_a.w;
        float ks_b = k_vals[0]*st4_b.x + k_vals[1]*st4_b.y + k_vals[2]*st4_b.z + k_vals[3]*st4_b.w;
        float qs_b = q_vals[0]*st4_b.x + q_vals[1]*st4_b.y + q_vals[2]*st4_b.z + q_vals[3]*st4_b.w;
        float ks_c = k_vals[0]*st4_c.x + k_vals[1]*st4_c.y + k_vals[2]*st4_c.z + k_vals[3]*st4_c.w;
        float qs_c = q_vals[0]*st4_c.x + q_vals[1]*st4_c.y + q_vals[2]*st4_c.z + q_vals[3]*st4_c.w;
        float ks_d = k_vals[0]*st4_d.x + k_vals[1]*st4_d.y + k_vals[2]*st4_d.z + k_vals[3]*st4_d.w;
        float qs_d = q_vals[0]*st4_d.x + q_vals[1]*st4_d.y + q_vals[2]*st4_d.z + q_vals[3]*st4_d.w;

        // Interleaved warp reductions (all 8 reductions for better ILP)
        for (int offset = 16; offset > 0; offset >>= 1) {
            ks_a += __shfl_down_sync(0xffffffff, ks_a, offset);
            ks_b += __shfl_down_sync(0xffffffff, ks_b, offset);
            ks_c += __shfl_down_sync(0xffffffff, ks_c, offset);
            ks_d += __shfl_down_sync(0xffffffff, ks_d, offset);
            qs_a += __shfl_down_sync(0xffffffff, qs_a, offset);
            qs_b += __shfl_down_sync(0xffffffff, qs_b, offset);
            qs_c += __shfl_down_sync(0xffffffff, qs_c, offset);
            qs_d += __shfl_down_sync(0xffffffff, qs_d, offset);
        }

        // Broadcast ks values to all lanes
        ks_a = __shfl_sync(0xffffffff, ks_a, 0);
        ks_b = __shfl_sync(0xffffffff, ks_b, 0);
        ks_c = __shfl_sync(0xffffffff, ks_c, 0);
        ks_d = __shfl_sync(0xffffffff, ks_d, 0);

        // Compute residuals
        float res_a = beta * (s_v[vi_a] - g * ks_a);
        float res_b = beta * (s_v[vi_b] - g * ks_b);
        float res_c = beta * (s_v[vi_c] - g * ks_c);
        float res_d = beta * (s_v[vi_d] - g * ks_d);

        // Write new states (float4 stores with default writeback caching for L2 residency)
        float4 new_a = make_float4(
            g*st4_a.x + k_vals[0]*res_a, g*st4_a.y + k_vals[1]*res_a,
            g*st4_a.z + k_vals[2]*res_a, g*st4_a.w + k_vals[3]*res_a);
        float4 new_b = make_float4(
            g*st4_b.x + k_vals[0]*res_b, g*st4_b.y + k_vals[1]*res_b,
            g*st4_b.z + k_vals[2]*res_b, g*st4_b.w + k_vals[3]*res_b);
        float4 new_c = make_float4(
            g*st4_c.x + k_vals[0]*res_c, g*st4_c.y + k_vals[1]*res_c,
            g*st4_c.z + k_vals[2]*res_c, g*st4_c.w + k_vals[3]*res_c);
        float4 new_d = make_float4(
            g*st4_d.x + k_vals[0]*res_d, g*st4_d.y + k_vals[1]*res_d,
            g*st4_d.z + k_vals[2]*res_d, g*st4_d.w + k_vals[3]*res_d);
        *reinterpret_cast<float4*>(new_state_base + vi_a * HEAD_DIM + lane * 4) = new_a;
        *reinterpret_cast<float4*>(new_state_base + vi_b * HEAD_DIM + lane * 4) = new_b;
        *reinterpret_cast<float4*>(new_state_base + vi_c * HEAD_DIM + lane * 4) = new_c;
        *reinterpret_cast<float4*>(new_state_base + vi_d * HEAD_DIM + lane * 4) = new_d;

        // Lane 0 writes outputs
        if (lane == 0) {
            out_base[vi_a] = __float2bfloat16(scale * (g * qs_a + qk_dot * res_a));
            out_base[vi_b] = __float2bfloat16(scale * (g * qs_b + qk_dot * res_b));
            out_base[vi_c] = __float2bfloat16(scale * (g * qs_c + qk_dot * res_c));
            out_base[vi_d] = __float2bfloat16(scale * (g * qs_d + qk_dot * res_d));
        }
    }
}

// TVM FFI entry point (DPS style)
void gdn_decode(
    tvm::ffi::TensorView q,         // [B, 1, 4, 128] bf16
    tvm::ffi::TensorView k,         // [B, 1, 4, 128] bf16
    tvm::ffi::TensorView v,         // [B, 1, 8, 128] bf16
    tvm::ffi::TensorView state,     // [B, 8, 128, 128] f32
    tvm::ffi::TensorView A_log,     // [8] f32
    tvm::ffi::TensorView a,         // [B, 1, 8] bf16
    tvm::ffi::TensorView dt_bias,   // [8] f32
    tvm::ffi::TensorView b_gate,    // [B, 1, 8] bf16
    double scale,
    tvm::ffi::TensorView output,    // [B, 1, 8, 128] bf16
    tvm::ffi::TensorView new_state  // [B, 8, 128, 128] f32
) {
    int batch_size = q.size(0);

    // Choose split factor to increase SM utilization at small batch sizes
    int split_factor;
    if (batch_size <= 2) {
        split_factor = 8;   // 16 V-rows per block, 4 per warp (1 iteration of 4-row pipeline)
    } else if (batch_size <= 4) {
        split_factor = 4;   // 32 V-rows per block, 8 per warp
    } else if (batch_size <= 16) {
        split_factor = 2;   // 64 V-rows per block, 16 per warp
    } else {
        split_factor = 1;   // 128 V-rows per block, 32 per warp (original)
    }

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    // Use 8 warps (256 threads) for large batches (B>16, sf=1) to better utilize SMs
    int block_size = (batch_size > 16) ? 256 : 128;

    dim3 grid(batch_size * NUM_V_HEADS * split_factor);
    dim3 block(block_size);

    gdn_decode_kernel<<<grid, block, 0, stream>>>(
        static_cast<const bf16*>(q.data_ptr()),
        static_cast<const bf16*>(k.data_ptr()),
        static_cast<const bf16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const bf16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const bf16*>(b_gate.data_ptr()),
        static_cast<float>(scale),
        static_cast<bf16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()),
        batch_size,
        split_factor
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, gdn_decode);
