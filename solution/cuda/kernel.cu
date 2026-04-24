/*
 * GDN Decode Kernel: gdn_decode_qk4_v8_d128_k_last
 *
 * Single-token decode with recurrent state update.
 * State layout: [B, H=8, V=128, K=128] float32 (k-last)
 * GVA: q_heads=4, k_heads=4, v_heads=8 (2 v_heads per q/k head)
 *
 * Fully templated kernel: ROWS_PER_WARP determines all block/split parameters
 * at compile time. All integer division/modulo uses power-of-2 constants.
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
constexpr int V_PER_Q = NUM_V_HEADS / NUM_Q_HEADS;
constexpr int BLOCK_SIZE = 128;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;

__device__ __forceinline__ float softplus(float x) {
    return log1pf(expf(x));
}

// A4: Inline PTX for read-only global load with bypass
__device__ __forceinline__ float4 ld_global_nc_f4(const float4* ptr) {
    float4 result;
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w)
        : "l"(ptr));
    return result;
}

template<int ROWS_PER_WARP>
__global__ void __launch_bounds__(BLOCK_SIZE, 9) gdn_decode_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const bf16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const bf16* __restrict__ b_gate,
    const float scale,
    bf16* __restrict__ output,
    float* __restrict__ new_state,
    int batch_size
) {
    constexpr int ROWS_PER_BLOCK = ROWS_PER_WARP * NUM_WARPS;
    constexpr int SPLIT_FACTOR = HEAD_DIM / ROWS_PER_BLOCK;
    constexpr int HEADS_X_SPLIT = NUM_V_HEADS * SPLIT_FACTOR;

    const int idx = blockIdx.x;
    const int batch = idx / HEADS_X_SPLIT;
    const int remainder = idx - batch * HEADS_X_SPLIT;
    const int vh = remainder / SPLIT_FACTOR;
    const int split_id = remainder - vh * SPLIT_FACTOR;
    const int qkh = vh / V_PER_Q;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    float g, beta, beta_g;
    if (lane == 0) {
        float a_val = __bfloat162float(a[batch * NUM_V_HEADS + vh]);
        float dt_val = dt_bias[vh];
        float A_val = A_log[vh];
        g = expf(-expf(A_val) * softplus(a_val + dt_val));
        beta = 1.0f / (1.0f + expf(-__bfloat162float(b_gate[batch * NUM_V_HEADS + vh])));
        beta_g = beta * g;
    }
    g = __shfl_sync(0xffffffff, g, 0);
    beta = __shfl_sync(0xffffffff, beta, 0);
    beta_g = __shfl_sync(0xffffffff, beta_g, 0);

    const int k_base = batch * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM + (lane << 2);
    float k_vals[4];
    {
        uint2 packed = *reinterpret_cast<const uint2*>(k + k_base);
        __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
        __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162*>(&packed.y);
        float2 lof = __bfloat1622float2(lo);
        float2 hif = __bfloat1622float2(hi);
        k_vals[0] = lof.x; k_vals[1] = lof.y;
        k_vals[2] = hif.x; k_vals[3] = hif.y;
    }

    const int q_base = batch * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM + (lane << 2);
    float q_vals[4];
    {
        uint2 packed = *reinterpret_cast<const uint2*>(q + q_base);
        __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
        __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162*>(&packed.y);
        float2 lof = __bfloat1622float2(lo);
        float2 hif = __bfloat1622float2(hi);
        q_vals[0] = lof.x; q_vals[1] = lof.y;
        q_vals[2] = hif.x; q_vals[3] = hif.y;
    }

    float qk_local = q_vals[0]*k_vals[0] + q_vals[1]*k_vals[1]
                   + q_vals[2]*k_vals[2] + q_vals[3]*k_vals[3];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
    float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);
    float scale_g = scale * g;
    float scale_qk = scale * qk_dot;

    __shared__ float s_v[HEAD_DIM];
    s_v[tid] = __bfloat162float(v[batch * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + tid]);
    __syncthreads();

    const int state_hv_offset = (batch * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    const float* state_base = state + state_hv_offset;
    float* new_state_base = new_state + state_hv_offset;
    bf16* out_base = output + batch * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM;

    const int vi_start = split_id * ROWS_PER_BLOCK + warp_id * ROWS_PER_WARP;
    const int lane4 = lane << 2;

    // H2.5: Dual-buffer prefetch for 8-row lookahead
    // curr_*: rows [vi_off, vi_off+4)
    // next_*: rows [vi_off+4, vi_off+8)
    float4 curr_a = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + vi_start * HEAD_DIM + lane4));
    float4 curr_b = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_start + 1) * HEAD_DIM + lane4));
    float4 curr_c = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_start + 2) * HEAD_DIM + lane4));
    float4 curr_d = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_start + 3) * HEAD_DIM + lane4));

    float4 next_a = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_start + 4) * HEAD_DIM + lane4));
    float4 next_b = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_start + 5) * HEAD_DIM + lane4));
    float4 next_c = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_start + 6) * HEAD_DIM + lane4));
    float4 next_d = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_start + 7) * HEAD_DIM + lane4));

    #pragma unroll
    for (int vi_off = 0; vi_off < ROWS_PER_WARP; vi_off += 4) {
        const int vi_a = vi_start + vi_off;

        float4 st4_a = curr_a;
        float4 st4_b = curr_b;
        float4 st4_c = curr_c;
        float4 st4_d = curr_d;

        // Rotate buffers for next iteration
        curr_a = next_a;
        curr_b = next_b;
        curr_c = next_c;
        curr_d = next_d;

        // Prefetch 8 rows ahead
        if (vi_off + 8 < ROWS_PER_WARP) {
            next_a = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_a + 8) * HEAD_DIM + lane4));
            next_b = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_a + 9) * HEAD_DIM + lane4));
            next_c = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_a + 10) * HEAD_DIM + lane4));
            next_d = ld_global_nc_f4(reinterpret_cast<const float4*>(state_base + (vi_a + 11) * HEAD_DIM + lane4));
        }

        float ks_a = k_vals[0]*st4_a.x + k_vals[1]*st4_a.y + k_vals[2]*st4_a.z + k_vals[3]*st4_a.w;
        float qs_a = q_vals[0]*st4_a.x + q_vals[1]*st4_a.y + q_vals[2]*st4_a.z + q_vals[3]*st4_a.w;
        float ks_b = k_vals[0]*st4_b.x + k_vals[1]*st4_b.y + k_vals[2]*st4_b.z + k_vals[3]*st4_b.w;
        float qs_b = q_vals[0]*st4_b.x + q_vals[1]*st4_b.y + q_vals[2]*st4_b.z + q_vals[3]*st4_b.w;
        float ks_c = k_vals[0]*st4_c.x + k_vals[1]*st4_c.y + k_vals[2]*st4_c.z + k_vals[3]*st4_c.w;
        float qs_c = q_vals[0]*st4_c.x + q_vals[1]*st4_c.y + q_vals[2]*st4_c.z + q_vals[3]*st4_c.w;
        float ks_d = k_vals[0]*st4_d.x + k_vals[1]*st4_d.y + k_vals[2]*st4_d.z + k_vals[3]*st4_d.w;
        float qs_d = q_vals[0]*st4_d.x + q_vals[1]*st4_d.y + q_vals[2]*st4_d.z + q_vals[3]*st4_d.w;

        #pragma unroll
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

        ks_a = __shfl_sync(0xffffffff, ks_a, 0);
        ks_b = __shfl_sync(0xffffffff, ks_b, 0);
        ks_c = __shfl_sync(0xffffffff, ks_c, 0);
        ks_d = __shfl_sync(0xffffffff, ks_d, 0);

        float res_a = beta * s_v[vi_a] - beta_g * ks_a;
        float res_b = beta * s_v[vi_a + 1] - beta_g * ks_b;
        float res_c = beta * s_v[vi_a + 2] - beta_g * ks_c;
        float res_d = beta * s_v[vi_a + 3] - beta_g * ks_d;

        *reinterpret_cast<float4*>(new_state_base + vi_a * HEAD_DIM + lane4) = make_float4(
            g*st4_a.x + k_vals[0]*res_a, g*st4_a.y + k_vals[1]*res_a,
            g*st4_a.z + k_vals[2]*res_a, g*st4_a.w + k_vals[3]*res_a);
        *reinterpret_cast<float4*>(new_state_base + (vi_a+1) * HEAD_DIM + lane4) = make_float4(
            g*st4_b.x + k_vals[0]*res_b, g*st4_b.y + k_vals[1]*res_b,
            g*st4_b.z + k_vals[2]*res_b, g*st4_b.w + k_vals[3]*res_b);
        *reinterpret_cast<float4*>(new_state_base + (vi_a+2) * HEAD_DIM + lane4) = make_float4(
            g*st4_c.x + k_vals[0]*res_c, g*st4_c.y + k_vals[1]*res_c,
            g*st4_c.z + k_vals[2]*res_c, g*st4_c.w + k_vals[3]*res_c);
        *reinterpret_cast<float4*>(new_state_base + (vi_a+3) * HEAD_DIM + lane4) = make_float4(
            g*st4_d.x + k_vals[0]*res_d, g*st4_d.y + k_vals[1]*res_d,
            g*st4_d.z + k_vals[2]*res_d, g*st4_d.w + k_vals[3]*res_d);

        if (lane == 0) {
            out_base[vi_a]   = __float2bfloat16(scale_g * qs_a + scale_qk * res_a);
            out_base[vi_a+1] = __float2bfloat16(scale_g * qs_b + scale_qk * res_b);
            out_base[vi_a+2] = __float2bfloat16(scale_g * qs_c + scale_qk * res_c);
            out_base[vi_a+3] = __float2bfloat16(scale_g * qs_d + scale_qk * res_d);
        }
    }
}

static bool g_l2_persistence_setup = false;
static size_t g_persist_max_bytes = 0;
static cudaStream_t g_last_stream = nullptr;
static const void* g_last_state_ptr = nullptr;
static size_t g_last_state_bytes = 0;

static __forceinline__ void setup_l2_persistence(cudaStream_t stream, const void* state_ptr, size_t state_bytes) {
    if (__builtin_expect(!g_l2_persistence_setup, 0)) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        g_persist_max_bytes = prop.persistingL2CacheMaxSize;
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, g_persist_max_bytes);
        g_l2_persistence_setup = true;
    }
    if (stream == g_last_stream && state_ptr == g_last_state_ptr && state_bytes == g_last_state_bytes) {
        return;
    }
    g_last_stream = stream;
    g_last_state_ptr = state_ptr;
    g_last_state_bytes = state_bytes;
    float hit_ratio = state_bytes > 0 ? fminf((float)g_persist_max_bytes / (float)state_bytes, 1.0f) : 1.0f;
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = const_cast<void*>(state_ptr);
    attr.accessPolicyWindow.num_bytes = state_bytes;
    attr.accessPolicyWindow.hitRatio  = hit_ratio;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}

void gdn_decode(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView A_log,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias,
    tvm::ffi::TensorView b_gate,
    double scale,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView new_state
) {
    int batch_size = q.size(0);

    int split_factor;
    if (batch_size <= 2) split_factor = 8;
    else if (batch_size < 32) split_factor = 8;
    else split_factor = 4;
    int rows_per_warp = HEAD_DIM / split_factor / NUM_WARPS;

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    size_t state_bytes = (size_t)batch_size * NUM_V_HEADS * HEAD_DIM * HEAD_DIM * sizeof(float);
    setup_l2_persistence(stream, state.data_ptr(), state_bytes);

    dim3 grid(batch_size * NUM_V_HEADS * split_factor);
    dim3 block(BLOCK_SIZE);

    #define LAUNCH(RPW) \
        gdn_decode_kernel<RPW><<<grid, block, 0, stream>>>( \
            static_cast<const bf16*>(q.data_ptr()), \
            static_cast<const bf16*>(k.data_ptr()), \
            static_cast<const bf16*>(v.data_ptr()), \
            static_cast<const float*>(state.data_ptr()), \
            static_cast<const float*>(A_log.data_ptr()), \
            static_cast<const bf16*>(a.data_ptr()), \
            static_cast<const float*>(dt_bias.data_ptr()), \
            static_cast<const bf16*>(b_gate.data_ptr()), \
            static_cast<float>(scale), \
            static_cast<bf16*>(output.data_ptr()), \
            static_cast<float*>(new_state.data_ptr()), \
            batch_size \
        )

    switch (rows_per_warp) {
        case 4:  LAUNCH(4);  break;
        case 8:  LAUNCH(8);  break;
        case 16: LAUNCH(16); break;
        default: LAUNCH(8);  break;
    }
    #undef LAUNCH
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, gdn_decode);