import torch
import triton
import triton.language as tl
import math


@triton.jit
def _gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    scale_val,
    output_ptr, new_state_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // NUM_V_HEADS
    h_idx = pid % NUM_V_HEADS
    qk_h = h_idx // (NUM_V_HEADS // NUM_Q_HEADS)

    a_val = tl.load(a_ptr + b_idx * NUM_V_HEADS + h_idx).to(tl.float32)
    dt_val = tl.load(dt_bias_ptr + h_idx).to(tl.float32)
    A_val = tl.load(A_log_ptr + h_idx).to(tl.float32)
    b_val = tl.load(b_ptr + b_idx * NUM_V_HEADS + h_idx).to(tl.float32)

    x = a_val + dt_val
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_val) * sp)
    beta = tl.sigmoid(b_val)

    offs_d = tl.arange(0, HEAD_SIZE)
    qk_base = b_idx * NUM_Q_HEADS * HEAD_SIZE + qk_h * HEAD_SIZE
    q_vec = tl.load(q_ptr + qk_base + offs_d).to(tl.float32)
    k_vec = tl.load(k_ptr + qk_base + offs_d).to(tl.float32)

    s_head = (b_idx * NUM_V_HEADS + h_idx) * HEAD_SIZE * HEAD_SIZE
    v_head = b_idx * NUM_V_HEADS * HEAD_SIZE + h_idx * HEAD_SIZE

    for v_off in range(0, HEAD_SIZE, BLOCK_V):
        offs_v = v_off + tl.arange(0, BLOCK_V)
        s_idx = s_head + offs_v[:, None] * HEAD_SIZE + offs_d[None, :]

        s = tl.load(state_ptr + s_idx).to(tl.float32)
        gs = g * s

        old_v = tl.sum(k_vec[None, :] * gs, axis=1)

        v_tile = tl.load(v_ptr + v_head + offs_v).to(tl.float32)
        delta = beta * (v_tile - old_v)

        ns = gs + k_vec[None, :] * delta[:, None]

        o = scale_val * tl.sum(q_vec[None, :] * ns, axis=1)

        tl.store(output_ptr + v_head + offs_v, o.to(tl.bfloat16))
        tl.store(new_state_ptr + s_idx, ns)


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    B, T, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[2]

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    q_f = q.squeeze(1).contiguous()
    k_f = k.squeeze(1).contiguous()
    v_f = v.squeeze(1).contiguous()
    a_f = a.squeeze(1).contiguous()
    b_f = b.squeeze(1).contiguous()
    o_f = output.squeeze(1)

    if state is None:
        state = torch.zeros(B, num_v_heads, head_size, head_size,
                            dtype=torch.float32, device=q.device)

    _gdn_decode_kernel[(B * num_v_heads,)](
        q_f, k_f, v_f, state,
        A_log, a_f, dt_bias, b_f,
        scale,
        o_f, new_state,
        NUM_Q_HEADS=num_q_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_SIZE=head_size,
        BLOCK_V=32,
    )
