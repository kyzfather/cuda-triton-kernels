import torch
import triton
import triton.language as tl

# 长上下文计算量大，kv显存占用大
# 正常self-attention的计算复杂度是o(l^2), sliding-window attention是o(l*w)
# kvcache显存占用o(l), o(w)常数，decode时候只需要维持一个w长度的kvcache就行
@triton.autotune(
    configs=[
        triton.Config({"Br":32, "Bc":32}, num_warps=4, num_stages=3),
        # triton.Config({"Br":32, "Bc":64}, num_warps=8, num_stages=3),
    ],
    key=["M", "d", "window_size"]
)
@triton.jit
def sliding_window_attention(
    q_ptr, k_ptr, v_ptr, output_ptr, 
    M, d, window_size,
    stride_q0, stride_q1,
    stride_k0, stride_k1,
    stride_v0, stride_v1,
    stride_o0, stride_o1,
    scale, d_padded: tl.constexpr,
    Br: tl.constexpr, Bc: tl.constexpr
):
    m_id = tl.program_id(axis=0)
    m_offsets = m_id * Br + tl.arange(0, Br)
    k_offsets = tl.arange(0, d_padded)
    q_ptrs = q_ptr + (m_offsets * stride_q0)[:, None] + (k_offsets * stride_q1)[None, :]
    q_mask = (m_offsets < M)[:, None] & (k_offsets < d)[None, :]
    q_data = tl.load(q_ptrs, q_mask, other=0.0)


    m_old = tl.full((Br,), value=-float("inf"), dtype=tl.float32)
    d_old = tl.zeros((Br,), dtype=tl.float32)
    acc = tl.zeros((Br, d_padded), dtype=tl.float32)
    # 只遍历这个窗口，不要整个序列都遍历
    lo = tl.maximum(m_id * Br - window_size, 0)
    hi = tl.minimum(m_id * Br + Br + window_size + 1, M)
    for i in range(lo, hi, Bc):
        n_offsets = i + tl.arange(0, Bc)
        k_ptrs = k_ptr + (n_offsets * stride_k0)[:, None] + (k_offsets * stride_k1)[None, :]
        v_ptrs = v_ptr + (n_offsets * stride_v0)[:, None] + (k_offsets * stride_v1)[None, :]

        kv_mask = (n_offsets < M)[:, None] & (k_offsets < d)[None, :]  
        k_data = tl.load(k_ptrs, kv_mask, other=0.0)
        v_data = tl.load(v_ptrs, kv_mask, other=0.0)

        mask0 = (m_offsets - window_size)[:, None] <= (n_offsets)[None, :] 
        mask1 = (m_offsets + window_size)[:, None] >= (n_offsets)[None, :]
        mask = mask0 & mask1 & (m_offsets < M)[:, None] & (n_offsets < M)[None, :]

        qk_scale = tl.dot(q_data, tl.trans(k_data), input_precision="ieee") * scale
        qk_scale = tl.where(mask, qk_scale, -float("inf"))
        m_new = tl.maximum(m_old, tl.max(qk_scale, axis=1))
        p = tl.exp(qk_scale - m_new[:, None])
        delta = tl.exp(m_old - m_new)
        d_new = d_old * delta + tl.sum(p, axis=1)
        acc *= delta[:, None]
        acc = tl.dot(p, v_data, acc, input_precision="ieee")

        m_old = m_new
        d_old = d_new
    d_old_safe = tl.where(d_old > 0, d_old, 1.0)
    acc /= d_old_safe[:, None]
    output_ptrs = output_ptr + (m_offsets * stride_o0)[:, None] + (k_offsets * stride_o1)[None, :]
    output_mask = (m_offsets < M)[:, None] & (k_offsets < d)[None, :]
    tl.store(output_ptrs, acc, output_mask)

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    window_size: int,
):
    stride_q0, stride_q1 = Q.stride(0), Q.stride(1)
    stride_k0, stride_k1 = K.stride(0), K.stride(1)
    stride_v0, stride_v1 = V.stride(0), V.stride(1)
    stride_o0, stride_o1 = output.stride(0), output.stride(1)
    scale = d ** -0.5
    d_padded = max(16, triton.next_power_of_2(d))
    grid = lambda meta: (triton.cdiv(M, meta["Br"]), )
    sliding_window_attention[grid](
        Q, K, V, output, M, d, window_size,
        stride_q0, stride_q1,
        stride_k0, stride_k1,
        stride_v0, stride_v1,
        stride_o0, stride_o1,
        scale, d_padded
    )
    
