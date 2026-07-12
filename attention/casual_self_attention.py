import torch
import triton
import triton.language as tl

@triton.jit 
def causal_self_attention(
    Q, K, V, output, 
    M, d,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    stride_om, stride_ok,
    scale,
    Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr
):
    pid = tl.program_id(axis=0)

    q_m_offsets = pid * Br + tl.arange(0, Br)
    k_offsets = tl.arange(0, Bk)
    q_ptrs = Q + (q_m_offsets * stride_qm)[:, None] + (k_offsets * stride_qk)[None, :]
    q_mask = (q_m_offsets < M)[:, None] & (k_offsets < d)[None, :]
    q_data = tl.load(q_ptrs, mask=q_mask, other=0.0)

    steps = (M + Bc - 1) // Bc 
    m_old = tl.full((Br,), dtype=tl.float32, value=-float("inf"))
    d_old = tl.zeros((Br,), dtype=tl.float32)
    acc = tl.zeros((Br, Bk), dtype=tl.float32)
    for i in range(steps):
        n_offsets = i * Bc + tl.arange(0, Bc)

        k_ptrs = K + (n_offsets * stride_kn)[:, None] + (k_offsets * stride_kk)[None, :]
        v_ptrs = V + (n_offsets * stride_vn)[:, None] + (k_offsets * stride_vk)[None, :]
        kv_mask = (n_offsets < M)[:, None] & (k_offsets < d)[None, :]

        k_data = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v_data = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        
        mask = (q_m_offsets < M)[:, None] & (n_offsets < M)[None, :]
        casual_mask = mask & (q_m_offsets[:, None] >= n_offsets[None, :])

        qk_scale = tl.dot(q_data, tl.trans(k_data)) * scale
        qk_scale = tl.where(casual_mask, qk_scale, -float("inf"))
        m_new = tl.maximum(m_old, tl.max(qk_scale, axis=1))
        p = tl.exp(qk_scale - m_new[:, None])
        d_new = d_old * delta + tl.sum(p, axis=1)
        delta = tl.exp(m_old - m_new)
        acc *= delta[:, None]
        acc = tl.dot(p, v_data, acc)

        m_old = m_new
        d_old = d_new
    acc /= d_old[:, None]
    output_ptrs = output + (q_m_offsets * stride_om)[:, None] + (k_offsets * stride_ok)[None, :]
    output_mask = (q_m_offsets < M)[:, None] & (k_offsets < d)[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    stride_qm, stride_qk = Q.stride(0), Q.stride(1)
    stride_kn, stride_kk = K.stride(0), K.stride(1)
    stride_vn, stride_vk = V.stride(0), V.stride(1)
    stride_om, stride_ok = output.stride(0), output.stride(1)
    Bk = max(16, triton.next_power_of_2(d))
    Br = 64
    Bc = 64
    grid = (triton.cdiv(M, Br), )
    scale = d ** -0.5
    causal_self_attention[grid](
        Q, K, V, output, M, d, 
        stride_qm, stride_qk,
        stride_kn, stride_kk,
        stride_vn, stride_vk,
        stride_om, stride_ok,
        scale,
        Br, Bc, Bk
    )
