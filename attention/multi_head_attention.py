import torch
import triton
import triton.language as tl

# grid(num_heads, seq_len/br, d_model/N/bc)
# flashAttention的推导公式感觉需要记住
# 就是头这里再划分block,不同的block执行呗.
# 原本可能是4096, bc=128, 4096/128.现在是4096/32
# grid(num_heads, seq_len/br)
@triton.autotune(
    configs=[
        triton.Config({"Br": 128, "Bc": 64}, num_warps=8, num_stages=3),
        triton.Config({"Br": 64,  "Bc": 64}, num_warps=4, num_stages=3),
        triton.Config({"Br": 64,  "Bc": 128}, num_warps=8, num_stages=2),
        triton.Config({"Br": 32,  "Bc": 32}, num_warps=4, num_stages=2),
    ],
    key=["N", "head_dim"],
)
@triton.jit
def mha_kernel(
    Q, K, V, output,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    stride_om, stride_on,
    N, d_model, h, scale, head_dim,
    Br: tl.constexpr, Bc: tl.constexpr, Bd: tl.constexpr
):
    heads_id = tl.program_id(axis=0)
    row_id = tl.program_id(axis=1)

    head_start = head_dim * heads_id
    head_end = head_start + head_dim

    q_m_offsets = row_id * Br + tl.arange(0, Br)
    k_offsets = head_start + tl.arange(0, Bd) # Br == dim_per_head, k轴方向一次运算, k通常比较小128

    steps = (N + Bc - 1) // Bc
    m_old = tl.full((Br,), dtype=tl.float32, value=-float("inf"))
    d_old = tl.zeros((Br,), dtype=tl.float32)
    acc = tl.zeros((Br, Bd), dtype=tl.float32)
    for i in range(steps):
        n_offsets = i * Bc + tl.arange(0, Bc)

        Q_ptrs = Q + (q_m_offsets * stride_qm)[:, None] + (k_offsets * stride_qk)[None, :]
        K_ptrs = K + (n_offsets * stride_kn)[None, :] + (k_offsets * stride_kk)[:, None]
        V_ptrs = V + (n_offsets * stride_vn)[:, None] + (k_offsets * stride_vk)[None, :]

        q_mask = (q_m_offsets < N)[:, None] & (k_offsets < head_end)[None, :]
        k_mask = (n_offsets < N)[None, :] & (k_offsets < head_end)[:, None]
        v_mask = (n_offsets < N)[:, None] & (k_offsets < head_end)[None, :]

        Q_data = tl.load(Q_ptrs, mask=q_mask, other=0.0)
        K_data = tl.load(K_ptrs, mask=k_mask, other=0.0)

        # m_new = max(m_old, m_tile)
        # d_new = d_old * exp(m_old - m_new) + sum(exp(x - m_new))
        # o_new = {o_old * d_old * exp(m_old - m_new) + matmul(exp(x - m_new) * v)} / di 
        qk = tl.dot(Q_data, K_data)
        qk_scale = qk * scale
        qk_scale = tl.where((q_m_offsets < N)[:, None] & (n_offsets < N)[None, :], qk_scale, -float("inf"))
        m_new = tl.maximum(m_old, tl.max(qk_scale, axis=1))
        new_qk = tl.exp(qk_scale - m_new[:, None])
        delta = tl.exp(m_old - m_new)
        d_new = d_old * delta + tl.sum(new_qk, axis=1)
        acc *= delta[:, None]

        V_data = tl.load(V_ptrs, mask=v_mask, other=0.0)
        acc = tl.dot(new_qk, V_data, acc)

        m_old = m_new
        d_old = d_new
    acc /= d_old[:, None]
    output_ptrs = output + (q_m_offsets * stride_om)[:, None] + (k_offsets * stride_on)[None, :]
    output_mask = (q_m_offsets < N)[:, None] & (k_offsets < head_end)[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    stride_qm, stride_qk = Q.stride(0), Q.stride(1)
    stride_kn, stride_kk = K.stride(0), K.stride(1)
    stride_vn, stride_vk = V.stride(0), V.stride(1)
    stride_om, stride_on = output.stride(0), output.stride(1)
    head_dim = d_model // h 
    qk_scale = head_dim ** -0.5
    grid = lambda meta: (h, triton.cdiv(N, meta["Br"]))
    Bd = max(16, triton.next_power_of_2(head_dim))  # 防止h太大, 导致head_dim太小, tl.dot出现问题
    mha_kernel[grid](
        Q, K, V, output,
        stride_qm, stride_qk,
        stride_kn, stride_kk,
        stride_vn, stride_vk,
        stride_om, stride_on,
        N, d_model, h, qk_scale, head_dim,
        Bd=Bd
    )
