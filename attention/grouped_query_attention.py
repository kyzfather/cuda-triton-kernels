import torch
import triton
import triton.language as tl

'''
如果是prefill阶段, 一般是采用调整grid的顺序, 被动依赖l2cache的优化。
或者是一个block处理一个group的q head, 但是这样寄存器压力比较大,
可能需要调小Br或者Bc来减少register spill。
如果是decode阶段, 可以主动,因为M很小,寄存器压力小,
可以主动的让一个block处理一个group的q head来优化内存带宽的压力。
'''
# shape: (seq_len, num_heads, head_dim)
@triton.autotune(
    configs=[
        triton.Config({"Br":64, "Bc":64}, num_warps=4, num_stages=3),
        triton.Config({"Br":128, "Bc":64}, num_warps=8, num_stages=3),
        triton.Config({"Br":128, "Bc":32}, num_warps=4, num_stages=4)
    ],
    key=["seq_len", "head_dim"]
)
@triton.jit
def grouped_query_attention(
    Q, K, V, output,
    num_q_heads, num_kv_heads,
    seq_len, head_dim, scale,
    stride_qm, stride_qk,
    stride_kn, stride_kk, 
    stride_vn, stride_vk,
    stride_om, stride_on,
    Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr
):
    m_id = tl.program_id(axis=0)
    heads_id = tl.program_id(axis=1)
    group_size = num_q_heads // num_kv_heads
    group_id = heads_id // group_size

    q_head_start = heads_id * head_dim
    q_head_end = q_head_start + head_dim

    kv_head_start = group_id * head_dim
    kv_head_end = kv_head_start + head_dim

    m_offsets = m_id * Br + tl.arange(0, Br)
    q_k_offsets = q_head_start + tl.arange(0, Bk)
    kv_k_offsets = kv_head_start + tl.arange(0, Bk)

    q_ptrs = Q + (m_offsets * stride_qm)[:, None] + (q_k_offsets * stride_qk)[None, :]
    q_mask = (m_offsets < seq_len)[:, None] & (q_k_offsets < q_head_end)[None, :]
    q_data = tl.load(q_ptrs, mask=q_mask, other=0.0)

    steps = (seq_len + Bc - 1) // Bc
    m_old = tl.full((Br, ), dtype=tl.float32, value=-float("inf"))
    d_old = tl.zeros((Br, ), dtype=tl.float32)
    acc = tl.zeros((Br, Bk), dtype=tl.float32)
    for i in range(steps):
        n_offsets = i * Bc + tl.arange(0, Bc)

        k_ptrs = K + (n_offsets * stride_kn)[:, None] + (kv_k_offsets * stride_kk)[None, :]
        v_ptrs = V + (n_offsets * stride_vn)[:, None] + (kv_k_offsets * stride_vk)[None, :]

        k_mask = (n_offsets < seq_len)[:, None] & (kv_k_offsets < kv_head_end)[None, :]
        v_mask = (n_offsets < seq_len)[:, None] & (kv_k_offsets < kv_head_end)[None, :]

        k_data = tl.load(k_ptrs, mask=k_mask, other=0.0)
        mask = (m_offsets < seq_len)[:, None] & (n_offsets < seq_len)[None, :]

        qk_scale = tl.where(mask, tl.dot(q_data, tl.trans(k_data)) * scale, -float("inf"))
        m_new = tl.maximum(m_old, tl.max(qk_scale, axis=1))
        qk_scale = tl.exp(qk_scale - m_new[:, None])
        delta = tl.exp(m_old - m_new)
        d_new = d_old * delta + tl.sum(qk_scale, axis=1)

        v_data = tl.load(v_ptrs, mask=v_mask, other=0.0)
        acc = acc * delta[:, None]
        acc = tl.dot(qk_scale, v_data, acc)

        m_old = m_new
        d_old = d_new
    acc = acc / d_old[:, None]
    output_ptrs = output + (m_offsets * stride_om)[:, None] + (q_k_offsets * stride_on)[None, :]
    output_mask = (m_offsets < seq_len)[:, None] & (q_k_offsets < q_head_end)[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)

# shape:(num_heads, seq_len, head_dim)
# Bk = max(16, next_power_of_2(head_dim)), 防止测试用例中head_dim太小
@triton.autotune(
    configs=[
        triton.Config({"Br":64, "Bc":64}, num_warps=4, num_stages=3),
        triton.Config({"Br":128, "Bc":64}, num_warps=8, num_stages=3),
        triton.Config({"Br":128, "Bc":32}, num_warps=4, num_stages=4)
    ],
    key=["seq_len", "head_dim"]
)
@triton.jit
def grouped_query_attention_v2(
    Q, K, V, output,
    num_q_heads, num_kv_heads,
    seq_len, head_dim, scale,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk, 
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_on,
    Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr 
):
    m_id = tl.program_id(axis=0)
    heads_id = tl.program_id(axis=1)
    group_size = num_q_heads // num_kv_heads
    group_id = heads_id // group_size

    q_head_start = heads_id * stride_qh
    k_head_start = group_id * stride_kh
    v_head_start = group_id * stride_vh
    # kv_head_end = kv_head_start + head_dim

    m_offsets = m_id * Br + tl.arange(0, Br)
    k_offsets = tl.arange(0, Bk)

    q_ptrs = Q + q_head_start + (m_offsets * stride_qm)[:, None] + (k_offsets * stride_qk)[None, :]
    q_mask = (m_offsets < seq_len)[:, None] & (k_offsets < head_dim)[None, :]
    q_data = tl.load(q_ptrs, mask=q_mask, other=0.0)

    steps = (seq_len + Bc - 1) // Bc
    m_old = tl.full((Br, ), dtype=tl.float32, value=-float("inf"))
    d_old = tl.zeros((Br, ), dtype=tl.float32)
    acc = tl.zeros((Br, Bk), dtype=tl.float32)
    for i in range(steps):
        n_offsets = i * Bc + tl.arange(0, Bc)

        k_ptrs = K + k_head_start + (n_offsets * stride_kn)[:, None] + (k_offsets * stride_kk)[None, :]
        v_ptrs = V + v_head_start + (n_offsets * stride_vn)[:, None] + (k_offsets * stride_vk)[None, :]

        k_mask = (n_offsets < seq_len)[:, None] & (k_offsets < head_dim)[None, :]
        v_mask = (n_offsets < seq_len)[:, None] & (k_offsets < head_dim)[None, :]

        k_data = tl.load(k_ptrs, mask=k_mask, other=0.0)
        mask = (m_offsets < seq_len)[:, None] & (n_offsets < seq_len)[None, :]

        qk_scale = tl.where(mask, tl.dot(q_data, tl.trans(k_data)) * scale, -float("inf"))
        m_new = tl.maximum(m_old, tl.max(qk_scale, axis=1))
        qk_scale = tl.exp(qk_scale - m_new[:, None])
        delta = tl.exp(m_old - m_new)
        d_new = d_old * delta + tl.sum(qk_scale, axis=1)

        v_data = tl.load(v_ptrs, mask=v_mask, other=0.0)
        acc = acc * delta[:, None]
        acc = tl.dot(qk_scale, v_data, acc)

        m_old = m_new
        d_old = d_new
    acc = acc / d_old[:, None]
    output_head_start = heads_id * stride_oh
    output_ptrs = output + output_head_start + (m_offsets * stride_om)[:, None] + (k_offsets * stride_on)[None, :]
    output_mask = (m_offsets < seq_len)[:, None] & (k_offsets < head_dim)[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
):
    stride_qh, stride_qm, stride_qk = Q.stride(0), Q.stride(1), Q.stride(2)
    stride_kh, stride_kn, stride_kk = K.stride(0), K.stride(1), K.stride(2)
    stride_vh, stride_vn, stride_vk = V.stride(0), V.stride(1), V.stride(2)
    stride_oh, stride_om, stride_on = output.stride(0), output.stride(1), output.stride(2)
    group_size = num_q_heads // num_kv_heads
    grid = lambda meta: (triton.cdiv(seq_len, meta["Br"]), num_q_heads)
    scale = head_dim ** -0.5
    Bk = max(16, triton.next_power_of_2(head_dim))
    grouped_query_attention_v2[grid](
        Q, K, V, output, num_q_heads, num_kv_heads, 
        seq_len, head_dim, scale,
        stride_qh, stride_qm, stride_qk,
        stride_kh, stride_kn, stride_kk,
        stride_vh, stride_vn, stride_vk,
        stride_oh, stride_om, stride_on,
        Bk=Bk
    )

