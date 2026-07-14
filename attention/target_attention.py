import torch
import triton
import triton.language as tl

# target attention
# q:候选商品，序列长度1000
# kv: 用户行为序列，最大序列长度100
# embedding_size: 64(推荐场景的特征向量表示维度不大，不像llm里比如4096)
# 因为kv这里比较小，sm里shared memory放得下，所以不需要分块计算
@triton.jit
def fused_target_attention(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_q0, stride_q1,
    stride_k0, stride_k1,
    stride_v0, stride_v1,
    stride_o0, stride_o1,
    q_seq_len, kv_seq_len, scale,
    kv_seq_len_padded: tl.constexpr, embedding_size: tl.constexpr,
    Br: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    m_offsets = pid * Br + tl.arange(0, Br)
    # 分块比较小，sm放得下，所以不递推，单次运算好
    k_offsets = tl.arange(0, embedding_size)
    n_offsets = tl.arange(0, kv_seq_len_padded)
    q_ptrs = q_ptr + (m_offsets * stride_q0)[:, None] + (k_offsets * stride_q1)[None, :]
    k_ptrs = k_ptr + (n_offsets * stride_k0)[:, None] + (k_offsets * stride_k1)[None, :]
    v_ptrs = v_ptr + (n_offsets * stride_v0)[:, None] + (k_offsets * stride_v1)[None, :]
    o_ptrs = o_ptr + (m_offsets * stride_o0)[:, None] + (k_offsets * stride_o1)[None, :]

    q_mask = (m_offsets < q_seq_len)[:, None]
    kv_mask = (n_offsets < kv_seq_len)[:, None] 
    o_mask = (m_offsets < q_seq_len)[:, None]
    mask = (m_offsets < q_seq_len)[:, None] & (n_offsets < kv_seq_len)[None, :]

    q = tl.load(q_ptrs, q_mask, other=0.0)
    k = tl.load(k_ptrs, kv_mask, other=0.0)
    v = tl.load(v_ptrs, kv_mask, other=0.0)

    qk_scale = tl.dot(q, tl.trans(k)) * scale
    qk_scale = tl.where(mask, qk_scale, -float("inf"))
    max_val = tl.max(qk_scale, axis=1)
    qk_scale = tl.exp(qk_scale - max_val[:, None])
    sum_val = tl.sum(qk_scale, axis=1)
    p = qk_scale / sum_val[:, None]
    acc = tl.dot(p, v)

    tl.store(o_ptrs, acc.to(tl.float16), mask=o_mask)


    
def solve(
    q, k, v, o,
    q_seq_len, kv_seq_len, embedding_size
):
    kv_seq_len_padded = max(16, triton.next_power_of_2(kv_seq_len))
    Br = 16
    grid = (triton.cdiv(q_seq_len, Br), )
    scale = embedding_size ** -0.5

    stride_q0, stride_q1 = q.stride(0), q.stride(1)
    stride_k0, stride_k1 = k.stride(0), k.stride(1)
    stride_v0, stride_v1 = v.stride(0), v.stride(1)
    stride_o0, stride_o1 = o.stride(0), o.stride(1)
    fused_target_attention[grid](q, k, v, o, stride_q0, stride_q1, stride_k0, stride_k1,
                                 stride_v0, stride_v1, stride_o0, stride_o1, 
                                 q_seq_len, kv_seq_len, scale, kv_seq_len_padded, embedding_size, Br)


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    q_seq_len = 1000
    kv_seq_len = 128
    embedding_size = 32

    q = torch.randn(q_seq_len, embedding_size, device=device, dtype=dtype)
    k = torch.randn(kv_seq_len, embedding_size, device=device, dtype=dtype)
    v = torch.randn(kv_seq_len, embedding_size, device=device, dtype=dtype)
    o = torch.empty((q_seq_len, embedding_size), device=device, dtype=dtype)

    solve(q, k, v, o, q_seq_len, kv_seq_len, embedding_size)
    torch.cuda.synchronize()