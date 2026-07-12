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
    # (16, 64) * (64, 128) = (16, 128)
    # (16, 128) * (128, 64) = (16, 64)
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
    k = tl.load(k_ptrs, k_mask, other=0.0)
    v = tl.load(v_ptrs, v_mask, other=0.0)

    qk_scale = tl.dot(q, tl.trans(k)) * scale
    qk_scale = tl.where(mask, qk_scale, -float("inf"))
    max_val = tl.max(qk_scale, axis=1)
    qk_scale = tl.exp(qk_scale - max_val[:, None])
    sum_val = tl.sum(qk_scale, axis=1)
    p = qk_scale / sum_val[:, None]
    acc = tl.dot(p, v)

    tl.store(o_ptrs, p, mask=o_mask)

    


def solve():
    pass