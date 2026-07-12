import torch
import triton
import triton.language as tl

# m=1, split-k
# 这里m比较小，用不上tensorcore
# 如果是GQA，如果group_size=16, 可以沿m维度拼接，利用上tensorcore
# @triton.autotune(
#     configs=[
#         triton.Config({"Bc":64}, num_warps=4, num_stages=3),
#         triton.Config({"Bc":128}, num_warps=4, num_stages=3)
#     ],
#     key=["seq_len", "head_dim"]
# )
@triton.jit
def int8_flash_decoding(
    Q, K_int8, V_int8, k_scale, v_scale,
    m_tmp, d_tmp, o_tmp,
    num_heads, seq_len, head_dim: tl.constexpr,
    stride_q0, stride_q1,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_ksh, stride_ksn,
    stride_vsh, stride_vsn,
    stride_mh, stride_mn,
    stride_dh, stride_dn,
    stride_oh, stride_o1, stride_o2,
    scale,
    Bc: tl.constexpr
):
    n_id = tl.program_id(axis=0)
    heads_id = tl.program_id(axis=1)

    q_heads_start = heads_id * stride_q0
    k_heads_start = heads_id * stride_kh
    v_heads_start = heads_id * stride_vh
    ks_heads_start = heads_id * stride_ksh
    vs_heads_start = heads_id * stride_vsh

    k_offsets = tl.arange(0, head_dim)
    n_offsets = n_id * Bc + tl.arange(0, Bc)

    q_ptrs = Q + q_heads_start + k_offsets * stride_q1
    q_data = tl.load(q_ptrs)

    k_ptrs = K_int8 + k_heads_start + (n_offsets * stride_kn)[:, None] + (k_offsets * stride_kk)[None, :]
    ks_ptrs = k_scale + ks_heads_start + (n_offsets * stride_ksn)
    k_mask = (n_offsets < seq_len)
    k_data = tl.load(k_ptrs, mask=k_mask[:, None], other=0)
    ks_data = tl.load(ks_ptrs, mask=k_mask, other=0.0)
    k_dequant = k_data.to(tl.float16) * ks_data[:, None]


    v_ptrs = V_int8 + v_heads_start + (n_offsets * stride_vn)[:, None] + (k_offsets * stride_vk)[None, :]
    vs_ptrs = v_scale + vs_heads_start + (n_offsets * stride_vsn)
    v_mask = (n_offsets < seq_len)
    v_data = tl.load(v_ptrs, mask=v_mask[:, None], other=0)
    vs_data = tl.load(vs_ptrs, mask=v_mask, other=0.0)
    v_dequant = v_data.to(tl.float16) * vs_data[:, None]


    # m_tile 
    # d_tile = sum(exp(x - mi))
    # o_tile = matmul(exp(x - mi), v)
    qk_scale = tl.sum(q_data[None, :] * k_dequant, axis=-1) * scale
    qk_scale = tl.where(n_offsets < seq_len, qk_scale, -float("inf"))
    m_tile = tl.max(qk_scale, axis=0)
    p = tl.exp(qk_scale - m_tile)
    d_tile = tl.sum(p, axis=0)
    acc = tl.sum(p[:, None] * v_dequant, axis=0) # (Br, ) * (Br, head_dim)

    # （num_heads, seq_len // Bc) 每个head有seq_len // Bc个的局部结果
    m_ptrs = m_tmp + heads_id * stride_mh + n_id * stride_mn
    d_ptrs= d_tmp + heads_id * stride_dh + n_id * stride_dn
    o_ptrs = o_tmp + heads_id * stride_oh + n_id * stride_o1 + tl.arange(0, head_dim) * stride_o2
    tl.store(m_ptrs, m_tile.to(tl.float32))
    tl.store(d_ptrs, d_tile.to(tl.float32))
    tl.store(o_ptrs, acc.to(tl.float32))

# num_heads个头，那么就启动num_heads个block, 每个block负责一个头
@triton.jit
def flash_decoding_reduce(
    m_tmp, d_tmp, o_tmp,
    res,
    stride_mh, stride_mn,
    stride_dh, stride_dn,
    stride_oh, stride_o1, stride_o2,
    stride_r0, stride_r1,
    count, head_dim: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr # block_size = triton.next_power_of(seq_len, Br)
):
    heads_id = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    m_ptrs = m_tmp + heads_id * stride_mh + offsets
    d_ptrs = d_tmp + heads_id * stride_dh + offsets
    o_ptrs = (o_tmp + heads_id * stride_oh + (offsets * stride_o1)[:, None] 
             + (tl.arange(0, head_dim) * stride_o2)[None, :])

    # 想一下全局规约怎么
    # m_global = max(m_i)
    # d_global = sum(di * exp(mi - m_global))
    # o_global = sum(oi * exp(mi - m_global)) / d_global
    mask = offsets < count
    m_data = tl.load(m_ptrs, mask, other=-float("inf"))
    d_data = tl.load(d_ptrs, mask, other=0.0)
    o_mask = (offsets < count)[:, None]
    o_data = tl.load(o_ptrs, o_mask, other=0.0)

    m_global = tl.max(m_data, axis=0)
    m_exp = tl.exp(m_data - m_global)
    d_global = tl.sum(d_data * m_exp, axis=0)
    o_global = tl.sum(o_data * m_exp[:, None], axis=0) / d_global
    
    res_ptrs = res + heads_id * stride_r0 + tl.arange(0, head_dim) * stride_r1
    tl.store(res_ptrs, o_global.to(tl.float16))

# Q, K_int8, V_int8, k_scale, v_scale, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K_int8: torch.Tensor,
    V_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    stride_q0, stride_q1 = Q.stride(0), Q.stride(1)
    stride_k0, stride_k1, stride_k2 = K_int8.stride(0), K_int8.stride(1), K_int8.stride(2)
    stride_v0, stride_v1, stride_v2 = V_int8.stride(0), V_int8.stride(1), V_int8.stride(2)
    stride_ks0, stride_ks1 = k_scale.stride(0), k_scale.stride(1)
    stride_vs0, stride_vs1 = v_scale.stride(0), v_scale.stride(1)

    Bc = 64
    cnt = triton.cdiv(seq_len, Bc)
    grid0 = (cnt, num_heads)
    m_tmp = torch.empty((num_heads, cnt), dtype=torch.float32, device=Q.device)
    d_tmp = torch.empty((num_heads, cnt), dtype=torch.float32, device=Q.device)
    o_tmp = torch.empty((num_heads, cnt, head_dim), dtype=torch.float32, device=Q.device)
    stride_m0, stride_m1 = m_tmp.stride(0), m_tmp.stride(1)
    stride_d0, stride_d1 = d_tmp.stride(0), d_tmp.stride(1)
    stride_oh, stride_o1, stride_o2 = o_tmp.stride(0), o_tmp.stride(1), o_tmp.stride(2)
    stride_r0, stride_r1 = output.stride(0), output.stride(1)

    scale = head_dim ** -0.5

    int8_flash_decoding[grid0](
        Q, K_int8, V_int8, k_scale, v_scale,
        m_tmp, d_tmp, o_tmp, 
        num_heads, seq_len, head_dim, 
        stride_q0, stride_q1,
        stride_k0, stride_k1, stride_k2,
        stride_v0, stride_v1, stride_v2,
        stride_ks0, stride_ks1,
        stride_vs0, stride_vs1,
        stride_m0, stride_m1,
        stride_d0, stride_d1,
        stride_oh, stride_o1, stride_o2,
        scale,
        Bc=Bc
    )

    grid1 = (num_heads, )
    BLOCK_SIZE = triton.next_power_of_2(cnt)
    flash_decoding_reduce[grid1](
        m_tmp, d_tmp, o_tmp, 
        output, 
        stride_m0, stride_m1,
        stride_d0, stride_d1,
        stride_oh, stride_o1, stride_o2,
        stride_r0, stride_r1,
        cnt, head_dim, BLOCK_SIZE
    )
