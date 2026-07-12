import torch
import triton
import triton.language as tl

# 按照naive的方式写下吧，就是矩阵乘
# 先让K和V相乘，乘完的结果写回gemm. 再做Q和KV的矩阵乘
# 然后还要做Q和向量sum(k)的 矩阵向量乘
@triton.jit
def map(data):
    data = tl.where(data > 0, data + 1, tl.exp(data))
    return data

@triton.jit
def gemm_split_k(
    k_ptr, v_ptr, o_ptr, k_sum_ptr,
    N, head_dim, head_dim_padded: tl.constexpr,
    stride_k0, stride_k1,
    stride_v0, stride_v1,
    stride_o0, stride_o1,
    Bn: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    n_per_split = N // SPLIT_K
    steps = (n_per_split + Bn - 1) // Bn
    start = pid * n_per_split
    end = start +  n_per_split
    end = tl.where(pid == SPLIT_K - 1, N, end)

    offsets = tl.arange(0, head_dim_padded)

    acc = tl.zeros((head_dim_padded, head_dim_padded), dtype=tl.float32)
    k_sum = tl.zeros((head_dim_padded,), dtype=tl.float32)
    for i in range(steps):
        n_offsets = start + i * Bn + tl.arange(0, Bn)
        k_ptrs = k_ptr + (n_offsets * stride_k0)[:, None] + (offsets * stride_k1)[None, :]
        v_ptrs = v_ptr + (n_offsets * stride_v0)[:, None] + (offsets * stride_v1)[None, :]
        
        tmp_mask = (n_offsets < end)[:, None]
        mask = (n_offsets < N)[:, None] & (offsets < head_dim)[None, :] & tmp_mask

        k_raw_data = tl.load(k_ptrs, mask=mask, other=0.0)
        k_data = tl.where(mask, map(k_raw_data), 0.0)
        v_data = tl.load(v_ptrs, mask=mask, other=0.0) 
        acc = tl.dot(tl.trans(k_data), v_data, acc)
        k_sum += tl.sum(k_data, axis=0)
    # split k, k不是很大，感觉这里可以atomic add
    o_ptrs = o_ptr + (offsets * stride_o0)[:, None] + (offsets * stride_o1)[None, :]
    k_sum_ptrs = k_sum_ptr + offsets
    o_mask = (offsets < head_dim)[:, None] & (offsets < head_dim)[None, :]
    ks_mask = offsets < head_dim
    tl.atomic_add(o_ptrs, acc, mask=o_mask)
    tl.atomic_add(k_sum_ptrs, k_sum, mask=ks_mask)

@triton.jit
def linear_attention(
    q_ptr, kv_ptr, k_sum_ptr, output_ptr,
    N, head_dim, head_dim_padded: tl.constexpr,
    stride_q0, stride_q1,
    stride_kv0, stride_kv1,
    stride_o0, stride_o1,
    Bn: tl.constexpr
):
    pid = tl.program_id(axis=0)
    n_offsets = pid * Bn + tl.arange(0, Bn)
    k_offsets = tl.arange(0, head_dim_padded)

    # head_dim <= 128, 矩阵k轴一次计算
    q_ptrs = q_ptr + (n_offsets * stride_q0)[:, None] + (k_offsets * stride_q1)[None, :]
    kv_ptrs = kv_ptr + (k_offsets * stride_kv0)[:, None] + (k_offsets * stride_kv1)[None, :]
    ks_ptrs = k_sum_ptr + k_offsets

    q_mask = (n_offsets < N)[:, None] & (k_offsets < head_dim)[None, :]
    kv_mask = (k_offsets < head_dim)[:, None] & (k_offsets < head_dim)[None, :]
    ks_mask = k_offsets < head_dim
    q_raw_data = tl.load(q_ptrs, mask=q_mask, other=0.0) # (Bn, head_dim) * (head_dim, head_dim)
    q_data = tl.where(q_mask, map(q_raw_data), 0.0)
    kv_data = tl.load(kv_ptrs, mask=kv_mask, other=0.0)
    ks_data = tl.load(ks_ptrs, mask=ks_mask, other=0.0)

    tmp0 = tl.dot(q_data, kv_data)
    tmp1 = tl.sum(q_data * ks_data[None, :], axis=1, keep_dims=True)
    res = tmp0 / (tmp1 + 1e-6)
    output_ptrs = output_ptr + (n_offsets * stride_o0)[:, None]  + (k_offsets * stride_o1)[None, :]
    tl.store(output_ptrs, res, mask=q_mask)


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    stride_q0, stride_q1 = Q.stride(0), Q.stride(1)
    stride_k0, stride_k1 = K.stride(0), K.stride(1)
    stride_v0, stride_v1 = V.stride(0), V.stride(1)
    stride_o0, stride_o1 = output.stride(0), output.stride(1)
    SPLIT_K = min(8, M)
    Bn = 32
    d_padded = max(16, triton.next_power_of_2(d))
    kv = torch.zeros((d, d), dtype=K.dtype, device=K.device)
    k_sum = torch.zeros((d,), dtype=K.dtype, device=K.device)
    stride_kv0, stride_kv1 = kv.stride(0), kv.stride(1)
    grid0 = (SPLIT_K,)
    gemm_split_k[grid0](
        K, V, kv, k_sum,
        M, d, d_padded,
        stride_k0, stride_k1,
        stride_v0, stride_v1,
        stride_kv0, stride_kv1,
        Bn, SPLIT_K
    )
    grid1 = (triton.cdiv(M, Bn),)
    linear_attention[grid1](
        Q, kv, k_sum, output,
        M, d, d_padded,
        stride_q0, stride_q1,
        stride_kv0, stride_kv1,
        stride_o0, stride_o1,
        Bn
    )

