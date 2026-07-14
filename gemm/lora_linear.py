import torch
import triton
import triton.language as tl

# (256, 4096) * (4096, 64) = (256, 64)  X*A
# (256, 64) * (64, 4096) = (256, 4096)  X*A*B
# (256, 4096) * (4096, 4096) = (256, 4096) X*W
# kernel0：算X*A，写到gmem.
# kernel1: fuse, X*W的分块和(X*A)*B的分块，相加再写到gmem
@triton.autotune(
    configs=[
        triton.Config({"Br": 64, "Bc": 64, "Bk": 32, "SPLIT_K": 4}, num_warps=4, num_stages=3),
        triton.Config({"Br": 64, "Bc": 64, "Bk": 32, "SPLIT_K": 8}, num_warps=8, num_stages=3),
        triton.Config({"Br": 32, "Bc": 64, "Bk": 64, "SPLIT_K": 4}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
    reset_to_zero=["C"], 
)
@triton.jit
def split_k_gemm( 
    A, B, C, # 注意这里B没有转置
    M, N, K,
    stride_am, stride_ak, # 为什么要用stride而不用mnk, 因为可能padding
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    m_id = tl.program_id(axis=0)
    n_id = tl.program_id(axis=1)
    k_id = tl.program_id(axis=2)

    k_per_split = K // SPLIT_K
    steps = (k_per_split + Bk - 1) // Bk
    k_start = k_id * k_per_split
    k_end = k_start + k_per_split

    m_offsets = m_id * Br + tl.arange(0, Br)
    n_offsets = n_id * Bc + tl.arange(0, Bc)
    C_ptrs = C + (m_offsets * stride_cm)[:, None] + (n_offsets * stride_cn)[None, :]
    c_mask = (m_offsets < M)[:, None] & (n_offsets < N)[None, :]

    acc = tl.zeros((Br, Bc), dtype=tl.float32)
    for i in range(steps):
        k_offsets = k_start + i * Bk + tl.arange(0, Bk)
        A_ptrs = A + (m_offsets * stride_am)[:, None] + (k_offsets * stride_ak)[None, :]
        B_ptrs = B + (n_offsets * stride_bn)[None, :] + (k_offsets * stride_bk)[:, None]  # 注意这里的写法
        
        a_mask = (m_offsets < M)[:, None] & (k_offsets < k_end)[None, :]
        b_mask = (k_offsets < k_end)[:, None] & (n_offsets < N)[None, :] # 注意这里的写法

        A_data = tl.load(A_ptrs, mask=a_mask, other=0.0)
        B_data = tl.load(B_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(A_data, B_data, acc)

    tl.atomic_add(C_ptrs, acc, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({"Br": 64,  "Bc": 64,  "Bk": 32}, num_warps=4, num_stages=3),
        triton.Config({"Br": 128, "Bc": 64,  "Bk": 32}, num_warps=4, num_stages=3),
        triton.Config({"Br": 64,  "Bc": 128, "Bk": 32}, num_warps=4, num_stages=4),
        triton.Config({"Br": 128, "Bc": 128, "Bk": 32}, num_warps=8, num_stages=3),
        triton.Config({"Br": 128, "Bc": 128, "Bk": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "d_in", "d_out", "rank"],
)
@triton.jit
def lora_fused_kernel(
    XA, B, X, W, output,
    M, rank, d_in, d_out,
    stride_xam, stride_xak,
    stride_bn, stride_bk,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    lora_scale, rank_padded: tl.constexpr,
    Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr,
):   
    m_id = tl.program_id(axis=0)
    n_id = tl.program_id(axis=1)

    m_offsets = m_id * Br + tl.arange(0, Br)
    n_offsets = n_id * Bc + tl.arange(0, Bc)

    steps = (d_in + Bk - 1) // Bk
    acc = tl.zeros((Br, Bc), dtype=tl.float32)
    for i in range(steps):
        k_offsets = i * Bk + tl.arange(0, Bk)

        X_ptrs = X + (m_offsets * stride_xm)[:, None] + (k_offsets * stride_xk)[None, :]
        W_ptrs = W + (n_offsets * stride_wn)[None, :] + (k_offsets * stride_wk)[:, None]

        x_mask = (m_offsets < M)[:, None] & (k_offsets < d_in)[None, :]
        w_mask = (k_offsets < d_in)[:, None] & (n_offsets < d_out)[None, :]
        
        x_data = tl.load(X_ptrs, mask=x_mask, other=0.0)
        w_data = tl.load(W_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_data, w_data, acc)


    # xa*b的计算，k维度一次计算完，因为k=rank<=256
    offsets = tl.arange(0, rank_padded)
    XA_ptrs = XA + (m_offsets * stride_xam)[:, None] + (offsets * stride_xak)[None, :]
    B_ptrs = B + (offsets * stride_bk)[:, None] + (n_offsets * stride_bn)[None, :]
    xa_mask = (m_offsets < M)[:, None] & (offsets < rank)[None, :]
    b_mask = (offsets < rank)[:, None] & (n_offsets < d_out)[None, :]
    xa_data = tl.load(XA_ptrs, mask=xa_mask, other=0.0)
    b_data = tl.load(B_ptrs, mask=b_mask, other=0.0)

    acc += tl.dot(xa_data, b_data) * lora_scale

    output_ptrs = output + (m_offsets * stride_om)[:, None] + (n_offsets * stride_on)[None, :]
    output_mask = (m_offsets < M)[:, None] & (n_offsets < d_out)[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)



# x, W, A, B, output are tensors on the GPU
def solve(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    output: torch.Tensor,
    batch: int,
    d_in: int,
    d_out: int,
    rank: int,
    lora_scale: float,
):
    grid0 = lambda meta: (
        triton.cdiv(batch, meta["Br"]), 
        triton.cdiv(rank, meta["Bc"]),
        meta["SPLIT_K"],
    )

    grid1 = lambda meta: (
        triton.cdiv(batch, meta["Br"]),
        triton.cdiv(d_out, meta["Bc"])
    )
    
    rank_padded = max(16, triton.next_power_of_2(rank))

    xa = torch.zeros((batch, rank), dtype=torch.float32, device=x.device)
    split_k_gemm[grid0](x, A, xa, batch, rank, d_in, x.stride(0), x.stride(1), 
                        A.stride(0), A.stride(1), xa.stride(0), xa.stride(1))
    lora_fused_kernel[grid1](xa, B, x, W, output, batch, rank, d_in, d_out, 
                            xa.stride(0), xa.stride(1), B.stride(0), B.stride(1),
                            x.stride(0), x.stride(1), W.stride(0), W.stride(1),
                            output.stride(0), output.stride(1), lora_scale, rank_padded)

