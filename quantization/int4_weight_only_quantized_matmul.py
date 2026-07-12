import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=(
        triton.Config({"Br":64, "Bc":64, "Bk":32}, num_warps=4, num_stages=3),
        triton.Config({"Br":128, "Bc":128, "Bk":32}, num_warps=8, num_stages=3),
    ),
    key=["M", "N", "K"]
)
@triton.jit
def w4a16(
    x, w_q, scales, y,
    M, N, K,
    group_size: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sk,
    stride_ym, stride_yn,
    Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr,
):
    m_id = tl.program_id(axis=0)
    n_id = tl.program_id(axis=1)
    m_offsets = m_id * Br + tl.arange(0, Br)
    n_offsets = n_id * Bc + tl.arange(0, Bc)

    steps = (K + Bk - 1) // Bk
    # 还是分块矩阵乘
    # 读出x和w_q的对应部分。根据group_size读出scale的对应部分。对w_q进行反量化
    acc = tl.zeros((Br, Bc), dtype=tl.float32)
    for i in range(steps):
        k_offsets = i * Bk + tl.arange(0, Bk)
        
        w_k_offsets = k_offsets // 2

        scales_k_offsets = k_offsets // group_size

        x_ptrs = x + (m_offsets * stride_xm)[:, None] + (k_offsets * stride_xk)[None, :]
        w_ptrs = w_q + (n_offsets * stride_wn)[:, None] + (w_k_offsets * stride_wk)[None, :]
        scales_ptrs = scales + (n_offsets * stride_sn)[:, None] + (scales_k_offsets * stride_sk)[None, :]

        x_mask = (m_offsets < M)[:, None] & (k_offsets < K)[None, :]
        w_mask = (n_offsets < N)[:, None] & (w_k_offsets < K // 2)[None, :] # uint8
        scales_mask = (n_offsets < N)[:, None] & (scales_k_offsets < K // group_size)[None, :]

        x_data = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_data = tl.load(w_ptrs, mask=w_mask, other=0)
        scales_data = tl.load(scales_ptrs, mask=scales_mask, other=0.0)

        # dequantization
        # 利用w_data和scales_data对权重进行反量化
        # 不知道怎么将w_data拆开，以及将scales_data按照group_size进行广播
        w_unpacked = tl.where((k_offsets % 2 == 0)[None, :], (w_data >> 4) & 0xf, w_data & 0xf)
        w_unpacked_fp16 = w_unpacked.to(tl.float16) - 8.0
        w_dequant = w_unpacked_fp16 * scales_data

        acc = tl.dot(x_data, tl.trans(w_dequant), acc)
    output_ptrs = y + (m_offsets * stride_ym)[:, None] + (n_offsets * stride_yn)[None, :]
    output_mask = (m_offsets < M)[:, None] & (n_offsets < N)[None, :]
    tl.store(output_ptrs, acc.to(tl.float16), mask=output_mask)
        

# x, w_q, scales, y are tensors on the GPU
def solve(
    x: torch.Tensor, # fp16
    w_q: torch.Tensor, # uint8
    scales: torch.Tensor, # fp16
    y: torch.Tensor, # fp16
    M: int,
    N: int,
    K: int,
    group_size: int,
):
    stride_xm, stride_xk = x.stride(0), x.stride(1)
    stride_wn, stride_wk = w_q.stride(0), w_q.stride(1)
    stride_sn, stride_sk = scales.stride(0), scales.stride(1)
    stride_ym, stride_yn = y.stride(0), y.stride(1)
    grid = lambda meta: (triton.cdiv(M, meta["Br"]), triton.cdiv(N, meta["Bc"]))
    w4a16[grid](
        x, w_q, scales, y, 
        M, N, K,
        group_size,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_sn, stride_sk,
        stride_ym, stride_yn
    )
