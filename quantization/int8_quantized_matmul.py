import torch
import triton
import triton.language as tl


@triton.jit
def int8_quantized_matmul(a, b, c, M, N, K, scale_A, scale_B, scale_C, z_a, z_b, z_c,
                         Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    col_id = tl.program_id(axis=1)

    a_row_offsets = row_id * Br + tl.arange(0, Br)
    b_col_offsets = col_id * Bc + tl.arange(0, Bc)
    a_row_mask = a_row_offsets < M
    b_col_mask = b_col_offsets < N

    steps = (K + Bk - 1) // Bk
    acc = tl.zeros([Br, Bc], dtype=tl.int32)
    for i in range(steps):  
        a_col_offsets = i * Bk + tl.arange(0, Bk)
        b_row_offsets = i * Bk + tl.arange(0, Bk)

        a_col_mask = a_col_offsets < K
        b_row_mask = b_row_offsets < K

        a_ptrs = a + a_row_offsets[:, None] * K + a_col_offsets
        b_ptrs = b + b_row_offsets[:, None] * N + b_col_offsets

        a_mask = a_row_mask[:, None] & a_col_mask[None, :]
        b_mask = b_row_mask[:, None] & b_col_mask[None, :]

        a_data = tl.load(a_ptrs, a_mask, other=0.0)
        b_data = tl.load(b_ptrs, b_mask, other=0.0)

        a_data = (a_data - z_a).to(tl.int8)
        b_data = (b_data - z_b).to(tl.int8)
        acc = tl.dot(a_data, b_data, acc, out_dtype=tl.int32)
        
    
    scaled = acc.to(tl.float32) * scale_A * scale_B / scale_C
    rounded = tl.where()
    tmp = tl.clamp(tl.math.round(acc.to(tl.float32) * scale_A * scale_B / scale_C) + z_c, -128, 127)


    c_ptrs = c + (a_row_offsets * N)[:, None] + b_col_offsets[None, :]
    c_mask = a_row_mask[:, None] & b_col_mask[None, :]
    tl.store(c_ptrs, tmp.to(tl.int8), mask=c_mask)
    

# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_C: int,
):
    Br = 64
    Bc = 64
    Bk = 64
    BLOCK_SIZE = 256
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    int8_quantized_matmul[grid](a, b, c, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C,
                                Br, Bc, Bk, BLOCK_SIZE)
