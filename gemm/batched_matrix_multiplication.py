import torch
import triton
import triton.language as tl

# 写完这个写下grouped matrix multiplication         
@triton.jit
def batched_matmul(a, b, c, M, N, K, 
                  Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    batch_id = tl.program_id(axis=0)
    row_id = tl.program_id(axis=1)
    col_id = tl.program_id(axis=2)

    batch_start_a = a + batch_id * M * K
    a_row_offset = row_id * Br + tl.arange(0, Br)

    batch_start_b = b + batch_id * K * N
    b_col_offset = col_id * Bc + tl.arange(0, Bc)

    steps = (K + Bk - 1) // Bk
    acc = tl.zeros([Br, Bc], dtype=tl.float32)
    for i in range(steps):
        a_col_offset = i * Bk + tl.arange(0, Bk)
        b_row_offset = i * Bk + tl.arange(0, Bk)

        a_mask = (a_row_offset[:, None] < M) & (a_col_offset[None, :] < K) 
        b_mask = (b_row_offset[:, None] < K) & (b_col_offset[None, :] < N)

        a_ptrs = batch_start_a + a_row_offset[:, None] * K + a_col_offset[None, :]
        b_ptrs = batch_start_b + b_row_offset[:, None] * N + b_col_offset[None, :]

        a_data = tl.load(a_ptrs, a_mask, other=0.0)
        b_data = tl.load(b_ptrs, b_mask, other=0.0)

        acc = tl.dot(a_data, b_data, acc, allow_tf32=False)

    c_ptrs = c + batch_id * M * N + a_row_offset[:, None] * N + b_col_offset[None, :]
    c_mask = (a_row_offset[:, None] < M) & (b_col_offset[None, :] < N)
    tl.store(c_ptrs, acc, c_mask)



# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    Br = 16
    Bc = 64
    Bk = 64
    BLOCK_SIZE = 128
    grid = (BATCH, triton.cdiv(M, Br), triton.cdiv(N, Bc))
    batched_matmul[grid](a, b, c, M, N, K, Br, Bc, Bk, BLOCK_SIZE)