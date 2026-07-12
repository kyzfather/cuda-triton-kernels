import torch
import triton
import triton.language as tl

@triton.jit
def split_k_matmul(
    a, b, c,
    M, N, K,
    Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr,
    num_splits: tl.constexpr
):
    m_id = tl.program_id(axis=0)
    n_id = tl.program_id(axis=1)
    k_id = tl.program_id(axis=2)
    k_per_split = K // num_splits

    m_offsets = m_id * Br + tl.arange(0, Br)
    n_offsets = n_id * Bc + tl.arange(0, Bc)
    k_start = k_id * k_per_split
    k_end = k_start + k_per_split
    c_ptrs = c + (m_offsets * N)[:, None] + n_offsets[None, :]
    c_mask = (m_offsets < M)[:, None] & (n_offsets < N)[None, :]

    steps = (k_per_split + Bk - 1) // Bk
    acc = tl.zeros((Br, Bc), dtype=tl.float32)
    for i in range(steps):
        k_offsets = k_start + i * Bk + tl.arange(0, Bk)
        a_ptrs = a + (m_offsets * K)[:, None] + k_offsets[None, :]
        b_ptrs = b + (k_offsets * N)[:, None] + n_offsets[None, :]

        a_mask = (m_offsets < M)[:, None] & (k_offsets < k_end)[None, :]
        b_mask = (k_offsets < k_end)[:, None] & (n_offsets < N)[None, :]

        a_data = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_data = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a_data, b_data, acc)

    tl.atomic_add(c_ptrs, acc, mask=c_mask)


def solve(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, 
    M: int, N: int, K: int
):
    Br = 128
    Bc = 128
    Bk = 32
    num_splits = 8
    grid = (triton.cdiv(M, Br), triton.cdiv(N, Bc), num_splits)
    split_k_matmul[grid](a, b, c, M, N, K, Br, Bc, Bk, num_splits)