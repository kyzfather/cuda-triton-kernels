import torch
import triton
import triton.language as tl

# v2是看了triton官方的示例后，凭记忆写的
# 假设batch和head都是1，不考虑mask
@triton.jit
def attention_v2(q, k, v, o, M, N, K, qk_scale, 
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_SIZE: tl.constexpr):  
    pid = tl.program_id(axis=0)
    offsets_k = tl.arange(0, BLOCK_K)
    col_mask = offsets_k < K
    q_ptrs = q + pid * BLOCK_M * K + (tl.arange(0, BLOCK_M) * K)[:, None] + offsets_k[None, :]
    q_row_mask = (pid * BLOCK_M + tl.arange(0, BLOCK_M)) < M
    q_data = tl.load(q_ptrs, q_row_mask[:, None] & col_mask[None, :], other=0.0)
    # steps = (M + BLOCK_M - 1) // BLOCK_M
    steps = (N + BLOCK_N - 1) // BLOCK_N
    m_old = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    d_old = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    for i in range(steps):
        # mi = max(mi-1, m_tile)
        # di = di-1 * exp(mi-1 - mi) + sum(exp(xj - mi))
        # oi = {oi-1 * di-1 * exp(mi-1 - mi) + sum[exp(xj - mi) * V]} / di   
        k_ptrs = k + i * BLOCK_N * K + (tl.arange(0, BLOCK_N) * K)[:, None] + offsets_k[None, :]
        k_row_mask = (i * BLOCK_N + tl.arange(0, BLOCK_N)) < N
        k_data = tl.load(k_ptrs, k_row_mask[:, None] & col_mask[None, :], other=0.0)

        v_ptrs = v + i * BLOCK_N * K + (tl.arange(0, BLOCK_N) * K)[:, None] + offsets_k[None, :]
        v_data = tl.load(v_ptrs, k_row_mask[:, None] & col_mask[None, :], other=0.0)

        qk = tl.dot(q_data, tl.trans(k_data))
        qk = tl.where(k_row_mask[None, :], qk * qk_scale, -float("inf"))
        m_tile = tl.max(qk, axis=1)
        m_new = tl.maximum(m_old, m_tile)
        alpha = tl.exp(m_old - m_new)
        qk = tl.exp(qk - m_new[:, None])
        # qk = qk * qk_scale - m_new[:, None]
        sum_val = tl.sum(qk, axis=1)
        d_new = d_old * alpha + sum_val
        acc *= alpha[:, None]
        acc = tl.dot(qk, v_data, acc)

        m_old = m_new
        d_old = d_new
    acc = acc / d_old[:, None]    
    o_ptrs = o + pid * BLOCK_M * K + (tl.arange(0, BLOCK_M) * K)[:, None] + offsets_k[None, :]
    tl.store(o_ptrs, acc.to(tl.float16), mask=(q_row_mask[:, None] & col_mask[None, :]))


# 带mask的版本（还不是变长的版本）
# 下三角，当前的seq不能看见后面的seq
@triton.jit
def attention_mask(q, k, v, o, M, N, K, qk_scale, 
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_SIZE: tl.constexpr):  
    pid = tl.program_id(axis=0)
    offsets_k = tl.arange(0, BLOCK_K)
    col_mask = offsets_k < K
    q_ptrs = q + pid * BLOCK_M * K + (tl.arange(0, BLOCK_M) * K)[:, None] + offsets_k[None, :]
    q_row_mask = (pid * BLOCK_M + tl.arange(0, BLOCK_M)) < M
    q_data = tl.load(q_ptrs, q_row_mask[:, None] & col_mask[None, :], other=0.0)
    # steps = (M + BLOCK_M - 1) // BLOCK_M
    steps = (N + BLOCK_N - 1) // BLOCK_N
    m_old = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    d_old = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    offsets_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    for i in range(steps):
        softmax_mask = offsets_m[:, None] >= (i * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        # mi = max(mi-1, m_tile)
        # di = di-1 * exp(mi-1 - mi) + sum(exp(xj - mi))
        # oi = {oi-1 * di-1 * exp(mi-1 - mi) + sum[exp(xj - mi) * V]} / di   
        k_ptrs = k + i * BLOCK_N * K + (tl.arange(0, BLOCK_N) * K)[:, None] + offsets_k[None, :]
        k_row_mask = (i * BLOCK_N + tl.arange(0, BLOCK_N)) < N
        k_data = tl.load(k_ptrs, k_row_mask[:, None] & col_mask[None, :], other=0.0)

        v_ptrs = v + i * BLOCK_N * K + (tl.arange(0, BLOCK_N) * K)[:, None] + offsets_k[None, :]
        v_data = tl.load(v_ptrs, k_row_mask[:, None] & col_mask[None, :], other=0.0)

        qk = tl.dot(q_data, tl.trans(k_data))
        qk = tl.where(softmax_mask, qk * qk_scale, -float("inf"))  # attention mask
        qk = tl.where(k_row_mask[None, :], qk, -float("inf")) 
        m_tile = tl.max(qk, axis=1)
        m_new = tl.maximum(m_old, m_tile)
        alpha = tl.exp(m_old - m_new)
        qk = tl.exp(qk - m_new[:, None])
        # qk = qk * qk_scale - m_new[:, None]
        sum_val = tl.sum(qk, axis=1)
        d_new = d_old * alpha + sum_val
        acc *= alpha[:, None]
        acc = tl.dot(qk, v_data, acc)

        m_old = m_new
        d_old = d_new
    acc = acc / d_old[:, None]    
    o_ptrs = o + pid * BLOCK_M * K + (tl.arange(0, BLOCK_M) * K)[:, None] + offsets_k[None, :]
    tl.store(o_ptrs, acc, mask=(q_row_mask[:, None] & col_mask[None, :]))    


def solve(q: torch.tensor, k: torch.tensor, v: torch.tensor,
          o: torch.tensor, M: int, N: int, K: int):
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = max(16, triton.next_power_of_2(K))
    BLOCK_SIZE = 256
    grid = (triton.cdiv(M, BLOCK_M), )
    qk_scale = K ** -0.5
    attention_v2[grid](q, k, v, o, M, N, K, qk_scale, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_SIZE)

if __name__ == "__main__":
    M, N, K = 4096, 4096, 1288
 
    device = "cuda"
    dtype = torch.float16
 
    q = torch.randn(M, K, device=device, dtype=dtype)
    k = torch.randn(N, K, device=device, dtype=dtype)
    v = torch.randn(N, K, device=device, dtype=dtype)
    o = torch.empty(M, K, device=device, dtype=dtype)
 
    solve(q, k, v, o, M, N, K)
    torch.cuda.synchronize()    