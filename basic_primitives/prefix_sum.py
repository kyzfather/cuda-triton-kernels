import torch
import triton
import triton.language as tl

# 感觉确实claude在处理代码方面能力更强
# 这里的写法基本就是将之前的cuda的写法写成triton的写法，用tl.cumsum来替代自己实现

@triton.jit
def scan(data_ptr, block_sums_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data_ptrs = data_ptr + offsets
    mask = offsets < N
    data = tl.load(data_ptrs, mask=mask, other=0.0)

    prefix_sum = tl.cumsum(data, axis=0)
    sum = tl.sum(data, axis=0)

    tl.store(data_ptrs, prefix_sum, mask=mask)
    tl.store(block_sums_ptr + pid, sum)

@triton.jit
def add(data_ptr, block_sums_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        return
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data_ptrs = data_ptr + offsets
    mask = offsets < N
    data = tl.load(data_ptrs, mask=mask, other=0.0)
    block_sum = tl.load(block_sums_ptr + pid - 1)
    data += block_sum
    tl.store(data_ptrs, data, mask=mask)

def prefix_sum(data: torch.Tensor, n: int):
    BLOCK_SIZE = 512
    GRID_SIZE = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    block_sums = torch.empty((GRID_SIZE,), dtype=data.dtype, device=data.device)
    if GRID_SIZE == 1:
        scan[(GRID_SIZE,)](data, block_sums, n, BLOCK_SIZE=BLOCK_SIZE)
        return
    
    scan[(GRID_SIZE,)](data, block_sums, n, BLOCK_SIZE=BLOCK_SIZE)
    prefix_sum(block_sums, GRID_SIZE) # 递归，需要求出block_sums的prefix sum
    add[(GRID_SIZE,)](data, block_sums, n, BLOCK_SIZE=BLOCK_SIZE)


# data and output are tensors on the GPU
def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    output.copy_(data)
    prefix_sum(output, n)
