import torch
import triton
import triton.language as tl


@triton.jit
def swiglu(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets1 = offsets + N // 2
    mask = offsets < (N // 2)
    input_data = tl.load(input + offsets, mask)
    input_data1 = tl.load(input + offsets1, mask)
    output_data = input_data / (tl.exp(-input_data) + 1) * input_data1
    tl.store(output + offsets, output_data, mask)
# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    swiglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
