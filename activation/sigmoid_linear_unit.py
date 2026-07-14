import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_data = tl.load(input + offsets, mask)
    input_data = input_data / (tl.exp(-1 * input_data) + 1.0)
    tl.store(output + offsets, input_data, mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    silu_kernel[grid](input, output, N, BLOCK_SIZE)
