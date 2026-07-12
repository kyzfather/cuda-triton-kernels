import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_data = tl.load(input + offsets, mask)
    input_data = tl.where(input_data > 0, input_data, 0.01 * input_data)
    tl.store(output + offsets, input_data, mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](input, output, N, BLOCK_SIZE)
