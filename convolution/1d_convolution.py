import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(input, kernel, output, input_size, 
                  kernel_size, padding_kernel_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets_m = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_n = tl.arange(0, padding_kernel_size)
    input_ptrs = input + offsets_m[:, None] + offsets_n[None, :]
    mask = (offsets_m < input_size - kernel_size + 1)[:, None] & (offsets_n < kernel_size)[None, :]
    input_data = tl.load(input_ptrs, mask, other=0.0)
    kernel_ptrs = kernel + tl.arange(0, padding_kernel_size)
    kernel_mask = tl.arange(0, padding_kernel_size) < kernel_size
    kernel_data = tl.load(kernel_ptrs, kernel_mask, other=0.0)
    input_data = input_data * kernel_data[None, :]
    res = tl.sum(input_data, axis=1)
    output_ptrs = output + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < input_size - kernel_size + 1
    tl.store(output_ptrs, res, mask=output_mask)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)
    padding_kernel_size = triton.next_power_of_2(kernel_size)

    conv1d_kernel[grid](input, kernel, output, input_size, kernel_size, padding_kernel_size, BLOCK_SIZE)
