import torch
import triton
import triton.language as tl


# 之前写法有问题，这个能通过。
@triton.jit
def topk_kernel(
    x_ptr, y_ptr,
    n_elements, out_elements, k,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    sorted_x = tl.sort(x, descending=True)

    out_offsets = pid * k + tl.arange(0, BLOCK_SIZE) # 最后一个block可能大于这里y_ptr的边界啊
    store_mask = (tl.arange(0, BLOCK_SIZE) < k) & (out_offsets < out_elements)
    tl.store(y_ptr + out_offsets, sorted_x, mask=store_mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    BLOCK_SIZE = 2048
    current_N = N
    current_input = input

    while True:
        grid_size = triton.cdiv(current_N, BLOCK_SIZE)
        if grid_size == 1:
            topk_kernel[(grid_size, )](
                current_input, output,
                current_N, k, k, 
                BLOCK_SIZE
            )
            break
        else:
            out_elements = grid_size * k
            next_input = torch.empty(out_elements, dtype=input.dtype, device=input.device)
            topk_kernel[(grid_size,)](
                current_input, next_input, 
                current_N, out_elements, k,
                BLOCK_SIZE
            )

        current_input = next_input
        current_N = out_elements

