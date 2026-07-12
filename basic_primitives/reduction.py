import torch
import triton
import triton.language as tl

@triton.jit
def reduction(input, output, N, BLOCK_SIZE: tl.constexpr):
    # 感觉triton的reduce是不是比较好写啊
    # cuda的reduce是block内先在warp内reduce，然后再不同的warp进行reduce
    # triton直接一个sum函数就reduce了

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    input_data = tl.load(input + offsets, mask=mask, other=0.0)
    sum = tl.sum(input_data, axis=0)
    tl.store(output + pid, sum)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    block_num = triton.cdiv(N, BLOCK_SIZE)
    tmp = torch.empty(block_num, dtype=torch.float32, device=input.device)
    grid = (block_num, )
    reduction[grid](input, tmp, N, BLOCK_SIZE)
    output[0] = torch.sum(tmp)

