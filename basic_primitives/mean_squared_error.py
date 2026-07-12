import torch
import triton
import triton.language as tl

# 为什么这种题也能算medium, 有些easy的题目比这还难
# 这里的N很大，block规约后在host上进行规约?
@triton.jit
def mean_squared_error(predictions, targets, output_array, process_count, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE * process_count
    sum = 0.0
    for i in range(process_count):
        offsets = start + tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offsets < N
        predictions_ptrs = predictions + offsets
        predictions_data = tl.load(predictions_ptrs, mask, other=0.0)
        targets_ptrs = targets + offsets
        targets_data = tl.load(targets_ptrs, mask, other=0.0)
        sub = predictions_data - targets_data
        tmp = tl.sum(sub * sub, axis=0)
        sum += tmp
    tl.store(output_array + pid, sum)

# predictions, targets, mse are tensors on the GPU
def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    PROCESS_COUNT = 8
    BLOCK_NUMS = triton.cdiv(N, BLOCK_SIZE * PROCESS_COUNT)
    grid = (BLOCK_NUMS, )
    output_array = torch.zeros([BLOCK_NUMS], dtype=torch.float32, device=predictions.device)
    mean_squared_error[grid](predictions, targets, output_array, PROCESS_COUNT, N, BLOCK_SIZE)
    mse[0] = torch.sum(output_array) / N
