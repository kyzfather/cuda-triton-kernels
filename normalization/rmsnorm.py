import torch
import triton
import triton.language as tl

# 1维的 需要reduce sum求和
# 如果是每个block计算一部分的话，那是需要写到global memory里的
# 用一个block：
# 用多个block: 好像没法写，多个block没法同步
@triton.jit
def rmsnorm(input, gamma, beta, output, N, eps, BLOCK_SIZE: tl.constexpr):
    steps = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # triton写需要两趟啊？第一趟求平方和，第二趟逐元素计算
    sum = 0.0 # 对应每个线程的寄存器
    # 因为每一行可能很大，比如4096，还可能更大，如果不分steps，那么寄存器是不够的，寄存器压力比较大
    for i in range(steps):
        input_ptrs = input + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < N
        input_data = tl.load(input_ptrs, mask, other=0.0)
        sum += tl.sum(input_data * input_data, axis=0)
    rms = tl.math.sqrt(sum / N + eps)
    for i in range(steps):
        input_ptrs = input + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < N
        input_data = tl.load(input_ptrs, mask, other=0.0)
        output_data = gamma * input_data / rms + beta
        output_ptrs = output + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(output_ptrs, output_data, mask)

# 有个点一直没弄清，BLOCK_SIZE并不是triton设置的block的线程数，而是每次加载的数据量，
# 注意，sm的寄存器数量一般是65536个32位寄存器， 64kb * 4 = 256kb
# 所以triton的block size不大于8096， 
@triton.jit
def rmsnorm_1(input, gamma, beta, output, N, eps, BLOCK_SIZE: tl.constexpr):
    input_ptrs = input + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N
    input_data = tl.load(input_ptrs, mask=mask, other=0.0)
    sum = tl.sum(input_data * input_data, axis=0)
    rms = tl.math.sqrt(sum / N + eps)
    output_data = gamma * input_data / rms + beta
    output_ptrs = output + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptrs, output_data, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    BLOCK_SIZE = triton.next_power_of_2(N)
    rmsnorm[(1, )](input, gamma, beta, output, N, eps, BLOCK_SIZE)
