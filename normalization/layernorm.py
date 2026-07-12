import torch
import triton
import triton.language as tl


# triton的实现，寄存器抗input, 只需要一次load，不需要welford
# 但是如果不用寄存器抗的话，那么求均值load一次，计算方差又load一次
@triton.jit
def layernorm(input, gamma, beta, output, eps, N, BLOCK_SIZE: tl.constexpr):
    input_ptrs = input + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N
    input_data = tl.load(input_ptrs, mask, other=0.0)

    mean = tl.sum(input_data, axis=0) / N
    x_mu = tl.where(mask, input_data - mean, 0.0)
    var = tl.sum(x_mu * x_mu, axis=0) / N
    rsqrt_var = tl.math.rsqrt(var + eps)

    res = gamma * x_mu * rsqrt_var + beta

    output_ptrs = output + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptrs, res, mask=mask)


def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    BLOCK_SIZE = triton.next_power_of_2(N)
    layernorm[(1, )](input, gamma, beta, output, eps, N, BLOCK_SIZE)
