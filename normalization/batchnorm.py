import torch
import triton
import triton.language as tl

# 将之前的layernorm的行读取，改成列读取?
# 每个block负责一列，就是一个channel，非合并内存访问，性能比较差
# batchnorm的难度比layernorm的难度大
@triton.jit 
def batch_norm_naive(input, gamma, beta, output, N, C, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    input_ptrs = input + pid + tl.arange(0, BLOCK_SIZE) * C
    mask = tl.arange(0, BLOCK_SIZE) < N

    input_data = tl.load(input_ptrs, mask, other=0.0)

    mean = tl.sum(input_data, axis=0) / N
    x_mu = tl.where(mask, input_data - mean, 0.0)
    var = tl.sum(x_mu * x_mu, axis=0) / N
    rsqrt_val = tl.math.rsqrt(var + eps)
    gamma_val = tl.load(gamma + pid)
    beta_val = tl.load(beta + pid)
    output_data = gamma_val * x_mu * rsqrt_val + beta_val

    output_ptrs = output + pid + tl.arange(0, BLOCK_SIZE) * C
    tl.store(output_ptrs, output_data, mask)
    
# 每个block负责多个channel, 沿着batch维度使用welford递推   
# 单个元素的递推
# m_n = m_n-1 + (x_n - m_n-1) / n
# s_n = s_n-1 + (x_n - m_n-1)(x_n - m_n)

# 分块的递推
# m_x = m_a + (m_b - m_a) * n_b / (n_a + n_b)
# s_x = s_a + s_b + (m_b - m_a)^2 * n_a * n_b  / (n_a + n_b)
@triton.jit
def batch_norm_welford(input_ptr, gamma_ptr, beta_ptr, output_ptr, N, C, eps,
               Br: tl.constexpr, Bc: tl.constexpr):
    pid = tl.program_id(axis=0)
    steps = (N + Br - 1) // Br
    col_offsets = Bc * pid + tl.arange(0, Bc)

    count_old = 0
    mean_old = tl.zeros([Bc], dtype=tl.float32)
    s_old = tl.zeros([Bc], dtype=tl.float32)
    for i in range(steps):
        row_offsets = i * Br + tl.arange(0, Br)
        input_ptrs = input_ptr + row_offsets[:, None] * C + col_offsets[None, :]
        mask = (row_offsets < N)[:, None] & (col_offsets < C)[None, :]
        input_data = tl.load(input_ptrs, mask=mask, other=0.0) 

        count = tl.minimum(Br, N - i * Br) # 计算当前分块真实的元素数量
        mean = tl.sum(input_data, axis=0) / count
        x_mu = tl.where(mask, input_data - mean[None, :], 0.0)
        s = tl.sum(x_mu * x_mu, axis=0)
        delta = mean - mean_old
        total = count_old + count
        mean = mean_old + delta * count / total
        s = s_old + s + delta * delta * count_old * count / total

        mean_old = mean
        count_old = total
        s_old = s
    
    rsqrt_var = tl.math.rsqrt(s_old / count_old + eps)
    gamma_ptrs = gamma_ptr + pid * Bc + tl.arange(0, Bc)
    beta_ptrs = beta_ptr + pid * Bc + tl.arange(0, Bc)
    col_mask = col_offsets < C
    gamma = tl.load(gamma_ptrs, mask=col_mask, other=0.0)
    beta = tl.load(beta_ptrs, mask=col_mask, other=0.0)

    for i in range(steps):
        row_offsets = i * Br + tl.arange(0, Br)
        input_ptrs = input_ptr + row_offsets[:, None] * C + col_offsets[None, :]
        mask = (row_offsets < N)[:, None] & (col_offsets < C)[None, :]
        input_data = tl.load(input_ptrs, mask=mask, other=0.0)  

        x_mu = tl.where(mask, input_data - mean_old[None, :], 0.0)
        res = gamma[None, :] * x_mu * rsqrt_var[None, :] + beta[None, :]

        output_ptrs = output_ptr + row_offsets[:, None] * C + col_offsets[None, :]
        mask = (row_offsets < N)[:, None] & (col_offsets < C)[None, :]
        tl.store(output_ptrs, res, mask=mask)



# input, gamma, beta, output are tensors on the GPU
def solve_naive(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (C, )
    batch_norm_naive[grid](input, gamma, beta, output, N, C, eps, BLOCK_SIZE)


def solve_welford(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    Br = 32
    Bc = 128
    grid = (triton.cdiv(C, 128), )
    batch_norm_welford[grid](input, gamma, beta, output, N, C, eps, Br, Bc)

if __name__ == "__main__":
    # 大模型常见的中等偏大尺寸 (Batch * SeqLen = N, HiddenSize = C)
    N, C = 4096, 4096
    eps = 1e-5
    device = "cuda"

    # 初始化测试数据 (使用 float32 确保精度比对严格)
    torch.manual_seed(0)
    X = torch.randn(N, C, device=device, dtype=torch.float32)
    gamma = torch.randn(C, device=device, dtype=torch.float32)
    beta = torch.randn(C, device=device, dtype=torch.float32)

    out_naive = torch.zeros_like(X)
    out_welford = torch.zeros_like(X)

    # 验证正确性：对比 PyTorch 官方原生实现
    # 注意：PyTorch 的 BatchNorm1d 默认期望输入是 [N, C]
    bn_torch = torch.nn.BatchNorm1d(C, eps=eps, affine=True, track_running_stats=False).to(device)
    bn_torch.weight.data.copy_(gamma)
    bn_torch.bias.data.copy_(beta)
    ref_out = bn_torch(X)

    # 运行各自的算子
    solve_naive(X, gamma, beta, out_naive, N, C, eps)
    solve_welford(X, gamma, beta, out_welford, N, C, eps)

    # 正确性断言
    assert torch.allclose(out_naive, ref_out, atol=1e-4), "Naive 实现结果错误！"
    assert torch.allclose(out_welford, ref_out, atol=1e-4), "Welford 实现结果错误！"
    print("🎉 正确性验证通过！开始准备 NCU 性能捕获...")

    # 为了能让 NCU 稳定抓取，我们在 GPU 上循环执行，排除单次冷启动的干扰
    # 你可以在 NCU 命令行中过滤对应的 kernel 名字
    for _ in range(10):
        solve_naive(X, gamma, beta, out_naive, N, C, eps)
        solve_welford(X, gamma, beta, out_welford, N, C, eps)