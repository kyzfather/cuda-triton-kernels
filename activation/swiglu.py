import torch
import triton
import triton.language as tl
import argparse
import time

# 实现1：两阶段，启动两个kernel，tensorcore
@triton.jit 
def swiglu0(
    x, W_gate, W_up, 
    output, M, d_model, d_ffn,
    Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr
):
    # 两次矩阵乘，一次求sigmoid，逐元素乘，矩阵乘
    # 关键是怎么fuse，感觉不是很好fuse
    # 感觉down之前的可以fuse下，总共启动两个kernel?
    row_id = tl.program_id(axis=0)
    col_id = tl.program_id(axis=1)
    
    m_offsets = row_id * Br + tl.arange(0, Br)
    n_offsets = col_id * Bc + tl.arange(0, Bc)
    mask_m = m_offsets < M
    mask_n = n_offsets < d_ffn    
    
    steps = (d_model + Bk - 1) // Bk
    acc_gate = tl.zeros((Br, Bc), dtype=tl.float32)
    acc_up = tl.zeros((Br, Bc), dtype=tl.float32)
    for i in range(steps):
        k_offsets = i * Bk + tl.arange(0, Bk)

        x_ptrs = x + (m_offsets * d_model)[:, None] + k_offsets[None, :]
        W_gate_ptrs = W_gate + (k_offsets * d_ffn)[:, None] + n_offsets[None, :]
        W_up_ptrs = W_up + (k_offsets * d_ffn)[:, None] + n_offsets[None, :]


        mask_k = k_offsets < d_model
        mask0 = mask_m[:, None] & mask_k[None, :]
        mask1 = mask_k[:, None] & mask_n[None, :]

        x_data = tl.load(x_ptrs, mask=mask0, other=0.0)
        W_gate_data = tl.load(W_gate_ptrs, mask=mask1, other=0.0)
        W_up_data = tl.load(W_up_ptrs, mask=mask1, other=0.0)

        acc_gate = tl.dot(x_data, W_gate_data, acc_gate)
        acc_up = tl.dot(x_data, W_up_data, acc_up)

    acc_gate = acc_gate * tl.sigmoid(acc_gate)
    res = acc_gate * acc_up
    output_ptrs = output + (m_offsets * d_ffn)[:, None] + n_offsets[None, :] 
    mask_output = mask_m[:, None] & mask_n[None, :]
    tl.store(output_ptrs, res, mask=mask_output)

@triton.jit
def gemm(
    input, weight, output,
    M, N, K,
    Br: tl.constexpr, Bc: tl.constexpr, Bk: tl.constexpr
):
    row_id = tl.program_id(axis=0)
    col_id = tl.program_id(axis=1)
    
    steps = (K + Bk - 1) // Bk
    m_offsets = row_id * Br + tl.arange(0, Br)
    n_offsets = col_id * Bc + tl.arange(0, Bc)
    
    acc = tl.zeros((Br, Bc), dtype=tl.float32)
    for i in range(steps):
        k_offsets = i * Bk + tl.arange(0, Bk)

        input_ptrs = input + (m_offsets * K)[:, None] + k_offsets[None, :]
        weight_ptrs = weight + (k_offsets * N)[:, None] + n_offsets[None, :]
        
        mask0 = (m_offsets < M)[:, None] & (k_offsets < K)[None, :]
        mask1 = (k_offsets < K)[:, None] & (n_offsets < N)[None, :]

        input_data = tl.load(input_ptrs, mask=mask0, other=0.0)
        weight_data = tl.load(weight_ptrs, mask=mask1, other=0.0)

        acc = tl.dot(input_data, weight_data, acc)

    output_ptrs = output + (m_offsets * N)[:, None] + n_offsets[None, :]
    output_mask = (m_offsets < M)[:, None] & (n_offsets < N)[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)


@triton.jit
def swiglu_activation_kernel(
    input, output,
    M, d_ffn, BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(axis=0)
    col_id = tl.program_id(axis=1)
    input_gate_ptrs = input + row_id * d_ffn * 2 + col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_up_ptrs = input + row_id * d_ffn * 2 + d_ffn + col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < d_ffn

    input_gate_data = tl.load(input_gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    input_up_data = tl.load(input_up_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    res = input_gate_data * tl.sigmoid(input_gate_data) * input_up_data
    output_ptrs = output + row_id * d_ffn + col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptrs, res, mask=mask)


    # steps = (d_ffn + BLOCK_SIZE - 1) // BLOCK_SIZE
    # for i in range(steps):
    #     input_gate_ptrs = input_gate_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    #     input_up_ptrs = input_up_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
    #     mask = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < d_ffn

    #     input_gate_data = tl.load(input_gate_ptrs, mask=mask, other=0.0)
    #     input_up_data = tl.load(input_up_ptrs, mask=mask, other=0.0)
        
    #     res = input_gate_data * tl.sigmoid(input_gate_data) * input_up_data
    #     output_ptrs = output + row_id * d_ffn + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    #     tl.store(output_ptrs, res, mask=mask)


 
        
# 两个kernel: (gemm + activation) -> gemm
def solve0(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    d_ffn: int,
):
    Br = 128
    Bc = 128
    Bk = 32
    grid0 = (triton.cdiv(M, Br), triton.cdiv(d_ffn, Bc))
    grid1 = (triton.cdiv(M, Br), triton.cdiv(d_model, Bc))
    tmp = torch.empty((M, d_ffn), dtype=x.dtype, device=x.device)
    swiglu0[grid0](x, W_gate, W_up, tmp, M, d_model, d_ffn, Br, Bc, Bk)
    gemm[grid1](tmp, W_down, output, M, d_model, d_ffn, Br, Bc, Bk)

# 三个kernel：gemm -> activation -> gemm
def solve1(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    d_ffn: int,
):
    W_gate_up_cat = torch.cat([W_gate, W_up], dim=1) # gate与up的权重拼接一起进行矩阵乘运算
    tmp0 = torch.zeros([M, 2 * d_ffn], dtype=x.dtype, device=x.device)
    tmp1 = torch.zeros([M, d_ffn], dtype=x.dtype, device=x.device)
    Br = 128
    Bc = 128
    Bk = 32
    BLOCK_SIZE = 512

    grid0 = (triton.cdiv(M, Br), triton.cdiv(2 * d_ffn, Bc))
    gemm[grid0](x, W_gate_up_cat, tmp0, M, 2 * d_ffn, d_model, Br, Bc, Bk)

    grid1 = (M, triton.cdiv(d_ffn, BLOCK_SIZE))
    swiglu_activation_kernel[grid1](tmp0, tmp1, M, d_ffn, BLOCK_SIZE)

    grid2 = (triton.cdiv(M, Br), triton.cdiv(d_model, Bc))
    gemm[grid2](tmp1, W_down, output, M, d_model, d_ffn, Br, Bc, Bk)

def benchmark(name, func, x, W_gate, W_up, W_down, out,
              M, d_model, d_ffn,
              warmup=20, repeat=100):

    # warmup
    for _ in range(warmup):
        func(x, W_gate, W_up, W_down, out, M, d_model, d_ffn)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        func(x, W_gate, W_up, W_down, out, M, d_model, d_ffn)
    end.record()

    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / repeat
    print(f"{name:<8} Average Latency: {ms:.3f} ms")

    return ms    


# 时间分析：python swiglu.py
# ncu分析：ncu --set full --profile-from-start off -o report_solve0 python swiglu.py --mode solve0 --ncu
if __name__ == "__main__":
    import argparse
    import time
    import torch.nn.functional as F

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["solve0", "solve1"], required=True)
    parser.add_argument("--ncu", action="store_true", help="NCU分析模式：仅运行一次")
    args = parser.parse_args()

    # 1. 设置维度与设备
    M, d_model, d_ffn = 512, 4096, 14336
    device = "cuda"
    dtype = torch.float32
    
    torch.manual_seed(0)
    x = torch.randn(M, d_model, device=device, dtype=dtype)
    W_gate = torch.randn(d_model, d_ffn, device=device, dtype=dtype)
    W_up = torch.randn(d_model, d_ffn, device=device, dtype=dtype)
    W_down = torch.randn(d_ffn, d_model, device=device, dtype=dtype)
    out = torch.empty((M, d_model), device=device, dtype=dtype)

    # 2. 根据模式执行
    if args.ncu:
        # --- NCU模式：禁止循环，只做单次调用，用于 Profiling ---
        func = solve0 if args.mode == "solve0" else solve1

        # warm up
        for i in range(20):
            func(x, W_gate, W_up, W_down, out, M, d_model, d_ffn)

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

        func(x, W_gate, W_up, W_down, out, M, d_model, d_ffn)   

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        # --- Benchmark模式：包含预热与循环，用于测总耗时 ---
        print("=" * 60)
        print("Benchmark")
        print("=" * 60)

        out0 = torch.empty((M, d_model), device=device, dtype=dtype)
        out1 = torch.empty((M, d_model), device=device, dtype=dtype)

        ms0 = benchmark(
            "solve0", 
            solve0,
            x,
            W_gate,
            W_up,
            W_down,
            out0,
            M,
            d_model,
            d_ffn,
        )

        ms1 = benchmark(
            "solve1",
            solve1,
            x,
            W_gate,
            W_up,
            W_down,
            out1,
            M,
            d_model,
            d_ffn,
        )