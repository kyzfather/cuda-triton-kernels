import torch
import triton
import triton.language as tl

# N非常大 每个block自己计算一部分 写到gmem里 最后由一个block reduce sum
# 这里triton有点不知道怎么写，因为不能操作线程。每个线程搞一个fp32的acc 然后线程累加自己的
# 然后block内累加
@triton.jit 
def fp16_dot_product(A, B, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    A_ptrs = A + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    B_ptrs = B + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < N
    A_data = tl.load(A_ptrs, mask=mask, other=0.0)
    B_data = tl.load(B_ptrs, mask=mask, other=0.0)
    A_data = A_data.to(tl.float32)
    B_data = B_data.to(tl.float32)
    res = tl.sum(A_data * B_data, axis=0)
    output_ptrs = output + pid
    tl.store(output_ptrs, res)

@triton.jit
def reduce_sum(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_ptrs = input + offsets
    mask = offsets < N
    input_data = tl.load(input_ptrs, mask=mask, other=0.0)
    sum = tl.sum(input_data, axis=0)
    output_ptrs = output + pid
    tl.store(output_ptrs, sum)


@triton.jit
def fp16_dot_product_optimized(A, B, N, res, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    stride = num_jobs * BLOCK_SIZE
    start = pid * BLOCK_SIZE
    acc = 0.0
    for i in range(start, N, stride):
        A_ptrs = A + i + tl.arange(0, BLOCK_SIZE)
        B_ptrs = B + i + tl.arange(0, BLOCK_SIZE)

        mask = (i + tl.arange(0, BLOCK_SIZE)) < N
        A_data = tl.load(A_ptrs, mask=mask, other=0.0)
        B_data = tl.load(B_ptrs, mask=mask, other=0.0)
        A_data = A_data.to(tl.float32)
        B_data = B_data.to(tl.float32)
        acc += tl.sum(A_data * B_data, axis=0)
    tl.atomic_add(res + tl.arange(0, 1), acc)

# A, B, result are tensors on the GPU
def solve0(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    BLOCK_SIZE = 4096
    current_N = N
    block_nums = triton.cdiv(current_N, BLOCK_SIZE)
    tmp = torch.zeros([block_nums], dtype=torch.float32, device=A.device)
    fp16_dot_product[(block_nums, )](A, B, tmp, N, BLOCK_SIZE)
    current_N = block_nums
    current_input = tmp
    while current_N > 1:
        block_nums = triton.cdiv(current_N, BLOCK_SIZE)
        grid = (block_nums, )
        output = torch.zeros([block_nums], dtype=torch.float32, device=A.device)
        reduce_sum[grid](current_input, output, current_N, BLOCK_SIZE)
        current_input = output
        current_N = block_nums
    result[0] = current_input[0].to(torch.float16)


def solve1(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    BLOCK_SIZE = 8192
    block_nums = min(256, triton.cdiv(N, 8192))
    tmp = torch.zeros([1], dtype=torch.float32, device=A.device)
    fp16_dot_product_optimized[(block_nums, )](A, B, N, tmp, BLOCK_SIZE)
    result[0] = tmp[0].to(torch.float16)

def benchmark(name, func, A, B, result, N,
              warmup=20, repeat=100):

    # warmup
    for _ in range(warmup):
        func(A, B, result, N)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        func(A, B, result, N)
    end.record()

    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / repeat
    print(f"{name:<8} Average Latency: {ms:.3f} ms")

    return ms    

# 时间分析：python xxx.py
# ncu分析：ncu --set full --profile-from-start off -o report_solve0 python xxx.py --mode solve0 --ncu
if __name__ == "__main__":
    import argparse
    import time
    import torch.nn.functional as F

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["solve0", "solve1"])
    parser.add_argument("--ncu", action="store_true", help="NCU分析模式：仅运行一次")
    args = parser.parse_args()

    # 1. 设置维度与设备
    N = 100000000
    device = "cuda"
    dtype = torch.float16
    
    torch.manual_seed(0)
    A = torch.randn(N, device=device, dtype=dtype)
    B = torch.randn(N, device=device, dtype=dtype)
    res = torch.empty((1), device=device, dtype=dtype)

    # 2. 根据模式执行
    if args.ncu:
        # --- NCU模式：禁止循环，只做单次调用，用于 Profiling ---
        func = solve0 if args.mode == "solve0" else solve1

        # warm up
        for i in range(20):
            func(A, B, res, N)

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

        func(A, B, res, N)   

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        # --- Benchmark模式：包含预热与循环，用于测总耗时 ---
        print("=" * 60)
        print("Benchmark")
        print("=" * 60)

        out0 = torch.empty((1), device=device, dtype=dtype)
        out1 = torch.empty((1), device=device, dtype=dtype)

        ms0 = benchmark("solve0", solve0, A, B, res, N)

        ms1 = benchmark("solve1", solve1, A, B, res, N)    