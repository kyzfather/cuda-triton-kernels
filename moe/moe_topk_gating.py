import torch
import triton
import triton.language as tl

@triton.jit
def moe_topk_gating_kernel(
    logits_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    M, E, k: tl.constexpr,
    E_PADDED: tl.constexpr,
    K_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # 自己写一遍吧，gemini pro还是可以的，问gemini flash和chatgpt都是好多遍都给不出正确的答案
    # topk selection还有这个moe topk gating都是一两遍就可以问出正确的答案

    # 总结：topk不要用sort，sort返回不了下标
    # 循环，使用argmax返回下标，使用max返回最大值。然后就是一般不要通过索引的方式访问寄存器
    # 可以通过tl.where加掩码的方式，来下标索引寄存器，但是这里没有用寄存器。而是用的gmem
    row_id = tl.program_id(axis=0)
    logits_ptrs = logits_ptr + row_id * E + tl.arange(0, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < E
    logits_vals = tl.load(logits_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(logits_vals, axis=0)

    # 循环，找topk的val和idx，在循环的过程中求sum
    sum = 0.0
    for i in tl.static_range(k):
        idx = tl.argmax(logits_vals, axis=0)
        val = tl.max(logits_vals, axis=0)
        val = tl.exp(val - max_val)

        weights_ptrs = topk_weights_ptr + row_id * k + i
        indices_ptrs = topk_indices_ptr + row_id * k + i

        # 这里的weight和indices都不能用寄存器来索引，这里只能存到gmem里吗？
        tl.store(weights_ptrs, val)
        tl.store(indices_ptrs, idx)

        sum += val
        logits_vals = tl.where(offsets==idx, -float("inf"), logits_vals)

    for i in tl.static_range(k):
        weights_ptrs = topk_weights_ptr + row_id * k + i
        val = tl.load(weights_ptrs)
        tl.store(weights_ptrs, val / sum)


@triton.jit
def moe_topk_gating_kernel_register(
    logits_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    M, E, k: tl.constexpr,
    E_PADDED: tl.constexpr,
    K_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(axis=0)
    logits_ptrs = logits_ptr + row_id * E + tl.arange(0, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < E
    logits_vals = tl.load(logits_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(logits_vals, axis=0)

    topk_vals = tl.full((K_PADDED,), -float("inf"), dtype=tl.float32)
    topk_indices = tl.full((K_PADDED,), -1, dtype=tl.int32)

    offsets_k = tl.arange(0, K_PADDED)
    for i in tl.static_range(k):
        idx = tl.argmax(logits_vals, axis=0)
        val = tl.max(logits_vals, axis=0)

        topk_vals = tl.where(offsets_k==i, val - max_val, topk_vals) 
        topk_indices = tl.where(offsets_k==i, idx, topk_indices)
        logits_vals = tl.where(offsets==idx, -float("inf"), logits_vals)

    topk_mask = offsets_k < k
    topk_vals = tl.exp(topk_vals)
    sum_val = tl.sum(topk_vals, axis=0)
    topk_vals = topk_vals / sum_val

    weights_ptrs = topk_weights_ptr + row_id * k + tl.arange(0, K_PADDED)
    indices_ptrs = topk_indices_ptr + row_id * k + tl.arange(0, K_PADDED)
    tl.store(weights_ptrs, topk_vals, mask=topk_mask)
    tl.store(indices_ptrs, topk_indices, mask=topk_mask)


def launch(
    kernel,
    logits,
    topk_weights,
    topk_indices,
    M, E, k
):
    E_PADDED = triton.next_power_of_2(E)
    K_PADDED = triton.next_power_of_2(k)
    BLOCK_SIZE = max(16, E_PADDED)

    grid = (M,)

    kernel[grid](
        logits,
        topk_weights,
        topk_indices,
        M,
        E,
        k,
        E_PADDED,
        K_PADDED,
        BLOCK_SIZE,
    )

def check_correctness(M=4096, E=128, k=4):
    torch.manual_seed(0)

    logits = torch.randn(M, E, device="cuda")

    w1 = torch.empty((M, k), device="cuda")
    i1 = torch.empty((M, k), dtype=torch.int32, device="cuda")

    w2 = torch.empty((M, k), device="cuda")
    i2 = torch.empty((M, k), dtype=torch.int32, device="cuda")

    launch_kernel(
        moe_topk_gating_kernel,
        logits,
        w1,
        i1,
        M,
        E,
        k,
    )

    launch_kernel(
        moe_topk_gating_kernel_register,
        logits,
        w2,
        i2,
        M,
        E,
        k,
    )

    torch.cuda.synchronize()

    print("weight max diff:", (w1 - w2).abs().max().item())
    print("index equal:", torch.equal(i1, i2))

def benchmark(
    kernel,
    name,
    M=32768,
    E=128,
    k=4,
    warmup=50,
    iters=200,
):
    logits = torch.randn(
        (M, E),
        device="cuda",
        dtype=torch.float32,
    )

    weights = torch.empty(
        (M, k),
        device="cuda",
        dtype=torch.float32,
    )

    indices = torch.empty(
        (M, k),
        device="cuda",
        dtype=torch.int32,
    )

    # warmup
    for _ in range(warmup):
        launch_kernel(
            kernel,
            logits,
            weights,
            indices,
            M,
            E,
            k,
        )

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iters):
        launch_kernel(
            kernel,
            logits,
            weights,
            indices,
            M,
            E,
            k,
        )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(
        f"{name:<12}: "
        f"{elapsed / iters * 1e6:.2f} us"
    )


if __name__ == "__main__":

    check_correctness()

    benchmark(
        moe_topk_gating_kernel,
        "gmem",
    )

    benchmark(
        moe_topk_gating_kernel_register,
        "register",
    )        


