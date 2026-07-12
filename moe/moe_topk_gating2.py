import torch
import triton
import triton.language as tl
 
# 再写一遍
# 想变强就得花费时间和精力，你用时间去砸，就能比别人强。
@triton.jit
def moe_topk_gating_kernel(
    logits_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    M, E, k,
    logits_stride0, logits_stride1,
    topk_weights_stride0, topk_weights_stride1,
    topk_indices_stride0, topk_indices_stride1,
):
    tid = tl.program_id(axis=0)
    logits_ptrs = logits_ptr + tid * logits_stride0 + tl.arange(0, E) * logits_stride1
    logits_data = tl.load(logits_ptrs)

    # k比较小，没必要使用sort进行排序下标，直接循环k次，每次找最大值和最大值的下标即可
    # 但是这个找到的值放哪里？
    # triton不太好处理动态的寄存器数组，这里只能使用gmem，多一次load store
    sum_val = 0.0
    for i in range(k):
        max_val = tl.max(logits_data, axis=0)
        idx = tl.argmax(logits_data, axis=0)
        vals.append(max_val)
        indice.append(idx)
        sum_val += tl.exp(max_val)
        tl.where(logits_data == max_val, -float("inf"), logits_data)

        weight_ptrs = topk_weights_ptr + tid * topk_weights_stride0 + i * topk_weights_stride1
        indices_ptrs = topk_indices_ptr + tid * topk_indices_stride0 + i * topk_indices_stride1
        tl.store(weight_ptrs, max_val)
        tl.store(indices_ptrs, idx)

    for i in range(k):
        weight_ptrs = topk_weights_ptr + tid * topk_weights_stride0 + i * topk_weights_stride1
        val = tl.load(weight_ptrs)
        tl.store(weight_ptrs, val / sum_val)



# logits, topk_weights, topk_indices are tensors on the GPU
def solve(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    M: int,
    E: int,
    k: int,
):
    pass
