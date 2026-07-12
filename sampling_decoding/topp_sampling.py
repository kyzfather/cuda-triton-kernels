import torch
import triton
import triton.language as tl

# 一维的，词表大小<=50,000
# 单个block处理不了5w的数据，那么求max
# softmax是单个block做吗？还是并行的split-k，开block并行，最后单block规约？
# 求出softmax后，也没法单个block求最大值，难道只能用sort？
# 但是单个block也没法sort
@triton.jit
def top_p_sampling_kernel(
    logits_ptr,
    p_ptr, seed_ptr, sampled_token_ptr,
    vocab_size
):


@triton.jit
def kernel0(
    logits_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    logits_ptrs = logits_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 


def solve(
    logits: torch.Tensor,
    p: torch.Tensor,
    seed: torch.Tensor,
    sampled_token: torch.Tensor,
    vocab_size: int,
):
    pass
