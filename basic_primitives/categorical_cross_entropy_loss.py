import torch
import triton
import triton.language as tl

# 思路：每个block负责一行的exp，以及reduce求sum
# 因为N可能有10000，要求整个batch的平均loss，这里就写到host memory上吧
# 否则的话每个block写到gmem, 然后最后还需要从gmem读进行reduce
@triton.jit 
def cross_entropy_loss(logits, true_labels, loss_array, N, C, padded_c: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    logits_ptrs = logits + pid * C + tl.arange(0, padded_c)
    mask = tl.arange(0, padded_c) < C
    logits_data = tl.load(logits_ptrs, mask, other=-float("inf"))

    labels_ptr = true_labels + pid
    label = tl.load(labels_ptr) # 单个元素的加载，这样写可以吗
    true_label_logits = tl.load(logits + pid * C + label)

    max_val = tl.max(logits_data, axis=0)
    loss = tl.log(tl.sum(tl.exp(logits_data - max_val), axis=0)) + max_val - true_label_logits

    tl.store(loss_array + pid, loss)



# logits, true_labels, loss are tensors on the GPU
def solve(logits: torch.Tensor, true_labels: torch.Tensor, loss: torch.Tensor, N: int, C: int):
    BLOCK_SIZE = 256
    padded_c = triton.next_power_of_2(C)
    loss_array = torch.zeros([N], dtype=torch.float32, device=logits.device)
    grid = (N, )
    cross_entropy_loss[grid](logits, true_labels, loss_array, N, C, padded_c, BLOCK_SIZE)
    loss[0] = torch.sum(loss_array) / N
