import torch

## triton要实现的太多了，online softmax, sort, prefix sum scan
## 这题应该是用torch写的

def solve(
    logits: torch.Tensor,
    p: torch.Tensor,
    seed: torch.Tensor,
    sampled_token: torch.Tensor,
    vocab_size: int,
):
    probs = torch.softmax(logits[:vocab_size], dim=0)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    cdf = torch.cumsum(sorted_probs, dim=0)

    cutoff = torch.searchsorted(cdf, p)
    cutoff = torch.clamp(cutoff, max=vocab_size - 1)

    nucleus_probs = sorted_probs[: cutoff+1]
    nucleus_indices = sorted_indices[: cutoff+1]

    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    g = torch.Generator(device=logits.device)
    g.manual_seed(int(seed.item()))

    idx = torch.multinomial(
        nucleus_probs,
        1,
        generator=g,
    ).item()

    sampled_token[0] = nucleus_indices[idx]