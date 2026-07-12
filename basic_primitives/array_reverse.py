import torch
import triton
import triton.language as tl


@triton.jit
def reverse_kernel(input, N, BLOCK_SIZE: tl.constexpr):
    # 感觉难度不大，倒着读一遍，然后写到id对应的位置
    # 如果一个线程只读一个元素呢，会有问题吗？
    # 如果一个block操作的快，已经读写好了，另一个block的线程还没有读取，就出现问题了
    # syncthreads是block内warp的同步 好像没有block的同步
    # 我是看了下面block启动设置除以了2想到这个的，那应该是每个线程读取两个元素，每个线程自己负责翻转
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N // 2)
    input_ptrs = input + offsets

    offsets_reverse = N - 1 - offsets
    input_reverse_ptrs = input + offsets_reverse

    data = tl.load(input_ptrs, mask)
    data_reverse = tl.load(input_reverse_ptrs, mask)

    tl.store(input_ptrs, data_reverse, mask)
    tl.store(input_reverse_ptrs, data, mask)


@triton.jit
def reverse_kernel_1(input, N, BLOCK_SIZE: tl.constexpr):
    # 上面的那个版本，都后面一半的时候是这样读的： N-1 N-2 N-3 
    # 这样倒着连续的，虽然也可以合并内存访问，但稍微有些代价
    # 其次，这里N - 1 - offsets， 因为合并内存访问要求是128字节对齐的，这里N可能导致很多的block的读取都是跨128字节的
    # 就需要两次load, 浪费显存带宽
    # 新的方法：每个block读取正向的block和对称的那个block 这两个block，然后将后面的那个block内部进行翻转，然后交换两个block的store
    pid = tl.program_id(axis=0)
    block_num = torch.cdiv(N // 2, BLOCK_SIZE)
    pid0 = block_num - 1 - pid
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    reverse_offsets = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 不行，分析了下，最后边那个block有问题
    mask = offsets < 

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)

    reverse_kernel[grid](input, N, BLOCK_SIZE)
