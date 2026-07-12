import torch
import triton
import triton.language as tl

# 假设二维的的input进行softmax
@triton.jit
def softmax_kernel(input, output, row_stride, col_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input + pid * row_stride + offsets
    ouput_ptrs = output + pid * row_stride + offsets
    # 一趟算出max和sum
    # m_new = max(m_old, m_tile)
    # a = exp(m_old - m_new)
    # d = d_old * a + sum(exp(x_tile - m_new)) 
    m_old = -float('inf')
    d_old = 0.0
    steps = (row_stride + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(triton.cdiv(row_stride, steps)):
        mask = (offsets + i * BLOCK_SIZE) < row_stride
        tmp_ptrs = input_ptrs + i * BLOCK_SIZE
        input_data = tl.load(tmp_ptrs, mask=mask, other=-float('inf'))
        m_tile = tl.max(input_data, axis=0)
        m_new = tl.maximum(m_tile, m_old) # 不太清楚这里调用的函数对不对
        a = tl.exp(m_old - m_new)
        d = d_old * a + tl.sum(tl.exp(input_data - m_new), axis=0)
        d_old = d
        m_old = m_new
    for i in range(triton.cdiv(row_stride, BLOCK_SIZE)):
        mask = (offsets + i * BLOCK_SIZE) < row_stride
        tmp_ptrs = input_ptrs + i * BLOCK_SIZE
        input_data = tl.load(tmp_ptrs, mask=mask, other=-float('inf'))
        input_data = tl.exp(input_data - m_old) / d_old
        output_tmp_ptrs = output_ptrs + i * BLOCK_SIZE
        tl.store(output_tmp_ptrs, input_data, mask=mask)


def softmax(input: torch.tensor):
    orig_shape = input.shape
    N = orig_shape[-1]
    M = input.numel() // N
    input_2d = input.view(-1, N)
    output_2d = torch.empty_like(input_2d)

    row_stride = input_2d.stride(0)
    col_stride = input_2d.stride(1)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))

    grid = (M,)
    softmax_kernel[grid](
        input_2d, input_2d
        row_stride=row_stride,
        col_stride=col_stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
  