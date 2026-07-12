import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    Q_ptr, cos_ptr, sin_ptr, out_ptr,
    M, D,
    stride_qm, stride_qd,
    stride_cosm, stride_cosd,
    stride_sinm, stride_sind,
    stride_outm, stride_outd,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)

    if row_idx >= M:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D
    
    row_q_ptr = Q_ptr + row_idx * stride_qm
    row_cos_ptr = cos_ptr + row_idx * stride_cosm
    row_sin_ptr = sin_ptr + row_idx * stride_sinm
    row_out_ptr = out_ptr + row_idx * stride_outm

    q = tl.load(row_q_ptr + col_offsets * stride_qd, mask=mask, other=0.0)
    cos = tl.load(row_cos_ptr + col_offsets * stride_cosd, mask=mask, other=0.0)
    sin = tl.load(row_sin_ptr + col_offsets * stride_sind, mask=mask, other=0.0)

    half_D = D // 2


    rotate_offsets = tl.where(col_offsets < half_D, col_offsets + half_D, col_offsets - half_D)
    q_rotated = tl.load(row_q_ptr + rotate_offsets * stride_qd, mask=mask, other=0.0)
    q_rotated = tl.where(col_offsets < half_D, -q_rotated, q_rotated)
    
    output_row = q * cos + q_rotated * sin

    tl.store(row_out_ptr + col_offsets * stride_outd, output_row, mask=mask)

def solve(
    Q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output: torch.Tensor,
    M: int,
    D: int 
):
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (M, )
    rope_kernel[grid](
        Q, cos, sin, output,
        M, D,
        Q.stride(0), Q.stride(1),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output