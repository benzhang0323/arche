"""
Triton implementation of block_reduce_sum.

This implementation reduces each row of a 2D tensor by summing across the
last dimension.

x       : [B, N]

returns : [B]

This kernel is intentionally written to be readable and reusable as a
reference implementation of block-local summation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _block_reduce_sum_kernel(
    x_ptr,
    out_ptr,
    N,
    stride_xb,
    stride_xn,
    stride_ob,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    x_ptrs = x_ptr + pid_b * stride_xb + offs_n * stride_xn
    vals = tl.load(x_ptrs, mask=mask_n, other=0.0)
    vals = vals.to(tl.float32)

    acc = tl.sum(vals, axis=0)

    out_ptr = out_ptr + pid_b * stride_ob
    tl.store(out_ptr, acc)


def block_reduce_sum(x: torch.Tensor):
    """
    Apply block-local summation to a 2D tensor.

    x       : [B, N]

    returns : [B]
    """

    B, N = x.shape

    out = torch.empty((B,), device=x.device, dtype=torch.float32)

    BLOCK_N = triton.next_power_of_2(N)

    grid = (B,)

    _block_reduce_sum_kernel[grid](
        x,
        out,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        BLOCK_N=BLOCK_N,
    )

    return out