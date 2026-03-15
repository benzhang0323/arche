"""
Triton implementation of block_edge_mask.

This implementation applies boundary masking to a 2D tensor.

x           : [M, N]
valid_M     : logical valid rows
valid_N     : logical valid cols

returns     : [M, N]

Positions outside the logical valid region are replaced with fill_value.

This kernel is intentionally written to be readable and reusable as a
reference implementation of block-edge masking.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _block_edge_mask_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    valid_M,
    valid_N,
    fill_value,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    in_bounds = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)

    valid = (offs_m[:, None] < valid_M) & (offs_n[None, :] < valid_N)
    out_vals = tl.where(valid, vals, fill_value)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out_vals, mask=in_bounds)


def block_edge_mask(x: torch.Tensor, valid_M: int, valid_N: int, fill_value: float = 0.0):
    """
    Apply block-edge masking to a 2D tensor.

    x           : [M, N]
    valid_M     : logical valid rows
    valid_N     : logical valid cols

    returns     : [M, N]
    """

    M, N = x.shape

    out = torch.empty_like(x)

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _block_edge_mask_kernel[grid](
        x,
        out,
        M,
        N,
        valid_M,
        valid_N,
        fill_value,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out