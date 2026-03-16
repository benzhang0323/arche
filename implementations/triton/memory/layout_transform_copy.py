"""
Triton implementation of layout_transform_copy.

This implementation realizes layout transform copy as a 2D transpose-like copy.

src     : [M, N]

returns : [N, M]

This kernel is intentionally written to be readable and reusable as a
reference implementation of layout-changing memory movement.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layout_transform_copy_kernel(
    src_ptr,
    out_ptr,
    M,
    N,
    stride_sm,
    stride_sn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    src_ptrs = (
        src_ptr
        + offs_m[:, None] * stride_sm
        + offs_n[None, :] * stride_sn
    )

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    vals = tl.load(src_ptrs, mask=mask, other=0.0)

    out_ptrs = (
        out_ptr
        + offs_n[:, None] * stride_om
        + offs_m[None, :] * stride_on
    )

    out_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(out_ptrs, vals.trans(), mask=out_mask)


def layout_transform_copy(src: torch.Tensor):
    """
    Copy data while transforming layout through a transpose-like mapping.

    src     : [M, N]

    returns : [N, M]
    """

    M, N = src.shape

    out = torch.empty((N, M), device=src.device, dtype=src.dtype)

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _layout_transform_copy_kernel[grid](
        src,
        out,
        M,
        N,
        src.stride(0),
        src.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out