"""
Triton implementation of welford_variance.

This implementation computes row-wise mean and variance for a 2D tensor.

x       : [B, N]

returns : mean [B], var [B]

Statistics are accumulated in fp32.

This kernel is intentionally written to be readable and reusable as a
reference implementation of Welford-style variance computation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _welford_variance_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    N,
    stride_xb,
    stride_xn,
    stride_mb,
    stride_vb,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    x_ptrs = x_ptr + pid_b * stride_xb + offs_n * stride_xn
    vals = tl.load(x_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    count = tl.sum(mask_n.to(tl.float32), axis=0)
    mean = tl.sum(vals, axis=0) / count

    centered = tl.where(mask_n, vals - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / count

    mean_ptr = mean_ptr + pid_b * stride_mb
    var_ptr = var_ptr + pid_b * stride_vb

    tl.store(mean_ptr, mean)
    tl.store(var_ptr, var)


def welford_variance(x: torch.Tensor):
    """
    Compute row-wise mean and variance for a 2D tensor.

    x       : [B, N]

    returns : mean [B], var [B]
    """

    B, N = x.shape

    mean = torch.empty((B,), device=x.device, dtype=torch.float32)
    var = torch.empty((B,), device=x.device, dtype=torch.float32)

    BLOCK_N = triton.next_power_of_2(N)

    grid = (B,)

    _welford_variance_kernel[grid](
        x,
        mean,
        var,
        N,
        x.stride(0),
        x.stride(1),
        mean.stride(0),
        var.stride(0),
        BLOCK_N=BLOCK_N,
    )

    return mean, var