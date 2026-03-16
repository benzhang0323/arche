"""
Triton implementation of online_softmax.

This implementation computes row-wise softmax using a numerically stable online
reduction over the last dimension.

scores  : [B, N]

returns : [B, N]

This kernel is intentionally written to be readable and reusable as a
reference implementation of online softmax.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _online_softmax_pass1_kernel(
    scores_ptr,
    max_ptr,
    sum_ptr,
    N,
    stride_sb,
    stride_sn,
    stride_mb,
    stride_ub,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    row_start = scores_ptr + pid_b * stride_sb

    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.full((), 0.0, tl.float32)

    for start_n in range(0, N, BLOCK_N):
        idx_n = start_n + offs_n
        mask = idx_n < N

        vals = tl.load(
            row_start + idx_n * stride_sn,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)

        tile_max = tl.max(vals, axis=0)
        new_max = tl.maximum(running_max, tile_max)

        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum = running_sum + tl.sum(tl.exp(vals - new_max), axis=0)
        running_max = new_max

    tl.store(max_ptr + pid_b * stride_mb, running_max)
    tl.store(sum_ptr + pid_b * stride_ub, running_sum)


@triton.jit
def _online_softmax_pass2_kernel(
    scores_ptr,
    out_ptr,
    max_ptr,
    sum_ptr,
    N,
    stride_sb,
    stride_sn,
    stride_ob,
    stride_on,
    stride_mb,
    stride_ub,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_tile = tl.program_id(1)

    offs_n = pid_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    row_start = scores_ptr + pid_b * stride_sb
    out_start = out_ptr + pid_b * stride_ob

    row_max = tl.load(max_ptr + pid_b * stride_mb).to(tl.float32)
    row_sum = tl.load(sum_ptr + pid_b * stride_ub).to(tl.float32)

    vals = tl.load(
        row_start + offs_n * stride_sn,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    probs = tl.exp(vals - row_max) / row_sum

    tl.store(out_start + offs_n * stride_on, probs, mask=mask)


def online_softmax(scores: torch.Tensor):
    """
    Compute row-wise online softmax.

    scores  : [B, N]

    returns : [B, N]
    """
    B, N = scores.shape

    out = torch.empty_like(scores)
    row_max = torch.empty((B,), device=scores.device, dtype=torch.float32)
    row_sum = torch.empty((B,), device=scores.device, dtype=torch.float32)

    BLOCK_N = 128

    _online_softmax_pass1_kernel[(B,)](
        scores,
        row_max,
        row_sum,
        N,
        scores.stride(0),
        scores.stride(1),
        row_max.stride(0),
        row_sum.stride(0),
        BLOCK_N=BLOCK_N,
    )

    _online_softmax_pass2_kernel[(B, triton.cdiv(N, BLOCK_N))](
        scores,
        out,
        row_max,
        row_sum,
        N,
        scores.stride(0),
        scores.stride(1),
        out.stride(0),
        out.stride(1),
        row_max.stride(0),
        row_sum.stride(0),
        BLOCK_N=BLOCK_N,
    )

    return out