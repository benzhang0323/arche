"""
Triton implementation of ragged_mask.

This implementation applies per-batch validity masking to a score tensor.

scores   : [B, Q, K]
lengths  : [B]

returns  : [B, Q, K]

Positions with k >= lengths[b] are masked to -inf.

This kernel is intentionally written to be readable and reusable as a
reference implementation of ragged validity masking.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ragged_mask_kernel(
    scores_ptr,
    lengths_ptr,
    out_ptr,
    Q,
    K,
    stride_sb,
    stride_sq,
    stride_sk,
    stride_ob,
    stride_oq,
    stride_ok,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_bounds = (offs_q[:, None] < Q) & (offs_k[None, :] < K)

    valid_len = tl.load(lengths_ptr + pid_b)

    score_ptrs = (
        scores_ptr
        + pid_b * stride_sb
        + offs_q[:, None] * stride_sq
        + offs_k[None, :] * stride_sk
    )

    vals = tl.load(score_ptrs, mask=mask_bounds, other=0.0)

    ragged_valid = offs_k[None, :] < valid_len
    out_vals = tl.where(ragged_valid, vals, float("-inf"))

    out_ptrs = (
        out_ptr
        + pid_b * stride_ob
        + offs_q[:, None] * stride_oq
        + offs_k[None, :] * stride_ok
    )

    tl.store(out_ptrs, out_vals, mask=mask_bounds)


def ragged_mask(scores: torch.Tensor, lengths: torch.Tensor):
    """
    Apply per-batch validity masking to a score tensor.

    scores   : [B, Q, K]
    lengths  : [B]

    returns  : [B, Q, K]
    """

    B, Q, K = scores.shape

    out = torch.empty_like(scores)

    BLOCK_Q = 32
    BLOCK_K = 32

    grid = (
        B,
        triton.cdiv(Q, BLOCK_Q),
        triton.cdiv(K, BLOCK_K),
    )

    _ragged_mask_kernel[grid](
        scores,
        lengths,
        out,
        Q,
        K,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
    )

    return out