"""
Triton implementation of batched_sparse_row_gather.

This implementation realizes batched sparse row retrieval using index lists
over token-major storage.

rows    : [num_rows, dim]
indices : [B, K]

Invalid indices are represented with -1 and produce zero rows.

This kernel is intentionally written to be readable and reusable as a
reference gather implementation for sparse row retrieval.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batched_sparse_row_gather_kernel(
    rows_ptr,
    idx_ptr,
    out_ptr,
    num_rows,
    dim,
    stride_rn,
    stride_rd,
    stride_ib,
    stride_ik,
    stride_ob,
    stride_ok,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)

    idx = tl.load(idx_ptr + pid_b * stride_ib + pid_k * stride_ik)

    valid = (idx >= 0) & (idx < num_rows)

    safe_idx = tl.where(valid, idx, 0)

    row_ptrs = (
        rows_ptr
        + safe_idx * stride_rn
        + offs_d * stride_rd
    )

    vals = tl.load(row_ptrs, mask=valid & (offs_d < dim), other=0.0)

    out_ptrs = (
        out_ptr
        + pid_b * stride_ob
        + pid_k * stride_ok
        + offs_d * stride_od
    )

    tl.store(out_ptrs, vals, mask=offs_d < dim)


def batched_sparse_row_gather(rows: torch.Tensor, indices: torch.Tensor):
    """
    Gather rows from token-major storage using batched sparse indices.

    rows    : [num_rows, dim]
    indices : [B, K]

    returns : [B, K, dim]
    """

    num_rows, dim = rows.shape
    B, K = indices.shape

    out = torch.empty((B, K, dim), device=rows.device, dtype=rows.dtype)

    BLOCK_D = triton.next_power_of_2(dim)

    grid = (B, K)

    _batched_sparse_row_gather_kernel[grid](
        rows,
        indices,
        out,
        num_rows,
        dim,
        rows.stride(0),
        rows.stride(1),
        indices.stride(0),
        indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_D=BLOCK_D,
    )

    return out