"""
Triton implementation of paged_kv_gather.

This implementation realizes paged KV row retrieval using token indices
over flattened paged storage.

kv_pages: [num_pages, page_size, dim]
indices : [B, K]

Invalid indices are represented with -1 and produce zero rows.

This kernel is intentionally written to be readable and reusable as a
reference gather implementation for paged KV storage.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_kv_gather_kernel(
    kv_ptr,
    idx_ptr,
    out_ptr,
    total_rows,
    dim,
    stride_kvp,
    stride_kvs,
    stride_kvd,
    stride_ib,
    stride_ik,
    stride_ob,
    stride_ok,
    stride_od,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)

    idx = tl.load(idx_ptr + pid_b * stride_ib + pid_k * stride_ik)

    valid = (idx >= 0) & (idx < total_rows)

    safe_idx = tl.where(valid, idx, 0)

    page = safe_idx // PAGE_SIZE
    offset = safe_idx % PAGE_SIZE

    kv_ptrs = (
        kv_ptr
        + page * stride_kvp
        + offset * stride_kvs
        + offs_d * stride_kvd
    )

    vals = tl.load(kv_ptrs, mask=valid & (offs_d < dim), other=0.0)

    out_ptrs = (
        out_ptr
        + pid_b * stride_ob
        + pid_k * stride_ok
        + offs_d * stride_od
    )

    tl.store(out_ptrs, vals, mask=offs_d < dim)


def paged_kv_gather(kv_pages: torch.Tensor, indices: torch.Tensor):
    """
    Gather rows from paged KV storage.

    kv_pages : [num_pages, page_size, dim]
    indices  : [B, K]

    returns  : [B, K, dim]
    """

    num_pages, page_size, dim = kv_pages.shape
    B, K = indices.shape

    total_rows = num_pages * page_size

    out = torch.empty((B, K, dim), device=kv_pages.device, dtype=kv_pages.dtype)

    BLOCK_D = triton.next_power_of_2(dim)

    grid = (B, K)

    _paged_kv_gather_kernel[grid](
        kv_pages,
        indices,
        out,
        total_rows,
        dim,
        kv_pages.stride(0),
        kv_pages.stride(1),
        kv_pages.stride(2),
        indices.stride(0),
        indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        PAGE_SIZE=page_size,
        BLOCK_D=BLOCK_D,
    )

    return out