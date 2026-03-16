import torch


def batched_sparse_row_gather_ref(rows: torch.Tensor, indices: torch.Tensor):
    """
    Reference implementation of batched sparse row gather.

    rows    : [num_rows, dim]
    indices : [B, K]

    returns : [B, K, dim]
    """

    num_rows, dim = rows.shape
    B, K = indices.shape

    out = torch.zeros((B, K, dim), device=rows.device, dtype=rows.dtype)

    valid = (indices >= 0) & (indices < num_rows)

    out[valid] = rows[indices[valid]]

    return out