import torch


def paged_kv_gather_ref(kv_pages: torch.Tensor, indices: torch.Tensor):
    """
    Reference implementation of paged KV gather.

    kv_pages : [num_pages, page_size, dim]
    indices  : [B, K]

    returns  : [B, K, dim]
    """

    num_pages, page_size, dim = kv_pages.shape
    total_rows = num_pages * page_size

    flat = kv_pages.reshape(total_rows, dim)

    B, K = indices.shape

    out = torch.zeros((B, K, dim), device=kv_pages.device, dtype=kv_pages.dtype)

    valid = (indices >= 0) & (indices < total_rows)

    out[valid] = flat[indices[valid]]

    return out