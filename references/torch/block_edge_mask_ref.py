import torch


def block_edge_mask_ref(x: torch.Tensor, valid_M: int, valid_N: int, fill_value: float = 0.0):
    """
    Reference implementation of block edge mask.

    x           : [M, N]
    valid_M     : logical valid rows
    valid_N     : logical valid cols

    returns     : [M, N]
    """

    M, N = x.shape

    offs_m = torch.arange(M, device=x.device).view(M, 1)
    offs_n = torch.arange(N, device=x.device).view(1, N)

    valid = (offs_m < valid_M) & (offs_n < valid_N)

    out = x.clone()
    out = out.masked_fill(~valid, fill_value)

    return out