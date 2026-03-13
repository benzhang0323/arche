import torch


def block_reduce_sum_ref(x: torch.Tensor):
    """
    Reference implementation of block reduce sum.

    x       : [B, N]

    returns : [B]
    """

    return x.sum(dim=-1)