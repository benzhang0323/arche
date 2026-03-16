import torch


def layout_transform_copy_ref(src: torch.Tensor):
    """
    Reference implementation of layout transform copy.

    src     : [M, N]

    returns : [N, M]
    """

    M, N = src.shape

    out = torch.empty((N, M), device=src.device, dtype=src.dtype)

    out.copy_(src.transpose(0, 1))

    return out