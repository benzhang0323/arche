import torch


def welford_variance_ref(x: torch.Tensor):
    """
    Reference implementation of Welford variance.

    x       : [B, N]

    returns : mean [B], var [B]
    """

    x_fp32 = x.float()
    mean = x_fp32.mean(dim=-1)
    var = x_fp32.var(dim=-1, unbiased=False)

    return mean, var