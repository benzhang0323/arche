import torch


def online_softmax_ref(scores: torch.Tensor):
    """
    Reference implementation of online softmax.

    scores  : [B, N]

    returns : [B, N]
    """

    return torch.softmax(scores, dim=-1)