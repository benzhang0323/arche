import torch


def ragged_mask_ref(scores: torch.Tensor, lengths: torch.Tensor):
    """
    Reference implementation of ragged mask.

    scores   : [B, Q, K]
    lengths  : [B]

    returns  : [B, Q, K]
    """

    B, Q, K = scores.shape

    k_idx = torch.arange(K, device=scores.device).view(1, 1, K)
    valid = k_idx < lengths.view(B, 1, 1)

    out = scores.clone()
    out = out.masked_fill(~valid, float("-inf"))

    return out