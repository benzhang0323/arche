import torch


def causal_mask_ref(scores: torch.Tensor):
    """
    Reference implementation of causal mask.

    scores  : [B, Q, K]

    returns : [B, Q, K]
    """

    B, Q, K = scores.shape

    q_idx = torch.arange(Q, device=scores.device).view(Q, 1)
    k_idx = torch.arange(K, device=scores.device).view(1, K)
    valid = k_idx <= q_idx

    out = scores.clone()
    out = out.masked_fill(~valid.unsqueeze(0), float("-inf"))

    return out