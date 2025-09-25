import torch
import torch.nn.functional as F
import math

def sinusoidal_embedding(
    x: torch.Tensor,
    dim: int
) -> torch.Tensor:
    """
    Compute sinusoidal positional embeddings.

    Args:
        x (Tensor): LongTensor or FloatTensor of shape (B,) or (B,1) with integer positions.
        dim (int): Embedding dimensionality.

    Returns:
        Tensor of shape (B, dim).
    """
    # squeeze (B,1) → (B,)
    if x.dim() == 2 and x.size(-1) == 1:
        x = x.squeeze(-1)
    device = x.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
    args = x.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)  # (B, 2*half)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)
