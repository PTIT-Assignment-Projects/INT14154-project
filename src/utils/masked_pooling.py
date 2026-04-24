"""
Masked pooling utilities for fixing the attention_mask bug.
All RNN models pass attention_mask to forward() but were ignoring it
in pooling (padding tokens polluted pooled representations).
"""
import torch


def masked_mean_pool(tensor, attention_mask):
    """Masked mean pooling over sequence dimension.
    Args:
        tensor: (batch, seq_len, hidden)
        attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
    Returns:
        (batch, hidden)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()
    mask_sum = mask_expanded.sum(dim=1).clamp(min=1)
    sum_embeds = (tensor * mask_expanded).sum(dim=1)
    return sum_embeds / mask_sum


def masked_max_pool(tensor, attention_mask):
    """Masked max pooling over sequence dimension.
    Sets padding positions to -inf before max to exclude them.
    Args:
        tensor: (batch, seq_len, hidden)
        attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
    Returns:
        (batch, hidden)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()
    tensor_masked = tensor.masked_fill(mask_expanded == 0, float('-inf'))
    out, _ = tensor_masked.max(dim=1)
    return out