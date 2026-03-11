import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Simple Additive Attention (also known as Bahdanau or MLP attention) 
    to pool sequence outputs into a single context vector.
    """
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, mask=None):
        # hidden_states: (batch_size, seq_len, hidden_size)
        # mask: (batch_size, seq_len) - 1 for valid tokens, 0 for padding
        
        # Energy: (batch_size, seq_len, hidden_size)
        energy = torch.tanh(self.projection(hidden_states))
        
        # Weights: (batch_size, seq_len, 1)
        weights = self.v(energy)
        
        if mask is not None:
            # Masking: set weight for padding tokens to large negative value
            # mask: (batch_size, seq_len) -> (batch_size, seq_len, 1)
            mask = mask.unsqueeze(-1).float()
            weights = weights.masked_fill(mask == 0, -1e9)
            
        # Softmax over sequence length
        weights = F.softmax(weights, dim=1)
        
        # Context vector: (batch_size, hidden_size)
        context = torch.sum(weights * hidden_states, dim=1)
        
        return context, weights
