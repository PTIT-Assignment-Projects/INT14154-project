"""
Attention-enhanced LSTM (Unidirectional).
Combines the standard LSTM with self-attention pooling.
"""
import torch
import torch.nn as nn
from .lstm_cell import OwnLSTMCell
from .attention import SelfAttention


class AttentionLSTM(nn.Module):
    """
    Unidirectional LSTM with Attention mechanism.
    Instead of simple average pooling, uses attention to learn
    which time steps are most important for classification.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 num_layers: int, num_classes: int, dropout: float):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = embedding_dim if i == 0 else hidden_size
            self.layers.append(OwnLSTMCell(layer_input_size, hidden_size))

        # Attention layer applied to LSTM outputs
        self.attention = SelfAttention(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()

        # 1. Embedding
        out = self.embedding(x)

        # 2. Sequential processing through LSTM layers
        for layer_idx in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c = torch.zeros(batch_size, self.hidden_size).to(x.device)

            layer_outputs = []
            for t in range(seq_len):
                h, c = self.layers[layer_idx](out[:, t, :], (h, c))
                layer_outputs.append(h)

            out = torch.stack(layer_outputs, dim=1)

            if layer_idx < self.num_layers - 1:
                out = self.dropout(out)

        # 3. Attention pooling (instead of avg pooling)
        context, attn_weights = self.attention(out, mask=attention_mask)

        # 4. Classification
        context = self.dropout(context)
        logits = self.fc(context)

        return logits
