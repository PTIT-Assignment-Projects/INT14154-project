import torch
import torch.nn as nn
from .lstm_cell import OwnLSTMCell
from ..utils.masked_pooling import masked_mean_pool


class OwnLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super(OwnLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = embedding_dim if i == 0 else hidden_size
            self.layers.append(OwnLSTMCell(layer_input_size, hidden_size))

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()

        out = self.embedding(x)

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

        if attention_mask is not None:
            pooled_out = masked_mean_pool(out, attention_mask)
        else:
            pooled_out = torch.mean(out, dim=1)

        pooled_out = self.dropout(pooled_out)
        logits = self.fc(pooled_out)

        return logits