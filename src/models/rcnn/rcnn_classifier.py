import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.lstm.lstm_cell import OwnLSTMCell
from ..utils.masked_pooling import masked_max_pool


class OwnRCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super(OwnRCNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_size = embedding_dim if i == 0 else hidden_size * 2
            self.fwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))
            self.bwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))

        self.fusion = nn.Linear(hidden_size * 2 + embedding_dim, hidden_size * 2)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()

        embeds = self.embedding(x)

        rnn_out = embeds
        for layer_idx in range(self.num_layers):
            h_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            h_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)

            fwd_outputs = []
            bwd_outputs = [None] * seq_len

            for t in range(seq_len):
                h_fwd, c_fwd = self.fwd_layers[layer_idx](rnn_out[:, t, :], (h_fwd, c_fwd))
                fwd_outputs.append(h_fwd)

            for t in range(seq_len - 1, -1, -1):
                h_bwd, c_bwd = self.bwd_layers[layer_idx](rnn_out[:, t, :], (h_bwd, c_bwd))
                bwd_outputs[t] = h_bwd

            fwd_outputs = torch.stack(fwd_outputs, dim=1)
            bwd_outputs = torch.stack(bwd_outputs, dim=1)

            rnn_out = torch.cat((fwd_outputs, bwd_outputs), dim=2)

            if layer_idx < self.num_layers - 1:
                rnn_out = self.dropout(rnn_out)

        combined = torch.cat((rnn_out[:, :, :self.hidden_size], embeds, rnn_out[:, :, self.hidden_size:]), dim=2)

        latent = torch.tanh(self.fusion(combined))

        if attention_mask is not None:
            out = masked_max_pool(latent, attention_mask)
        else:
            out, _ = latent.max(dim=1)

        out = self.dropout(out)
        logits = self.fc(out)

        return logits


class EnhancedRCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super(EnhancedRCNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_size = embedding_dim if i == 0 else hidden_size * 2
            self.fwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))
            self.bwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))

        self.fusion = nn.Linear(hidden_size * 2 + embedding_dim, hidden_size * 2)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_final = nn.Linear(hidden_size, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()

        embeds = self.embedding(x)

        rnn_out = embeds
        for layer_idx in range(self.num_layers):
            h_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            h_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)

            fwd_outputs = []
            bwd_outputs = [None] * seq_len

            for t in range(seq_len):
                h_fwd, c_fwd = self.fwd_layers[layer_idx](rnn_out[:, t, :], (h_fwd, c_fwd))
                fwd_outputs.append(h_fwd)

            for t in range(seq_len - 1, -1, -1):
                h_bwd, c_bwd = self.bwd_layers[layer_idx](rnn_out[:, t, :], (h_bwd, c_bwd))
                bwd_outputs[t] = h_bwd

            fwd_outputs = torch.stack(fwd_outputs, dim=1)
            bwd_outputs = torch.stack(bwd_outputs, dim=1)

            rnn_out = torch.cat((fwd_outputs, bwd_outputs), dim=2)

            if layer_idx < self.num_layers - 1:
                rnn_out = self.dropout(rnn_out)

        combined = torch.cat((rnn_out[:, :, :self.hidden_size], embeds, rnn_out[:, :, self.hidden_size:]), dim=2)

        latent = torch.tanh(self.fusion(combined))

        if attention_mask is not None:
            out = masked_max_pool(latent, attention_mask)
        else:
            out, _ = latent.max(dim=1)

        out = torch.relu(self.fc_hidden(out))
        out = self.dropout(out)
        logits = self.fc_final(out)

        return logits