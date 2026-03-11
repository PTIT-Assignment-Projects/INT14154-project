import torch
import torch.nn as nn
from src.models.lstm.lstm_cell import OwnLSTMCell
from src.models.lstm.attention import SelfAttention

class AttentionBiLSTM(nn.Module):
    """
    Bidirectional LSTM with Attention mechanism implemented from scratch.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super(AttentionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # List of layers, each with forward and backward cells
        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = embedding_dim if i == 0 else hidden_size * 2
            self.fwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))
            self.bwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))

        # Attention layer - applied to the concatenated outputs (hidden_size * 2)
        self.attention = SelfAttention(hidden_size * 2)
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head - takes the context vector from attention
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, attention_mask=None):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # 1. Get embeddings
        out = self.embedding(x)
        
        # 2. Sequential Processing through BiLSTM layers
        for layer_idx in range(self.num_layers):
            h_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            h_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            
            fwd_outputs = []
            bwd_outputs = [None] * seq_len
            
            # Forward pass
            for t in range(seq_len):
                h_fwd, c_fwd = self.fwd_layers[layer_idx](out[:, t, :], (h_fwd, c_fwd))
                fwd_outputs.append(h_fwd)
            
            # Backward pass
            for t in range(seq_len - 1, -1, -1):
                h_bwd, c_bwd = self.bwd_layers[layer_idx](out[:, t, :], (h_bwd, c_bwd))
                bwd_outputs[t] = h_bwd
                
            fwd_outputs = torch.stack(fwd_outputs, dim=1)
            bwd_outputs = torch.stack(bwd_outputs, dim=1)
            
            # Concatenate fwd and bwd: (batch_size, seq_len, hidden_size * 2)
            out = torch.cat((fwd_outputs, bwd_outputs), dim=2)
            
            if layer_idx < self.num_layers - 1:
                out = self.dropout(out)
        
        # 3. Attention Pooling
        # context: (batch_size, hidden_size * 2)
        context, attn_weights = self.attention(out, mask=attention_mask)
        
        # 4. Classification
        context = self.dropout(context)
        logits = self.fc(context)
        
        return logits
