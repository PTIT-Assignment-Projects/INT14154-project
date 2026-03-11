import torch
import torch.nn as nn
from src.models.lstm.lstm_cell import OwnLSTMCell

class OwnBiLSTM(nn.Module):
    """
    Bidirectional LSTM implemented from scratch for sequence classification.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super(OwnBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        # Embedding layer convert token indices to (batch_size, seq_len, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # List of layers, each with forward and backward cells
        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # First layer input is embedding_dim, subsequent layers take concatenated fwd+bwd output (hidden_size * 2)
            layer_input_size = embedding_dim if i == 0 else hidden_size * 2
            self.fwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))
            self.bwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))

        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        # We take the output of the last BiLSTM layer (hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, attention_mask=None):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # 1. Get embeddings
        # out: (batch_size, seq_len, embedding_dim)
        out = self.embedding(x)
        
        # Iterate over BiLSTM layers
        for layer_idx in range(self.num_layers):
            # Initialize hidden and cell states for both directions
            h_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_fwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            h_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_bwd = torch.zeros(batch_size, self.hidden_size).to(x.device)
            
            fwd_outputs = []
            # Pre-allocate for backward outputs to maintain sequence order
            bwd_outputs = [None] * seq_len
            
            # Forward pass: t goes from 0 to seq_len-1
            for t in range(seq_len):
                h_fwd, c_fwd = self.fwd_layers[layer_idx](out[:, t, :], (h_fwd, c_fwd))
                fwd_outputs.append(h_fwd)
            
            # Backward pass: t goes from seq_len-1 to 0
            for t in range(seq_len - 1, -1, -1):
                h_bwd, c_bwd = self.bwd_layers[layer_idx](out[:, t, :], (h_bwd, c_bwd))
                bwd_outputs[t] = h_bwd
                
            # Stack results: (batch_size, seq_len, hidden_size)
            fwd_outputs = torch.stack(fwd_outputs, dim=1)
            bwd_outputs = torch.stack(bwd_outputs, dim=1)
            
            # Combine forward and backward hidden states
            # out: (batch_size, seq_len, hidden_size * 2)
            out = torch.cat((fwd_outputs, bwd_outputs), dim=2)
            
            # Apply dropout between layers (if multi-layer)
            if layer_idx < self.num_layers - 1:
                out = self.dropout(out)
        
        # 2. Sequence Pooling (Global Average Pooling)
        # Reduces (batch_size, seq_len, hidden_size * 2) -> (batch_size, hidden_size * 2)
        pooled_out = torch.mean(out, dim=1)
        
        # 3. Final Dropout and Classification
        pooled_out = self.dropout(pooled_out)
        logits = self.fc(pooled_out)
        
        return logits