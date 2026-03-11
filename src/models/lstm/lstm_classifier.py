import torch
import torch.nn as nn
from .lstm_cell import OwnLSTMCell

class OwnLSTM(nn.Module):
    """
    Unidirectional LSTM implemented from scratch for sequence classification.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super(OwnLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # List of layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = embedding_dim if i == 0 else hidden_size
            self.layers.append(OwnLSTMCell(layer_input_size, hidden_size))

        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # 1. Get embeddings
        # out: (batch_size, seq_len, embedding_dim)
        out = self.embedding(x)
        
        # Iterate over LSTM layers
        for layer_idx in range(self.num_layers):
            # Initialize hidden and cell states
            h = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c = torch.zeros(batch_size, self.hidden_size).to(x.device)
            
            layer_outputs = []
            
            # Forward pass through time
            for t in range(seq_len):
                h, c = self.layers[layer_idx](out[:, t, :], (h, c))
                layer_outputs.append(h)
            
            # Stack results: (batch_size, seq_len, hidden_size)
            out = torch.stack(layer_outputs, dim=1)
            
            # Apply dropout between layers
            if layer_idx < self.num_layers - 1:
                out = self.dropout(out)
        
        # 2. Sequence Pooling (Global Average Pooling)
        # Reduces (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        pooled_out = torch.mean(out, dim=1)
        
        # 3. Final Dropout and Classification
        pooled_out = self.dropout(pooled_out)
        logits = self.fc(pooled_out)
        
        return logits
