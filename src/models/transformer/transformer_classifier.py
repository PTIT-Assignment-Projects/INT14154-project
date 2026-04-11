import torch
import torch.nn as nn
from .transformer_components import PositionalEncoding, TransformerEncoderBlock

class OwnTransformer(nn.Module):
    """
    Transformer Encoder for sequence classification implemented from scratch.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout, num_heads=8):
        super(OwnTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        
        # d_model in transformer will be the embedding_dim
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=embedding_dim, 
                num_heads=num_heads, 
                feedforward_dim=hidden_size * 2, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, attention_mask=None):
        # x: [batch_size, seq_len]
        
        # 1. Embedding + Positional Encoding
        out = self.embedding(x) # [batch_size, seq_len, embedding_dim]
        out = self.pos_encoding(out)
        out = self.dropout(out)
        
        # 2. Pass through Transformer Layers
        for layer in self.encoder_layers:
            out = layer(out, mask=attention_mask)
            
        # 3. Pooling (Global Average Pooling)
        # We could also use a [CLS] token, but for simplicity we'll pool
        pooled_out = torch.mean(out, dim=1)
        
        # 4. Classification
        logits = self.fc(pooled_out)
        
        return logits
