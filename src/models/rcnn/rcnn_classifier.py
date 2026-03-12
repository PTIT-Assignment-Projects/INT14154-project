import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.lstm.lstm_cell import OwnLSTMCell

class OwnRCNN(nn.Module):
    """
    Recurrent Convolutional Neural Network (RCNN) implemented from scratch.
    Combines BiLSTM context with original embeddings followed by Max Pooling.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super(OwnRCNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Recurrent Part (BiLSTM)
        # We only implement 1 layer of BiLSTM for standard RCNN, 
        # but we use self.num_layers if requested.
        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = embedding_dim if i == 0 else hidden_size * 2
            self.fwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))
            self.bwd_layers.append(OwnLSTMCell(layer_input_size, hidden_size))

        # 3. Fusion Layer (Convolutional part in RCNN)
        # Standard RCNN concatenates [left_context, embedding, right_context]
        # input_size = hidden_size (fwd) + embedding_dim + hidden_size (bwd)
        self.fusion = nn.Linear(hidden_size * 2 + embedding_dim, hidden_size * 2)
        
        self.dropout = nn.Dropout(dropout)
        
        # 4. Classification Head
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, attention_mask=None):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # 1. Get embeddings
        # embeds: (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(x)
        
        # 2. BiLSTM context
        # We'll pass through all BiLSTM layers
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

        # 3. RCNN Fusion: Concatenate [fwd_context, original_embedding, bwd_context]
        # rnn_out is already [fwd, bwd]. We need to concat with 'embeds'
        # combined: (batch_size, seq_len, hidden_size * 2 + embedding_dim)
        combined = torch.cat((rnn_out[:, :, :self.hidden_size], embeds, rnn_out[:, :, self.hidden_size:]), dim=2)
        
        # Pass through fusion layer + tanh
        # latent: (batch_size, seq_len, hidden_size * 2)
        latent = torch.tanh(self.fusion(combined))
        
        # 4. Global Max Pooling (The "Convolutional" part)
        # We pool over the sequence dimension (dim 1)
        # out: (batch_size, hidden_size * 2)
        out, _ = torch.max(latent, dim=1)
        
        # 5. Final Classification
        out = self.dropout(out)
        logits = self.fc(out)
        
        return logits
