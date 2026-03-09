import torch
import torch.nn as nn

class ToxicBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2, n_classes=6, dropout=0.3):
        """
        Bidirectional LSTM for multi-label text classification.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of character/word embeddings.
            hidden_dim (int): Number of hidden units in LSTM.
            n_layers (int): Number of LSTM layers.
            n_classes (int): Number of target labels (6 for toxic classification).
            dropout (float): Dropout probability.
        """
        super(ToxicBiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # hidden_dim * 2 because of bidirectionality
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Tensor of input IDs (batch_size, seq_len).
            attention_mask (torch.Tensor): Optional attention mask.
            
        Returns:
            torch.Tensor: Raw logits for each class (batch_size, n_classes).
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(input_ids))
        
        # We don't strictly need pack_padded_sequence here if we use global pooling, 
        # but it's good practice. For now, simple BiLSTM + Max Pooling.
        
        # output: (batch_size, seq_len, hidden_dim * 2)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Global Max Pooling over the sequence length dimension
        # (batch_size, seq_len, hidden_dim * 2) -> (batch_size, hidden_dim * 2)
        pooled, _ = torch.max(output, dim=1)
        
        # Final classification head
        logits = self.fc(self.dropout(pooled))
        
        return logits


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
