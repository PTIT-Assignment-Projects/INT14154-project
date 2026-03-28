import json
import os

def create_notebook():
    cells = []

    def add_markdown(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")]
        })

    add_markdown("# Toxic Comment Classification Challenge\n\nThis notebook contains the complete project for the Toxic Comment Classification challenge. It includes custom implementations of LSTM, BiLSTM, GRU, RCNN, and Transformer models from scratch, along with optimizations like Class-Weighted Loss, AdamW, Gradient Clipping, and Learning Rate Scheduling.")

    add_markdown("## 1. Setup & Imports")
    add_code("""!pip install transformers tqdm scikit-learn pandas numpy matplotlib
    
import os
import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, accuracy_score
from transformers import AutoTokenizer""")

    add_markdown("## 2. Configuration & Constants\nConfigure your paths and hyperparameters here.")
    add_code("""# Kaggle paths (adjust these if running locally)
# For Kaggle, data is usually in '/kaggle/input/jigsaw-toxic-comment-classification-challenge/'
DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_CSV_PATH = "submission.csv"

# If running locally or you want to use a specific path, you can overwrite them here:
# TRAIN_CSV_PATH = "datasets/train.csv"
# TEST_CSV_PATH = "datasets/test.csv"
# SUBMISSION_CSV_PATH = "submission.csv"

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COLUMN = 'comment_text'

# ========== Core Hyperparameters ==========
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10          # Early Stopping will handle when to really stop
MAX_LEN = 128
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# ========== Optimization Hyperparameters ==========
WEIGHT_DECAY = 1e-4            # L2 regularization strength for AdamW
WARMUP_RATIO = 0.1             # 10% of total steps used for LR warmup
EARLY_STOPPING_PATIENCE = 3    # Stop after 3 epochs without improvement
GRADIENT_CLIP_MAX_NORM = 1.0   # Maximum gradient norm for clipping

# Model type identifiers
BILSTM_MODEL = 'bilstm'
ATTENTION_BILSTM_MODEL = 'attention_bilstm'
LSTM_MODEL = 'lstm'
GRU_MODEL = 'gru'
RCNN_MODEL = 'rcnn'
TRANSFORMER_MODEL = 'transformer'""")

    add_markdown("## 3. Utilities (Metrics, LR Scheduler, Early Stopping)")
    add_code("""def compute_metrics(all_labels: np.ndarray, all_probs: np.ndarray, threshold: float = 0.5):
    all_preds = (all_probs >= threshold).astype(int)
    metrics = {}
    
    try:
        metrics['roc_auc_macro'] = roc_auc_score(all_labels, all_probs, average='macro')
        metrics['roc_auc_per_class'] = roc_auc_score(all_labels, all_probs, average=None)
    except ValueError:
        metrics['roc_auc_macro'] = 0.0
        metrics['roc_auc_per_class'] = np.zeros(all_labels.shape[1])

    metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['subset_accuracy'] = accuracy_score(all_labels, all_preds)
    return metrics

def print_classification_report(all_labels: np.ndarray, all_probs: np.ndarray,
                                 label_names: list, threshold: float = 0.5):
    all_preds = (all_probs >= threshold).astype(int)
    print("\\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))
    
    try:
        per_class_auc = roc_auc_score(all_labels, all_probs, average=None)
        print("\\nPer-Class ROC-AUC:")
        for name, auc in zip(label_names, per_class_auc):
            print(f"  {name:20s}: {auc:.4f}")
        print(f"  {'MACRO AVERAGE':20s}: {np.mean(per_class_auc):.4f}")
    except ValueError:
        print("\\nCould not compute per-class AUC (possibly missing classes in batch)")
    print("=" * 60)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        current_score = -score if self.mode == 'min' else score

        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(model)
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  ⏳ EarlyStopping: {self.counter}/{self.patience} (no improvement)")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  🛑 EarlyStopping triggered! Restoring best model...")
                self._restore_checkpoint(model)
        else:
            if self.verbose:
                print(f"  ✅ EarlyStopping: metric improved")
            self.best_score = current_score
            self.counter = 0
            self._save_checkpoint(model)
        return self.early_stop

    def _save_checkpoint(self, model):
        self.best_model_state = copy.deepcopy(model.state_dict())

    def _restore_checkpoint(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.scheduler = LambdaLR(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()""")

    add_markdown("## 4. Preprocessing & Dataset")
    add_code("""class TextProcessor:
    def __init__(self, model_name="distilbert-base-uncased", max_len=128):
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, texts, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.fillna("").tolist()
        elif isinstance(texts, list):
            texts = [str(t) if t is not None else "" for t in texts]

        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors=return_tensors
        )


class ToxicDataset(Dataset):
    def __init__(self, df, processor, text_column=TEXT_COLUMN, label_columns=LABEL_COLUMNS):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.text_column = text_column
        self.label_columns = label_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_column]) if pd.notnull(row[self.text_column]) else ""

        if all(col in self.df.columns for col in self.label_columns):
            labels = torch.tensor(row[self.label_columns].values.astype(float), dtype=torch.float)
        else:
            labels = torch.tensor([])

        encoding = self.processor.tokenize(text, return_tensors="pt")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }""")

    add_markdown("## 5. Model Core Components (Cells & Attention)")
    add_code("""class OwnLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(OwnLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.W_ih)
        nn.init.orthogonal_(self.W_hh)
        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)

    def forward(self, x, states):
        h_prev, c_prev = states
        gates = (torch.matmul(x, self.W_ih) + self.b_ih) + (torch.matmul(h_prev, self.W_hh) + self.b_hh)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        c_next = f_gate * c_prev + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)
        return h_next, c_next


class OwnGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(OwnGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(input_size, 2 * hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, 2 * hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(2 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(2 * hidden_size))
        self.W_in = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_in = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hn = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.W_ih); nn.init.orthogonal_(self.W_hh)
        nn.init.orthogonal_(self.W_in); nn.init.orthogonal_(self.W_hn)
        nn.init.zeros_(self.b_ih); nn.init.zeros_(self.b_hh)
        nn.init.zeros_(self.b_in); nn.init.zeros_(self.b_hn)

    def forward(self, x, h_prev):
        gates = (torch.matmul(x, self.W_ih) + self.b_ih) + (torch.matmul(h_prev, self.W_hh) + self.b_hh)
        z_gate, r_gate = gates.chunk(2, 1)
        z_gate, r_gate = torch.sigmoid(z_gate), torch.sigmoid(r_gate)
        n_gate = torch.tanh((torch.matmul(x, self.W_in) + self.b_in) + r_gate * (torch.matmul(h_prev, self.W_hn) + self.b_hn))
        h_next = (1 - z_gate) * n_gate + z_gate * h_prev
        return h_next


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, mask=None):
        energy = torch.tanh(self.projection(hidden_states))
        weights = self.v(energy)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * hidden_states, dim=1)
        return context, weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, feedforward_dim, dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(feedforward_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out = self.attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x""")

    add_markdown("## 6. Model Classifiers")
    add_code("""class OwnLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout):
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
        
        pooled_out = torch.mean(out, dim=1)
        return self.fc(self.dropout(pooled_out))


class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(OwnLSTMCell(embedding_dim if i == 0 else hidden_size, hidden_size))

        self.attention = SelfAttention(hidden_size)
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

        context, _ = self.attention(out, mask=attention_mask)
        return self.fc(self.dropout(context))


class OwnBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout):
        super(OwnBiLSTM, self).__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.fwd_layers, self.bwd_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            lin = embedding_dim if i == 0 else hidden_size * 2
            self.fwd_layers.append(OwnLSTMCell(lin, hidden_size))
            self.bwd_layers.append(OwnLSTMCell(lin, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()
        out = self.embedding(x)
        
        for layer_idx in range(self.num_layers):
            h_f, c_f = torch.zeros(batch_size, self.hidden_size).to(x.device), torch.zeros(batch_size, self.hidden_size).to(x.device)
            h_b, c_b = torch.zeros(batch_size, self.hidden_size).to(x.device), torch.zeros(batch_size, self.hidden_size).to(x.device)
            
            fwd_outputs, bwd_outputs = [], [None] * seq_len
            for t in range(seq_len):
                h_f, c_f = self.fwd_layers[layer_idx](out[:, t, :], (h_f, c_f))
                fwd_outputs.append(h_f)
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = self.bwd_layers[layer_idx](out[:, t, :], (h_b, c_b))
                bwd_outputs[t] = h_b
                
            out = torch.cat((torch.stack(fwd_outputs, dim=1), torch.stack(bwd_outputs, dim=1)), dim=2)
            if layer_idx < self.num_layers - 1: out = self.dropout(out)
        
        return self.fc(self.dropout(torch.mean(out, dim=1)))


class AttentionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout):
        super(AttentionBiLSTM, self).__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.fwd_layers, self.bwd_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            lin = embedding_dim if i == 0 else hidden_size * 2
            self.fwd_layers.append(OwnLSTMCell(lin, hidden_size))
            self.bwd_layers.append(OwnLSTMCell(lin, hidden_size))

        self.attention = SelfAttention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()
        out = self.embedding(x)
        
        for layer_idx in range(self.num_layers):
            h_f, c_f = torch.zeros(batch_size, self.hidden_size).to(x.device), torch.zeros(batch_size, self.hidden_size).to(x.device)
            h_b, c_b = torch.zeros(batch_size, self.hidden_size).to(x.device), torch.zeros(batch_size, self.hidden_size).to(x.device)
            
            fwd_outputs, bwd_outputs = [], [None] * seq_len
            for t in range(seq_len):
                h_f, c_f = self.fwd_layers[layer_idx](out[:, t, :], (h_f, c_f))
                fwd_outputs.append(h_f)
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = self.bwd_layers[layer_idx](out[:, t, :], (h_b, c_b))
                bwd_outputs[t] = h_b
                
            out = torch.cat((torch.stack(fwd_outputs, dim=1), torch.stack(bwd_outputs, dim=1)), dim=2)
            if layer_idx < self.num_layers - 1: out = self.dropout(out)
        
        context, _ = self.attention(out, mask=attention_mask)
        return self.fc(self.dropout(context))


class OwnGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout):
        super(OwnGRU, self).__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(OwnGRUCell(embedding_dim if i == 0 else hidden_size, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()
        out = self.embedding(x)
        
        for layer_idx in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_size).to(x.device)
            layer_outputs = []
            for t in range(seq_len):
                h = self.layers[layer_idx](out[:, t, :], h)
                layer_outputs.append(h)
            out = torch.stack(layer_outputs, dim=1)
            if layer_idx < self.num_layers - 1: out = self.dropout(out)
        
        return self.fc(self.dropout(torch.mean(out, dim=1)))


class OwnRCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout):
        super(OwnRCNN, self).__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.fwd_layers, self.bwd_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            lin = embedding_dim if i == 0 else hidden_size * 2
            self.fwd_layers.append(OwnLSTMCell(lin, hidden_size))
            self.bwd_layers.append(OwnLSTMCell(lin, hidden_size))

        self.fusion = nn.Linear(hidden_size * 2 + embedding_dim, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()
        embeds = self.embedding(x)
        rnn_out = embeds
        
        for layer_idx in range(self.num_layers):
            h_f, c_f = torch.zeros(batch_size, self.hidden_size).to(x.device), torch.zeros(batch_size, self.hidden_size).to(x.device)
            h_b, c_b = torch.zeros(batch_size, self.hidden_size).to(x.device), torch.zeros(batch_size, self.hidden_size).to(x.device)
            fwd_outputs, bwd_outputs = [], [None] * seq_len
            
            for t in range(seq_len):
                h_f, c_f = self.fwd_layers[layer_idx](rnn_out[:, t, :], (h_f, c_f))
                fwd_outputs.append(h_f)
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = self.bwd_layers[layer_idx](rnn_out[:, t, :], (h_b, c_b))
                bwd_outputs[t] = h_b
                
            rnn_out = torch.cat((torch.stack(fwd_outputs, dim=1), torch.stack(bwd_outputs, dim=1)), dim=2)
            if layer_idx < self.num_layers - 1: rnn_out = self.dropout(rnn_out)

        combined = torch.cat((rnn_out[:, :, :self.hidden_size], embeds, rnn_out[:, :, self.hidden_size:]), dim=2)
        latent = torch.tanh(self.fusion(combined))
        out, _ = torch.max(latent, dim=1)
        return self.fc(self.dropout(out))


class OwnTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout, num_heads=8):
        super(OwnTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim, num_heads, hidden_size * 2, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, attention_mask=None):
        out = self.dropout(self.pos_encoding(self.embedding(x)))
        for layer in self.encoder_layers:
            out = layer(out, mask=attention_mask)
        return self.fc(torch.mean(out, dim=1))""")

    add_markdown("## 7. Training & Evaluation Logic")
    add_code("""def generate_submission(model, processor, device, submission_path):
    print(f"Generating submission to {submission_path}...")
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
    except FileNotFoundError:
        print(f"Test data not found at {TEST_CSV_PATH}. Make sure you are in a Kaggle environment or paths are correct.")
        return
        
    test_dataset = ToxicDataset(test_df, processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            all_preds.append(torch.sigmoid(logits).cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()

    submission_df = pd.DataFrame(all_preds, columns=LABEL_COLUMNS)
    submission_df.insert(0, 'id', test_df['id'].values)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved as '{submission_path}'.")


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0)


def compute_class_weights(df, label_columns):
    pos_counts = df[label_columns].sum()
    neg_counts = len(df) - pos_counts
    pos_weights = neg_counts / pos_counts.clip(lower=1)
    
    print("\\nClass weights applied to handle highly imbalanced data:")
    for name, weight in zip(label_columns, pos_weights):
        print(f"  {name:20s}: {weight:.2f}")
    return torch.tensor(pos_weights.values, dtype=torch.float)


def build_model(model_type, vocab_size):
    config = {
        'vocab_size': vocab_size,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'num_classes': len(LABEL_COLUMNS),
        'dropout': DROPOUT
    }
    
    if model_type == BILSTM_MODEL: return OwnBiLSTM(**config)
    elif model_type == ATTENTION_BILSTM_MODEL: return AttentionBiLSTM(**config)
    elif model_type == GRU_MODEL: return OwnGRU(**config)
    elif model_type == RCNN_MODEL: return OwnRCNN(**config)
    elif model_type == TRANSFORMER_MODEL: return OwnTransformer(**config)
    elif model_type == LSTM_MODEL: return OwnLSTM(**config)
    else: raise ValueError(f"Unsupported model_type: {model_type}")""")

    add_markdown("## 8. Main Execution")
    add_code("""def main(model_type=BILSTM_MODEL, sample_size=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========== 1. Load Data ==========
    print("Loading data...")
    try:
        df = pd.read_csv(TRAIN_CSV_PATH)
    except FileNotFoundError:
        print(f"Could not load data from {TRAIN_CSV_PATH}. Ensure paths are correct.")
        return
        
    # Option to use subset for faster iteration
    if sample_size and sample_size < len(df):
        print(f"Found {len(df)} rows, taking a random sample of {sample_size} for training...")
        df = df.sample(sample_size, random_state=42)

    # ========== 2. Preprocessing & Datasets ==========
    print("Initializing tokenizers...")
    processor = TextProcessor(model_name="distilbert-base-uncased", max_len=MAX_LEN)
    full_dataset = ToxicDataset(df, processor=processor)

    # ========== 3. Splitting ==========
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ========== 4. Initialize Model ==========
    print(f"Building {model_type} model...")
    model = build_model(model_type, processor.tokenizer.vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ========== 5. Loss & Optimizer ==========
    pos_weights = compute_class_weights(df, LABEL_COLUMNS).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=int(WARMUP_RATIO * total_steps), total_steps=total_steps)
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='min')

    # ========== 6. Training Loop ==========
    print(f"\\nStarting training for {model_type}...")
    train_losses, val_losses, val_aucs, learning_rates = [], [], [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss, val_labels, val_probs = evaluate_model(model, val_loader, criterion, device)
        metrics = compute_metrics(val_labels, val_probs)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_aucs.append(metrics['roc_auc_macro'])

        print(f"\\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val ROC-AUC: {metrics['roc_auc_macro']:.4f}")

        if early_stopper(avg_val_loss, model):
            print(f"\\n🛑 Early stopping triggered at epoch {epoch+1}!")
            break

    # ========== 7. Evaluation & Plots ==========
    print("\\nFINAL EVALUATION ON VALIDATION SET (Best Model)")
    _, final_labels, final_probs = evaluate_model(model, val_loader, criterion, device)
    print_classification_report(final_labels, final_probs, LABEL_COLUMNS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs_range, train_losses, label='Train Loss', marker='o')
    axes[0].plot(epochs_range, val_losses, label='Val Loss', marker='s')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(epochs_range, val_aucs, label='Val ROC-AUC', color='green', marker='D')
    axes[1].set_title('ROC-AUC')
    axes[1].legend()

    axes[2].plot(epochs_range, learning_rates, label='Learning Rate', color='red', marker='^')
    axes[2].set_title('LR Schedule')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

    # ========== 8. Generate Submission ==========
    generate_submission(model, processor, device, SUBMISSION_CSV_PATH)
    print("Done!")

# ==========================================
# Run training (you can pick your model here)
# Options: 'lstm', 'bilstm', 'attention_bilstm', 'gru', 'rcnn', 'transformer'
# Note: sample_size=10000 is used to test run. Set to None to train on the full ~160k dataset.
# ==========================================
main(model_type=BILSTM_MODEL, sample_size=10000)""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_path = "/home/duongvct/Documents/workspace/PTIT/Y4T2/y4t2_intro_deep_learning_project/kaggle_toxic_comment_notebook.ipynb"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
        
    print(f"Created notebook at {output_path}")

create_notebook()
