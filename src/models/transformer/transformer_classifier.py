import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

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

        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        return self.W_o(context)


class PreLNTransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, feedforward_dim, dropout):
        super(PreLNTransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, x_norm, x_norm, mask=mask)
        x = x + self.dropout(attn_out)

        x_norm2 = self.norm2(x)
        ff_out = self.ff(x_norm2)
        x = x + self.dropout(ff_out)
        return x


class OwnTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout, num_heads=8):
        super(OwnTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.encoder_layers = nn.ModuleList([
            PreLNTransformerEncoderBlock(
                d_model=embedding_dim,
                num_heads=num_heads,
                feedforward_dim=hidden_size * 2,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, attention_mask=None):
        batch_size = x.size(0)

        out = self.embedding(x)
        out = self.pos_encoding(out)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        out = torch.cat([cls_tokens, out], dim=1)

        for layer in self.encoder_layers:
            out = layer(out, mask=attention_mask)

        cls_output = out[:, 0, :]

        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)

        return logits