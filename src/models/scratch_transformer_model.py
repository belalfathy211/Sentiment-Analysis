import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#PE
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,embed_dim,2).float() * (-math.log(10000.0)/embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

#MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.linear_output = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask = None):
        batch_size = q.size(0)

        Q = self.linear_q(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.linear_k(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.linear_v(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return self.linear_output(context)

#FFN
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

#Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, dropout = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_head)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x

#Model
class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, ff_dim, n_layers, output_dim, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx= pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, text, mask=None):
        x = self.embedding(text)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        # Masked Global Average Pooling
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).float()
            embedding_sum = torch.sum(x * expanded_mask, dim= 1)
            actual_len = torch.clamp(expanded_mask.sum(dim=1), min= 1.0)
            pooled = embedding_sum / actual_len
        else:
            pooled = x.mean(dim=1)

        return self.fc(self.dropout(pooled))