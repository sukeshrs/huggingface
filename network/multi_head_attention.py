import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Linear projections
        query = self.linear_q(query).view(
            batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, seq_len,
                                      self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(
            batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Check shapes
        print(
            f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(self.d_k)

        # Check shapes
        print(f"Scores shape: {scores.shape}")

        if mask is not None:
            # Adjust mask dimensions to [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.softmax(scores)
        attn = self.attn_dropout(attn)

        x = torch.matmul(attn, value).transpose(1, 2).contiguous()

        # Check shapes
        print(f"Attention output shape before view: {x.shape}")

        x = x.view(batch_size, seq_len, self.d_model)

        # Check shapes
        print(f"Attention output shape after view: {x.shape}")

        return self.linear_out(x)
