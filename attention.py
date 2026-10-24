import math
import torch
import torch.nn as nn


class MyAttention(nn.Module):
    """Drop-in replacement for HF's GPT2Attention. Reuses its pretrained c_attn / c_proj."""

    def __init__(self, orig):
        super().__init__()
        self.c_attn = orig.c_attn      # Conv1D 768 -> 2304 (fused Q, K, V)
        self.c_proj = orig.c_proj      # Conv1D 768 -> 768 (output projection)
        self.n_heads = orig.num_heads  # 12
        self.head_dim = orig.head_dim  # 64
        self.embed_dim = orig.embed_dim  # 768

    def forward(self, hidden_states, **kwargs):
        # hidden_states: [B, T, 768], already passed through ln_1
        B, T, _ = hidden_states.shape

        # fused projection -> split into q, k, v               each [B, T, 768]
        q, k, v = self.c_attn(hidden_states).split(self.embed_dim, dim=2)

        # split heads                                          each [B, 12, T, 64]
        q = torch.reshape(q, (B, T, self.n_heads, self.head_dim))
        q = torch.permute(q, (0, 2, 1, 3))

        k = torch.reshape(k, (B, T, self.n_heads, self.head_dim))
        k = torch.permute(k, (0, 2, 1, 3))

        v = torch.reshape(v, (B, T, self.n_heads, self.head_dim))
        v = torch.permute(v, (0, 2, 1, 3))

        # scores = q @ k^T / sqrt(head_dim)                    [B, 12, T, T]
        attn_scores = torch.div(q @ k.transpose(-2, -1), math.sqrt(self.head_dim))

        # causal mask (-inf above the diagonal), softmax over the last dim
        upper_tri = torch.triu(torch.ones(T, T, dtype=torch.bool, device=hidden_states.device), diagonal=1)
        attn_scores = attn_scores.masked_fill(upper_tri, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # weights @ v                                          [B, 12, T, 64]
        attn_out = attn_weights @ v

        # merge heads                                          [B, T, 768]
        attn_out = torch.permute(attn_out, (0, 2, 1, 3))
        attn_out = torch.reshape(attn_out, (B, T, self.embed_dim))

        # output projection                                    [B, T, 768]
        out = self.c_proj(attn_out)
        return out, None  # GPT2Block unpacks (attn_output, attn_weights)
