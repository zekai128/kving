import math
import torch
import torch.nn as nn


class SerialPreallocKVAttention(nn.Module):
    """Drop-in replacement for HF's GPT2Attention. Reuses its pretrained c_attn / c_proj."""

    def __init__(self, orig):
        super().__init__()
        self.GPT2_MAX_SEQ_LEN = 1024
        self.c_attn = orig.c_attn      # Conv1D 768 -> 2304 (fused Q, K, V)
        self.c_proj = orig.c_proj      # Conv1D 768 -> 768 (output projection)
        self.n_heads = orig.num_heads  # 12
        self.head_dim = orig.head_dim  # 64
        self.embed_dim = orig.embed_dim  # 768
        self.k_cache = torch.full((1, orig.num_heads, self.GPT2_MAX_SEQ_LEN, orig.head_dim), 0.0, device=orig.c_attn.weight.device)
        self.v_cache = torch.full((1, orig.num_heads, self.GPT2_MAX_SEQ_LEN, orig.head_dim), 0.0, device=orig.c_attn.weight.device)

    def forward(self, hidden_states, prefill=False, cache_pos = 0, **kwargs):
        # hidden_states: [B, T, 768], already passed through ln_1
        if (prefill):
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

            # store the prompt's k, v at positions 0..T-1
            self.k_cache[:, :, 0:T, :] = k
            self.v_cache[:, :, 0:T, :] = v

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
        else:
            B, T, _ = hidden_states.shape                         # T == 1 in decode

            # fused projection -> split into q, k, v               each [B, 1, 768]
            q, k, v = self.c_attn(hidden_states).split(self.embed_dim, dim=2)

            # split heads                                          each [B, 12, 1, 64]
            q = torch.reshape(q, (B, T, self.n_heads, self.head_dim))
            q = torch.permute(q, (0, 2, 1, 3))

            k = torch.reshape(k, (B, T, self.n_heads, self.head_dim))
            k = torch.permute(k, (0, 2, 1, 3))

            v = torch.reshape(v, (B, T, self.n_heads, self.head_dim))
            v = torch.permute(v, (0, 2, 1, 3))

            # store the new token's k, v at position cache_pos
            self.k_cache[:, :, cache_pos:cache_pos + T, :] = k
            self.v_cache[:, :, cache_pos:cache_pos + T, :] = v

            # every cached position so far, including the new token  each [B, 12, cache_pos+1, 64]
            k_all = self.k_cache[:, :, :cache_pos + T, :]
            v_all = self.v_cache[:, :, :cache_pos + T, :]

            # scores = q @ k_all^T / sqrt(head_dim)                [B, 12, 1, cache_pos+1]
            # no mask: the new token may attend to every cached position
            attn_scores = torch.div(q @ k_all.transpose(-2, -1), math.sqrt(self.head_dim))
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # weights @ v_all                                      [B, 12, 1, 64]
            attn_out = attn_weights @ v_all

            # merge heads                                          [B, 1, 768]
            attn_out = torch.permute(attn_out, (0, 2, 1, 3))
            attn_out = torch.reshape(attn_out, (B, T, self.embed_dim))

            # output projection                                    [B, 1, 768]
            out = self.c_proj(attn_out)
            return out, None
