import math
import torch
import torch.nn as nn


class StaticBatchPreallocKVAttention(nn.Module):
    """Drop-in replacement for HF's GPT2Attention with a batched, preallocated KV cache.

    Step 2 (static batching): up to max_batch requests run together, left-padded so every
    row's last token is at the same position. Row i of the cache belongs to request i of
    the current batch. Same buffer for the whole run; each new batch overwrites the rows.

    Extra kwargs (passed through model(...) like prefill / cache_pos in step 1):
        prefill:   True for the prompt pass, False for single-token decode
        cache_pos: decode only, the cache column to write (shared by every row thanks to left padding)
        pad_len:   [B] long, number of left-pad tokens in each row (pads are always a prefix).
                   Fixed for the life of a batch; the engine passes the same tensor every step.
    Note: HF also passes its own `attention_mask` kwarg; it lands in **kwargs and is ignored.
    """

    GPT2_MAX_SEQ_LEN = 1024

    def __init__(self, orig, max_batch: int = 8):
        super().__init__()
        self.c_attn = orig.c_attn      # Conv1D 768 -> 2304 (fused Q, K, V)
        self.c_proj = orig.c_proj      # Conv1D 768 -> 768 (output projection)
        self.n_heads = orig.num_heads  # 12
        self.head_dim = orig.head_dim  # 64
        self.embed_dim = orig.embed_dim  # 768
        self.max_batch = max_batch
        device = orig.c_attn.weight.device

        # TODO: allocate k_cache and v_cache with room for max_batch rows
        self.k_cache = torch.full((max_batch, orig.num_heads, self.GPT2_MAX_SEQ_LEN, orig.head_dim), 0.0, device=orig.c_attn.weight.device)
        self.v_cache = torch.full((max_batch, orig.num_heads, self.GPT2_MAX_SEQ_LEN, orig.head_dim), 0.0, device=orig.c_attn.weight.device)

    def _split_heads(self, x, B, T):
        # [B, T, 768] -> [B, 12, T, 64]
        return x.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x, B, T):
        # [B, 12, T, 64] -> [B, T, 768]
        return x.permute(0, 2, 1, 3).reshape(B, T, self.embed_dim)

    def forward(self, hidden_states, prefill=False, cache_pos=0, pad_len=None, **kwargs):
        # hidden_states: [B, T, 768], already passed through ln_1
        B, T, _ = hidden_states.shape

        q, k, v = self.c_attn(hidden_states).split(self.embed_dim, dim=2)   # each [B, T, 768]
        q = self._split_heads(q, B, T)                                      # each [B, 12, T, 64]
        k = self._split_heads(k, B, T)
        v = self._split_heads(v, B, T)

        if prefill:
            # T = longest prompt in the batch (shorter ones are left-padded)

            # TODO: store this batch's k, v in the cache (which rows? which columns?)
            self.k_cache[0:B, :, 0:T, :] = k
            self.v_cache[0:B, :, 0:T, :] = v

            # scores = q @ k^T / sqrt(head_dim)                           [B, 12, T, T]
            attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # causal: query i may not see key j > i                        [T, T]
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=hidden_states.device), diagonal=1)
            # padding: key columns 0..pad_len[b]-1 are pad for row b       [B, T]
            cols = torch.arange(T, device=hidden_states.device)
            is_pad = cols[None, :] < pad_len[:, None]
            # one combined mask, True = masked                             [B, 1, T, T]
            mask = causal[None, None, :, :] | is_pad[:, None, None, :]

            # finite fill instead of -inf: fully masked rows (pad queries) become uniform
            # instead of NaN; in every other row the masked weights still underflow to exactly 0
            attn_scores = attn_scores.masked_fill(mask, torch.finfo(attn_scores.dtype).min)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            attn_out = attn_weights @ v                                    # [B, 12, T, 64]
        else:
            # T == 1; every row writes to the same column, cache_pos

            # store the new k, v at column cache_pos, only in this batch's rows
            self.k_cache[:B, :, cache_pos:cache_pos + T, :] = k
            self.v_cache[:B, :, cache_pos:cache_pos + T, :] = v

            # every cached column so far, including the new one           [B, 12, cache_pos+1, 64]
            n_cols = cache_pos + T
            k_all = self.k_cache[:B, :, :n_cols, :]
            v_all = self.v_cache[:B, :, :n_cols, :]

            # scores = q @ k_all^T / sqrt(head_dim)                       [B, 12, 1, cache_pos+1]
            attn_scores = (q @ k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # padding only (the new token may see every cached column)     [B, 1, 1, cache_pos+1]
            cols = torch.arange(n_cols, device=hidden_states.device)
            is_pad = cols[None, :] < pad_len[:, None]
            attn_scores = attn_scores.masked_fill(is_pad[:, None, None, :], torch.finfo(attn_scores.dtype).min)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            attn_out = attn_weights @ v_all                                # [B, 12, 1, 64]

        out = self.c_proj(self._merge_heads(attn_out, B, T))               # [B, T, 768]
        return out, None  # GPT2Block unpacks (attn_output, attn_weights)
