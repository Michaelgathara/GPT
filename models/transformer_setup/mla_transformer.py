# mla_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LatentAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

        # Learnable latent tokens
        self.latents = nn.Parameter(torch.randn(n_latent_vec, latent_dim))

        # Projections for keys, values, queries
        self.key_in = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_in = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_in = nn.Linear(latent_dim, head_dim, bias=False)

        self.dropout = nn.Dropout(dropout_prob)

        # KV caches
        self.k_cache = None
        self.v_cache = None
        self.query_cache = None

    def forward(self, input_tensor, latent=None, use_cache=False):
        batch_size, seq_len, _ = input_tensor.shape
        if latent is None:
            latent = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Query
        if use_cache and self.query_cache is not None:
            queries = self.query_cache
        else:
            queries = self.query_in(latent)
            if use_cache:
                self.query_cache = queries

        # Key/Value
        if use_cache:
            new_k = self.key_in(input_tensor)
            new_v = self.value_in(input_tensor)
            if self.k_cache is not None:
                keys = torch.cat([self.k_cache, new_k], dim=1)
                values = torch.cat([self.v_cache, new_v], dim=1)
            else:
                keys, values = new_k, new_v
            # Truncate
            if keys.size(1) > self.max_seq_len:
                keys = keys[:, -self.max_seq_len:]
                values = values[:, -self.max_seq_len:]
            self.k_cache, self.v_cache = keys, values
        else:
            if seq_len > self.max_seq_len:
                inp = input_tensor[:, -self.max_seq_len:]
            else:
                inp = input_tensor
            keys = self.key_in(inp)
            values = self.value_in(inp)
            self.clear_cache()

        # Attention
        scores = queries @ keys.transpose(-2, -1) * (self.head_dim ** -0.5)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ values
        return out

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.query_cache = None

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()
        self.heads = nn.ModuleList([
            LatentAttentionHead(embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, latent=None, use_cache=False):
        head_outs = [h(x, latent, use_cache) for h in self.heads]
        concat = torch.cat(head_outs, dim=-1)
        return self.dropout(self.out_proj(concat))

    def clear_cache(self):
        for h in self.heads:
            h.clear_cache()

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout_prob):
        super().__init__()
        hidden = (int(4 * embed_dim * 2/3) + 7)//8*8
        self.w1 = nn.Linear(embed_dim, hidden, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        fused = gate * up
        return self.dropout(self.w2(fused))

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadLatentAttention(num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec)
        self.norm_latent = nn.LayerNorm(embed_dim)

        # static projection
        self.latent_to_seq = nn.Linear(n_latent_vec, max_seq_len, bias=False)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, latent=None, use_cache=False):
        b, seq_len, _ = x.shape
        # attention path
        y = self.norm1(x)
        attn = self.attn(y, latent, use_cache)
        attn = self.norm_latent(attn)
        # project back
        proj = self.latent_to_seq(attn.transpose(1,2))  # (b, embed_dim, seq_len)
        proj = proj.transpose(1,2)
        x = x + self.dropout(proj)
        # feedforward
        y2 = self.norm2(x)
        x = x + self.dropout(self.ff(y2))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout_prob, latent_dim, n_latent_vec, use_checkpoint=False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, max_seq_len, dropout_prob, latent_dim, n_latent_vec)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.token_emb.weight

        if use_checkpoint:
            for blk in self.blocks:
                blk.use_checkpointing = True

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    def clear_all_caches(self):
        for blk in self.blocks:
            blk.attn.clear_cache()

    def forward(self, idx, targets=None, latent=None, use_cache=False):
        b, seq_len = idx.shape
        if seq_len > self.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len")
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(seq_len, device=idx.device))
        x = tok + pos
        for blk in self.blocks:
            x = blk(x, latent, use_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            tgt_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        self.clear_all_caches()
        for _ in range(max_new_tokens):
            inp = idx if idx.size(1)<=self.max_seq_len else idx[:,-self.max_seq_len:]
            logits, _ = self(inp, use_cache=True)
            logits = logits[:,-1,:] / temperature
            if top_k:
                v,_=torch.topk(logits, top_k)
                logits[logits<v[:,-1].unsqueeze(-1)] = -float('Inf')
            elif top_p:
                sorted_logits, idxs = torch.sort(logits, descending=True)
                cumprob = torch.cumsum(F.softmax(sorted_logits,dim=-1),dim=-1)
                mask = cumprob>top_p
                mask[...,1:]=mask[...,:-1]
                mask[...,0]=False
                logits[mask.scatter(1, idxs, mask)] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs,1)
            idx = torch.cat([idx, nxt], dim=1)
        self.train()
        return idx
