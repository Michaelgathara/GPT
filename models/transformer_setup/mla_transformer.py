# Refactored mla_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LatentAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.latents = nn.Parameter(torch.randn(n_latent_vec, latent_dim))
        self.key_in = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_in = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_in = nn.Linear(latent_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.clear_cache()
        # Check if flash attention is available
        self.has_flash_attn = False
        try:
            from flash_attn import flash_attn_unpadded
            self.flash_attn_unpadded = flash_attn_unpadded
            self.has_flash_attn = True
            print("Flash Attention is available and will be used")
        except ImportError:
            print("Flash Attention not available, falling back to standard attention")

    def forward(self, input_tensor, latent=None, use_cache=False):
        batch_size, seq_len, _ = input_tensor.shape
        if latent is None:
            latent = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        if use_cache and self.query_cache is not None:
            queries = self.query_cache
        else:
            queries = self.query_in(latent)
            if use_cache:
                self.query_cache = queries
        if use_cache:
            new_k = self.key_in(input_tensor)
            new_v = self.value_in(input_tensor)
            if self.k_cache is not None:
                keys = torch.cat([self.k_cache, new_k], dim=1)
                values = torch.cat([self.v_cache, new_v], dim=1)
            else:
                keys, values = new_k, new_v
            if keys.size(1) > self.max_seq_len:
                keys = keys[:, -self.max_seq_len:]
                values = values[:, -self.max_seq_len:]
            self.k_cache, self.v_cache = keys, values
        else:
            inp = input_tensor[:, -self.max_seq_len:] if seq_len > self.max_seq_len else input_tensor
            keys = self.key_in(inp)
            values = self.value_in(inp)
            self.clear_cache()
            
        # Compute attention using flash attention if available, otherwise use standard attention
        if self.has_flash_attn:
            # reshape for flash_attn_unpadded which expects (B, S, H)
            # Here our batch already contains only one head, so no need to reshape for num_heads
            Q = queries
            K = keys
            V = values
            
            # flash_attn_unpadded expects contiguous tensors
            Q = Q.contiguous()
            K = K.contiguous()
            V = V.contiguous()
            
            # Generate sequence length tensors
            Q_lengths = torch.full((Q.shape[0],), Q.shape[1], device=Q.device, dtype=torch.int32)
            K_lengths = torch.full((K.shape[0],), K.shape[1], device=K.device, dtype=torch.int32)
            
            # Call flash attention - note that it's non-causal for latent attention
            dropout_p = self.dropout.p if self.training else 0.0
            output = self.flash_attn_unpadded(
                Q, K, V,
                Q_lengths=Q_lengths,
                K_lengths=K_lengths,
                causal=False,
                dropout_p=dropout_p
            )
            return output
        else:
            # Original attention mechanism
            scores = queries @ keys.transpose(-2, -1) * (self.head_dim ** -0.5)
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            return weights @ values

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

    def forward(self, input_tensor, latent=None, use_cache=False):
        outs = [h(input_tensor, latent, use_cache) for h in self.heads]
        x = torch.cat(outs, dim=-1)
        return self.dropout(self.out_proj(x))

    def clear_cache(self):
        for h in self.heads:
            h.clear_cache()

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout_prob):
        super().__init__()
        hidden = int(4 * embed_dim * 2 / 3)
        hidden = (hidden + 7) // 8 * 8
        self.w1 = nn.Linear(embed_dim, hidden, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.dropout(self.w2(gate * up))

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        head_dim = embed_dim // num_heads
        self.attn = MultiHeadLatentAttention(num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec)
        self.norm_latent = nn.LayerNorm(embed_dim)
        self.latent_to_seq = nn.Linear(n_latent_vec, max_seq_len, bias=False)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, latent=None, use_cache=False):
        b, seq_len, _ = x.size()
        h = self.attn(self.norm1(x), latent, use_cache)
        h = self.norm_latent(h).transpose(1,2)
        proj = self.latent_to_seq(h)[:,:,:seq_len].transpose(1,2)
        x = x + self.dropout(proj)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout_prob, latent_dim, n_latent_vec, use_gradient_checkpoint=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, max_seq_len, dropout_prob, latent_dim, n_latent_vec)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        if use_gradient_checkpoint:
            for blk in self.blocks:
                blk.use_checkpointing = True
        self.clear_all_caches()
        self.apply(self._init_weights)
        print(f"Model initialized with {sum(p.numel() for p in self.parameters()):,} params")

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, latent=None, use_cache=False):
        b, seq_len = idx.size()
        x = self.token_emb(idx) + self.pos_emb(torch.arange(seq_len, device=idx.device))
        for blk in self.blocks:
            x = blk(x, latent, use_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            loss = F.cross_entropy(logits_flat, targets.view(-1), ignore_index=-1)
        return logits, loss

    def clear_all_caches(self):
        for blk in self.blocks:
            blk.attn.clear_cache()

    @torch.no_grad()
    def generate(self, idx, max_new, max_seq_len, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        self.clear_all_caches()
        for _ in range(max_new):
            inp = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
            logits, _ = self(inp, use_cache=True)
            logits = logits[:, -1] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:,[-1]]] = -float('Inf')
            elif top_p:
                sorted_logits, idxs = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum > top_p
                mask[...,1:] &= ~mask[...,:-1]
                logits[mask.scatter(1, idxs, mask)] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        self.train()
        return idx
