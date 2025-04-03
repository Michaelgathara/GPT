import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

class LatentAttentionHead(nn.Module):
    def __init__(self, embed_dim, latent_dim, head_dim, max_seq_len, dropout_prob):
        super().__init__()
        self.latent_proj = nn.Linear(embed_dim, latent_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.wuk = nn.Linear(latent_dim, head_dim, bias=False)
        self.wuv = nn.Linear(latent_dim, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, latent):
        _, seq_len, _ = x.shape

        queries = self.query_proj(x)

        keys = self.wuk(latent)
        values = self.wuv(latent)
        
        attn_scores = queries @ keys.transpose(-2, -1) * (keys.shape[-1] ** -0.5)
        mask = self.tril[:seq_len, :seq_len].unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = attn_weights @ values
        return output

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, latent_dim, head_dim, max_seq_len, dropout_prob):
        super().__init__()

        self.latent_proj = nn.Linear(embed_dim, latent_dim, bias=False)

        self.heads = nn.ModuleList([
            LatentAttentionHead(embed_dim, latent_dim, head_dim, max_seq_len, dropout_prob)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, latent_kv_cache=None, return_latent=False):
        computed_latent = self.latent_proj(x)
        latent = computed_latent if latent_kv_cache is None else x

        if latent_kv_cache is not None:
            latent = torch.cat([latent_kv_cache, latent], dim=1)

        head_outputs = [head(x, latent) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.dropout(self.projection(concatenated))

        if return_latent:
            return output, computed_latent
        return output

# improved FeedForward with SwiGLU activation (better than ReLU)
class FeedForward(nn.Module):
    """feedforward network with SwiGLU activation"""
    
    def __init__(self, embed_dim, dropout_prob):
        super().__init__()
        # SwiGLU architecture (similar to what's used in modern LLMs)
        self.w1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.w2 = nn.Linear(embed_dim, 4 * embed_dim)
        self.w3 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_tensor):
        # SwiGLU activation: SwiGLU(x) = Swish(xW1) ⊗ (xW2)
        w1_out = self.w1(input_tensor)
        swish = w1_out * torch.sigmoid(w1_out)
        gate = self.w2(input_tensor)
        x = swish * gate
        x = self.w3(x)
        return self.dropout(x)


class Block(nn.Module):
    """transformer block with optional gradient checkpointing"""
    
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout_prob, latent_dim):
        super().__init__()
        head_dim = embed_dim // num_heads
        
        self.self_attention = MultiHeadLatentAttention(
            num_heads, embed_dim, latent_dim, head_dim, max_seq_len, dropout_prob
        )
        
        self.feed_forward = FeedForward(embed_dim, dropout_prob)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # flag for gradient checkpointing
        self.use_checkpointing = False
    
    def forward(self, input_tensor, latent_kv_cache=None, return_latent=False):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        normed_input1 = self.layer_norm1(input_tensor)

        if self.use_checkpointing and self.training:
            attn_output, new_latent = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.self_attention),
                normed_input1,
                latent_kv_cache,
                return_latent
            )
        else:
            attn_output, new_latent = self.self_attention(
                normed_input1, latent_kv_cache=latent_kv_cache, return_latent=True
            )

        residual1 = input_tensor + attn_output

        normed_input2 = self.layer_norm2(residual1)

        if self.use_checkpointing and self.training:
            ffwd_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.feed_forward),
                normed_input2
            )
        else:
            ffwd_output = self.feed_forward(normed_input2)

        output_tensor = residual1 + ffwd_output

        if return_latent:
            return output_tensor, new_latent
        return output_tensor

class TransformerModel(nn.Module):
    """Transformer-based language model using Multi-Head Latent Attention"""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len,
                dropout_prob, latent_dim, use_gradient_checkpoint=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Create blocks with MLA
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, max_seq_len, dropout_prob, latent_dim)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if use_gradient_checkpoint:
            for block in self.blocks:
                block.use_checkpointing = True

        self.apply(self._init_weights)
        print(f"Model initialized with {self.get_num_params():,} parameters")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None, latent_kv_cache=None, return_latent_cache=False):
        batch_size, seq_len = idx.shape
        token_embeddings = self.token_embedding(idx)
        positions = torch.arange(seq_len, device=idx.device)
        pos_embeddings = self.position_embedding(positions)
        x = token_embeddings + pos_embeddings

        # Initialize updated cache as list
        updated_latent_cache = [] if return_latent_cache else None

        for i, block in enumerate(self.blocks):
            # Get per-layer cache if available
            cache_i = latent_kv_cache[i] if latent_kv_cache is not None else None
            x, latent_out = block(x, latent_kv_cache=cache_i, return_latent=True)

            if return_latent_cache:
                if cache_i is not None:
                    latent_out = torch.cat([cache_i, latent_out], dim=1)  # [B, T_cache + 1, D_latent]
                updated_latent_cache.append(latent_out)

        x = self.layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(batch_size * seq_len, -1)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)

        if return_latent_cache:
            return logits, updated_latent_cache
        else:
            return logits, loss

    def generate(self, idx, max_new_tokens, max_seq_len, temperature=1.0, top_k=None):
<<<<<<< HEAD
        latent_kv_cache = None  # initialize empty cache

=======
        # Make sure idx is long for embedding lookup
        idx = idx.to(dtype=torch.long)
        
>>>>>>> fdb740de7918c194fdc6aae1f0382436b13495d4
        for _ in range(max_new_tokens):
            # Crop context
            idx_cond = idx[:, -max_seq_len:]
            
<<<<<<< HEAD
            # Forward pass with cache (cache grows every step)
            logits, latent_kv_cache = self.forward(idx_cond, latent_kv_cache=latent_kv_cache, return_latent_cache=True)
            
            # Get last token logits
=======
            # Forward pass with appropriate dtype handling
            with torch.amp.autocast('cuda'):
                logits, _ = self(idx_cond)
                
            # Focus on last time step
>>>>>>> fdb740de7918c194fdc6aae1f0382436b13495d4
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# try:
#     from flash_attn import flash_attn_func
#     HAS_FLASH_ATTN = True
#     print("Flash Attention is available!")
# except ImportError:
#     HAS_FLASH_ATTN = False
#     print("Flash Attention is not available, falling back to standard attention")

# class FlashAttentionHead(nn.Module):
#     """single head of self-attention using Flash Attention when available"""
#     # apparently flash attention is one of those things that can just not be avail
    
#     def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob):
#         super().__init__()
#         self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
#         self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
#         self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
#         self.dropout = nn.Dropout(dropout_prob)
#         self.use_flash = HAS_FLASH_ATTN

#     def forward(self, input_tensor):
#         # input_tensor: (batch_size, seq_len, embed_dim)
#         batch_size, seq_len, embed_dim = input_tensor.shape
        
#         keys = self.key_proj(input_tensor)     # shape: (batch_size, seq_len, head_dim)
#         queries = self.query_proj(input_tensor)  # shape: (batch_size, seq_len, head_dim)
#         values = self.value_proj(input_tensor)   # shape: (batch_size, seq_len, head_dim)
        
#         if self.use_flash and seq_len <= 1024:  # Flash attention has seq length limitations
#             # reshape for flash attention which expects (batch, seqlen, nheads, headdim)
#             # for single head, we use nheads=1
#             q = queries.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
#             k = keys.unsqueeze(2)     # [batch_size, seq_len, 1, head_dim]
#             v = values.unsqueeze(2)   # [batch_size, seq_len, 1, head_dim]
            
#             # flash attention with causal mask
#             output = flash_attn_func(q, k, v, causal=True)
            
#             # reshape back to original dimensions
#             output = output.squeeze(2)  # [batch_size, seq_len, head_dim]
#         else:
#             # standard attention implementation with explicit causal mask
#             attention_scores = (queries @ keys.transpose(-2, -1)) * (keys.shape[-1] ** -0.5)
#             # apply causal masking
#             attention_scores = attention_scores.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
#             attention_weights = F.softmax(attention_scores, dim=-1)
#             attention_weights = self.dropout(attention_weights)
#             output = attention_weights @ values
        
#         return output

# class MultiHead(nn.Module):
#     def __init__(self, num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, use_flash_attn=False):
#         super().__init__()
        
#         head_class = FlashAttentionHead if (HAS_FLASH_ATTN and use_flash_attn) else Head
        
#         self.heads = nn.ModuleList([
#             head_class(embed_dim, head_dim, max_seq_len, dropout_prob)
#             for _ in range(num_heads)
#         ])
        
#         self.projection = nn.Linear(num_heads * head_dim, embed_dim)
#         self.dropout = nn.Dropout(dropout_prob)
    
#     def forward(self, input_tensor):
#         head_outputs = [head(input_tensor) for head in self.heads]
#         concatenated_heads = torch.cat(head_outputs, dim=-1)
#         projected_output = self.projection(concatenated_heads)
#         output_tensor = self.dropout(projected_output)
#         return output_tensor

# class Head(nn.Module):    
#     def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob):
#         super().__init__()
#         self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
#         self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
#         self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, input_tensor):
#         batch_size, seq_len, embed_dim = input_tensor.shape
        
#         keys = self.key_proj(input_tensor)
#         queries = self.query_proj(input_tensor)
#         values = self.value_proj(input_tensor)
        
#         attention_scores = queries @ keys.transpose(-2, -1) * (keys.shape[-1] ** -0.5)
#         attention_scores = attention_scores.masked_fill(
#             self.tril[:seq_len, :seq_len] == 0, float('-inf')
#         )
        
#         attention_weights = F.softmax(attention_scores, dim=-1)
#         attention_weights = self.dropout(attention_weights)
#         output_tensor = attention_weights @ values
        
#         return output_tensor

# class TransformerModel(nn.Module):
#     """transformer-based language model with gradient checkpointing support"""
    
#     def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout_prob, 
#                 use_gradient_checkpoint=False, use_flash_attn=False):
#         super().__init__()
#         # token embedding
#         self.token_embedding = nn.Embedding(vocab_size, embed_dim)
#         # position embedding
#         self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
#         # create transformer blocks
#         self.blocks = nn.ModuleList([
#             Block(embed_dim, num_heads, max_seq_len, dropout_prob, use_flash_attn)
#             for _ in range(num_layers)
#         ])
        
#         # final layer norm
#         self.layer_norm = nn.LayerNorm(embed_dim)
#         # language modeling head
#         self.lm_head = nn.Linear(embed_dim, vocab_size)
        
#         # apply gradient checkpointing to all blocks if enabled
#         if use_gradient_checkpoint:
#             for block in self.blocks:
#                 block.use_checkpointing = True
        
#         # initialize weights (important for stable training)
#         self.apply(self._init_weights)
        
#         # log model size (helpful for debugging)
#         print(f"Model initialized with {self.get_num_params():,} parameters")
    
#     def get_num_params(self):
#         return sum(p.numel() for p in self.parameters())
    
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#         elif isinstance(module, nn.LayerNorm):
#             torch.nn.init.zeros_(module.bias)
#             torch.nn.init.ones_(module.weight)

#     def forward(self, idx, targets=None):
#         batch_size, seq_len = idx.shape
        
#         # token embeddings
#         token_embeddings = self.token_embedding(idx)
        
#         # positional embeddings
#         positions = torch.arange(seq_len, device=idx.device)
#         pos_embeddings = self.position_embedding(positions)
        
#         # combine token and positional embeddings
#         x = token_embeddings + pos_embeddings
        
#         # pass through transformer blocks
#         for block in self.blocks:
#             x = block(x)
        
#         # apply final layer norm
#         x = self.layer_norm(x)
        
#         # compute logits
#         logits = self.lm_head(x)
        
#         # compute loss if targets provided
#         loss = None
#         if targets is not None:
#             logits_flat = logits.view(batch_size * seq_len, -1)
#             targets_flat = targets.view(batch_size * seq_len)
#             loss = F.cross_entropy(logits_flat, targets_flat)
        
#         return logits, loss

#     def generate(self, idx, max_new_tokens, max_seq_len, temperature=1.0, top_k=None):
#         """Generate text with more sampling options"""
#         for _ in range(max_new_tokens):
#             # Crop context to max_seq_len
#             idx_cond = idx[:, -max_seq_len:]
#             # Get logits
#             logits, _ = self(idx_cond)
#             # Focus on last time step
#             logits = logits[:, -1, :] / temperature
            
#             # Optional top-k sampling
#             if top_k is not None:
#                 v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#                 logits[logits < v[:, [-1]]] = -float('Inf')
            
#             # Get probabilities
#             probs = F.softmax(logits, dim=-1)
#             # Sample
#             idx_next = torch.multinomial(probs, num_samples=1)
#             # Append
#             idx = torch.cat((idx, idx_next), dim=1)
        
#         return idx