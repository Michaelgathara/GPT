import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# from torch.utils.tensorboard import SummaryWriter # ended up not using this much

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
    print("Flash Attention is available!")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention is not available, falling back to standard attention")


class FlashAttentionHead(nn.Module):
    """single head of self-attention using Flash Attention when available"""

    # apparently flash attention is one of those things that can just not be avail

    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob):
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout_prob)
        self.use_flash = HAS_FLASH_ATTN

    def forward(self, input_tensor):
        # input_tensor: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = input_tensor.shape

        keys = self.key_proj(input_tensor)  # shape: (batch_size, seq_len, head_dim)
        queries = self.query_proj(
            input_tensor
        )  # shape: (batch_size, seq_len, head_dim)
        values = self.value_proj(input_tensor)  # shape: (batch_size, seq_len, head_dim)

        if (
            self.use_flash and seq_len <= 1024
        ):  # Flash attention has seq length limitations
            # reshape for flash attention which expects (batch, seqlen, nheads, headdim)
            # for single head, we use nheads=1
            q = queries.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
            k = keys.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
            v = values.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]

            # flash attention with causal mask
            output = flash_attn_func(q, k, v, causal=True)

            # reshape back to original dimensions
            output = output.squeeze(2)  # [batch_size, seq_len, head_dim]
        else:
            # standard attention implementation with explicit causal mask
            attention_scores = (queries @ keys.transpose(-2, -1)) * (
                keys.shape[-1] ** -0.5
            )
            # apply causal masking
            attention_scores = attention_scores.masked_fill(
                self.tril[:seq_len, :seq_len] == 0, float("-inf")
            )
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            output = attention_weights @ values

        return output


class MultiHead(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        head_dim,
        max_seq_len,
        dropout_prob,
        use_flash_attn=False,
    ):
        super().__init__()

        head_class = FlashAttentionHead if (HAS_FLASH_ATTN and use_flash_attn) else Head

        self.heads = nn.ModuleList(
            [
                head_class(embed_dim, head_dim, max_seq_len, dropout_prob)
                for _ in range(num_heads)
            ]
        )

        self.projection = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_tensor):
        head_outputs = [head(input_tensor) for head in self.heads]
        concatenated_heads = torch.cat(head_outputs, dim=-1)
        projected_output = self.projection(concatenated_heads)
        output_tensor = self.dropout(projected_output)
        return output_tensor


class Head(nn.Module):
    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob):
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_tensor):
        batch_size, seq_len, embed_dim = input_tensor.shape

        keys = self.key_proj(input_tensor)
        queries = self.query_proj(input_tensor)
        values = self.value_proj(input_tensor)

        attention_scores = queries @ keys.transpose(-2, -1) * (keys.shape[-1] ** -0.5)
        attention_scores = attention_scores.masked_fill(
            self.tril[:seq_len, :seq_len] == 0, float("-inf")
        )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output_tensor = attention_weights @ values

        return output_tensor


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
        swish = self.w1(input_tensor) * torch.sigmoid(self.w1(input_tensor))
        gate = self.w2(input_tensor)
        x = swish * gate
        x = self.w3(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        max_seq_len,
        dropout_prob,
        use_flash_attn=False,
        is_mla_block=False,
        n_latent_vec=0,
        latent_dim=0,
        mla_dropout_prob=0.0,
    ):
        super().__init__()
        self.use_checkpointing = False
        self.is_mla_block = is_mla_block and (n_latent_vec > 0) and (latent_dim > 0)

        head_dim_sa = embed_dim // num_heads
        self.self_attention = MultiHead(
            num_heads, embed_dim, head_dim_sa, max_seq_len, dropout_prob, use_flash_attn
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        if self.is_mla_block:
            self.mla_cross_attention = MultiHeadLatentAttention(
                num_heads=num_heads,
                query_dim=embed_dim,
                context_dim=latent_dim,
                embed_dim=embed_dim,
                dropout_prob=mla_dropout_prob,
                use_flash_attn=use_flash_attn,
            )
            self.layer_norm_mla_input = nn.LayerNorm(embed_dim)
            self.layer_norm_mla_context = nn.LayerNorm(latent_dim)

        self.feed_forward = FeedForward(embed_dim, dropout_prob)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, input_tensor, latent_context_vectors=None):
        sa_normed_input = self.layer_norm1(input_tensor)
        if self.use_checkpointing and self.training:
            sa_output = torch.utils.checkpoint.checkpoint(
                self.self_attention,
                sa_normed_input,
                use_reentrant=(
                    False if hasattr(torch.utils.checkpoint, "use_reentrant") else True
                ),
            )
        else:
            sa_output = self.self_attention(sa_normed_input)
        x = input_tensor + sa_output

        if self.is_mla_block and latent_context_vectors is not None:
            mla_query_input = self.layer_norm_mla_input(x)
            mla_context_input = self.layer_norm_mla_context(latent_context_vectors)

            if self.use_checkpointing and self.training:
                mla_output = torch.utils.checkpoint.checkpoint(
                    self.mla_cross_attention,
                    mla_query_input,
                    mla_context_input,
                    use_reentrant=(
                        False
                        if hasattr(torch.utils.checkpoint, "use_reentrant")
                        else True
                    ),
                )
            else:
                mla_output = self.mla_cross_attention(
                    mla_query_input, mla_context_input
                )
            x = x + mla_output

        ffn_normed_input = self.layer_norm2(x)
        if self.use_checkpointing and self.training:
            ffwd_output = torch.utils.checkpoint.checkpoint(
                self.feed_forward,
                ffn_normed_input,
                use_reentrant=(
                    False if hasattr(torch.utils.checkpoint, "use_reentrant") else True
                ),
            )
        else:
            ffwd_output = self.feed_forward(ffn_normed_input)
        output_tensor = x + ffwd_output

        return output_tensor


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        max_seq_len,
        dropout_prob,
        use_gradient_checkpoint=False,
        use_flash_attn=False,
        n_latent_vec=0,
        latent_dim=0,
        use_mla_in_blocks=None,
        mla_dropout_prob=0.0,
    ):
        super().__init__()
        self.config_params = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "dropout_prob": dropout_prob,
            "use_flash_attn": use_flash_attn,
            "n_latent_vec": n_latent_vec,
            "latent_dim": latent_dim,
            "mla_dropout_prob": mla_dropout_prob,
        }
        if use_mla_in_blocks is None:
            use_mla_in_blocks = []

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        if n_latent_vec > 0 and latent_dim > 0:
            self.latent_context_vectors = nn.Parameter(
                torch.randn(1, n_latent_vec, latent_dim)
            )
        else:
            self.latent_context_vectors = None

        # Create transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            is_mla_block = i in use_mla_in_blocks
            block = Block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout_prob=dropout_prob,
                use_flash_attn=use_flash_attn,
                is_mla_block=(
                    is_mla_block if self.latent_context_vectors is not None else False
                ),
                n_latent_vec=n_latent_vec,
                latent_dim=latent_dim,
                mla_dropout_prob=mla_dropout_prob,
            )
            self.blocks.append(block)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if use_gradient_checkpoint:
            for block in self.blocks:
                block.use_checkpointing = True

        self.apply(self._init_weights)
        print(f"Model initialized with {self.get_num_params():,} parameters")
        if self.latent_context_vectors is not None:
            print(
                f"Using {n_latent_vec} latent context vectors of dimension {latent_dim}."
            )
            print(f"MLA enabled in blocks: {use_mla_in_blocks}")
            
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

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        idx_cloned = idx.clone()
        token_embeddings = self.token_embedding(idx_cloned)
        positions = torch.arange(seq_len, device=idx.device)
        pos_embeddings = self.position_embedding(positions)
        x = token_embeddings + pos_embeddings

        # Pass latent_context_vectors to each block
        # The block itself will decide if/how to use them
        latents_for_blocks = (
            self.latent_context_vectors.repeat(batch_size, 1, 1)
            if self.latent_context_vectors is not None
            else None
        )

        for block in self.blocks:
            x = block(x, latents_for_blocks)  # Modified block forward signature

        x = self.layer_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            targets = targets.to(idx.device)
            logits_flat = logits.view(batch_size * seq_len, -1)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, idx, max_new_tokens, max_seq_len, temperature=1.0, top_k=None):
        # Make sure idx is long for embedding lookup
        idx = idx.to(dtype=torch.long)

        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            idx_cond = idx[:, -max_seq_len:]

            # Forward pass with appropriate dtype handling
            with torch.amp.autocast("cuda"):
                logits, _ = self(idx_cond)

            # Focus on last time step
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class CrossAttentionHead(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim,
        head_dim,
        dropout_prob,
        use_flash_attn=False,
        max_seq_len_kv=1024,
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, head_dim, bias=False)
        self.key_proj = nn.Linear(context_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(context_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.use_flash = HAS_FLASH_ATTN and use_flash_attn
        # FlashAttention for cross-attention is possible but requires careful handling of inputs (q, k, v packing)
        # For simplicity, this uses standard attention.

    def forward(self, query_input, context_input):
        # query_input: (batch_size, seq_len_q, query_dim)
        # context_input: (batch_size, seq_len_kv, context_dim)
        Q = self.query_proj(query_input)  # (B, T_q, head_dim)
        K = self.key_proj(context_input)  # (B, T_kv, head_dim)
        V = self.value_proj(context_input)  # (B, T_kv, head_dim)

        # Scaled dot-product attention
        attention_scores = (Q @ K.transpose(-2, -1)) * (K.shape[-1] ** -0.5)
        # No causal mask typically applied in cross-attention unless q has causality relative to k/v
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = attention_weights @ V  # (B, T_q, head_dim)
        return output


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        query_dim,
        context_dim,
        embed_dim,
        dropout_prob,
        use_flash_attn=False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList(
            [
                CrossAttentionHead(
                    query_dim, context_dim, head_dim, dropout_prob, use_flash_attn
                )
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(
            embed_dim, embed_dim
        )  # embed_dim is num_heads * head_dim
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query_input, context_input):
        # query_input: (B, T_q, query_dim) -> e.g., token sequence (B, T, n_embd)
        # context_input: (B, T_kv, context_dim) -> e.g., latent vectors (B, n_latent_vec, latent_dim)
        head_outputs = [head(query_input, context_input) for head in self.heads]
        concatenated_heads = torch.cat(head_outputs, dim=-1)  # (B, T_q, embed_dim)
        projected_output = self.projection(concatenated_heads)
        output = self.dropout(projected_output)
        return output
