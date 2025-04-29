# models/transformer_setup/mla_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# # Removed unused distributed/dataloader imports from this specific file
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from torch.utils.tensorboard import SummaryWriter

class LatentAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim # Store head_dim

        # Latent tokens (learnable parameter), *0.02 to be consistent with other weights
        self.latents = nn.Parameter(torch.randn(n_latent_vec, latent_dim) * 0.02)

        # Linear transformation of input tokens to K/V space
        self.key_in = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_in = nn.Linear(embed_dim, head_dim, bias=False)

        # Linear transformation of latent tokens to query space
        self.query_in = nn.Linear(latent_dim, head_dim, bias=False)

        self.dropout = nn.Dropout(dropout_prob)

        # --- KV Cache Attributes ---
        self.k_cache = None
        self.v_cache = None
        self.query_cache = None # Keep existing query cache

    def forward(self, input_tensor, latent=None, use_cache=False):
        """
        Forward pass with KV Caching.

        Args:
            input_tensor: Shape (batch_size, seq_len, embed_dim)
            latent: Optional pre-computed latent tensor.
            use_cache: Boolean indicating whether to use KV caching (for generation).

        Returns:
            Output tensor of shape (batch_size, n_latent_vec, head_dim)
        """
        batch_size, seq_len, _ = input_tensor.shape

        # --- Query Computation ---
        # Expand latent to match batch size if latent is not passed as an argument
        if latent is None:
            latent = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Use cached queries if available and requested
        # Note: Queries depend on latents, not input_tensor sequence directly
        if use_cache and self.query_cache is not None:
            queries = self.query_cache
        else:
            queries = self.query_in(latent) # Shape: (batch_size, n_latent_vec, head_dim)
            if use_cache:
                self.query_cache = queries

        # --- Key/Value Computation and Caching ---
        if use_cache:
            # Generation phase: Compute K/V only for the new tokens
            # Assuming input_tensor contains only the *new* token(s) after the first step
            new_keys = self.key_in(input_tensor)     # Shape: (batch_size, seq_len, head_dim)
            new_values = self.value_in(input_tensor) # Shape: (batch_size, seq_len, head_dim)

            # Retrieve cache
            cached_keys = self.k_cache
            cached_values = self.v_cache

            if cached_keys is not None and cached_values is not None:
                # Append new keys/values to the cache
                keys = torch.cat([cached_keys, new_keys], dim=1)
                values = torch.cat([cached_values, new_values], dim=1)

                # --- Crucial: Enforce max_seq_len on the cache ---
                # If cache exceeds max length, truncate the oldest tokens (from the start)
                current_cache_len = keys.shape[1]
                if current_cache_len > self.max_seq_len:
                    keys = keys[:, -self.max_seq_len:, :]
                    values = values[:, -self.max_seq_len:, :]

            else:
                # First step of generation, cache is empty
                keys = new_keys
                values = new_values
                # Ensure the initial prompt doesn't exceed max_seq_len
                if keys.shape[1] > self.max_seq_len:
                    keys = keys[:, -self.max_seq_len:, :]
                    values = values[:, -self.max_seq_len:, :]


            # Update the cache for the next step
            self.k_cache = keys
            self.v_cache = values

        else:
            # Training or non-cached forward pass: Compute K/V for the whole sequence
            # Ensure sequence length doesn't exceed max_seq_len
            if seq_len > self.max_seq_len:
                 input_tensor_truncated = input_tensor[:, -self.max_seq_len:, :]
                 keys = self.key_in(input_tensor_truncated)
                 values = self.value_in(input_tensor_truncated)
            else:
                 keys = self.key_in(input_tensor)   # Shape: (batch_size, seq_len, head_dim)
                 values = self.value_in(input_tensor) # Shape: (batch_size, seq_len, head_dim)

            # Clear cache during non-cached passes
            self.clear_cache()

        # --- Attention Calculation ---
        # Queries attend to the full sequence keys
        # queries shape: (batch_size, n_latent_vec, head_dim)
        # keys shape:    (batch_size, effective_seq_len, head_dim)
        # values shape:  (batch_size, effective_seq_len, head_dim)
        attention_scores = queries @ keys.transpose(-2, -1) * (keys.shape[-1] ** -0.5)
        # Note: No causal mask applied here as queries are from latents, not sequence positions

        attention_weights = F.softmax(attention_scores, dim=-1) # Softmax over key sequence length
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values based on attention weights
        # output = attention_weights @ values
        # attention_weights shape: (batch_size, n_latent_vec, effective_seq_len)
        # values shape:          (batch_size, effective_seq_len, head_dim)
        # output shape:          (batch_size, n_latent_vec, head_dim)
        output_tensor = attention_weights @ values

        return output_tensor

    def clear_cache(self):
        """Clears the KV and query cache."""
        self.k_cache = None
        self.v_cache = None
        self.query_cache = None

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim

        self.heads = nn.ModuleList([
            LatentAttentionHead(embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec)
            for _ in range(num_heads)
        ])

        # Projection maps concatenated head outputs back to embed_dim
        self.out_projection = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_tensor, latent=None, use_cache=False):
        """
        Forward pass for Multi-Head Latent Attention.

        Args:
            input_tensor: Shape (batch_size, seq_len, embed_dim)
            latent: Optional pre-computed latent tensor.
            use_cache: Boolean indicating whether to use KV caching.

        Returns:
            Output tensor of shape (batch_size, n_latent_vec, embed_dim)
        """
        head_outputs = [head(input_tensor, latent, use_cache) for head in self.heads]
        # Each head_output shape: (batch_size, n_latent_vec, head_dim)

        # Concatenate along the last dimension (head_dim)
        concat_heads = torch.cat(head_outputs, dim=-1)
        # concat_heads shape: (batch_size, n_latent_vec, num_heads * head_dim)

        # Project back to embed_dim
        projected_output = self.out_projection(concat_heads)
        # projected_output shape: (batch_size, n_latent_vec, embed_dim)

        return self.dropout(projected_output)

    def clear_cache(self):
        """Clears the cache in all attention heads."""
        for head in self.heads:
            head.clear_cache()


# FeedForward class remains the same (no changes needed here)
class FeedForward(nn.Module):
    """feedforward network with SwiGLU activation"""
    def __init__(self, embed_dim, dropout_prob):
        super().__init__()
        hidden_dim = int(4 * embed_dim * 2 / 3) # Follow Llama 2 SwiGLU sizing
        hidden_dim = (hidden_dim + 7) // 8 * 8 # Make multiple of 8 for efficiency

        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False) # gate_proj
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False) # down_proj
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False) # up_proj
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # SwiGLU activation: silu(xW1) * (xW3)
        gate = F.silu(self.w1(x)) # Swish-like activation (SiLU)
        up = self.w3(x)
        fuse = gate * up
        # Project down and apply dropout
        return self.dropout(self.w2(fuse))


class Block(nn.Module):
    """Transformer block using MultiHeadLatentAttention"""
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        head_dim = embed_dim // num_heads

        # Layer norm applied *before* attention and feedforward (Pre-LN)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadLatentAttention(
            num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec
        )

        # Layer norm before the projection layer
        self.norm_latent = nn.LayerNorm(embed_dim)

        # --- Modified Projection Logic ---
        # This layer projects the attended latent vectors back to the sequence space.
        # Input: (batch, n_latent_vec, embed_dim)
        # Output: (batch, seq_len, embed_dim)
        # We need a linear layer that can map n_latent_vec features to seq_len features
        # for each embedding dimension independently. This suggests applying a Linear
        # layer across the n_latent_vec dimension.
        # Let's use a Linear layer mapping n_latent_vec -> embed_dim (or similar)
        # This part is tricky and might need architecture adjustment based on desired behavior.
        # Original projection: nn.Linear(n_latent_vec, max_seq_len) applied after transpose.
        # Let's try projecting n_latent_vec -> embed_dim first, then map to seq_len.

        # Project latent vectors after attention back towards embed_dim structure if needed
        # The output of attention is (batch, n_latent_vec, embed_dim). This might already
        # be the correct shape to add back, depending on how residual connection is handled.
        # Let's re-evaluate the residual connection path.

        # If the goal is to *replace* the standard self-attention output with the
        # processed latent information projected back, the projection needs careful thought.
        # A simple projection might be nn.Linear(n_latent_vec, max_seq_len) but applied differently.

        # Initialize projection layer
        self.latent_to_seq_proj = nn.Linear(n_latent_vec, max_seq_len, bias=True) # Try this compared to before

        # FeedForward part
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)

        # flag for gradient checkpointing
        self.use_checkpointing = False
        self.max_seq_len = max_seq_len # Store max_seq_len


    def forward(self, input_tensor, latent=None, use_cache=False):
        """
        Forward pass for the Block with Pre-LN and modified projection.

        Args:
            input_tensor: Shape (batch_size, seq_len, embed_dim)
            latent: Optional pre-computed latent tensor.
            use_cache: Boolean indicating whether to use KV caching.

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = input_tensor.shape

        # --- Attention Path ---
        # Apply LayerNorm before attention (Pre-LN)
        normed_input = self.norm1(input_tensor)
        # Get attended latent vectors
        # attn_output shape: (batch_size, n_latent_vec, embed_dim)
        attn_output = self.self_attention(normed_input, latent, use_cache)

        # --- Projection back to Sequence ---
        # Apply LayerNorm to the attended latent vectors
        normed_attn_output = self.norm_latent(attn_output)

        # Project from (batch, n_latent_vec, embed_dim) back to (batch, seq_len, embed_dim)
        # We need to project along the n_latent_vec dimension to match seq_len
        # Reshape/Transpose needed: view as (batch * embed_dim, n_latent_vec) ? No.
        # Transpose: (batch, embed_dim, n_latent_vec)
        project_input = normed_attn_output.transpose(1, 2)

        # NOTE: Testing different implementation where projection layer is initialized in __init__
        # # Dynamically create or resize the projection layer if seq_len changes (unlikely during generation)
        # # Input features: n_latent_vec, Output features: seq_len
        # if not hasattr(self, 'latent_to_seq_proj') or self.latent_to_seq_proj.out_features != seq_len:
        #      # Use config's n_latent_vec for input size, current seq_len for output size
        #      n_latent_vec = self.self_attention.heads[0].latents.shape[0] # Get n_latent_vec from head
        #      self.latent_to_seq_proj = nn.Linear(n_latent_vec, seq_len, bias=True).to(project_input.device, project_input.dtype)

        # Apply projection: output (batch, embed_dim, seq_len)
        projected_latents = self.latent_to_seq_proj(project_input)

        # Take only necessary latents
        projected_latents = projected_latents[:, :, :seq_len]

        # Transpose back: output (batch, seq_len, embed_dim)
        projected_output = projected_latents.transpose(1, 2)

        # First residual connection: Add projected latent info to original input
        residual1 = input_tensor + self.dropout(projected_output)

        # --- FeedForward Path ---
        # Apply LayerNorm before FeedForward (Pre-LN)
        normed_residual1 = self.norm2(residual1)
        ff_output = self.feed_forward(normed_residual1)

        # Second residual connection
        output_tensor = residual1 + self.dropout(ff_output)

        return output_tensor


    # Define a separate function for checkpointing if needed, but forward is usually checkpointed
    # def _forward_impl(self, input_tensor, latent=None, use_cache=False):
    #     # ... implementation from forward ...


class TransformerModel(nn.Module):
    """Transformer model using MLA blocks"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout_prob, latent_dim, n_latent_vec, use_gradient_checkpoint=False):
        super().__init__()
        self.max_seq_len = max_seq_len # Store max_seq_len

        # token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # position embedding
        # Important: Increase position embedding size if max_seq_len can be exceeded
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # create transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, max_seq_len, dropout_prob, latent_dim, n_latent_vec)
            for _ in range(num_layers)
        ])

        # final layer norm
        self.norm = nn.LayerNorm(embed_dim) # Renamed from layer_norm to avoid conflict
        # language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False) # Often no bias in final LM head

        # Weight tying (optional but common)
        self.token_embedding.weight = self.lm_head.weight

        # apply gradient checkpointing to all blocks if enabled
        if use_gradient_checkpoint:
            for block in self.blocks:
                block.use_checkpointing = True

        # initialize weights (important for stable training)
        self.apply(self._init_weights)

        # log model size (helpful for debugging)
        print(f"Model initialized with {self.get_num_params():,} parameters")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        # Initialization strategy (e.g., from GPT-2)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(len(self.blocks))) # Scale std dev by num layers
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def forward(self, idx, targets=None, latent=None, use_cache=None):
        batch_size, seq_len = idx.shape

        # Ensure sequence length doesn't exceed position embedding table size
        if seq_len > self.max_seq_len:
             # This case should ideally be handled by truncation *before* forward pass
             # Or position embedding needs to be larger / handle extrapolation
             idx = idx[:, -self.max_seq_len:] # Keep last max_seq_len tokens
             seq_len = self.max_seq_len
             if targets is not None:
                 targets = targets[:, -self.max_seq_len:] # Truncate targets too


        # token embeddings
        token_embeddings = self.token_embedding(idx) # (batch, seq_len, embed_dim)

        # NOTE: comment out to modify position embedding handling to account for if cache is being used
        # # positional embeddings
        # positions = torch.arange(seq_len, device=idx.device) # (seq_len)
        # pos_embeddings = self.position_embedding(positions) # (seq_len, embed_dim)

        if use_cache:
            pos_ids = torch.arange(seq_len, device=idx.device) + (idx.shape[1] - seq_len)
        else:
            pos_ids = torch.arange(seq_len, device=idx.device)
        
        pos_embeddings = self.position_embedding(pos_ids)

        # combine token and positional embeddings (broadcasting pos_embeddings)
        x = token_embeddings + pos_embeddings # (batch, seq_len, embed_dim)

        # NOTE: Can comment out or edit the scale
        # adding noise during training
        if self.training:
            x = x + torch.randn_like(x) * 0.001

        # pass through transformer blocks
        for block in self.blocks:
            # Checkpointing should be handled within the block's forward
            x = block(x, latent=latent, use_cache=use_cache)

        # apply final layer norm
        x = self.norm(x) # Use self.norm

        # compute logits
        logits = self.lm_head(x) # (batch, seq_len, vocab_size)

        # compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross_entropy (expects N, C)
            logits_flat = logits.view(-1, logits.size(-1)) # (batch * seq_len, vocab_size)
            targets_flat = targets.view(-1)               # (batch * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1) # Use ignore_index if padding exists

        return logits, loss

    @torch.no_grad() # Ensure no gradients are computed during generation
    def generate(self, idx, max_new_tokens, max_seq_len, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text sequences using the model with KV caching.
        """
        self.eval() # Ensure model is in evaluation mode

        # Make sure idx is long tensor and on the correct device
        idx = idx.to(dtype=torch.long, device=self.token_embedding.weight.device)

        # Clear cache in all attention heads before starting generation
        for block in self.blocks:
             if hasattr(block.self_attention, 'clear_cache'):
                  block.self_attention.clear_cache()

        for _ in range(max_new_tokens):
            # --- Context Handling ---
            # Crop context to max_seq_len for the forward pass K/V cache size limit
            # The actual input 'idx' to self() might just be the last token if KV cache is full
            # idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
            # Let the KV cache handle the effective context length within the attention heads

            # Determine the input for the current step.
            # If KV cache is used, only the last token is needed for *new* K/V computation.
            # The forward pass inside LatentAttentionHead handles the cache logic.
            # The input `idx` here represents the full sequence generated so far.
            # We pass the relevant part to the forward method.
            # After the first step, we only need to process the last token to generate the next one.
            is_first_step = not any(hasattr(h, 'k_cache') and h.k_cache is not None for block in self.blocks for h in block.self_attention.heads)

            if is_first_step:
                 idx_input = idx # Process the initial prompt
            else:
                 idx_input = idx[:, -1:] # Process only the last generated token

            # Crop input if it exceeds max_seq_len (relevant mainly for the first step)
            idx_input = idx_input[:, -max_seq_len:]


            # --- Forward Pass ---
            # Use autocast for potential speedup/memory saving
            with torch.amp.autocast(device_type=self.token_embedding.weight.device.type):
                 # Pass use_cache=True to enable KV caching
                 logits, _ = self(idx_input, use_cache=True) # Pass only the necessary input


            # --- Logit Processing ---
            # Focus on the logits for the very last token generated
            logits = logits[:, -1, :] # Shape: (batch_size, vocab_size)

            # Apply temperature scaling
            if temperature != 1.0:
                 logits = logits / temperature

            # --- Top-K / Top-P Sampling ---
            if top_k is not None:
                 v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                 # Set logits below the k-th threshold to -inf
                 logits[logits < v[:, [-1]]] = -float('Inf')
            elif top_p is not None:
                 # Sort logits and compute cumulative probabilities
                 sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                 cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                 # Remove tokens with cumulative probability above the threshold (nucleus)
                 sorted_indices_to_remove = cumulative_probs > top_p
                 # Shift the indices to the right to keep the first token above threshold
                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                 sorted_indices_to_remove[..., 0] = 0 # Never remove the most likely token

                 # Scatter negative infinity onto the original logits
                 indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                 logits[indices_to_remove] = -float('Inf')


            # --- Sampling ---
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample the next token index
            idx_next = torch.multinomial(probs, num_samples=1) # Shape: (batch_size, 1)

            # --- Append and Loop ---
            # Append the sampled token index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        self.train() # Set model back to training mode if needed after generation
        return idx