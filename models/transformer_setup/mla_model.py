import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("Flash Attention is available!")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention is not available, falling back to standard attention")

'''
custom linear function for mla
'''
# class QuantizedLinear(nn.Linear):
#     def __init__(self, in_features, out_features, bias: bool = False, dtype=torch.bfloat16):
#         super().__init__(in_features, out_features, bias)
        
#         self.dtype = dtype
#         self.weight.data = self.weight.data.to(dtype)
        
#         # Check for quantization condition (mocking it for now as dtype is bfloat16)
#         if self.weight.element_size() == 1:  # This would be true for quantized weights
#             scale_out_features = (out_features + block_size - 1) // block_size
#             scale_in_features = (in_features + block_size - 1) // block_size
#             self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
#         else:
#             self.register_parameter("scale", None)
#         if bias:
#             self.bias = nn.Parameter(torch.empty(out_features))
#         else:
#             self.register_parameter("bias", None)

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         # Add dequantization logic if needed here
#         # For now, we use standard F.linear unless quantization is active
#         return F.linear(input, self.weight, self.bias)

class LatentAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()

        # Linear 1st arg is original dimensions and projects to 2nd arg lower dimension

        # Latent tokens (learnable parameter)
        self.latents = nn.Parameter(torch.randn(n_latent_vec, latent_dim))

        # Linear transformation of input tokens to K/V space
        self.key_in = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_in = nn.Linear(embed_dim, head_dim, bias=False)

        # Linear transformation of latent tokens to query space
        self.query_in = nn.Linear(latent_dim, head_dim, bias=False)
        

        # TODO: Used in flash transformer, maybe not necessary
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        
        self.dropout = nn.Dropout(dropout_prob)

        self.query_cache = None
        
        '''Our original implementation didn't use latent_dim properly. In MLA (Multi-headed Latent Attention), latent_dim represents the dimensionality of each latent query vector, creating a lower-dimensional bottleneck that reduces computation while maintaining model capacity. This is similar to DeepSeek's implementation where latent vectors act as learnable parameters that capture important patterns in the data.'''

        def forward(self, input_tensor, latent=None, use_cache=False):
            '''
            @param input_tensor - batch_size, seq_len, embed_dim --> keys and values
            '''
            _, seq_len, _ = input_tensor.shape

            # Expand latent to match batch size if latent is not passed as an argument
            if latent is None:
                latent = self.latents.unsqueeze(0).expand(input_tensor.size(0), -1, -1)
            
            if use_cache and self.query_cache is not None:
                queries = self.query_cache
            else:
                queries = self.query_in(latent)
                if use_cache:
                    self.query_cache = queries

            keys = self.key_in(input_tensor)
            values = self.values_in(input_tensor)

            attention_scores = queries @ keys.transpose(-2, -1) * (keys.shape[-1] ** -0.5)
            attention_scores = attention_scores.masked_fill(
                self.tril[:seq_len, :seq_len] == 0, float('-inf')
            )
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            output_tensor = attention_weights @ values
            return output_tensor
        
        def clear_cache(self):
            self.query_cache = None

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()

        self.heads = nn.ModuleList([LatentAttentionHead(embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec) for _ in range(num_heads)
        ])

        # For MultiHeadLatentAttention, we need to project from the concatenated head outputs
        # Each LatentAttentionHead outputs tensors of shape (batch_size, n_latent_vec, head_dim)
        # After concatenation, we have (batch_size, n_latent_vec, num_heads * head_dim)
        self.out_projection = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

        def forward(self, input_tensor, latent=None, use_cache=False):
            head_outputs = [head(input_tensor, latent, use_cache) for head in self.heads]

            concat_heads = torch.cat(head_outputs, dim=-1)
            return self.out_projection(concat_heads)
        
        def clear_cache(self):
            for head in self.heads:
                head.clear_cache()



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
    
    # TODO: may need to edit this
    def forward(self, input_tensor):
        # SwiGLU activation: SwiGLU(x) = Swish(xW1) ⊗ (xW2)
        swish = self.w1(input_tensor) * torch.sigmoid(self.w1(input_tensor))
        gate = self.w2(input_tensor)
        x = swish * gate
        x = self.w3(x)
        return self.dropout(x)


class Block(nn.Module):
    """transformer block with optional gradient checkpointing"""
    
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout_prob, latent_dim, n_latent_vec):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        head_dim = embed_dim // num_heads
        
        self.self_attention = MultiHeadLatentAttention(
            num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, latent_dim, n_latent_vec
        )
        
        self.feed_forward = FeedForward(embed_dim, dropout_prob)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        # flag for gradient checkpointing
        self.use_checkpointing = False

    
    def forward(self, input_tensor, latent=None, use_cache=False):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                # input = input_tensor, latent, and use_cache
                return module(inputs[0], inputs[1] if len(inputs) > 1 else None, inputs[2] if len(inputs) > 2 else False)
            return custom_forward
        
        # layer norm and attention with residual connection
        normed_input1 = self.layer_norm1(input_tensor)
        
        if self.use_checkpointing and self.training:
            attn_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.self_attention),
                normed_input1,
                use_reentrant=False
            )
        else:
            attn_output = self.self_attention(normed_input1)
            
        residual1 = input_tensor + attn_output
        
        # layer norm and feedforward with residual connection
        normed_input2 = self.layer_norm2(residual1)
        
        if self.use_checkpointing and self.training:
            ffwd_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.feed_forward),
                normed_input2
            )
        else:
            ffwd_output = self.feed_forward(normed_input2)
            
        output_tensor = residual1 + ffwd_output
        
        return output_tensor


class TransformerModel(nn.Module):
    """transformer-based language model with gradient checkpointing support"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout_prob, 
                 use_gradient_checkpoint=False, use_flash_attn=False):
        super().__init__()
        # token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # position embedding
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # create transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, max_seq_len, dropout_prob, use_flash_attn)
            for _ in range(num_layers)
        ])
        
        # final layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        # language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
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
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        
        # token embeddings
        token_embeddings = self.token_embedding(idx)
        
        # positional embeddings
        positions = torch.arange(seq_len, device=idx.device)
        pos_embeddings = self.position_embedding(positions)
        
        # combine token and positional embeddings
        x = token_embeddings + pos_embeddings
        
        # pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # apply final layer norm
        x = self.layer_norm(x)
        
        # compute logits
        logits = self.lm_head(x)
        
        # compute loss if targets provided
        loss = None
        if targets is not None:
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
            with torch.amp.autocast('cuda'):
                logits, _ = self(idx_cond)
                
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx