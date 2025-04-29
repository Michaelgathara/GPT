"""Hugging Face compatible implementation of CustomTransformerModel."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class CustomTransformerConfig(PretrainedConfig):
    model_type = "Custom_Flash_Attention_Transformer"
    
    def __init__(
        self, 
        vocab_size=50257,
        embed_dim=1536,
        num_heads=12,
        num_layers=24,
        max_seq_len=1024,
        dropout_prob=0.1,
        use_gradient_checkpoint=True,
        use_flash_attn=False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,         
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout_prob = dropout_prob
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.use_flash_attn = use_flash_attn
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

class FlashAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob):
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout_prob)
        self.use_flash = HAS_FLASH_ATTN

    def forward(self, input_tensor, attention_mask=None):
        batch_size, seq_len, embed_dim = input_tensor.shape
        
        keys = self.key_proj(input_tensor)     
        queries = self.query_proj(input_tensor)  # shape: (batch_size, seq_len, head_dim)
        values = self.value_proj(input_tensor)   # shape: (batch_size, seq_len, head_dim)
        
        if self.use_flash and seq_len <= 8192:  
            q = queries.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
            k = keys.unsqueeze(2)     # [batch_size, seq_len, 1, head_dim]
            v = values.unsqueeze(2)   # [batch_size, seq_len, 1, head_dim]
            
            output = flash_attn_func(q, k, v, causal=True)
            output = output.squeeze(2)  # [batch_size, seq_len, head_dim]
        else:
            attention_scores = (queries @ keys.transpose(-2, -1)) * (keys.shape[-1] ** -0.5)
            causal_mask = self.tril[:seq_len, :seq_len].to(device=input_tensor.device)
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
            
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :]
                attention_scores = attention_scores + attention_mask
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            output = attention_weights @ values
        
        return output

class Head(nn.Module):
    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob):
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        batch_size, seq_len, embed_dim = input_tensor.shape
        
        keys = self.key_proj(input_tensor)
        queries = self.query_proj(input_tensor)
        values = self.value_proj(input_tensor)
        attention_scores = queries @ keys.transpose(-2, -1) * (keys.shape[-1] ** -0.5)
        
        causal_mask = self.tril[:seq_len, :seq_len].to(device=input_tensor.device)
        attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_scores = attention_scores + attention_mask
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output_tensor = attention_weights @ values
        
        return output_tensor


class MultiHead(nn.Module):
    def __init__(self, num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, use_flash_attn=False):
        super().__init__()
        
        head_class = FlashAttentionHead if (HAS_FLASH_ATTN and use_flash_attn) else Head
        self.heads = nn.ModuleList([
            head_class(embed_dim, head_dim, max_seq_len, dropout_prob)
            for _ in range(num_heads)
        ])
        
        self.projection = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_tensor, attention_mask=None):
        head_outputs = [head(input_tensor, attention_mask) for head in self.heads]
        concatenated_heads = torch.cat(head_outputs, dim=-1)
        projected_output = self.projection(concatenated_heads)
        output_tensor = self.dropout(projected_output)
        return output_tensor


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout_prob):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.w2 = nn.Linear(embed_dim, 4 * embed_dim)
        self.w3 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_tensor):
        swish = self.w1(input_tensor) * torch.sigmoid(self.w1(input_tensor))
        gate = self.w2(input_tensor)
        x = swish * gate
        x = self.w3(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout_prob, use_flash_attn=False):
        super().__init__()
        head_dim = embed_dim // num_heads
        
        self.self_attention = MultiHead(
            num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, use_flash_attn
        )
        
        self.feed_forward = FeedForward(embed_dim, dropout_prob)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.use_checkpointing = False
    
    def forward(self, input_tensor, attention_mask=None):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0], inputs[1] if len(inputs) > 1 else None)
            return custom_forward
        
        normed_input1 = self.layer_norm1(input_tensor)
        
        if self.use_checkpointing and self.training:
            attn_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.self_attention),
                normed_input1,
                attention_mask,
                use_reentrant=False
            )
        else:
            attn_output = self.self_attention(normed_input1, attention_mask)
            
        residual1 = input_tensor + attn_output
        normed_input2 = self.layer_norm2(residual1)
        
        if self.use_checkpointing and self.training:
            ffwd_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.feed_forward),
                normed_input2,
                use_reentrant=False
            )
        else:
            ffwd_output = self.feed_forward(normed_input2)
            
        output_tensor = residual1 + ffwd_output
        return output_tensor


class CustomTransformerPreTrainedModel(PreTrainedModel):
    config_class = CustomTransformerConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    
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
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CustomTransformerModel):
            module.gradient_checkpointing = value
            for block in module.blocks:
                block.use_checkpointing = value


class CustomTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        self.blocks = nn.ModuleList([
            Block(
                config.embed_dim, 
                config.num_heads, 
                config.max_seq_len, 
                config.dropout_prob, 
                config.use_flash_attn
            )
            for _ in range(config.num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        self.gradient_checkpointing = config.use_gradient_checkpoint
        if self.gradient_checkpointing:
            for block in self.blocks:
                block.use_checkpointing = True
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_input_embeddings(self):
        return self.token_embedding
    
    def set_input_embeddings(self, value):
        self.token_embedding = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
            
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
            
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = inputs_embeds + position_embeds
        
        extended_attention_mask = None
        if attention_mask is not None:
            # Convert attention mask from [batch, seq_len] to [batch, 1, 1, seq_len]
            # 0 = masked, 1 = not masked
            # Convert to: 0 = not masked, -10000 = masked
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
            
        for block in self.blocks:
            hidden_states = block(hidden_states, extended_attention_mask)
            
        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
            
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )
        
    def generate(
        self,
        input_ids=None,
        max_new_tokens=None,
        max_seq_len=None,
        temperature=1.0,
        top_k=None,
        attention_mask=None,
        **kwargs
    ):
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            
        input_ids = input_ids.to(dtype=torch.long, device=self.lm_head.weight.device)
        
        if max_seq_len is None:
            max_seq_len = self.config.max_seq_len
            
        if max_new_tokens is None:
            max_new_tokens = 100  # Default value
            
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -max_seq_len:]
            
            outputs = self.forward(input_ids=idx_cond, attention_mask=attention_mask)
            logits = outputs.logits
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            
            if idx_next.item() == self.config.eos_token_id:
                break
        
        return input_ids


class CustomTransformerForCausalLM(CustomTransformerPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomTransformerModel(config)
        
        self.post_init()
        
    def get_output_embeddings(self):
        return self.transformer.lm_head
        
    def set_output_embeddings(self, new_embeddings):
        self.transformer.lm_head = new_embeddings
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )