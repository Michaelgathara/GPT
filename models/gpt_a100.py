import sys
import os
import json
import time
import logging
import math
import multiprocessing
import numpy as np

# torch imports and shit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from functools import partial
from typing import Optional

base_folder = os.path.abspath("..")
print(f"Your base folder is: {base_folder}")
sys.path.append(base_folder)
from data import get_wikitext_data, clean_textdata
from tokenization import get_tiktoken_tokenizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("transformer_training")


try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("Flash Attention is available!")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention is not available, falling back to standard attention")

# changed the config stuff to a class for better changing and stuff
class ModelConfig:
    def __init__(self):
        # model architecture
        self.batch_size = 64                # Batch size per GPU
        self.block_size = 512               # Context size
        self.n_embd = 512                  # Embedding dimension
        self.n_head = 8                    # Number of attention heads
        self.n_layer = 2                   # Number of transformer layers
        self.dropout = 0.1                  # Dropout rate
        
        # training parameters
        self.max_iters = 1000               # Number of iterations
        self.eval_interval = 100            # Evaluation interval
        self.learning_rate = 4e-4           # Learning rate
        self.eval_iters = 5                 # Evaluation iterations
        self.accumulation_steps = 4         # Gradient accumulation steps
        self.warmup_iters = 100             # Learning rate warmup iterations
        
        # Optimization flags
        self.gradient_checkpointing = False  # Use gradient checkpointing
        # Above does not work
        self.use_flash_attn = True          # Use Flash Attention if available
        
        self.checkpoint_dir = 'checkpoints' # Directory to save checkpoints
        self.log_dir = 'logs'               # Directory to save logs
        self.seed = 1337                    # Random seed

config = ModelConfig()

# Custom FlashAttention implementation for Heads
class FlashAttentionHead(nn.Module):
    """single head of self-attention using Flash Attention when available"""
    # apparently flash attention is one of those things that can just not be avail
    
    def __init__(self, embed_dim, head_dim, max_seq_len, dropout_prob):
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout_prob)
        self.use_flash = HAS_FLASH_ATTN

    def forward(self, input_tensor):
        # input_tensor: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = input_tensor.shape
        
        keys = self.key_proj(input_tensor)     # shape: (batch_size, seq_len, head_dim)
        queries = self.query_proj(input_tensor)  # shape: (batch_size, seq_len, head_dim)
        values = self.value_proj(input_tensor)   # shape: (batch_size, seq_len, head_dim)
        
        if self.use_flash and seq_len <= 1024:  # Flash attention has seq length limitations
            # reshape for flash attention which expects (batch, seqlen, nheads, headdim)
            # for single head, we use nheads=1
            q = queries.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
            k = keys.unsqueeze(2)     # [batch_size, seq_len, 1, head_dim]
            v = values.unsqueeze(2)   # [batch_size, seq_len, 1, head_dim]
            
            # flash attention with causal mask
            output = flash_attn_func(q, k, v, causal=True)
            
            # reshape back to original dimensions
            output = output.squeeze(2)  # [batch_size, seq_len, head_dim]
        else:
            # standard attention implementation with explicit causal mask
            attention_scores = (queries @ keys.transpose(-2, -1)) * (keys.shape[-1] ** -0.5)
            # apply causal masking
            attention_scores = attention_scores.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            output = attention_weights @ values
        
        return output


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
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_tensor):
        batch_size, seq_len, embed_dim = input_tensor.shape
        
        keys = self.key_proj(input_tensor)
        queries = self.query_proj(input_tensor)
        values = self.value_proj(input_tensor)
        
        attention_scores = queries @ keys.transpose(-2, -1) * (keys.shape[-1] ** -0.5)
        attention_scores = attention_scores.masked_fill(
            self.tril[:seq_len, :seq_len] == 0, float('-inf')
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
        swish = self.w1(input_tensor) * torch.sigmoid(self.w1(input_tensor) * 1.0)
        gate = self.w2(input_tensor)
        x = swish * gate
        x = self.w3(x)
        return self.dropout(x)


class Block(nn.Module):
    """transformer block with optional gradient checkpointing"""
    
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout_prob, use_flash_attn=False):
        super().__init__()
        head_dim = embed_dim // num_heads
        
        self.self_attention = MultiHead(
            num_heads, embed_dim, head_dim, max_seq_len, dropout_prob, use_flash_attn
        )
        
        self.feed_forward = FeedForward(embed_dim, dropout_prob)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # flag for gradient checkpointing
        self.use_checkpointing = False
    
    def forward(self, input_tensor):
        # use custom forward functions for gradient checkpointing
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0])
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
        """Generate text with more sampling options"""
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            idx_cond = idx[:, -max_seq_len:]
            # Get logits
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


# cosine learning rate scheduler with warmup
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_iters, max_iters):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.current_iter = 0
        
    def step(self):
        # linear warmup
        if self.current_iter < self.warmup_iters:
            lr_scale = min(1.0, float(self.current_iter + 1) / self.warmup_iters)
        # cosine decay phase
        else:
            progress = float(self.current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale
        
        self.current_iter += 1
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class TokenizedDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = x.clone()
        y[:-1] = x[1:]
        return x, y


def get_batch(dataloader):
    for x, y in dataloader:
        yield x, y


@torch.no_grad()
def estimate_loss(model, dataloaders, eval_iters):
    model.eval()
    losses = {}
    
    for split, dataloader in dataloaders.items():
        losses[split] = []
        for _ in range(eval_iters):
            try:
                # get batch from dataloader
                x, y = next(iter(dataloader))
                x, y = x.to(model.device), y.to(model.device)
                
                # Compute loss
                with torch.amp.autocast('cuda'):
                    _, loss = model(x, y)
                
                if loss.ndim > 0:
                    loss = loss.mean()
                
                losses[split].append(loss.item())
            except StopIteration:
                pass
    
    model.train()
    
    # average losses
    avg_losses = {split: np.mean(split_losses) if split_losses else 0.0 
                 for split, split_losses in losses.items()}
    
    return avg_losses


# main training function for a single GPU
def train(gpu_id, config, train_tensor, val_tensor, test_tensor, vocab_size):
    # set up distributed process group
    rank = gpu_id
    world_size = torch.cuda.device_count()
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    # set device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    # set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # create checkpoint directory
    if rank == 0:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=config.log_dir)
    
    # create datasets and samplers
    train_dataset = TokenizedDataset(train_tensor, config.block_size)
    val_dataset = TokenizedDataset(val_tensor, config.block_size)
    test_dataset = TokenizedDataset(test_tensor, config.block_size)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        sampler=val_sampler,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        sampler=test_sampler,
        pin_memory=True
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # create model
    model = TransformerModel(
        vocab_size=vocab_size,
        embed_dim=config.n_embd,
        num_heads=config.n_head,
        num_layers=config.n_layer,
        max_seq_len=config.block_size,
        dropout_prob=config.dropout,
        use_gradient_checkpoint=config.gradient_checkpointing,
        use_flash_attn=config.use_flash_attn
    )
    
    # move model to device
    model = model.to(device)
    
    # wrap model with DDP
    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False)
    model.device = device 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    # set initial learning rate for scheduler
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = config.learning_rate
    
    # create learning rate scheduler
    scheduler = CosineWarmupScheduler(optimizer, config.warmup_iters, config.max_iters)
    
    # create gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # zero gradients
    optimizer.zero_grad()
    
    # initialize training metrics
    iter_num = 0
    best_val_loss = float('inf')
    
    # training loop
    train_iter = iter(train_loader)
    
    # start timer
    start_time = time.time()
    
    # get the number of batches per epoch
    if rank == 0:
        print(f"Total iterations: {config.max_iters}")
        print(f"Batches per epoch: {len(train_loader)}")
        
    tokens_processed = 0
    
    # main training loop
    for iter_num in range(config.max_iters):
        iter_start_time = time.time()
        model.train()
        
        # update sampler for new epoch if needed
        if iter_num % len(train_loader) == 0:
            train_sampler.set_epoch(iter_num // len(train_loader))
        
        # get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        
        # mixed precision forward pass
        with torch.cuda.amp.autocast():
            logits, loss = model(x, y)
        
        if loss.ndim > 0:
            loss = loss.mean()
        
        # normalize loss by accumulation steps
        loss_value = loss.item()
        loss = loss / config.accumulation_steps
        
        # backward pass with scaled loss
        scaler.scale(loss).backward()
        
        # update model if accumulation steps reached
        if (iter_num + 1) % config.accumulation_steps == 0:
            # clip gradients (helps with training stability)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # step optimizer with scaled gradients
            scaler.step(optimizer)
            scaler.update()
            
            # step scheduler
            scheduler.step()
            
            # zero gradients
            optimizer.zero_grad(set_to_none=True)
        
        # update tokens processed
        tokens_processed += config.batch_size * config.block_size * world_size
        
        # logging
        if rank == 0:
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            
            # log basic metrics
            if iter_num % 10 == 0:
                lr = scheduler.get_lr()
                tokens_per_sec = config.batch_size * config.block_size * world_size / iter_time
                
                print(f"Iter {iter_num}: loss {loss_value:.4f}, lr {lr:.6f}, {tokens_per_sec:.2f} tokens/sec")
                
                # log to tensorboard
                writer.add_scalar('training/loss', loss_value, iter_num)
                writer.add_scalar('training/learning_rate', lr, iter_num)
                writer.add_scalar('training/tokens_per_sec', tokens_per_sec, iter_num)
                writer.add_scalar('training/tokens_processed', tokens_processed, iter_num)
            
            # evaluate model
            if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
                loss_dict = estimate_loss(model, dataloaders, config.eval_iters)
                
                print(f"Iter {iter_num}: train loss {loss_dict['train']:.4f}, val loss {loss_dict['val']:.4f}")
                
                # log evaluation metrics
                for split, loss_val in loss_dict.items():
                    writer.add_scalar(f'evaluation/{split}_loss', loss_val, iter_num)
                
                # save model if validation loss improved
                if loss_dict['val'] < best_val_loss:
                    best_val_loss = loss_dict['val']
                    
                    # save checkpoint
                    checkpoint = {
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(config)
                    }
                    
                    checkpoint_path = os.path.join(config.checkpoint_dir, f'best_model.pt')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"New best model saved with val loss: {best_val_loss:.4f}")
                
                # save periodic checkpoint
                if iter_num % (config.eval_interval * 5) == 0:
                    checkpoint = {
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'iter_num': iter_num,
                        'val_loss': loss_dict['val'],
                        'config': vars(config)
                    }
                    
                    checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_{iter_num}.pt')
                    torch.save(checkpoint, checkpoint_path)
    
    # end training
    end_time = time.time()
    total_time = end_time - start_time
    
    if rank == 0:
        print(f"Training completed in {total_time:.2f} seconds")
        
        # generate sample text
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_sequence = model.module.generate(context, max_new_tokens=200, max_seq_len=config.block_size)
        generated_ids = generated_sequence[0].tolist()
        
        print("Training completed!")
        
        # close tensorboard writer
        writer.close()
    
    # clean up
    dist.destroy_process_group()


# main function to setup distributed training
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
def main():    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    num_gpus = torch.cuda.device_count()
    print(f"Training with {num_gpus} GPUs")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    tokenizer = get_tiktoken_tokenizer()
    vocab_size = tokenizer.n_vocab
    
    dataset = get_wikitext_data()
    num_cores = multiprocessing.cpu_count()
    
    def clean_batch(examples):
        cleaned_texts = [clean_textdata(text) for text in examples["text"]]
        cleaned_texts = list(filter(None, cleaned_texts))
        return {"text": cleaned_texts}
    
    cleaned_dataset = dataset.map(
        clean_batch,
        batched=True,
        batch_size=10_000,
        num_proc=num_cores,
        desc="Cleaning text"
    )
    
    logger.info("Tokenizing dataset...")
    def tokenize_batch(examples, tokenizer):
        return {
            "input_ids": [tokenizer.encode(text) for text in examples["text"]]
        }
    
    tokenized_dataset = cleaned_dataset.map(
        tokenize_batch, 
        fn_kwargs={"tokenizer": tokenizer},
        batched=True, 
        batch_size=10_000,
        num_proc=num_cores,
        remove_columns=cleaned_dataset["train"].column_names,
        desc="Tokenizing"
    )
    
    logger.info("Chunking dataset...")
    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        
        total_length = (len(concatenated) // config.block_size) * config.block_size
        concatenated = concatenated[:total_length]
    
        return {"input_ids": [concatenated[i : i + config.block_size] 
                for i in range(0, total_length, config.block_size)]}
    
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True, 
        batch_size=config.block_size, 
        num_proc=num_cores,
        desc="Chunking"
    )
    
    tokenized_dataset_text = lm_dataset.filter(lambda x: any(token != 0 for token in x["input_ids"]))
    
    logger.info("Converting to tensors...")
    train_tensor = np.array(tokenized_dataset_text["train"]["input_ids"], dtype=np.int32)
    val_tensor = np.array(tokenized_dataset_text["validation"]["input_ids"], dtype=np.int32)
    test_tensor = np.array(tokenized_dataset_text["test"]["input_ids"], dtype=np.int32)
    
    train_data = torch.from_numpy(train_tensor).long()
    val_data = torch.from_numpy(val_tensor).long()
    test_data = torch.from_numpy(test_tensor).long()
    
    logger.info(f"Train Data: {train_data.shape}, {train_data.dtype}")
    logger.info(f"Val Data: {val_data.shape}, {val_data.dtype}")
    logger.info(f"Test Data: {test_data.shape}, {test_data.dtype}")
    logger.info(f"Vocabulary size: {vocab_size}")
    
    print(f"Train Data: {train_data.shape}, {train_data.dtype}")
    print(f"Val   Data: {val_data.shape}, {val_data.dtype}")
    print(f"Test  Data: {test_data.shape}, {test_data.dtype}")
    print(f"Vocabulary size: {vocab_size}")
    
    mp.spawn(
        train,
        args=(config, train_data, val_data, test_data, vocab_size),
        nprocs=num_gpus,
        join=True
    )


if __name__ == "__main__":
    main()