# keys are very low intrinsic dimension but maybe not values
# curse of dimensionality applies here -> high dimensional space just do not make euclidean sense
'''
whole point of attention is to dot product queries and keys to get different values so some have high values (attend to) and some have low values (ignore these tokens)

Softmax is just a scale so everything adds to 1

Having full length vectors (512) --> almost all attention scores are very similar

Keys and queries need to be low dimensional (<10)

Attention - query from the keys but reality is need positional embeddings (embed each word to dense vector and add embedding)

RoPE (Rotary position Embeddings) very popular (e.g. Llama) -> don't directly modify token embeddings in layers, right before you do attention you apply rotation matrix (matrix multiply) -> number of degrees you rotate depends on token position --> same for keys so you rotate both

    The only thing that matters in terms of how diff they are is how far apart query the key is (if 1 apart -> rotate almost same), (if far, amount of rotation very different) -> transformer learns to measure rotation differences if it wants to know how far the query and key are --> this is how it knows how some word came 5 before current word (or something like that)

    Do this right before you do query and key dot product multiplication
        Storying both in cache causes issue of keys looking very different

MLA solution is to separately do rotation/RoPE info and do base keys
    Vanilla rope rotates actual key
    MLA leaves the actual key alone --> extra rotation info and concatenate it --> model wants to learn how far a token is it just uses the last part that has the position info --> content of word uses left hand side (content portion)

Can say first x bit is meaning and next x bit is rotation part

'''

import sys
import os
import json
import time
import logging
import math
import multiprocessing
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# other imports
from tqdm import tqdm
from functools import partial
from typing import Optional

base_folder = os.path.abspath("..") #needed for accessing custom modules
print(f"Your base folder is: {base_folder}")
sys.path.append(base_folder)

# custom data module imports
from data import get_wikitext_data, clean_textdata, get_fineweb_data

# GPT tokenizer imports
'''
commented out for now to test custom BPE first
'''
# from tokenization import get_tiktoken_tokenizer
# from tokenizers import Tokenizer

# TOKENIZER_PATH = f"{base_folder}/tokenization/custom_tokenizer.json"
# tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# custom tokenizer
from tokenization.custom_tokenizer.trainer import load_tokenizer

tokenizer = load_tokenizer()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctim)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHander("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("transformer_training")


# transformer imports
from transformer_setup import ModelConfig, TransformerModel

config = ModelConfig() # hyperparameters

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
        # cosine decay
        else:
            progress = float(self.current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            lr_scale = 0.5 * (1.0 * math.cos(math.pi * progress))
        
        # apply scale to all param groups in optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale
        
        self.current_iter += 1
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr'] # lr of first parameter group

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

# function to estimate loss
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

                # compute loss
                with torch.amp.autocast('cuda'):
                    _, loss = model(x, y)
                
                if loss.ndim > 0:
                    loss = loss.mean()
                
                losses[split].append(loss.item())
            except StopIteration:
                pass
    
    model.train()

    avg_losses = {split: np.mean(split_losses) if split_losses else 0.0 for split, split_losses in losses.items()}

    return avg_losses

# TODO: edit this to call latent head 

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
        latent_dim=config.latent_dim,
        n_latent_vec=config.n_latent_vec,
        use_gradient_checkpoint=config.gradient_checkpointing,

    )
    
    # move model to device
    model = model.to(device)
    
    # wrap model with DDP
    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False)
    model.device = device 
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay, 
        betas=(config.beta1, config.beta2)
    )
    
    # set initial learning rate for scheduler
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = config.learning_rate
    
    # create learning rate scheduler
    scheduler = CosineWarmupScheduler(optimizer, config.warmup_iters, config.max_iters)
    
    # create gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
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
        logger.info(f"Main loop iteration: {iter_num}")
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
        with torch.amp.autocast('cuda'):
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
        # context_ids = context[0].tolist()
        # context_text = tokenizer.decode(context_ids)
        
        generated_sequence = model.module.generate(context, max_new_tokens=200, max_seq_len=config.block_size)
        generated_ids = generated_sequence[0].tolist()
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # print("Context:", context_text)
        print("Generated text:", decoded_text) 
        
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
    
    vocab_size = tokenizer.get_vocab_size()
    
    dataset = get_wikitext_data()
    # dataset = get_fineweb_data(1)
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
            "input_ids": [tokenizer.encode(text).ids for text in examples["text"]]
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