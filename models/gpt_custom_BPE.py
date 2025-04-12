import sys
import os
import json
import time
import logging
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from data import get_llama_nemotron_data, clean_textdata
from tokenization import load_tokenizer

tokenizer = load_tokenizer()

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

from transformer_setup import ModelConfig, FeedForward, Block, TransformerModel
config = ModelConfig()


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
        latent_dim=config.latent_dim,
        n_latent_vec=config.n_latent_vec
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

def process_batch(args):
    dataset, start_idx, end_idx = args
    try:
        batch_data = []
        chunk_size = 10000  # Process 10k examples at a time
        
        for chunk_start in range(start_idx, end_idx, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_idx)
            chunk_data = dataset.select(range(chunk_start, chunk_end))['input_ids']
            batch_data.extend(chunk_data)
        
        if not batch_data:
            return torch.zeros((0, config.block_size), dtype=torch.long)
        
        batch_data = [ids for ids in batch_data if ids]
        
        if not batch_data:
            return torch.zeros((0, config.block_size), dtype=torch.long)
        
        padded_batch = []
        for seq in batch_data:
            if len(seq) > config.block_size:
                padded_batch.append(seq[:config.block_size])
            elif len(seq) < config.block_size:
                padded_seq = seq + [0] * (config.block_size - len(seq))
                padded_batch.append(padded_seq)
            else:
                padded_batch.append(seq)
        
        return torch.tensor(padded_batch, dtype=torch.long)
    except Exception as e:
        print(f"Error in process_batch: {e} for range {start_idx}-{end_idx}")
        raise
    
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
    print(f"Vocab size: {vocab_size}")
    
    dataset = get_llama_nemotron_data()
    print(f"Dataset: {dataset}")
    num_cores = multiprocessing.cpu_count()
    
    def prepare_training_text(example):
        if isinstance(example["input"], list):
            input_text = " ".join([str(item) for item in example["input"]])
        else:
            input_text = str(example["input"])
        
        if isinstance(example["output"], list):
            output_text = " ".join([str(item) for item in example["output"]])
        else:
            output_text = str(example["output"])
        
        full_text = input_text + " " + output_text
        
        return {
            "text": full_text
        }
    
    processed_dataset = {}
    for split in dataset:
        processed_dataset[split] = dataset[split].map(
            prepare_training_text,
            remove_columns=dataset[split].column_names 
        )
    
    logger.info("Tokenizing dataset...")
    def tokenize_batch(examples, tokenizer):
        texts = [str(text) for text in examples["text"]]
        
        encoded = []
        for text in texts:
            try:
                encoded.append(tokenizer.encode(text).ids)
            except Exception as e:
                print(f"Error tokenizing text: {e}")
                encoded.append([])
        
        return {
            "input_ids": encoded
        }
    
    tokenized_dataset = {}
    for split in processed_dataset:
        tokenized_dataset[split] = processed_dataset[split].map(
            tokenize_batch, 
            fn_kwargs={"tokenizer": tokenizer},
            batched=True,
            batch_size=10_000,
            num_proc=num_cores,
            remove_columns=processed_dataset[split].column_names,
            desc=f"Tokenizing {split}"
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
    
    lm_dataset = {}
    for split in tokenized_dataset:
        lm_dataset[split] = tokenized_dataset[split].map(
            group_texts,
            batched=True, 
            batch_size=1000, 
            num_proc=num_cores,
            desc=f"Chunking {split}"
        )
    
    print(f"Dataset: \n{lm_dataset}")
    

    def convert_to_tensor_batches(dataset, num_workers=None, max_retries=3):
        # Use all available cores except one (to keep system responsive)
        if num_workers is None:
            num_workers = multiprocessing.cpu_count() - 1
        
        total_length = len(dataset)
        optimal_batch_size = max(1, total_length // (num_workers * 2))  # Ensure at least 2 batches per worker
        
        num_batches = (total_length + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"Converting dataset with {total_length} examples using {num_workers} workers")
        print(f"Using batch size of {optimal_batch_size} with {num_batches} batches")
        
        batch_args = []
        for i in range(0, total_length, optimal_batch_size):
            end_idx = min(i + optimal_batch_size, total_length)
            batch_args.append((dataset, i, end_idx))
        
        tensors = []
        failed_batches = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_batch, args): args for args in batch_args}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                            desc="Converting to tensors", unit="batch"):
                args = futures[future]
                try:
                    tensor_batch = future.result()
                    if tensor_batch.shape[0] > 0:  # Only add non-empty tensors
                        tensors.append(tensor_batch)
                except Exception as e:
                    print(f"Batch processing failed: {e}")
                    failed_batches.append(args)
        
        if failed_batches:
            print(f"Retrying {len(failed_batches)} failed batches sequentially...")
            for retry in range(max_retries):
                if not failed_batches:
                    break
                    
                still_failed = []
                for args in tqdm(failed_batches, desc=f"Retry {retry+1}/{max_retries}", unit="batch"):
                    try:
                        tensor_batch = process_batch(args)
                        if tensor_batch.shape[0] > 0:
                            tensors.append(tensor_batch)
                    except Exception as e:
                        print(f"Batch still failing: {e}")
                        still_failed.append(args)
                
                failed_batches = still_failed
        
        if tensors:
            try:
                result = torch.cat(tensors, dim=0)
                print(f"Final tensor shape: {result.shape}")
                return result
            except RuntimeError as e:
                print(f"Error concatenating tensors: {e}")
                # Fallback solution
                max_shape = max(t.shape[1] for t in tensors)
                compatible_tensors = []
                
                for t in tensors:
                    if t.shape[1] == max_shape:
                        compatible_tensors.append(t)
                    else:
                        # Pad or trim tensor to match max_shape
                        if t.shape[1] < max_shape:
                            padding = torch.zeros((t.shape[0], max_shape - t.shape[1]), dtype=t.dtype)
                            padded_t = torch.cat([t, padding], dim=1)
                            compatible_tensors.append(padded_t)
                        else:
                            compatible_tensors.append(t[:, :max_shape])
                
                return torch.cat(compatible_tensors, dim=0)
        else:
            return torch.tensor([], dtype=torch.long)

    val_size = len(lm_dataset['science']) // 2

    num_cores_to_use = multiprocessing.cpu_count() - 1

    train_data = convert_to_tensor_batches(
        lm_dataset['code'],
        num_workers=num_cores_to_use
    )

    val_data = convert_to_tensor_batches(
        lm_dataset['science'].select(range(val_size)),
        num_workers=num_cores_to_use
    )

    test_data = convert_to_tensor_batches(
        lm_dataset['science'].select(range(val_size, len(lm_dataset['science']))),
        num_workers=num_cores_to_use
    )

    print(f"Train Data: {train_data.shape}, {train_data.dtype}")
    print(f"Val   Data: {val_data.shape}, {val_data.dtype}")
    print(f"Test  Data: {test_data.shape}, {test_data.dtype}")
    
    mp.spawn(
        train,
        args=(config, train_data, val_data, test_data, vocab_size),
        nprocs=num_gpus,
        join=True
    )


if __name__ == "__main__":
    main()