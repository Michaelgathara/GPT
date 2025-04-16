import os
import sys
import time
import math
import random
import logging
import numpy as np
from tqdm import tqdm 
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset, DataLoader # We will need a custom IterableDataset

base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_folder)

from data.fineweb_data import get_fineweb_data
from tokenization.utils import load_fineweb_tokenizer
from transformer_setup import ModelConfig
from transformer_setup import TransformerModel as MLATransformerModel

log_dir = os.path.join(os.path.dirname(__file__), 'logs_fineweb')
checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints_fineweb')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "fineweb_pretrain.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FineWebPretraining")
workers = 0

# --- Helper Classes ---
# move to utils later

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_iters, max_iters):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.current_iter = 0
        self.initial_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]

    def step(self):
        lr_scale = 1.0 # Default scale
        if self.current_iter < self.warmup_iters:
            lr_scale = min(1.0, float(self.current_iter + 1) / self.warmup_iters)
        elif self.current_iter < self.max_iters: # Only decay if less than max_iters
            progress = float(self.current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            # Ensure progress doesn't go beyond 1.0 causing math domain error
            progress = min(1.0, max(0.0, progress))
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        # else: keep lr at minimum value (zero if cosine ends at pi)

        for i, param_group in enumerate(self.optimizer.param_groups):
             param_group['lr'] = self.initial_lrs[i] * lr_scale

        # Only increment if not past max_iters to keep LR stable after decay
        if self.current_iter < self.max_iters:
            self.current_iter += 1

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0

class FineWebDataset(IterableDataset):
    def __init__(self, dataset_iterable, tokenizer, block_size, 
                 cache_size=10000, seed=1337, shuffle_buffer=1000):
        self.iterable = dataset_iterable
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_size = cache_size
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        # Token IDs cache to avoid re-encoding common texts
        self.token_cache = {}
        self.bos_token_id = tokenizer.token_to_id("[BOS]")
        self.eos_token_id = tokenizer.token_to_id("[EOS]")
        
    def __iter__(self):
        buffer = []
        shuffle_buffer = []
        rng = random.Random(self.seed)
        
        for example in self.iterable:
            if not example or 'text' not in example or not isinstance(example['text'], str):
                continue
                
            try:
                # Check cache first to avoid redundant tokenization
                text = example['text']
                cache_key = hash(text[:100] + text[-100:] if len(text) > 200 else text)
                
                if cache_key in self.token_cache:
                    token_ids = self.token_cache[cache_key]
                else:
                    token_ids = self.tokenizer.encode(text).ids
                    # Limit cache size to prevent memory issues
                    if len(self.token_cache) < self.cache_size:
                        self.token_cache[cache_key] = token_ids
                
                if not token_ids:
                    continue
                    
                buffer.extend(token_ids)
                
                # Generate examples once buffer is large enough
                while len(buffer) >= self.block_size + 1:
                    # If shuffle_buffer enabled, add to shuffle buffer
                    if self.shuffle_buffer > 0:
                        chunk = buffer[:self.block_size + 1]
                        shuffle_buffer.append((chunk[:-1], chunk[1:]))
                        buffer = buffer[self.block_size:]
                        
                        # When shuffle buffer is full, yield a random example
                        if len(shuffle_buffer) >= self.shuffle_buffer:
                            idx = rng.randint(0, len(shuffle_buffer) - 1)
                            x, y = shuffle_buffer[idx]
                            shuffle_buffer[idx] = shuffle_buffer[-1]
                            shuffle_buffer.pop()
                            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
                    else:
                        # Direct yield without shuffling
                        chunk = buffer[:self.block_size + 1]
                        yield (torch.tensor(chunk[:-1], dtype=torch.long), 
                               torch.tensor(chunk[1:], dtype=torch.long))
                        buffer = buffer[self.block_size:]
                        
            except Exception as e:
                logging.warning(f"Skipping example: {e}")
                
        # Drain shuffle buffer
        while shuffle_buffer:
            idx = rng.randint(0, len(shuffle_buffer) - 1)
            x, y = shuffle_buffer[idx]
            shuffle_buffer[idx] = shuffle_buffer[-1]
            shuffle_buffer.pop()
            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

@torch.no_grad()
def estimate_loss(model, val_dataloader, eval_iters, device):
    model.eval()
    total_loss = 0.0
    iters_run = 0
    logger.info(f"Estimating validation loss over {eval_iters} iterations...")
    try:
        val_iterator = iter(val_dataloader)
        for i in range(eval_iters):
            try:
                x, y = next(val_iterator)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    logits, loss = model(x, targets=y) # Pass targets=y
                    if loss.ndim > 0: loss = loss.mean()
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    iters_run += 1
                else:
                    logger.warning(f"Estimate Loss: Encountered non-finite loss in iteration {i}. Skipping.")
            except StopIteration:
                logger.warning("Validation dataloader exhausted before completing eval_iters.")
                break
            except Exception as e:
                 logger.error(f"Error during validation iteration {i}: {e}")
                 break
    except Exception as e:
         logger.error(f"Error creating validation iterator: {e}")

    model.train()
    if iters_run > 0:
        avg_loss = total_loss / iters_run
        logger.info(f"Average validation loss over {iters_run} iterations: {avg_loss:.4f}")
        return avg_loss
    else:
        logger.warning("No validation iterations completed successfully.")
        return float('inf')


def main():
    logger.info("--- Starting FineWeb Pre-training Script ---")

    config = ModelConfig() # Load config from params.py
    logger.info("Loaded ModelConfig from transformer_setup.params:")
    logger.info(f"  batch_size (per device): {config.batch_size}")
    logger.info(f"  block_size: {config.block_size}")
    logger.info(f"  n_layer: {config.n_layer}")
    logger.info(f"  n_head: {config.n_head}")
    logger.info(f"  n_embd: {config.n_embd}")
    logger.info(f"  dropout: {config.dropout}")
    logger.info(f"  learning_rate: {config.learning_rate}")
    logger.info(f"  max_iters: {config.max_iters}")
    logger.info(f"  warmup_iters: {config.warmup_iters}")
    logger.info(f"  weight_decay: {config.weight_decay}")
    logger.info(f"  accumulation_steps: {config.accumulation_steps}")
    logger.info(f"  gradient_checkpointing: {config.gradient_checkpointing}")
    logger.info(f"  use_flash_attn: {config.use_flash_attn}") # Log Flash Attention setting
    logger.info(f"  latent_dim: {config.latent_dim}")
    logger.info(f"  n_latent_vec: {config.n_latent_vec}")
    global checkpoint_dir, log_dir
    checkpoint_dir = getattr(config, 'checkpoint_dir', checkpoint_dir)
    log_dir = getattr(config, 'log_dir', log_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f"  Checkpoint Dir: {checkpoint_dir}")
    logger.info(f"  Log Dir: {log_dir}")

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        logger.warning("CUDA not available. Training on CPU is not feasible.")
        device = torch.device("cpu")

    # Set seed using value from config
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    logger.info(f"Set random seed to: {config.seed}")

    # --- Tokenizer ---
    tokenizer = load_fineweb_tokenizer()
    config.vocab_size = tokenizer.get_vocab_size()

    logger.info("Setting up data loading using streaming IterableDataset...")
    try:
        # Load the main training stream
        logger.info("Loading FineWeb 'default' stream...")
        full_stream = get_fineweb_data(name="default", streaming=True, split="train")

        # --- Train/Validation Split Strategy (Simplified for Streaming) ---
        # Option 1 (No True Split): Use the same stream but evaluate periodically.
        # Option 2 (Interleaving - Complex): Use itertools or similar to split.
        # Option 3 (Dedicated Val Set): If a separate val set exists or can be downloaded.

        # Let's use Option 1 for simplicity now: Train and Validate on the same stream,
        # relying on estimate_loss evaluating only eval_iters batches.
        # This isn't ideal (validation sees training data) but avoids complex stream splitting.
        # For rigorous results, a separate validation stream/file is better.
        train_iterable = full_stream
        val_iterable = full_stream # Use the same iterable for validation estimates

        logger.info("Initializing FineWebDataset for training...")
        train_dataset = FineWebDataset(train_iterable, tokenizer, config.block_size, seed=config.seed)

        # For validation, we create a separate instance, though it iterates the same underlying stream.
        # estimate_loss will only consume eval_iters batches from it.
        logger.info("Initializing FineWebDataset for validation...")
        val_dataset = FineWebDataset(val_iterable, tokenizer, config.block_size, seed=config.seed + 1) # Use diff seed if shuffle added

        # Create DataLoaders
        # num_workers > 0 with IterableDatasets can be tricky. Start with 0.
        # pin_memory=True only works effectively with map-style datasets unless
        # tensors are created before yielding in IterableDataset. Set to False or test carefully.
        logger.info(f"Creating DataLoaders with batch_size={config.batch_size}, num_workers=0")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=workers, # Start with 0 for IterableDataset stability
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size, # Use same batch size for evaluation consistency
            num_workers=workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        logger.info("DataLoaders created.")

    except Exception as e:
        logger.error(f"Failed to setup data loading: {e}")
        return


    logger.info("Initializing MLA Transformer model...")
    model = MLATransformerModel(
         vocab_size=config.vocab_size,
         embed_dim=config.n_embd,
         num_heads=config.n_head,
         num_layers=config.n_layer,
         max_seq_len=config.block_size,
         dropout_prob=config.dropout,
         latent_dim=config.latent_dim,
         n_latent_vec=config.n_latent_vec,
         use_gradient_checkpoint=config.gradient_checkpointing
         # Pass use_flash_attn if your model constructor accepts it
    )
    model = torch.compile(model, backend="inductor")
    model.to(device)
    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {num_params:,} trainable parameters.")
        # Simple check based on params - rough estimate
        if 1e9 < num_params < 2e9:
             logger.info("Parameter count is roughly in the 1-2B range.")
        elif num_params >= 2e9:
             logger.warning(f"Parameter count ({num_params:,}) > 2B. Check config if target is ~1B.")
        else:
             logger.info(f"Parameter count ({num_params:,}) < 1B.")

    except Exception as e:
        logger.error(f"Could not calculate parameter count: {e}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    logger.info(f"Optimizer: AdamW with LR={config.learning_rate}, WD={config.weight_decay}")

    # --- Scheduler ---
    scheduler = CosineWarmupScheduler(optimizer, config.warmup_iters, config.max_iters)
    logger.info(f"Scheduler: CosineWarmup with warmup={config.warmup_iters}, max_iters={config.max_iters}")

    # --- Grad Scaler ---
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    logger.info(f"Gradient Scaler Enabled: {scaler.is_enabled()}")

    # --- Training Loop ---
    logger.info("--- Starting Training Loop ---")
    start_time = time.time()
    iter_num = 0
    best_val_loss = float('inf')
    # TODO: Implement checkpoint loading if resuming training

    train_iter = iter(train_loader)

    while iter_num < config.max_iters:
        iter_start_time = time.time()
        model.train()

        batch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(config.accumulation_steps):
            # Fetch batch - handle StopIteration if dataset size is finite (not typical for large streams)
            try:
                 # Use the single iterator created outside the loop
                 x, y = next(train_iter)
            except StopIteration:
                 logger.info("Training iterator stopped. Assuming end of data or defined epoch.")
                 # If you intend to loop multiple epochs over a smaller stream, re-initialize here:
                 # train_iter = iter(train_loader)
                 # x, y = next(train_iter)
                 # For large pre-training streams, often just train until max_iters
                 break # Exit accumulation loop if stream ends

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled(), dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
                if loss.ndim > 0: loss = loss.mean()
                loss = loss / config.accumulation_steps

            if torch.isfinite(loss):
                 batch_loss += loss.item() * config.accumulation_steps # Accumulate original scale loss for logging
                 scaler.scale(loss).backward()
            else:
                 logger.warning(f"Iter {iter_num}, MicroStep {micro_step}: Non-finite loss detected ({loss.item()}). Skipping backward.")
                 # This micro-step's gradients are lost. Consider if zeroing grads here is needed.
                 # If loss is consistently NaN/inf, training might diverge.
                 continue # Skip to next micro-step without backward

        # --- Optimizer Step (after accumulation) ---
        if not torch.isfinite(torch.tensor(batch_loss)): # Check if accumulated loss was valid
             logger.error(f"Iter {iter_num}: Skipping optimizer step due to non-finite loss accumulated.")
             optimizer.zero_grad(set_to_none=True) # Zero gradients accumulated (might be NaN/inf)
        else:
             scaler.unscale_(optimizer)
             grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
             scaler.step(optimizer)
             scaler.update()
             # Zero gradients *after* stepping optimizer
             optimizer.zero_grad(set_to_none=True)

        # Step scheduler regardless of optimizer step success? Usually yes.
        scheduler.step()


        # --- Logging ---
        if iter_num % 10 == 0: # Log every 10 iterations
            iter_end_time = time.time()
            elapsed_time = iter_end_time - iter_start_time
            tokens_per_step = config.batch_size * config.accumulation_steps * config.block_size
            effective_time_per_step = elapsed_time
            tokens_per_sec = tokens_per_step / effective_time_per_step if effective_time_per_step > 0 else 0
            current_lr = scheduler.get_lr()
            avg_loss_this_iter = batch_loss / config.accumulation_steps if torch.isfinite(torch.tensor(batch_loss)) else float('nan')


            logger.info(
                f"Iter: {iter_num}/{config.max_iters} | "
                f"Loss: {avg_loss_this_iter:.4f} | "
                # Only log grad_norm if optimizer step happened
                f"{f'Grad Norm: {grad_norm:.4f} | ' if torch.isfinite(torch.tensor(batch_loss)) else ''}"
                f"LR: {current_lr:.6f} | "
                f"Tokens/sec: {tokens_per_sec:,.0f} | "
                f"Time/iter: {effective_time_per_step:.3f}s"
            )

        # --- Evaluation & Checkpointing ---
        # Use eval_interval from config
        if iter_num > 0 and (iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1):
            logger.info(f"--- Running Evaluation at Iteration {iter_num} ---")
            # Pass val_loader, use eval_iters from config
            val_loss = estimate_loss(model, val_loader, config.eval_iters, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"*** New best validation loss: {best_val_loss:.4f} ***")
                # Save checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': vars(config)
                }
                best_ckpt_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, best_ckpt_path)
                logger.info(f"Saved best model checkpoint to {best_ckpt_path}")
            else:
                logger.info(f"Validation loss {val_loss:.4f} did not improve from best {best_val_loss:.4f}")

            # Periodic checkpoint
            if iter_num % (config.eval_interval * 5) == 0:
                periodic_ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{iter_num}.pt')
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'iter_num': iter_num,
                    'val_loss': val_loss,
                    'config': vars(config)
                }
                torch.save(checkpoint, periodic_ckpt_path)
                logger.info(f"Saved periodic checkpoint to {periodic_ckpt_path}")


        iter_num += 1
        # Check if training should stop early if stream ended before max_iters
        if 'StopIteration' in locals() and micro_step < config.accumulation_steps -1 :
             logger.warning(f"Training stream ended before completing max_iters. Stopping at iter {iter_num-1}.")
             break


    # --- End of Training ---
    total_training_time = time.time() - start_time
    logger.info("--- Training Finished ---")
    logger.info(f"Total Training Time: {total_training_time / 3600:.2f} hours")
    # TODO: Final evaluation on a separate test set/stream if available

if __name__ == "__main__":
    try:
        _ = ModelConfig()
    except Exception as e:
        logger.error(f"Failed to instantiate ModelConfig: {e}")
        sys.exit(1)

    main()