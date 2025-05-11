import sys
import os
import time
import logging
import math
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# Removed: DDP and distributed imports
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

# TODO: Implement typing and tqdm
from tqdm import tqdm
from functools import partial
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer

base_folder = os.path.abspath("..")
sys.path.append(base_folder)

from transformer_setup import ModelConfig, TransformerModel 

#  Logger setup 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_single_gpu.log"), # Log file for single GPU run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("transformer_single_gpu_training")

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    logger.info("Flash Attention is available!")
except ImportError:
    HAS_FLASH_ATTN = False
    logger.info("Flash Attention is not available, falling back to standard attention.")

config = ModelConfig()

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_iters, max_iters):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.current_iter = 0
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

    def step(self):
        lr_scale = 1.0
        if self.current_iter < self.warmup_iters:
            lr_scale = min(1.0, float(self.current_iter + 1) / self.warmup_iters)
        elif self.current_iter < self.max_iters:
            progress = float(self.current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            progress = max(0.0, min(1.0, progress))
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            lr_scale = 0.0 


        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale

        self.current_iter += 1

    def get_lr(self):
        if not self.optimizer.param_groups:
            return 0.0
        return self.optimizer.param_groups[0]['lr']

@torch.no_grad()
def estimate_loss(model, data_generator_factory, eval_iters, device, batch_size):
    logger.info("Estimating loss...")
    model.eval() 
    losses = {'val': []}
    val_generator = data_generator_factory()

    for k in range(eval_iters):
        try:
            batch_x_list = []
            batch_y_list = []
            for _ in range(batch_size):
                 try:
                     x, y = next(val_generator)
                     batch_x_list.append(x)
                     batch_y_list.append(y)
                 except StopIteration:
                     if not batch_x_list: # If generator ends before first item of batch
                         raise StopIteration # Propagate to outer loop
                     else:
                         break # Process the partial batch

            if not batch_x_list:
                 logger.warning("Validation data generator yielded no data for an iteration.")
                 continue # Skip this eval iteration if no data

            x_batch = torch.stack(batch_x_list).to(device)
            y_batch = torch.stack(batch_y_list).to(device)

            #  Forward pass 
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                _, loss = model(x_batch, y_batch)

            if loss.ndim > 0: # Average if loss is per-item
                loss = loss.mean()
            losses['val'].append(loss.item())

        except StopIteration:
            logger.warning(f"Validation data stream ended after {k} evaluation iterations.")
            break # Stop evaluation if the generator is exhausted

    model.train() # Set model back to training mode
    avg_losses = {split: np.mean(split_losses) if split_losses else float('nan')
                  for split, split_losses in losses.items()}
    logger.info(f"Estimated losses: {avg_losses}")
    return avg_losses

def create_token_generator(stream, tokenizer, block_size):
    logger.info(f"GENERATOR ENTRY: Tokenizer type: {type(tokenizer)}")
    logger.info(f"GENERATOR ENTRY: Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"GENERATOR ENTRY: Tokenizer pad token ID: {tokenizer.pad_token_id}")

    buffer = []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None or pad_token_id >= tokenizer.vocab_size:
        pad_token_id = tokenizer.vocab_size - 1  
        logger.warning(f"Tokenizer pad_token_id {tokenizer.pad_token_id} is invalid. Using {pad_token_id} as fallback.")

    current_vocab_size = tokenizer.vocab_size
    logger.info(f"Token generator starting iteration with Vocab Size: {current_vocab_size}...")
    count = 0
    skipped = 0
    invalid_targets_found = 0

    for example in stream: # Iterates through the tokenized stream
        if example and 'input_ids' in example:
            input_ids = example['input_ids']
            if not input_ids: # Skip if tokenization resulted in empty list
                 skipped += 1
                 continue
            buffer.extend(input_ids)
            while len(buffer) >= block_size:
                chunk = buffer[:block_size]
                buffer = buffer[block_size:]
                x = torch.tensor(chunk, dtype=torch.long)
                # Create target by shifting input, handling the last token prediction
                y = torch.cat((x[1:], torch.tensor([pad_token_id], dtype=torch.long)))

                if y.min() < 0 or y.max() >= current_vocab_size: 
                    invalid_targets_found += 1
                    max_val = y.max().item()
                    min_val = y.min().item()
                    # Log with the vocab size used in the check
                    logger.error(f"INVALID TARGET INDICES DETECTED in generator! Min: {min_val}, Max: {max_val}, Vocab Size: {current_vocab_size}. Skipping this chunk.")
                    if invalid_targets_found > 100:
                         logger.error("Stopping detailed logging of invalid targets after 100 occurrences.")
                    continue # Skip yielding this invalid chunk
                #  End Check 

                yield x, y
                count += 1
        else:
            skipped += 1

    logger.info(f"Token generator finished. Yielded {count} sequences, skipped {skipped} examples.")
    if invalid_targets_found > 0:
         logger.warning(f"Total invalid target chunks skipped: {invalid_targets_found}")
    # Handle any remaining partial block if desired

def train_single_gpu(config, vocab_size, tokenizer, train_stream, val_stream=None):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available. Training on CPU (will be very slow).")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(config.seed)
        # Potentially add deterministic flags if needed, might impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    logger.info("Initializing model...")
    model = TransformerModel(
        vocab_size=vocab_size,
        embed_dim=config.n_embd,
        num_heads=config.n_head,
        num_layers=config.n_layer,
        max_seq_len=config.block_size,
        dropout_prob=config.dropout,
        use_gradient_checkpoint=config.gradient_checkpointing,
        use_flash_attn=config.use_flash_attn and HAS_FLASH_ATTN 
    )
    model = model.to(device)
    logger.info(f"Model initialized with {model.get_num_params():,} parameters.")
    
    # Had a ton of trouble with using compile here, but according to the docs (https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) it should make the model just a bit faster
    # # TODO: read more on torch.compile
    # try:
    #     model = torch.compile(model)
    #     print("Model compiled successfully for training.")
    # except Exception as e:
    #     print(f"torch.compile failed during training setup: {e}. Proceeding without compiling.")

    logger.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )

    logger.info("Setting up scheduler and gradient scaler...")
    # Initialize scheduler - it will be advanced later if resuming
    scheduler = CosineWarmupScheduler(optimizer, config.warmup_iters, config.max_iters)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_iter = 0          # Start from iter 0 by default
    best_val_loss = float('inf') # Start with best_val_loss as infinity
    checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt') # Path to the checkpoint to resume from

    if os.path.isfile(checkpoint_path):
        logger.info(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        try:
            # Load checkpoint dictionary, mapping storage to the training device
            with torch.serialization.safe_globals([np.core.multiarray.scalar]):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # weights_only being false here is pretty important if you want to resume training

            # checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load model state
            # Handle potential issues if the saved model architecture doesn't match current
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded model state dict.")

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state dict.")

            # Load gradient scaler state
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("Loaded scaler state dict.")

            # Load iteration number and best validation loss
            # Start from the *next* iteration after the one saved
            start_iter = checkpoint.get('iter_num', 0) # Use .get for backward compatibility
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"Resuming from iteration {start_iter}. Best validation loss so far: {best_val_loss:.4f}")

            # **Important:** Advance the scheduler to the correct state
            # We need to simulate the steps the scheduler already took.
            logger.info(f"Advancing scheduler by {start_iter} steps...")
            # Ensure initial_lr is set in the optimizer group *before* stepping
            # The scheduler's __init__ already does this, but double-check if issues arise
            for _ in range(start_iter):
                 scheduler.step() # Step the original scheduler instance
            logger.info(f"Scheduler advanced. Current LR: {scheduler.get_lr():.6f}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint properly: {e}. Starting training from scratch.")
            start_iter = 0
            best_val_loss = float('inf')
            # Re-initialize scheduler if loading failed? Or assume it's okay from initial init.
            # Re-initializing optimizer/scaler might also be needed if partial load failed.
            # For simplicity, we proceed with fresh start values if any error occurs.
    else:
        logger.info(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")


    logger.info("Creating data generators...")
    def train_generator_factory():
        return create_token_generator(train_stream, tokenizer, config.block_size)

    def val_generator_factory():
        if val_stream:
            return create_token_generator(val_stream, tokenizer, config.block_size)
        else:
            return iter([])

    train_generator = train_generator_factory() # Initial generator for training

    logger.info(f"Starting training loop from iteration {start_iter} up to {config.max_iters} iterations...")
    model.train()
    tokens_processed = 0 # Note: This counter restarts, doesn't resume total token count
    start_time = time.time()

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    for iter_num in range(start_iter, config.max_iters):
        iter_start_time = time.time()

        micro_batch_losses = []
        # Zero gradients *before* starting accumulation for the new effective batch
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(config.accumulation_steps):
            try:
                # Collect a micro-batch
                batch_x_list = []
                batch_y_list = []
                for _ in range(config.batch_size):
                    try:
                         x, y = next(train_generator)
                         batch_x_list.append(x)
                         batch_y_list.append(y)
                    except StopIteration:
                         logger.warning(f"Training data stream ended during accumulation at iter {iter_num+1}, micro-step {micro_step+1}.")
                         if not batch_x_list:
                             raise StopIteration
                         else:
                             break # Process partial micro-batch

                if not batch_x_list:
                    logger.warning(f"No data yielded for micro-batch {micro_step+1} at iter {iter_num+1}.")
                    continue # Skip this micro-step if no data

                x_micro_batch = torch.stack(batch_x_list).to(device)
                y_micro_batch = torch.stack(batch_y_list).to(device)

                #  Forward/Backward Pass for Micro-batch 
                model.train()
                # Enable autocast only if on CUDA
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits, loss = model(x_micro_batch, y_micro_batch)
                    if loss.ndim > 0: loss = loss.mean()
                    loss = loss / config.accumulation_steps # Normalize loss

                # Multiply by accum_steps later for avg loss over effective batch
                micro_batch_losses.append(loss.item() * config.accumulation_steps)
                # Scale loss and backward
                scaler.scale(loss).backward()

                tokens_processed += x_micro_batch.numel()

            except StopIteration:
                logger.warning(f"Training data stream ended definitively at iteration {iter_num+1}.")
                # Force the outer loop to end after this iteration completes its optimizer step etc.
                config.max_iters = iter_num + 1 # Adjust max_iters to stop gracefully
                break # Exit accumulation loop

        if micro_batch_losses: # Proceed only if forward/backward passes happened
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        if micro_batch_losses: # Log only if an optimizer step happened
            avg_effective_loss = np.mean(micro_batch_losses) # Already scaled back up
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            if (iter_num + 1) % 10 == 0:
                lr = scheduler.get_lr()
                elapsed_time = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
                logger.info(f"Iter {iter_num+1}/{config.max_iters}: Loss {avg_effective_loss:.4f}, LR {lr:.6f}, Tokens/Sec {tokens_per_sec:.2f}, Elapsed {elapsed_time:.2f}s")

        # Check if it's time to evaluate or if it's the very last iteration
        if (iter_num + 1) % config.eval_interval == 0 or (iter_num + 1) == config.max_iters:
            if val_stream:
                 loss_dict = estimate_loss(model, val_generator_factory, config.eval_iters, device, config.batch_size)
                 current_val_loss = loss_dict.get('val', float('inf'))
                 logger.info(f"Iter {iter_num+1}: Val Loss {current_val_loss:.4f}")

                 # Save best model checkpoint
                 if current_val_loss < best_val_loss and current_val_loss < 4:
                     best_val_loss = current_val_loss
                     checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
                     checkpoint = {
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'scaler_state_dict': scaler.state_dict(),
                         'iter_num': iter_num + 1, # Save the completed iteration number
                         'best_val_loss': best_val_loss,
                         'config': vars(config)
                     }
                     torch.save(checkpoint, checkpoint_path)
                     logger.info(f"New best model saved with val loss: {best_val_loss:.4f} at {checkpoint_path}")
            else:
                 logger.info(f"Iter {iter_num+1}: No validation data provided, skipping evaluation.")
                 if (iter_num + 1) % (config.eval_interval * 5) == 0: # Save every 5 eval intervals
                      checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_{iter_num+1}.pt')
                      checkpoint = {
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'scaler_state_dict': scaler.state_dict(),
                          'iter_num': iter_num + 1,
                          'config': vars(config)
                       }
                      torch.save(checkpoint, checkpoint_path)
                      logger.info(f"Periodic checkpoint saved at {checkpoint_path}")

        if (iter_num + 1) == config.max_iters and micro_step < config.accumulation_steps -1 :
             logger.warning(f"Exiting training loop early due to end of data stream at iteration {iter_num+1}.")
             break # Exit outer loop

    logger.info(f"Training loop finished after {iter_num + 1} iterations.")

    #  Final Model Saving (Optional - consider if best_model is sufficient) 
    # final_checkpoint_path = os.path.join(config.checkpoint_dir, 'final_model.pt')
    # final_checkpoint = { ... same as best_model checkpoint ... }
    # torch.save(final_checkpoint, final_checkpoint_path)
    # logger.info(f"Final model state saved to {final_checkpoint_path}")

    logger.info("Generating sample text...")
    model.eval()
    start_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0)
    context = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

    try:
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            generated_sequence = model.generate(context, max_new_tokens=200, max_seq_len=config.block_size, temperature=0.8, top_k=50)
        generated_ids = generated_sequence[0].tolist()
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info(f"Generated text:\n{decoded_text}")
    except AttributeError:
        logger.warning("Model does not have a 'generate' method. Skipping text generation.")
    except Exception as e:
        logger.error(f"Error during text generation: {e}", exc_info=True) # Log traceback

if __name__ == "__main__":
    logger.info("Script started.")

    logger.info("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        logger.info("Setting tokenizer pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token 
    else:
        logger.info(f"Tokenizer already has a pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    vocab_size = tokenizer.vocab_size
    pad_id_used = tokenizer.pad_token_id
    logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}") # Should log 50257
    logger.info(f"Using PAD token ID: {pad_id_used}")     # Should log 50256

    #  Load Streamed Dataset 
    logger.info("Loading Fineweb dataset with streaming...")
    dataset_name = "HuggingFaceFW/fineweb-edu"
    # Use "sample-10BT" for quick testing, "default" for full run
    dataset_config = "default" # Or "sample-10BT"
    try:
        stream_dataset = load_dataset(dataset_name, dataset_config, streaming=True)
        train_stream_raw = stream_dataset['train']
        logger.warning("No 'validation' split found in the streaming dataset. Evaluation will be skipped unless manually configured.")
        val_stream_raw = train_stream_raw.take(1000) # Take first 1000 samples for validation
        train_stream_raw = train_stream_raw.skip(1000) # Skip these samples in training
        logger.info("Created pseudo-validation set by taking first 1000 samples from train stream.")


        logger.info(f"Dataset '{dataset_name}/{dataset_config}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Please ensure you have internet access and are logged in to Hugging Face if required (huggingface-cli login).")
        sys.exit(1) # Exit if dataset fails to load

    def tokenize_function(example, tokenizer):
        try:
            text = example.get('text', '')
            if not text: return None
            tokenized = tokenizer(text, truncation=False)
            return {"input_ids": tokenized['input_ids']}
        except Exception as e:
            logger.warning(f"Error tokenizing example: {e}. Skipping.")
            return None

    logger.info("Applying tokenization to train stream...")
    tokenized_train_stream = train_stream_raw.map(
        tokenize_function,
        fn_kwargs={'tokenizer': tokenizer}
    ).filter(lambda x: x is not None).shuffle(buffer_size=10_000, seed=config.seed)

    tokenized_val_stream = None
    if val_stream_raw:
        logger.info("Applying tokenization to validation stream...")
        tokenized_val_stream = val_stream_raw.map(
            tokenize_function,
            fn_kwargs={'tokenizer': tokenizer}
        ).filter(lambda x: x is not None)

    try:
        train_single_gpu(
            config=config,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            train_stream=tokenized_train_stream,
            val_stream=tokenized_val_stream # Pass the tokenized validation stream
        )
        logger.info("Training finished successfully.")
    except Exception as e:
        logger.exception("An error occurred during training.") # Log full traceback
        sys.exit(1)

    logger.info("Script finished.")