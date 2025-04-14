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
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter

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
def estimate_loss(model, dataloaders, eval_iters, device):
    model.eval()
    losses = {}
    
    for split, dataloader in dataloaders.items():
        losses[split] = []
        for _ in range(eval_iters):
            try:
                # get batch from dataloader
                x, y = next(iter(dataloader))
                x, y = x.to(device), y.to(device)
                
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
def train(config, train_tensor, val_tensor, test_tensor, vocab_size, device): # Accept device
    # No distributed setup needed
    # rank = 0 (implicitly)
    # world_size = 1 (implicitly)

    # set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # create checkpoint directory (no rank check needed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir=config.log_dir)

    # create datasets
    train_dataset = TokenizedDataset(train_tensor, config.block_size)
    val_dataset = TokenizedDataset(val_tensor, config.block_size)
    test_dataset = TokenizedDataset(test_tensor, config.block_size)

    # create dataloaders (use standard DataLoader, no samplers needed or use shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True, # Shuffle training data
        pin_memory=True if device.type == 'cuda' else False, # Use pin_memory only for CUDA
        num_workers=4 # Adjust as needed
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False, # No shuffle for validation
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False, # No shuffle for test
        pin_memory=True if device.type == 'cuda' else False
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
    # model.device = device # Set device attribute if estimate_loss relies on it

    # No DDP wrapping needed

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

    # create gradient scaler for mixed precision (only if using CUDA)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # zero gradients
    optimizer.zero_grad()

    # initialize training metrics
    iter_num = 0
    best_val_loss = float('inf')

    # training loop
    train_iter = iter(train_loader)

    # start timer
    start_time = time.time()

    # get the number of batches per epoch (no rank check needed)
    print(f"Total iterations: {config.max_iters}")
    print(f"Batches per epoch: {len(train_loader)}")

    tokens_processed = 0

    # main training loop
    for iter_num in range(config.max_iters):
        logger.info(f"Main loop iteration: {iter_num}")
        iter_start_time = time.time()
        model.train()

        # No sampler epoch setting needed

        # get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            # End of epoch, reshuffle data
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # mixed precision forward pass (use autocast only for CUDA)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits, loss = model(x, y)

        if loss.ndim > 0:
             loss = loss.mean() # Ensure loss is scalar

        # Check for NaN/inf loss
        if not torch.isfinite(loss):
             logger.error(f"Iteration {iter_num}: Encountered non-finite loss: {loss.item()}. Skipping update.")
             optimizer.zero_grad(set_to_none=True) # Zero gradients to prevent propagation
             continue # Skip the rest of the loop for this batch


        loss_value = loss.item() # Get scalar value for logging *before* division

        # normalize loss by accumulation steps
        loss = loss / config.accumulation_steps

        # backward pass with scaled loss
        scaler.scale(loss).backward()

        # update model if accumulation steps reached
        if (iter_num + 1) % config.accumulation_steps == 0:
            # clip gradients (helps with training stability) - unscale first!
            scaler.unscale_(optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # step optimizer with scaled gradients
            scaler.step(optimizer)
            scaler.update() # Update scaler for next iteration

            # step scheduler
            scheduler.step()

            # zero gradients
            optimizer.zero_grad(set_to_none=True)

        # update tokens processed (no world_size multiplication)
        tokens_processed += x.size(0) * x.size(1) # batch_size * block_size

        # logging (no rank check needed)
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time

        # log basic metrics
        if iter_num % 10 == 0:
            lr = scheduler.get_lr()
            # Calculate tokens/sec based on effective batch size over accumulation steps
            effective_batch_time = iter_time * config.accumulation_steps
            tokens_per_sec = (config.batch_size * config.block_size * config.accumulation_steps) / effective_batch_time if effective_batch_time > 0 else 0

            print(f"Iter {iter_num}: loss {loss_value:.4f}, lr {lr:.6f}, {tokens_per_sec:.2f} tokens/sec")

            # log to tensorboard
            # writer.add_scalar('training/loss', loss_value, iter_num)
            # writer.add_scalar('training/learning_rate', lr, iter_num)
            # writer.add_scalar('training/tokens_per_sec', tokens_per_sec, iter_num)
            # writer.add_scalar('training/tokens_processed', tokens_processed, iter_num)


        # evaluate model (no rank check needed)
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            # Pass device to estimate_loss
            loss_dict = estimate_loss(model, {'val': val_loader}, config.eval_iters, device) # Only evaluate validation set typically


            print(f"Iter {iter_num}: val loss {loss_dict['val']:.4f}")

            # log evaluation metrics
            # writer.add_scalar(f'evaluation/val_loss', loss_dict['val'], iter_num)

            # save model if validation loss improved
            if loss_dict['val'] < best_val_loss:
                best_val_loss = loss_dict['val']

                # save checkpoint (use model.state_dict())
                checkpoint = {
                    'model_state_dict': model.state_dict(), # No .module
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': vars(config)
                }

                checkpoint_path = os.path.join(config.checkpoint_dir, f'best_model_single.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f"New best model saved with val loss: {best_val_loss:.4f}")

            # save periodic checkpoint (use model.state_dict())
            if iter_num > 0 and iter_num % (config.eval_interval * 5) == 0: # Avoid saving at step 0 unless it's also eval interval
                 checkpoint = {
                     'model_state_dict': model.state_dict(), # No .module
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scaler_state_dict': scaler.state_dict(),
                     'iter_num': iter_num,
                     'val_loss': loss_dict['val'],
                     'config': vars(config)
                 }
                 checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_{iter_num}_single.pt')
                 torch.save(checkpoint, checkpoint_path)
                 print(f"Periodic checkpoint saved at iteration {iter_num}")
    # end training
    end_time = time.time()
    total_time = end_time - start_time

    # (no rank check needed)
    print(f"Training completed in {total_time:.2f} seconds")

    # Evaluate final test loss
    test_loss_dict = estimate_loss(model, {'test': test_loader}, len(test_loader), device) # Evaluate on all test data
    print(f"Final Test Loss: {test_loss_dict['test']:.4f}")
    # writer.add_scalar('evaluation/final_test_loss', test_loss_dict['test'], config.max_iters)

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with token 0

    try:
        generated_sequence = model.generate(context, max_new_tokens=200, max_seq_len=config.block_size)
        generated_ids = generated_sequence[0].tolist()
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("--------------------")
        print("Generated text sample:")
        print(decoded_text)
        print("--------------------")
    except AttributeError:
        print("Model does not have a 'generate' method implemented.")
    except Exception as e:
        print(f"Error during text generation: {e}")


    print("Training completed!")

    # close tensorboard writer
    # writer.close()

def convert_to_tensor_batches(dataset, batch_size=100_000):
    tensors = []
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(dataset), batch_size), 
                total=num_batches,
                desc="Converting to tensors",
                unit="batch"):
        end_idx = min(i + batch_size, len(dataset))
        batch = dataset.select(range(i, end_idx))['input_ids']
        tensors.append(torch.tensor(batch, dtype=torch.long))
    return torch.cat(tensors, dim=0)

def fast_convert_to_tensor(dataset):
    input_ids = dataset["input_ids"]
    array = np.array(input_ids, dtype=np.int64)
    tensor_data = torch.as_tensor(array, dtype=torch.long)
    return tensor_data

# main function to setup distributed training
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
def main():
    # No MASTER_ADDR or MASTER_PORT needed
    # No mp.set_start_method needed

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda:0") # Use the first available GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        # Fallback to CPU if needed, though training will be very slow
        print("WARNING: CUDA not available, falling back to CPU. Training will be extremely slow.")
        device = torch.device("cpu")


    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    # --- Data Loading and Preprocessing ---
    # This part remains largely the same, ensure num_proc is reasonable
    # dataset = get_llama_nemotron_data()
    # print(f"Dataset: {dataset}")
    # # num_cores = multiprocessing.cpu_count() # Can still use multiprocessing for map
    # num_cores = max(1, os.cpu_count() // 2) # Use half cores to leave resources

    # def prepare_training_text(example):
    #     # ... (same as before) ...
    #     if isinstance(example["input"], list):
    #          input_text = " ".join([str(item) for item in example["input"]])
    #     else:
    #          input_text = str(example["input"])

    #     if isinstance(example["output"], list):
    #          output_text = " ".join([str(item) for item in example["output"]])
    #     else:
    #          output_text = str(example["output"])

    #     # Combine input and output, maybe with a separator if needed by the model
    #     # Example: Add EOS token between input and output if desired
    #     # full_text = input_text + tokenizer.eos_token + output_text
    #     full_text = input_text + " " + output_text # Original simple concatenation

    #     return {"text": full_text}


    # processed_dataset = {}
    # for split in dataset:
    #     processed_dataset[split] = dataset[split].map(
    #         prepare_training_text,
    #         remove_columns=dataset[split].column_names,
    #         num_proc=num_cores, # Use multiprocessing for mapping
    #         desc=f"Preparing text for {split}"
    #     )

    # logger.info("Tokenizing dataset...")
    # def tokenize_batch(examples, tokenizer):
    #     # ... (same as before) ...
    #     texts = [str(text) for text in examples["text"]]

    #     encoded = []
    #     for text in texts:
    #          try:
    #              # Add EOS token if desired for sequence separation during training
    #              encoded_ids = tokenizer.encode(text).ids
    #              # Optionally add EOS token to the end of each sequence
    #              # encoded_ids.append(tokenizer.eos_token_id)
    #              encoded.append(encoded_ids)
    #          except Exception as e:
    #              print(f"Error tokenizing text: {e}")
    #              encoded.append([]) # Append empty list on error

    #     return {"input_ids": encoded}


    # tokenized_dataset = {}
    # for split in processed_dataset:
    #     tokenized_dataset[split] = processed_dataset[split].map(
    #         tokenize_batch,
    #         fn_kwargs={"tokenizer": tokenizer},
    #         batched=True,
    #         batch_size=10_000, # Adjust as needed
    #         num_proc=num_cores, # Use multiprocessing
    #         remove_columns=processed_dataset[split].column_names,
    #         desc=f"Tokenizing {split}"
    #     )

    # logger.info("Chunking dataset...")
    # def group_texts(examples):
    #     # ... (same as before, ensure it handles empty lists in input_ids) ...
    #     concatenated = []
    #     for ids in examples["input_ids"]:
    #         if ids: # Only extend if the list is not empty
    #              concatenated.extend(ids)
    #              # Optionally add EOS between concatenated sequences if not done in tokenize
    #              # concatenated.append(tokenizer.eos_token_id)

    #     # Calculate total length usable for full blocks
    #     total_length = (len(concatenated) // config.block_size) * config.block_size
    #     concatenated = concatenated[:total_length]

    #     # Ensure the result is a list of lists, even if empty
    #     result = [concatenated[i : i + config.block_size]
    #                for i in range(0, total_length, config.block_size)]

    #     # Return format expected by datasets.map (dict of lists)
    #     return {"input_ids": result}


    # lm_dataset = {}
    # for split in tokenized_dataset:
    #     # Filter out empty examples before chunking if necessary
    #     # filtered_split = tokenized_dataset[split].filter(lambda ex: len(ex['input_ids']) > 0)
    #     # lm_dataset[split] = filtered_split.map(
    #     lm_dataset[split] = tokenized_dataset[split].map(
    #         group_texts,
    #         batched=True,
    #         batch_size=1000, # Adjust as needed
    #         num_proc=num_cores, # Use multiprocessing
    #         desc=f"Chunking {split}"
    #     )
    #     # Remove empty chunks that might result from filtering/grouping issues
    #     lm_dataset[split] = lm_dataset[split].filter(lambda example: len(example['input_ids']) > 0)


    # print(f"Dataset after chunking: \n{lm_dataset}")
    # # Ensure 'science' split exists and has enough data
    # if 'science' not in lm_dataset or len(lm_dataset['science']) < 2:
    #      raise ValueError("Science split is missing or too small for val/test split.")

    # # Ensure val_size calculation is valid
    # val_size = max(1, len(lm_dataset['science']) // 2) # Ensure at least 1 example if possible
    # test_start_index = val_size


    # # Ensure test set is not empty
    # if test_start_index >= len(lm_dataset['science']):
    #      print("Warning: Validation set size calculation results in empty test set. Adjusting.")
    #      # Adjust val_size, e.g., take fewer for validation if dataset is small
    #      val_size = max(1, len(lm_dataset['science']) - 1)
    #      test_start_index = val_size
    #      if test_start_index >= len(lm_dataset['science']):
    #            raise ValueError("Cannot create non-empty validation and test splits from science data.")


    # --- Tensor Conversion ---
    # Define paths
    train_data_path = os.path.join(config.checkpoint_dir, 'train_data.pt')
    val_data_path = os.path.join(config.checkpoint_dir, 'val_data.pt')
    test_data_path = os.path.join(config.checkpoint_dir, 'test_data.pt')

    # Load or convert tensors
    train_data = None
    val_data = None
    test_data = None

    # Ensure checkpoint directory exists before trying to load/save
    os.makedirs(config.checkpoint_dir, exist_ok=True)


    if os.path.exists(train_data_path):
        print("Loading pre-converted training tensor from disk...")
        train_data = torch.load(train_data_path, map_location='cpu') # Load to CPU first
    if os.path.exists(val_data_path):
        print("Loading pre-converted validation tensor from disk...")
        val_data = torch.load(val_data_path, map_location='cpu')
    if os.path.exists(test_data_path):
        print("Loading pre-converted testing tensor from disk...")
        test_data = torch.load(test_data_path, map_location='cpu')

    print("Converting datasets to tensors (if not loaded)...")
    if train_data is None:
        if 'code' not in lm_dataset or len(lm_dataset['code']) == 0:
            raise ValueError("Code split is missing or empty, cannot create training tensor.")
        print("Converting train split ('code') to tensor...")
        start_time = time.time()
        train_data = fast_convert_to_tensor(lm_dataset['code']) # Use preferred conversion
        print(f"Train conversion took: {time.time() - start_time:.2f}s")
        print("Saving training tensor to disk...")
        torch.save(train_data, train_data_path)

    if val_data is None:
         print(f"Converting validation split ('science'[:{val_size}]) to tensor...")
         start_time = time.time()
         # Select range and ensure it's not empty
         val_slice_indices = range(val_size)
         if not val_slice_indices: raise ValueError("Validation split size is zero.")
         val_ds_slice = lm_dataset['science'].select(val_slice_indices)
         if len(val_ds_slice) == 0: raise ValueError("Validation slice resulted in empty dataset.")
         val_data = fast_convert_to_tensor(val_ds_slice)
         print(f"Validation conversion took: {time.time() - start_time:.2f}s")
         print("Saving validation tensor to disk...")
         torch.save(val_data, val_data_path)


    if test_data is None:
         print(f"Converting test split ('science'[{test_start_index}:]) to tensor...")
         start_time = time.time()
         # Select range and ensure it's not empty
         test_slice_indices = range(test_start_index, len(lm_dataset['science']))
         if not test_slice_indices: raise ValueError("Test split size is zero.")
         test_ds_slice = lm_dataset['science'].select(test_slice_indices)
         if len(test_ds_slice) == 0: raise ValueError("Test slice resulted in empty dataset.")
         test_data = fast_convert_to_tensor(test_ds_slice)
         print(f"Test conversion took: {time.time() - start_time:.2f}s")
         print("Saving testing tensor to disk...")
         torch.save(test_data, test_data_path)



    # Ensure tensors are loaded and have the expected type/shape
    if train_data is None or val_data is None or test_data is None:
        raise RuntimeError("Failed to load or convert one or more data tensors.")

    # Move loaded tensors to target device only when needed (inside train)
    # Or keep them on CPU and move batches in DataLoader/training loop

    # Log final shapes
    print(f"Train Data: {train_data.shape}, {train_data.dtype}")
    print(f"Val   Data: {val_data.shape}, {val_data.dtype}")
    print(f"Test  Data: {test_data.shape}, {test_data.dtype}")
    print(f"Vocabulary size: {vocab_size}")


    # --- Start Training ---
    # No mp.spawn needed, directly call train
    train(config, train_data, val_data, test_data, vocab_size, device)


if __name__ == "__main__":
    main()