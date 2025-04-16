#!/usr/bin/env python
"""
H100-Optimized Training Script for Qwen2-0.5B-Instruct
- Leverages H100 GPU capabilities including Tensor Cores and FP8/BF16 precision
- Implements efficient memory management with gradient checkpointing and offloading
- Uses DeepSpeed ZeRO optimization for large batch training
- Provides monitoring, logging, and checkpoint management
"""

import os
import sys
import time
import torch
import logging
import argparse
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training
)
from accelerate import Accelerator
import wandb
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-0.5B-Instruct with H100 optimizations")
    
    # Model and data parameters
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct", 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--data_dir", type=str, default="./preprocessed_data", 
                        help="Directory containing the preprocessed datasets")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save the model checkpoints")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=2, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=1, 
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, 
                        help="Ratio of steps for learning rate warmup")
    parser.add_argument("--max_seq_length", type=int, default=2048, 
                        help="Maximum sequence length")
    
    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true", 
                        help="Whether to use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=16, 
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout rate")
    
    # Optimization parameters
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"],
                        help="Training precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--use_deepspeed", action="store_true", 
                        help="Whether to use DeepSpeed")
    parser.add_argument("--deepspeed_stage", type=int, default=3, choices=[1, 2, 3], 
                        help="DeepSpeed ZeRO stage")
    
    # Logging and saving parameters
    parser.add_argument("--logging_steps", type=int, default=50, 
                        help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=500, 
                        help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=1000, 
                        help="Checkpoint saving steps")
    parser.add_argument("--save_total_limit", type=int, default=3, 
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use Weights & Biases for logging")
    
    return parser.parse_args()

def load_datasets(data_dir: str) -> Dict[str, Dataset]:
    logger.info(f"Loading datasets from {data_dir}")
    datasets = {}
    
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            datasets[split] = load_from_disk(split_path)
            logger.info(f"Loaded {split} dataset: {len(datasets[split])} examples")
        else:
            logger.warning(f"Could not find {split} dataset at {split_path}")
    
    if "train" not in datasets:
        raise ValueError("Training dataset not found")
    
    return datasets

def setup_lora(model, args) -> torch.nn.Module:
    logger.info("Setting up LoRA for parameter-efficient fine-tuning")
    
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
        "gate_proj", "up_proj", "down_proj"      # MLP modules
    ]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def create_deepspeed_config(args) -> Dict[str, Any]:
    """Create DeepSpeed configuration based on arguments"""
    logger.info(f"Creating DeepSpeed config with ZeRO-{args.deepspeed_stage}")
    
    # Base DeepSpeed configuration
    ds_config = {
        "fp16": {
            "enabled": args.precision == "fp16",
            "auto_cast": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": args.precision == "bf16"
        },
        "zero_optimization": {
            "stage": args.deepspeed_stage,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            } if args.deepspeed_stage == 3 else False,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            } if args.deepspeed_stage == 3 else False,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": args.grad_accum_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": args.logging_steps,
        "train_batch_size": args.batch_size * args.grad_accum_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "wall_clock_breakdown": False
    }
    
    return ds_config

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    accelerator = Accelerator()
    
    # Initialize wandb if enabled
    # if args.use_wandb and accelerator.is_main_process:
    #     run_name = f"qwen2-0.5b-nemotron-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    #     wandb.init(
    #         project="qwen2-nemotron-finetuning",
    #         name=run_name,
    #         config=vars(args)
    #     )
    #     logger.info(f"Initialized wandb with run name: {run_name}")
    
    if accelerator.is_main_process:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
        logger.info(f"Initialized TensorBoard in {os.path.join(args.output_dir, 'tensorboard')}")
    
    datasets = load_datasets(args.data_dir)
    train_dataset = datasets["train"]
    eval_dataset = datasets.get("validation")
    
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model: {args.model_name}")
    torch_dtype = torch.float32
    if args.precision == "fp16":
        torch_dtype = torch.float16
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16
    
    logger.info(f"Using {args.precision} precision")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    if args.use_lora:
        model = setup_lora(model, args)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    deepspeed_config = None
    if args.use_deepspeed:
        deepspeed_config = create_deepspeed_config(args)
        logger.info("Using DeepSpeed with config:")
        for key, value in deepspeed_config.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        report_to="wandb" if args.use_wandb else "tensorboard",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=4,
        group_by_length=True,  # More efficient batching
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        deepspeed=deepspeed_config,
        remove_unused_columns=False,  # Important for some model architectures
    )
    
    logger.info("Applying H100-specific optimizations")
    
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn
    
    model.config.use_cache = False  # Disable KV-cache for training
    
    logger.info(f"Model configuration: {model.config}")
    
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    if accelerator.is_main_process:
        class PerformanceMonitorCallback(trainer.callback_handler.__class__):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.start_time = time.time()
                self.step_start_time = time.time()
                self.steps_since_last_log = 0
                self.total_tokens = 0
                self.log_interval = 100  # Log performance every N steps
            
            def on_step_end(self, args, state, control, **kwargs):
                super().on_step_end(args, state, control, **kwargs)
                self.steps_since_last_log += 1
                
                batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
                tokens_per_batch = batch_size * args.max_seq_length
                self.total_tokens += tokens_per_batch
                
                if state.global_step % self.log_interval == 0:
                    elapsed = time.time() - self.step_start_time
                    tokens_per_second = (tokens_per_batch * self.steps_since_last_log) / elapsed
                    logger.info(f"Step {state.global_step}: {tokens_per_second:.2f} tokens/second")
                    
                    if args.report_to == "tensorboard":
                        tb_writer.add_scalar("performance/tokens_per_second", tokens_per_second, state.global_step)
                    # elif args.report_to == "wandb" and wandb.run is not None:
                    #     wandb.log({"performance/tokens_per_second": tokens_per_second}, step=state.global_step)
                    
                    # Reset counters
                    self.step_start_time = time.time()
                    self.steps_since_last_log = 0
            
            def on_train_end(self, args, state, control, **kwargs):
                super().on_train_end(args, state, control, **kwargs)
                total_time = time.time() - self.start_time
                avg_tokens_per_second = self.total_tokens / total_time
                logger.info(f"Training complete: Average {avg_tokens_per_second:.2f} tokens/second")
                
                if args.report_to == "tensorboard":
                    tb_writer.add_scalar("performance/avg_tokens_per_second", avg_tokens_per_second, state.global_step)
                    tb_writer.add_scalar("performance/total_training_time", total_time, state.global_step)
                # elif args.report_to == "wandb" and wandb.run is not None:
                #     wandb.log({
                #         "performance/avg_tokens_per_second": avg_tokens_per_second,
                #         "performance/total_training_time": total_time
                #     }, step=state.global_step)
        
        trainer.callback_handler = PerformanceMonitorCallback(
            trainer.callback_handler.callbacks,
            trainer.callback_handler.model,
            trainer.callback_handler.optimizer,
            trainer.callback_handler.lr_scheduler
        )
    
    logger.info("Starting training...")
    train_result = trainer.train()
    
    logger.info("Saving the final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    if eval_dataset:
        logger.info("Evaluating the final model...")
        eval_results = trainer.evaluate()
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)
    
    # if args.use_wandb and accelerator.is_main_process:
    #     wandb.finish()
    
    if accelerator.is_main_process:
        tb_writer.close()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()