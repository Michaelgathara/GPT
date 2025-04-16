import os
import torch
import numpy as np
import random
import logging
import multiprocessing

from typing import Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training
)
import wandb
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

workers = 20

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

# Config class for training parameters
class TrainingConfig:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        data_path: str = "./data",
        output_dir: str = "./output",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        num_train_epochs: int = 2,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        gradient_checkpointing: bool = True,
        save_total_limit: int = 3,
        max_seq_length: int = 2048,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 1000,
        seed: int = 42,
        mixed_precision: str = "bf16",  # H100 supports bf16 natively
        max_samples_per_dataset: Optional[int] = None,  # Set to limit samples per dataset
        use_wandb: bool = False,
        use_curriculum: bool = True,  # Use curriculum learning
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.save_total_limit = save_total_limit
        self.max_seq_length = max_seq_length
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.seed = seed
        self.mixed_precision = mixed_precision
        self.max_samples_per_dataset = max_samples_per_dataset
        self.use_wandb = use_wandb
        self.use_curriculum = use_curriculum

    def to_dict(self):
        return self.__dict__

def prepare_nemotron_dataset(config: TrainingConfig):
    logger.info("Loading Nemotron dataset...")
    
    def get_llama_nemotron_data():
        dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", "SFT", split=["code", "science"])
        return dataset
    
    dataset = get_llama_nemotron_data()
    
    logger.info(f"Dataset structure: {dataset}")
    for split in dataset:
        logger.info(f"Split: {split}, Features: {dataset[split].features}, Size: {len(dataset[split])}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    def process_nemotron_example(example):
        input_text = example["input"] if example["input"] else ""
        output_text = example["output"] if example["output"] else ""
        
        if example.get("reasoning") and example["reasoning"]:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ]
        
        formatted_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        quality_score = 2 if example.get("reasoning") and example["reasoning"] else 1
        
        return {
            "formatted_chat": formatted_chat,
            "quality_score": quality_score,
            "category": example.get("category", "unknown")
        }
    
    processed_datasets = {}
    
    for split_name, split_dataset in zip(['code', 'science'], dataset):
        # Sample if max_samples specified
        if config.max_samples_per_dataset is not None:
            split_dataset = split_dataset.shuffle(seed=config.seed).select(
                range(min(len(split_dataset), config.max_samples_per_dataset))
            )
        
        processed_split = split_dataset.map(
            process_nemotron_example,
            remove_columns=split_dataset.column_names,
            desc=f"Processing {split_name} split",
            num_proc=workers 
        )
        
        processed_datasets[split_name] = processed_split
    
    # Combine datasets
    combined_dataset = concatenate_datasets(
        [processed_datasets["code"], processed_datasets["science"]]
    )
    
    combined_dataset = combined_dataset.shuffle(seed=config.seed)
    
    train_val_split = combined_dataset.train_test_split(test_size=0.1, seed=config.seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    
    if config.use_curriculum:
        logger.info("Implementing curriculum learning...")
        train_dataset = implement_curriculum(train_dataset)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["formatted_chat"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt"
        )
    
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["formatted_chat", "quality_score", "category"],
        desc="Tokenizing train dataset",
        num_proc=workers
    )
    
    tokenized_val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["formatted_chat", "quality_score", "category"],
        desc="Tokenizing validation dataset",
        num_proc=workers
    )
    
    logger.info(f"Processed train dataset size: {len(tokenized_train_dataset)}")
    logger.info(f"Processed validation dataset size: {len(tokenized_val_dataset)}")
    
    return tokenized_train_dataset, tokenized_val_dataset, tokenizer

def implement_curriculum(dataset):
    def get_complexity(example):
        length = len(example["formatted_chat"])
        return (-example["quality_score"], length)
    
    dataset_list = list(dataset)
    dataset_list.sort(key=get_complexity)
    
    sorted_dataset = Dataset.from_dict({
        k: [example[k] for example in dataset_list]
        for k in dataset_list[0].keys()
    })
    
    return sorted_dataset

def concatenate_datasets(datasets):
    if len(datasets) == 1:
        return datasets[0]
    
    first_columns = set(datasets[0].column_names)
    for ds in datasets[1:]:
        assert set(ds.column_names) == first_columns, "All datasets must have the same columns"
    
    combined_dict = {col: [] for col in first_columns}
    
    for ds in datasets:
        for col in first_columns:
            combined_dict[col].extend(ds[col])
    
    return Dataset.from_dict(combined_dict)

def setup_lora_for_qwen2(config: TrainingConfig, model):
    logger.info("Setting up LoRA for Qwen2...")
    target_modules = [
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def main():
    # Initialize configuration
    config = TrainingConfig(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        per_device_train_batch_size=16,  # H100 has enough memory for larger batches
        gradient_accumulation_steps=2,   # We'll use this to get an effective batch size of 32
        learning_rate=2e-5,              # Appropriate for small-to-mid sized models
        num_train_epochs=2,              # Usually sufficient for instruct tuning
        mixed_precision="bf16",          # H100 supports bf16 well
        max_seq_length=2048,             # Qwen2 can handle 2048 sequence lengths
        max_samples_per_dataset=1000000, # Limit dataset size initially, can be increased
    )
    
    set_random_seed(config.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    if config.use_wandb:
        wandb.init(
            project="qwen2-0.5b-nemotron-finetuning",
            config=config.to_dict(),
        )
    
    train_dataset, val_dataset, tokenizer = prepare_nemotron_dataset(config)
    
    logger.info(f"Loading Qwen2 model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
        device_map="auto"  # This will automatically use all available GPUs
    )
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    model = setup_lora_for_qwen2(config, model)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.mixed_precision == "fp16",
        bf16=config.mixed_precision == "bf16",
        report_to="wandb" if config.use_wandb else "none",
        seed=config.seed,
        dataloader_num_workers=4,
        optim="adamw_torch",
        group_by_length=True,
        ddp_find_unused_parameters=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving the final model...")
    trainer.save_model(f"{config.output_dir}/final_model")
    
    logger.info("Evaluating the final model...")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    if config.use_wandb:
        wandb.finish()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()