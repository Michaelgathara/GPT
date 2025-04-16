import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NemotronPreprocessor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        output_dir: str = "./preprocessed_data",
        max_seq_length: int = 2048,
        seed: int = 42,
        test_size: float = 0.1,
        val_size: float = 0.1,
        max_samples_per_split: Optional[int] = None
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.test_size = test_size
        self.val_size = val_size
        self.max_samples_per_split = max_samples_per_split
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set random seed
        np.random.seed(seed)
    
    def load_nemotron_dataset(self) -> Dict[str, Dataset]:
        logger.info("Loading LLaMa Nemotron dataset...")
        
        def get_llama_nemotron_data():
            dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1", split=['code', 'science'])
            return dataset
        
        dataset = get_llama_nemotron_data()
        
        # Basic dataset info
        datasets_dict = {}
        for i, split_name in enumerate(['code', 'science']):
            datasets_dict[split_name] = dataset[i]
            logger.info(f"Split '{split_name}': {len(datasets_dict[split_name])} examples")
        
        # Sample if max_samples specified
        if self.max_samples_per_split is not None:
            for split_name in datasets_dict:
                if len(datasets_dict[split_name]) > self.max_samples_per_split:
                    logger.info(f"Sampling {self.max_samples_per_split} examples from '{split_name}' split")
                    datasets_dict[split_name] = datasets_dict[split_name].shuffle(seed=self.seed).select(
                        range(self.max_samples_per_split)
                    )
        
        return datasets_dict
    
    def analyze_dataset(self, datasets_dict: Dict[str, Dataset]) -> Dict[str, Any]:
        logger.info("Analyzing dataset characteristics...")
        
        analysis = {}
        
        for split_name, dataset in datasets_dict.items():
            split_analysis = {}
            
            # Sample size
            split_analysis["sample_size"] = len(dataset)
            
            # Analyze input lengths
            input_lengths = [len(example["input"]) if example["input"] else 0 for example in dataset]
            split_analysis["input_length"] = {
                "mean": np.mean(input_lengths),
                "median": np.median(input_lengths),
                "min": min(input_lengths),
                "max": max(input_lengths),
                "std": np.std(input_lengths)
            }
            
            # Analyze output lengths
            output_lengths = [len(example["output"]) if example["output"] else 0 for example in dataset]
            split_analysis["output_length"] = {
                "mean": np.mean(output_lengths),
                "median": np.median(output_lengths),
                "min": min(output_lengths),
                "max": max(output_lengths),
                "std": np.std(output_lengths)
            }
            
            # Category distribution
            categories = {}
            for example in dataset:
                category = example.get("category", "unknown")
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1
            
            split_analysis["category_distribution"] = {
                k: v / len(dataset) for k, v in categories.items()
            }
            
            # Reasoning path presence
            reasoning_count = sum(1 for example in dataset if example.get("reasoning") and example["reasoning"])
            split_analysis["reasoning_presence"] = reasoning_count / len(dataset)
            
            analysis[split_name] = split_analysis
        
        # Save analysis
        with open(os.path.join(self.output_dir, "dataset_analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def clean_and_filter_dataset(self, datasets_dict: Dict[str, Dataset]) -> Dict[str, Dataset]:
        logger.info("Cleaning and filtering dataset...")
        
        filtered_datasets = {}
        
        for split_name, dataset in datasets_dict.items():
            logger.info(f"Processing '{split_name}' split...")
            
            # Filter invalid examples
            def is_valid_example(example):
                # Ensure input and output are not empty
                if not example["input"] or not example["output"]:
                    return False
                
                # Check if input is too short (possibly noise)
                if len(example["input"]) < 10:
                    return False
                
                # Filter out very long sequences
                input_output_length = len(example["input"]) + len(example["output"])
                if input_output_length > self.max_seq_length * 0.9:  # Allow some buffer for formatting
                    return False
                
                return True
            
            filtered_dataset = dataset.filter(
                is_valid_example,
                desc=f"Filtering invalid examples from '{split_name}'"
            )
            
            logger.info(f"After filtering: {len(filtered_dataset)} examples (removed {len(dataset) - len(filtered_dataset)})")
            filtered_datasets[split_name] = filtered_dataset
        
        return filtered_datasets
    
    def deduplicate_dataset(self, datasets_dict: Dict[str, Dataset]) -> Dict[str, Dataset]:
        logger.info("Deduplicating dataset...")
        
        deduplicated_datasets = {}
        
        for split_name, dataset in datasets_dict.items():
            logger.info(f"Deduplicating '{split_name}' split...")
            
            # Create hashes for each example
            hashes = set()
            unique_indices = []
            
            for i, example in enumerate(tqdm(dataset, desc=f"Hashing examples in '{split_name}'")):
                # Create a hash of the input and output
                hash_str = f"{example['input']}_{example['output']}"
                hash_val = hashlib.md5(hash_str.encode()).hexdigest()
                
                if hash_val not in hashes:
                    hashes.add(hash_val)
                    unique_indices.append(i)
            
            deduplicated_dataset = dataset.select(unique_indices)
            logger.info(f"After deduplication: {len(deduplicated_dataset)} examples (removed {len(dataset) - len(deduplicated_dataset)})")
            
            deduplicated_datasets[split_name] = deduplicated_dataset
        
        return deduplicated_datasets
    
    def format_for_qwen2(self, datasets_dict: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """Format the dataset for Qwen2's expected chat format"""
        logger.info("Formatting dataset for Qwen2...")
        
        formatted_datasets = {}
        
        for split_name, dataset in datasets_dict.items():
            logger.info(f"Formatting '{split_name}' split...")
            
            def format_example(example):
                # Get input/output from example
                input_text = example["input"] if example["input"] else ""
                output_text = example["output"] if example["output"] else ""
                
                # Create messages structure
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_text}
                ]
                
                # Apply Qwen2's chat template
                formatted_chat = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # Add quality score based on if reasoning exists
                quality_score = 2 if example.get("reasoning") and example["reasoning"] else 1
                
                return {
                    "formatted_chat": formatted_chat,
                    "quality_score": quality_score,
                    "category": example.get("category", "unknown"),
                    "input": input_text,
                    "output": output_text
                }
            
            formatted_dataset = dataset.map(
                format_example,
                remove_columns=dataset.column_names,
                desc=f"Formatting '{split_name}' for Qwen2",
                num_proc=8
            )
            
            formatted_datasets[split_name] = formatted_dataset
        
        return formatted_datasets
    
    def create_train_val_test_split(self, datasets_dict: Dict[str, Dataset]) -> Dict[str, Dataset]:
        logger.info("Creating train/validation/test splits...")
        
        # Combine all datasets first
        combined_datasets = []
        for split_name, dataset in datasets_dict.items():
            combined_datasets.append(dataset)
        
        combined_dataset = concatenate_datasets(combined_datasets)
        logger.info(f"Combined dataset size: {len(combined_dataset)}")
        
        # Shuffle the dataset
        combined_dataset = combined_dataset.shuffle(seed=self.seed)
        
        # Calculate split sizes
        test_size_abs = int(len(combined_dataset) * self.test_size)
        val_size_abs = int(len(combined_dataset) * self.val_size)
        train_size_abs = len(combined_dataset) - test_size_abs - val_size_abs
        
        # Create splits
        train_dataset = combined_dataset.select(range(0, train_size_abs))
        val_dataset = combined_dataset.select(range(train_size_abs, train_size_abs + val_size_abs))
        test_dataset = combined_dataset.select(range(train_size_abs + val_size_abs, len(combined_dataset)))
        
        logger.info(f"Train split: {len(train_dataset)} examples")
        logger.info(f"Validation split: {len(val_dataset)} examples")
        logger.info(f"Test split: {len(test_dataset)} examples")
        
        return {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }
    
    def implement_curriculum(self, dataset: Dataset) -> Dataset:
        logger.info("Implementing curriculum learning...")
        
        # Convert to DataFrame for easier sorting
        df = pd.DataFrame({
            "formatted_chat": dataset["formatted_chat"],
            "quality_score": dataset["quality_score"],
            "category": dataset["category"],
            "input": dataset["input"],
            "output": dataset["output"],
            "length": [len(x) for x in dataset["formatted_chat"]]
        })
        
        # Sort by quality score (descending) and length (ascending)
        # This puts high-quality, shorter examples first
        df = df.sort_values(by=["quality_score", "length"], ascending=[False, True])
        
        # Convert back to Dataset
        sorted_dataset = Dataset.from_pandas(df)
        
        logger.info("Curriculum sorting complete")
        return sorted_dataset
    
    def save_processed_datasets(self, datasets_dict: Dict[str, Dataset]):
        logger.info("Saving processed datasets...")
        
        for split_name, dataset in datasets_dict.items():
            output_path = os.path.join(self.output_dir, split_name)
            dataset.save_to_disk(output_path)
            logger.info(f"Saved {split_name} dataset to {output_path}")
    
    def run_full_preprocessing(self):
        # 1. Load the dataset
        datasets_dict = self.load_nemotron_dataset()
        
        # 2. Analyze the dataset
        analysis = self.analyze_dataset(datasets_dict)
        
        # 3. Clean and filter the dataset
        filtered_datasets = self.clean_and_filter_dataset(datasets_dict)
        
        # 4. Deduplicate the dataset
        deduplicated_datasets = self.deduplicate_dataset(filtered_datasets)
        
        # 5. Format for Qwen2
        formatted_datasets = self.format_for_qwen2(deduplicated_datasets)
        
        # 6. Create train/val/test splits
        split_datasets = self.create_train_val_test_split(formatted_datasets)
        
        # 7. Implement curriculum learning for training dataset
        split_datasets["train"] = self.implement_curriculum(split_datasets["train"])
        
        # 8. Save processed datasets
        self.save_processed_datasets(split_datasets)
        
        logger.info("Preprocessing complete!")
        return split_datasets


if __name__ == "__main__":
    preprocessor = NemotronPreprocessor(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        output_dir="./preprocessed_data",
        max_seq_length=2048,
        max_samples_per_split=500000  # Set to None to use all data
    )
    
    processed_datasets = preprocessor.run_full_preprocessing()
    
    for split_name, dataset in processed_datasets.items():
        print(f"{split_name.capitalize()} dataset: {len(dataset)} examples")