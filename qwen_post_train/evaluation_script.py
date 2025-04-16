"""
Evaluation and Inference Script for Fine-tuned Qwen2-0.5B-Instruct Model
- Provides qualitative and quantitative evaluation on various tasks
- Implements MMLU, HumanEval, GSM8K evaluations for generalization testing
- Includes inference examples and comparison with base model
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        fine_tuned_model_path: str = "./output/final_model",
        output_dir: str = "./evaluation_results",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ):
        self.base_model_name = base_model_name
        self.fine_tuned_model_path = fine_tuned_model_path
        self.output_dir = output_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load fine-tuned model
        logger.info(f"Loading fine-tuned model from: {fine_tuned_model_path}")
        try:
            # First try to load as a PeftModel
            peft_config = PeftConfig.from_pretrained(fine_tuned_model_path)
            # Load the base model again for fine-tuned version to avoid GPU memory issues
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.fine_tuned_model = PeftModel.from_pretrained(
                base_model,
                fine_tuned_model_path
            )
            logger.info("Successfully loaded fine-tuned model as PeftModel")
        except Exception as e:
            logger.warning(f"Failed to load as PeftModel: {e}")
            logger.info("Trying to load as regular model")
            # Fall back to loading as a regular model
            self.fine_tuned_model = AutoModelForCausalLM.from_pretrained(
                fine_tuned_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logger.info("Successfully loaded fine-tuned model as regular model")
    
    def format_prompt(self, prompt: str) -> str:
        """Format a text prompt into the expected chat format"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate(self, model, prompt: str) -> str:
        """Generate a response from the model"""
        formatted_prompt = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=(self.temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the generated part (not the input)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    def compare_models(self, prompts: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Compare responses between base and fine-tuned models"""
        results = []
        
        for prompt in tqdm(prompts, desc="Comparing models"):
            base_response = self.generate(self.base_model, prompt)
            ft_response = self.generate(self.fine_tuned_model, prompt)
            
            results.append({
                "prompt": prompt,
                "base_model_response": base_response,
                "fine_tuned_response": ft_response
            })
        
        # Save results
        with open(os.path.join(self.output_dir, "model_comparison.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_mmlu(self, subjects: Optional[List[str]] = None, n_shots: int = 5) -> Dict[str, float]:
        """
        Evaluate the model on MMLU (Massive Multitask Language Understanding)
        """
        logger.info("Evaluating on MMLU...")
        
        # Load MMLU dataset
        mmlu_dataset = load_dataset("cais/mmlu", "all")
        
        # Filter subjects if specified
        if subjects is None:
            # Use a subset of subjects for quicker evaluation
            subjects = [
                "high_school_mathematics", 
                "college_mathematics",
                "high_school_computer_science",
                "college_computer_science", 
                "machine_learning"
            ]
        
        results = {}
        overall_scores = {"base": 0.0, "fine_tuned": 0.0}
        total_subjects = len(subjects)
        
        for subject in subjects:
            logger.info(f"Evaluating subject: {subject}")
            
            # Get the test set for this subject
            try:
                test_data = mmlu_dataset["test"].filter(lambda x: x["subject"] == subject)
            except Exception as e:
                logger.error(f"Error loading subject {subject}: {e}")
                continue
            
            # Use a subset for faster evaluation during development
            test_data = test_data.select(range(min(100, len(test_data))))
            
            # Prepare few-shot examples from dev set
            dev_data = mmlu_dataset["dev"].filter(lambda x: x["subject"] == subject)
            dev_data = dev_data.select(range(min(n_shots, len(dev_data))))
            
            few_shot_examples = ""
            for i, example in enumerate(dev_data):
                question = example["question"]
                options = [example[f"option_{i}"] for i in range(4)]
                answer_idx = "ABCD"[example["answer"]]
                
                few_shot_examples += f"Question: {question}\n"
                for j, option in enumerate(options):
                    few_shot_examples += f"{chr(65+j)}. {option}\n"
                few_shot_examples += f"Answer: {answer_idx}\n\n"
            
            # Evaluate both models
            subject_scores = {"base": 0, "fine_tuned": 0}
            total_questions = len(test_data)
            
            for i, example in enumerate(tqdm(test_data, desc=f"Evaluating {subject}")):
                question = example["question"]
                options = [example[f"option_{i}"] for i in range(4)]
                correct_idx = example["answer"]
                correct_answer = "ABCD"[correct_idx]
                
                # Format the prompt
                prompt = few_shot_examples + f"Question: {question}\n"
                for j, option in enumerate(options):
                    prompt += f"{chr(65+j)}. {option}\n"
                prompt += "Answer:"
                
                # Get predictions
                for model_name, model in [("base", self.base_model), ("fine_tuned", self.fine_tuned_model)]:
                    response = self.generate(model, prompt).strip()
                    # Extract the first letter as the answer
                    predicted_answer = response[0] if response else ""
                    
                    if predicted_answer == correct_answer:
                        subject_scores[model_name] += 1
            
            # Calculate accuracy
            base_accuracy = subject_scores["base"] / total_questions if total_questions > 0 else 0
            ft_accuracy = subject_scores["fine_tuned"] / total_questions if total_questions > 0 else 0
            
            results[subject] = {
                "base_model_accuracy": base_accuracy,
                "fine_tuned_accuracy": ft_accuracy,
                "improvement": ft_accuracy - base_accuracy
            }
            
            overall_scores["base"] += base_accuracy
            overall_scores["fine_tuned"] += ft_accuracy
            
            logger.info(f"Subject: {subject}")
            logger.info(f"  Base model accuracy: {base_accuracy:.4f}")
            logger.info(f"  Fine-tuned accuracy: {ft_accuracy:.4f}")
            logger.info(f"  Improvement: {ft_accuracy - base_accuracy:.4f}")
        
        # Calculate overall accuracy
        if total_subjects > 0:
            overall_scores["base"] /= total_subjects
            overall_scores["fine_tuned"] /= total_subjects
        
        results["overall"] = {
            "base_model_accuracy": overall_scores["base"],
            "fine_tuned_accuracy": overall_scores["fine_tuned"],
            "improvement": overall_scores["fine_tuned"] - overall_scores["base"]
        }
        
        logger.info(f"Overall MMLU Results:")
        logger.info(f"  Base model accuracy: {overall_scores['base']:.4f}")
        logger.info(f"  Fine-tuned accuracy: {overall_scores['fine_tuned']:.4f}")
        logger.info(f"  Improvement: {overall_scores['fine_tuned'] - overall_scores['base']:.4f}")
        
        # Save results
        with open(os.path.join(self.output_dir, "mmlu_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_humaneval(self, max_samples: int = 20) -> Dict[str, float]:
        """
        Evaluate the model on HumanEval (code generation)
        """
        logger.info("Evaluating on HumanEval...")
        
        # Load HumanEval dataset
        try:
            humaneval_dataset = load_dataset("openai_humaneval")["test"]
        except Exception as e:
            logger.error(f"Error loading HumanEval dataset: {e}")
            return {}
        
        # Use a subset for faster evaluation during development
        humaneval_dataset = humaneval_dataset.select(range(min(max_samples, len(humaneval_dataset))))
        
        results = []
        
        for item in tqdm(humaneval_dataset, desc="Evaluating HumanEval"):
            task_id = item["task_id"]
            prompt = item["prompt"]
            
            # Generate code from both models
            base_completion = self.generate(self.base_model, f"Complete the following Python function:\n\n{prompt}")
            ft_completion = self.generate(self.fine_tuned_model, f"Complete the following Python function:\n\n{prompt}")
            
            # Save the completions
            results.append({
                "task_id": task_id,
                "prompt": prompt,
                "base_completion": base_completion,
                "fine_tuned_completion": ft_completion
            })
        
        # Save results (we're not implementing actual execution testing here)
        with open(os.path.join(self.output_dir, "humaneval_completions.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved {len(results)} HumanEval completions")
        
        return {"samples_evaluated": len(results)}
    
    def evaluate_gsm8k(self, max_samples: int = 50) -> Dict[str, float]:
        """
        Evaluate the model on GSM8K (math reasoning)
        """
        logger.info("Evaluating on GSM8K...")
        
        # Load GSM8K dataset
        try:
            gsm8k_dataset = load_dataset("gsm8k", "main")["test"]
        except Exception as e:
            logger.error(f"Error loading GSM8K dataset: {e}")
            return {}
        
        # Use a subset for faster evaluation
        gsm8k_dataset = gsm8k_dataset.select(range(min(max_samples, len(gsm8k_dataset))))
        
        # Track correct answers
        correct_base = 0
        correct_ft = 0
        total = len(gsm8k_dataset)
        results = []
        
        for item in tqdm(gsm8k_dataset, desc="Evaluating GSM8K"):
            question = item["question"]
            answer_str = item["answer"]
            
            # Extract the final numerical answer from the answer string
            try:
                # Look for patterns like "The answer is X" or just "X" at the end
                import re
                final_answer = re.findall(r"The answer is (\d+)|(\d+)$", answer_str)
                if final_answer:
                    # Extract the matched group that contains the number
                    matched_groups = final_answer[0]
                    correct_answer = next(g for g in matched_groups if g)
                else:
                    # Fall back to getting the last number in the string
                    all_numbers = re.findall(r'\d+', answer_str)
                    correct_answer = all_numbers[-1] if all_numbers else None
            except Exception as e:
                logger.warning(f"Could not extract answer from: {answer_str}. Error: {e}")
                correct_answer = None
            
            # Format the prompt with chain-of-thought instruction
            prompt = f"""Solve this math problem step by step:
{question}

Show your work and explain each step clearly. At the end, provide the final numerical answer.
"""
            
            # Generate solutions from both models
            base_solution = self.generate(self.base_model, prompt)
            ft_solution = self.generate(self.fine_tuned_model, prompt)
            
            # Extract numerical answers from solutions
            base_numbers = re.findall(r'\d+', base_solution)
            ft_numbers = re.findall(r'\d+', ft_solution)
            
            base_answer = base_numbers[-1] if base_numbers else None
            ft_answer = ft_numbers[-1] if ft_numbers else None
            
            base_correct = base_answer == correct_answer if base_answer and correct_answer else False
            ft_correct = ft_answer == correct_answer if ft_answer and correct_answer else False
            
            if base_correct:
                correct_base += 1
            if ft_correct:
                correct_ft += 1
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "base_solution": base_solution,
                "base_answer": base_answer,
                "base_correct": base_correct,
                "ft_solution": ft_solution,
                "ft_answer": ft_answer,
                "ft_correct": ft_correct
            })
        
        base_accuracy = correct_base / total if total > 0 else 0
        ft_accuracy = correct_ft / total if total > 0 else 0
        
        summary = {
            "base_model_accuracy": base_accuracy,
            "fine_tuned_accuracy": ft_accuracy,
            "improvement": ft_accuracy - base_accuracy,
            "total_samples": total
        }
        
        logger.info(f"GSM8K Results:")
        logger.info(f"  Base model accuracy: {base_accuracy:.4f}")
        logger.info(f"  Fine-tuned accuracy: {ft_accuracy:.4f}")
        logger.info(f"  Improvement: {ft_accuracy - base_accuracy:.4f}")
        
        with open(os.path.join(self.output_dir, "gsm8k_results.json"), "w") as f:
            json.dump({"summary": summary, "samples": results}, f, indent=2)
        
        return summary
    
    def run_all_evaluations(self):
        code_prompts = [
            "Write a Python function to find the longest palindromic substring in a given string.",
            "Implement a binary search tree in JavaScript with insert, delete, and search operations.",
            "Create a simple REST API in Python using Flask that supports CRUD operations for a 'User' resource.",
            "Write a function to detect cycles in a linked list.",
            "Implement the merge sort algorithm and explain its time complexity."
        ]
        
        science_prompts = [
            "Explain the process of cellular respiration and its importance.",
            "Describe the structure of DNA and how it replicates.",
            "Explain the principles of quantum mechanics in simple terms.",
            "Describe the greenhouse effect and its impact on climate change.",
            "Explain how vaccines work to provide immunity against diseases."
        ]
        
        reasoning_prompts = [
            "If a train travels at 60 mph, how long will it take to travel 150 miles?",
            "A store is offering a 20% discount on all items. If an item originally costs $85, what is the sale price?",
            "If 8 workers can build a wall in 10 days, how many workers would be needed to build the same wall in 5 days?",
            "Explain step by step how to solve the Tower of Hanoi puzzle with 3 disks.",
            "A dice is rolled twice. What is the probability of getting a sum of 7?"
        ]
        
        # Combine all prompts
        all_prompts = code_prompts + science_prompts + reasoning_prompts
        
        # Run evaluations
        logger.info("Starting comprehensive evaluation...")
        
        # 1. Model comparison on sample prompts
        logger.info("Comparing models on sample prompts...")
        comparison_results = self.compare_models(all_prompts)
        
        # 2. MMLU evaluation (optional - can be slow)
        logger.info("Running MMLU evaluation...")
        mmlu_results = self.evaluate_mmlu(n_shots=3)
        
        # 3. HumanEval for code generation
        logger.info("Running HumanEval evaluation...")
        humaneval_results = self.evaluate_humaneval(max_samples=10)
        
        # 4. GSM8K for math reasoning
        logger.info("Running GSM8K evaluation...")
        gsm8k_results = self.evaluate_gsm8k(max_samples=20)
        
        summary = {
            "models": {
                "base_model": self.base_model_name,
                "fine_tuned_model": self.fine_tuned_model_path
            },
            "mmlu": mmlu_results.get("overall", {}),
            "gsm8k": gsm8k_results,
            "humaneval_samples": humaneval_results.get("samples_evaluated", 0),
            "prompt_comparisons": len(comparison_results)
        }
        
        with open(os.path.join(self.output_dir, "evaluation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Evaluation complete! Results saved to: " + self.output_dir)
        return summary


if __name__ == "__main__":
    evaluator = ModelEvaluator(
        base_model_name="Qwen/Qwen2-0.5B-Instruct",
        fine_tuned_model_path="./output/final_model",
        output_dir="./evaluation_results"
    )
    
    summary = evaluator.run_all_evaluations()
    print(json.dumps(summary, indent=2))