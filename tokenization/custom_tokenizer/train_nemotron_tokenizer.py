import os
import sys
import multiprocessing
from functools import partial

base_folder = os.path.abspath("../..")
sys.path.append(base_folder)

from data import get_llama_nemotron_data
from tokenization.custom_tokenizer.config import DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE, MIN_FREQUENCY, SPECIAL_TOKENS, NUM_CORES
from tokenization.custom_tokenizer.trainer import train_and_save_tokenizer

def extract_texts_for_tokenizer(dataset):
    all_texts = []
    
    for split in dataset:
        split_data = dataset[split]
        
        all_texts.extend(split_data['input'])
        all_texts.extend(split_data['output'])
    
    filtered_texts = [text for text in all_texts if text and len(text.strip()) > 0]
    
    print(f"Extracted {len(filtered_texts)} texts for tokenizer training")
    return filtered_texts

def main():
    print("Loading llama_nemotron dataset...")
    dataset = get_llama_nemotron_data()
    
    print("Extracting texts for tokenizer training...")
    texts = extract_texts_for_tokenizer(dataset)
    
    print(f"Saving extracted texts to {DATA_PATH}")
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    
    print("Training tokenizer...")
    train_and_save_tokenizer()
    
    print("Done!")

if __name__ == "__main__":
    main()