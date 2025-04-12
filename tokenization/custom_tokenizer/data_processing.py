import os, sys
base_folder = os.path.abspath("../..")
sys.path.append(base_folder)

from datasets import Dataset, concatenate_datasets
from tokenization.custom_tokenizer.config import DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE, MIN_FREQUENCY, SPECIAL_TOKENS, NUM_CORES
from data import get_llama_nemotron_data, clean_textdata

def load_dataset() -> Dataset:
    dataset = get_llama_nemotron_data()
    
    combined_dataset = concatenate_datasets([dataset['code'], dataset['science']])
    
    print(f"Total examples in combined dataset: {len(combined_dataset)}")
    return combined_dataset

def clean_and_save_dataset(dataset: Dataset):
    input_texts = dataset['input']
    output_texts = dataset['output']
    
    all_texts = input_texts + output_texts
    
    def clean_nemotron_text(text):
        text = text.strip()
        return text
    
    cleaned_texts = [clean_nemotron_text(text) for text in all_texts if text]
    
    print(f"Prepared {len(cleaned_texts)} texts for tokenizer training")
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_texts) + "\n")
    
    return DATA_PATH