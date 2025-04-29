import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from .config import DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE, MIN_FREQUENCY, SPECIAL_TOKENS

def create_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS
    )

    return tokenizer, trainer

def train_and_save_tokenizer():
    tokenizer, trainer = create_tokenizer()
    print(f"Data Path: {DATA_PATH}")
    
    # Check if training data exists, if not ask to download it
    if not os.path.exists(DATA_PATH):
        print(f"\nTraining data not found at {DATA_PATH}")
        response = input("Would you like to download the WikiText-103 dataset? (y/n): ").lower().strip()
        
        if response == 'y':
            print("Downloading WikiText-103 dataset...")
            try:
                from datasets import load_dataset
                ds_train = load_dataset("wikitext", "wikitext-103-v1", split="train")
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
                
                # Write the dataset to file
                with open(DATA_PATH, "w", encoding="utf-8") as f:
                    for example in ds_train:
                        f.write(example["text"].rstrip() + "\n")
                print(f"Wrote {len(ds_train)} documents to {DATA_PATH}")
            except ImportError:
                print("Please install the datasets library: pip install datasets")
                raise
            except Exception as e:
                print(f"Failed to download dataset: {str(e)}")
                raise
        else:
            print("Download cancelled. Please provide the training data manually at:", DATA_PATH)
            return

    # Train and save the tokenizer
    tokenizer.train([DATA_PATH], trainer)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer trained and saved at: {TOKENIZER_PATH}")

def load_tokenizer():
    """Loads the tokenizer from file if it exists; otherwise, trains and saves a new one."""
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer file not found at {TOKENIZER_PATH}. Training a new one...")
        train_and_save_tokenizer()

    # Load the tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"Tokenizer loaded from {TOKENIZER_PATH}")
    return tokenizer  