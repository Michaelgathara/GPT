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