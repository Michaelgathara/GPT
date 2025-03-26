
import os, sys

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

base_folder = os.path.abspath("../..")
sys.path.append(base_folder)
from tokenization.custom_tokenizer.config import DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE, MIN_FREQUENCY, SPECIAL_TOKENS, NUM_CORES

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
    
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]"))
        ]
    )
    
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer trained and saved at: {TOKENIZER_PATH}")
    
def load_tokenizer():
    # print(f"Tokenizer Path: {TOKENIZER_PATH}")
    # if not os.path.exists(TOKENIZER_PATH):
    #     print(f"Tokenizer file not found at {TOKENIZER_PATH}. Training a new one...")
    #     train_and_save_tokenizer()

    tokenizer = Tokenizer.from_file("/workspace/GPT/tokenization/llama_nemotron_tokenizer.json")
    print(f"Tokenizer loaded")
    return tokenizer