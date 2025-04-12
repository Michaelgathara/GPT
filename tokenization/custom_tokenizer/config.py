import os, sys
import multiprocessing

BASE_FOLDER = os.path.abspath("../..")
sys.path.append(BASE_FOLDER)

DATA_PATH = f"{BASE_FOLDER}/tokenization/llama_nemotron_train.txt"
TOKENIZER_PATH = f"{BASE_FOLDER}/tokenization/llama_nemotron_tokenizer.json"

NUM_CORES = max(1, multiprocessing.cpu_count())

VOCAB_SIZE = 32000  
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "[PAD]", 
    "[UNK]", 
    "[CLS]", 
    "[SEP]", 
    "[MASK]", 
    "[BOS]", 
    "[EOS]",
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<think>",
    "system",
    "user",
    "assistant"
]