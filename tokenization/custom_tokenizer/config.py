import os, sys
import multiprocessing

BASE_FOLDER = os.path.abspath("../..")
sys.path.append(BASE_FOLDER)

DATA_PATH = f"{BASE_FOLDER}/tokenization/fineweb_tokenizer.txt"
INEWEB_TOKENIZER_PATH = f"{BASE_FOLDER}/tokenization/fineweb_tokenizer.json"

NUM_CORES = max(1, multiprocessing.cpu_count()) - 1

VOCAB_SIZE = 32000  
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "[PAD]", 
    "[UNK]", 
    "[CLS]", 
    "[SEP]", 
    "[MASK]", 
    "[BOS]", 
    "[EOS]"
]