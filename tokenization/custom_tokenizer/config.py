import os
import multiprocessing

BASE_FOLDER = os.path.abspath("..")
print(f"Your base folder is: {BASE_FOLDER}")

DATA_PATH = f"{BASE_FOLDER}/tokenization/wikitext-103-train.txt"
TOKENIZER_PATH = f"{BASE_FOLDER}/tokenization/custom_tokenizer.json"

NUM_CORES = max(1, multiprocessing.cpu_count())

VOCAB_SIZE = 25000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]