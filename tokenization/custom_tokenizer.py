import os, sys
import multiprocessing

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

base_folder = os.path.abspath("..")
print(f"Your base folder is: {base_folder}")
sys.path.append(base_folder)
from data import get_wikitext_data, clean_textdata, save_data

DATA_PATH = f"{base_folder}/tokenization/wikitext-103-train.txt"
TOKENIZER_PATH = f"{base_folder}/tokenization/custom_tokenizer.json"

dataset = get_wikitext_data()

train_dataset = dataset["train"]

num_cores = max(1, multiprocessing.cpu_count())

print("Total lines:", len(dataset["train"]["text"]))

def clean_batch(examples):
    cleaned_texts = [clean_textdata(text) for text in examples["text"]]
    cleaned_texts = list(filter(None, cleaned_texts))
    return {"text": cleaned_texts}

cleaned_dataset = train_dataset.map(
    clean_batch,
    batched=True,
    batch_size=10_000,
    num_proc=num_cores,
    desc="Cleaning text"
)

print("Cleaned lines:", len(cleaned_dataset["text"]))
# save_data(cleaned_dataset, DATA_PATH)

with open(DATA_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_dataset["text"]) + "\n")

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=25000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
)

tokenizer.train(files=[DATA_PATH], trainer=trainer)
tokenizer.save(TOKENIZER_PATH)

print(f"Tokenizer trained and saved at: {TOKENIZER_PATH}")