import os
import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import time

base_folder = os.path.abspath("../..")
sys.path.append(base_folder)

from data import get_fineweb_data 
from tokenization.custom_tokenizer.config import (
    FINEWEB_TOKENIZER_PATH, 
    VOCAB_SIZE,
    MIN_FREQUENCY,
    SPECIAL_TOKENS
)
# --- Configuration ---
OUTPUT_TOKENIZER_PATH = FINEWEB_TOKENIZER_PATH
TARGET_VOCAB_SIZE = VOCAB_SIZE
TARGET_MIN_FREQUENCY = MIN_FREQUENCY 
TOKENS_TO_DEFINE = SPECIAL_TOKENS

# How many examples from the sample to train on (adjust based on memory/time)
# sample-10BT is ~10B tokens, processing even a fraction takes time.
# Let's aim for a smaller subset initially for faster iteration.
NUM_TRAINING_EXAMPLES = 5_000_000 # Example: Train on 5 million documents from the sample

# --- Data Iterator Function ---
def fine_web_iterator(dataset, num_examples):
    """
    Yields text examples from the streamed FineWeb dataset.
    """
    print(f"Starting iteration over FineWeb sample for {num_examples} examples...")
    count = 0
    start_time = time.time()
    try:
        for example in dataset:
            if example and 'text' in example and example['text']:
                yield example['text']
                count += 1
                if count % 100000 == 0: # Log progress
                     elapsed = time.time() - start_time
                     print(f"  ... processed {count}/{num_examples} examples ({elapsed:.2f}s)")
                if count >= num_examples:
                    print(f"Reached target number of examples ({num_examples}).")
                    break
            else:
                 print(f"Warning: Skipping empty or invalid example at count {count}")
    except Exception as e:
         print(f"Error during dataset iteration at example {count}: {e}")
    finally:
        print(f"Finished iteration after {count} examples.")
        if count < num_examples:
            print(f"Warning: Only yielded {count} examples, less than the target {num_examples}.")


# --- Tokenizer Training ---
def train_fineweb_tokenizer():
    print("Initializing Tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]")) 
    tokenizer.pre_tokenizer = Whitespace()

    print("Initializing Trainer...")
    trainer = BpeTrainer(
        vocab_size=TARGET_VOCAB_SIZE,
        min_frequency=TARGET_MIN_FREQUENCY,
        special_tokens=TOKENS_TO_DEFINE
    )

    print("Loading FineWeb dataset sample (streaming)...")
    try:
        dataset_sample = get_fineweb_data(name="sample-10BT", streaming=True, split="train")
    except Exception as e:
         print(f"Failed to load FineWeb sample: {e}")
         return

    data_iterator = fine_web_iterator(dataset_sample, NUM_TRAINING_EXAMPLES)

    print("Starting Tokenizer Training...")
    start_train_time = time.time()
    try:
        tokenizer.train_from_iterator(data_iterator, trainer=trainer, length=NUM_TRAINING_EXAMPLES)
    except Exception as e:
        print(f"Error during tokenizer training: {e}")
        return
    end_train_time = time.time()
    print(f"Training completed in {end_train_time - start_train_time:.2f} seconds.")


    print("Adding Post-Processor...")
    # 5. Add Post-Processor (Optional, but good practice)
    # Example: Add BOS/EOS tokens if needed for your training setup
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")

    if bos_token_id is not None and eos_token_id is not None:
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", bos_token_id),
                ("[EOS]", eos_token_id)
            ]
        )
    else:
        print("Warning: [BOS] or [EOS] not found in vocabulary after training. Skipping post-processor.")


    print(f"Saving tokenizer to: {OUTPUT_TOKENIZER_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(OUTPUT_TOKENIZER_PATH)

    print("Tokenizer training and saving finished!")

if __name__ == "__main__":
    train_fineweb_tokenizer()