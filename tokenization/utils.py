import os
from tokenizers import Tokenizer

base_folder = os.path.abspath("../")
sys.path.append(base_folder)

from tokenization.custom_tokenizer.config import FINEWEB_TOKENIZER_PATH

def load_fineweb_tokenizer() -> Tokenizer:
    path = FINEWEB_TOKENIZER_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FineWeb tokenizer file not found at {path}. "
            f"Please run the training script first (e.g., train_fineweb_tokenizer.py)."
        )
    print(f"Loading FineWeb tokenizer from: {path}")
    try:
        tokenizer = Tokenizer.from_file(path)
        print("FineWeb tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer from {path}: {e}")
        raise

# Example Usage:
# if __name__ == '__main__':
#     try:
#         fw_tokenizer = load_fineweb_tokenizer()
#         print(f"Tokenizer Vocab Size: {fw_tokenizer.get_vocab_size()}")
#         sample_text = "This is a test sentence using the FineWeb tokenizer."
#         encoded = fw_tokenizer.encode(sample_text)
#         print(f"Sample Text: '{sample_text}'")
#         print(f"Encoded Tokens: {encoded.tokens}")
#         print(f"Encoded IDs: {encoded.ids}")
#     except FileNotFoundError as e:
#         print(e)
#     except Exception as e:
#         print(f"An error occurred: {e}")