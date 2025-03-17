# GPT token-level model
import multiprocessing

from data import get_wikitext_data, save_data, load_data
from tokenization import get_tiktoken_tokenizer, tiktoken_tokenize_dataset

DATA_PATH = "/workspace/GPT/data/tiktoken_tokenized_wikitext"
num_cores = multiprocessing.cpu_count()

tokenizer = get_tiktoken_tokenizer()

dataset = get_wikitext_data()
tokenized_dataset = tiktoken_tokenize_dataset(dataset, tokenizer, num_cores)
print(tokenized_dataset)

save_data(tokenized_dataset, DATA_PATH)


tokenized_dataset = load_data(DATA_PATH)
#######################################################
# GPT-2 Tokenizer Code

# import multiprocessing

# from data import get_wikitext_data, save_data, load_data
# from models.tokenizer import get_gpt2_tokenizer, tokenize_dataset

# DATA_PATH = "/workspace/GPT/data/tokenized_wikitext"
# num_cores = multiprocessing.cpu_count()

# tokenizer = get_gpt2_tokenizer()

# dataset = get_wikitext_data()
# tokenized_dataset = tokenize_dataset(dataset, tokenizer, num_cores)
# print(tokenized_dataset)
# save_data(tokenized_dataset, DATA_PATH)

# tokenized_dataset = load_data(DATA_PATH)

############################################################
# Example tokenization

# from models import get_tokenizer

# tokenizer = get_tokenizer()
# tokenizer.pad_token = tokenizer.unk_token

# print("Vocab size:", tokenizer.vocab_size)
# print("Unknown token:", tokenizer.unk_token)
# print("Pad token:", tokenizer.pad_token)

# dataset = get_wikitext_data()

# print(dataset["train"][4])

# sample_text = dataset["train"][4]["text"]
# tokens = tokenizer.tokenize(sample_text)
# token_ids = tokenizer.encode(sample_text)

# print("Original Text:", sample_text)
# print("Tokenized:", tokens)
# print("Token IDs:", token_ids)