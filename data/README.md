# Data Module

This module provides easy access to two datasets:

1. `get_fineweb_data()`: Loads the HuggingFaceFW/fineweb-edu dataset (1.43B training rows)
2. `get_wikitext_data()`: Loads the Salesforce/wikitext-103-raw-v1 dataset (1.8M training rows)

## Quick Usage

```python
# Import the functions
from data import get_fineweb_data, get_wikitext_data

# Load the 10 billion token sample
fineweb_sample_dataset = get_fineweb_data(type=1)
print(fineweb_sample_dataset)

# To load the full dataset (requires significant disk space and download time)
# In our code, we stream this dataset to help with disk space constraints
fineweb_full_dataset = get_fineweb_data(type=0)
print(fineweb_full_dataset)

wikitext_dataset = get_wikitext_data()
print(wikitext_dataset)

train_data = wikitext_dataset['train']
save_data(train_data, "./wikitext_train_saved")
loaded_train_data = load_data("./wikitext_train_saved")
```

## Dataset Details

### Wikitext Dataset
WikiText-103 is a widely used benchmark dataset composed of high-quality articles from Wikipedia. It's smaller than FineWeb but excellent for testing and ablation studies.
- Link: https://huggingface.co/datasets/Salesforce/wikitext
- Splits:
    - Training: 1.8M rows
    - Validation: 3.76k rows
    - Test: 4.36k rows

### FineWeb-Edu Dataset
FineWeb-Edu is a massive ~1.5 trillion token dataset derived from the CommonCrawl web scrape, specifically filtered for high-quality educational content. It's an excellent resource for pre-training robust language models. The version used here is processed by Hugging Face.
- Link: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Size:
    - Full Training: 1.43B rows
    - Subsets: ~10M rows

## Setup Requirements

1. Install the Hugging Face CLI:
```bash
pip install -U "huggingface_hub[cli]"
```

2. Login to Hugging Face (required for FineWeb-Edu access):
```bash
huggingface-cli login
```

## Data Loading for Training
The main training script (`models/gpt_training.py`) is primarily set up to use streamed datasets from Hugging Face (like FineWeb-Edu). The data loading functions (`get_fineweb_data`, `get_wikitext_data`) return Hugging Face Dataset or DatasetDict objects, which are then further processed (tokenized and batched) within the training script.

Refer to [models/README.md](../models/README.md) for details on how these datasets are consumed during model training.

## (Optional) Using Your Own Dataset
While this project provides helpers for FineWeb and WikiText, you can adapt the principles for your own custom text datasets:

1. Format: Your data should ideally be in plain text files, with one document or segment per line, or in a format easily loadable by Hugging Face Datasets (e.g., JSON lines, CSV).
2. Loading: Use Hugging Face load_dataset to load your custom files.
3. Cleaning: Apply a clean_textdata function with your own custom cleaning logic to each example in your dataset using the .map() method. `This is dependent on the type of data you have, some datasets require more cleaning than others`
