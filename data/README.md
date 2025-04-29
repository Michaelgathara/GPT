# Data Module

This module provides easy access to two datasets:

1. `get_fineweb_data()`: Loads the HuggingFaceFW/fineweb-edu dataset (1.43B training rows)
2. `get_wikitext_data()`: Loads the Salesforce/wikitext-103-raw-v1 dataset (1.8M training rows)

## Quick Usage

```python
# Import the functions
from data import get_fineweb_data, get_wikitext_data

# Load fineweb dataset
fineweb = get_fineweb_data()  # Returns HuggingFaceFW/fineweb-edu "default" config

# Load wikitext dataset
wikitext = get_wikitext_data()  # Returns wikitext-103-raw-v1 with train/test/val splits
```

## Dataset Details

### Wikitext Dataset
- Link: https://huggingface.co/datasets/Salesforce/wikitext
- Splits:
    - Training: 1.8M rows
    - Validation: 3.76k rows
    - Test: 4.36k rows

### FineWeb-Edu Dataset
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