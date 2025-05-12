# GPT From Scratch

A PyTorch implementation of a GPT-style transformer language model trained from scratch, featuring modern training optimizations and both custom BPE tokenization and inference capabilities.

[![Open In Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/purelyunfunctionalai/gibberishgpt)

## Features

- **Modern Transformer Architecture**: Implementation based on the GPT architecture with SwiGLU activation functions
- **Advanced Optimizations**:
  - Flash Attention for improved performance on compatible hardware
  - Mixed precision training (FP16)
  - Gradient checkpointing (optional)
- **Custom BPE Tokenization**: Uses the GPT2 tokenizer with a 52k vocab size
- **Performance Monitoring**: TensorBoard integration for tracking training metrics (deprecated for now)
- **Dataset**: Supports dataset options like (FineWeb-Edu)[https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu]

## Model Architecture
The main model trained under this repo has the following parameters:
- Embedding dimension: 1536
- Number of attention heads: 12
- Number of transformer layers: 24
- Context size: 1024 tokens
- Total parameters: ~1.1B

## Documentation
1.  **[Foundational Concepts & Further Reading](./PAPERS.md)**: (Optional) Understand the key research behind LLMs.
2.  **[Setting Up Your Data](./data/README.md)**: Learn how to acquire and prepare the datasets.
3.  **[Understanding Tokenization](./tokenization/README.md)**: Discover how text is converted into a format models can understand and how to train your tokenizer.
4.  **[The Model Architecture](./models/README.md)**: Dive into the components of our Transformer model.
5.  **[Training the Model](./training/README.md)**: Step-by-step guide to train your LLM. *(Note: Training scripts are currently in `models/` but will be documented as if in `training/` as per the new structure)*
6.  **[Using Your Trained Model - Inference](./models/README.md#inference)**: Generate text with your trained model. *(This might be a subsection of the models README or a separate scripts README later)*
7.  **[Evaluating Model Performance](./evaluation/README.md)**: Assess the quality of your trained LLM.
8.  **[Working with Hugging Face](./hugging_face/README.md)**: Make your model compatible with and share it on the Hugging Face Hub.
9.  **[Utility Scripts](./scripts/README.md)**: Explore helper scripts for various tasks.
10. **[Testing Your Code](./tests/README.md)**: (Future) Learn about the testing setup.


## Requirements
1. Install `uv` from (here)[https://docs.astral.sh/uv/getting-started/installation/]

2. Sync libraries, 
```bash
uv sync
```

```bash
# Core dependencies, if they did not sync
uv pip install torch
uv pip install flash-attn --no-build-isolation  # this might fails, but the error will tell you exactly how to fix it

# Hugging Face access (for datasets)
huggingface-cli login
```

## Project Structure

- **transformer_setup/**: Core transformer model implementation
  - `params.py`: Model configuration
  - `transformer.py`: Transformer model classes including attention mechanisms
- **tokenization/**: Tokenizer training and utilities
  - `custom_tokenizer/`: BPE tokenizer implementation
- **data/**: Dataset loading and preprocessing
  - `wikitext_data.py`: Load and process WikiText-103
  - `fineweb_data.py`: Load and process FineWeb datasets
  - `clean_text.py`: Text cleaning utilities
- **models/**: Training and inference scripts
  - `gpt.py`: Main training script
  - `inference.py`: Text generation script for trained models
- **evaluation/**: Testing and evaluation scripts
  - `eval_perplexity.py`: Calculate perplexity score for saved model

## Usage

### Training

To train the model from scratch:

```bash
source .venv/bin/activate
cd models/
python3 gpt.py
# to run in the background and out prints/logs to a file
nohup python3 -u gpt.py > train.log 2>&1 &
```

This will:
1. Download and preprocess the FineWeb-Edu dataset
2. Load up the GPT2 tokenizer
3. Start streaming and tokenizing the dataset
4. Train the transformer model
5. Save checkpoints to the `checkpoints_1B/` directory

### Inference

To generate text with a trained model:

```bash
python3 inference.py checkpoints_1B/best_model.pt
```

This will:
1. Load the checkpoint pt file
2. Allow you to enter prompts and generate continuations
3. Exit when you type 'exit'

## Training Details

- **Optimizer**: AdamW with weight decay and cosine learning rate scheduling
- **Batch size**: 72 per GPU (configurable)
- **Learning rate**: 6e-4 with warmup
- **Gradient accumulation**: 4 steps
- **Mixed precision**: FP16 training enabled
- **Evaluation**: Every 1000 iterations on validation set

## Acknowledgements

This implementation draws inspiration from:
- The GPT architecture by OpenAI
- "Attention Is All You Need" (Vaswani et al., 2017)
- The Flash Attention implementation
- Hugging Face's tokenizers and datasets libraries

## License

MIT