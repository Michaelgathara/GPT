# GPT From Scratch

A PyTorch implementation of a GPT-style transformer language model trained from scratch, featuring modern training optimizations and both custom BPE tokenization and inference capabilities.

[![Open In Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/purelyunfunctionalai/gibberishgpt)

## Features

- **Modern Transformer Architecture**: Implementation based on the GPT architecture with SwiGLU activation functions
- **Advanced Optimizations**:
  - Flash Attention for improved performance on compatible hardware
  - Mixed precision training (FP16)
  - Gradient checkpointing (optional)
  - Distributed training with DDP (DataParallel)
- **Custom BPE Tokenization**: Uses a custom-trained BPE tokenizer with 25K vocab size
- **Performance Monitoring**: TensorBoard integration for tracking training metrics
- **Flexible Training**: Supports dataset options including WikiText-103 and FineWeb-Edu

## Model Architecture

- Embedding dimension: 768
- Number of attention heads: 12
- Number of transformer layers: 12
- Context size: 512 tokens
- Total parameters: ~152M

## Requirements

```bash
# Core dependencies
uv pip install torch
uv sync --no-build-isolation  # Optional, for hardware that supports it

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
- **gpt_custom_BPE.py**: Main training script
- **inference.py**: Text generation script for trained models

## Usage

### Training

To train the model from scratch:

```bash
source .venv/bin/activate
cd models/
python gpt_custom_BPE.py
```

This will:
1. Download and preprocess the WikiText dataset
2. Train (or load) a BPE tokenizer
3. Tokenize the dataset
4. Train the transformer model using distributed data parallel
5. Save checkpoints to the `checkpoints/` directory

### Inference

To generate text with a trained model:

```bash
python inference.py
```

This will:
1. Load a checkpoint (you'll be prompted to choose one)
2. Allow you to enter prompts and generate continuations
3. Exit when you type 'exit'

## Training Details

- **Optimizer**: AdamW with weight decay and cosine learning rate scheduling
- **Batch size**: 72 per GPU (configurable)
- **Learning rate**: 1e-3 with warmup
- **Gradient accumulation**: 4 steps
- **Mixed precision**: FP16 training enabled
- **Evaluation**: Every 100 iterations on validation set

## Acknowledgements

This implementation draws inspiration from:
- The GPT architecture by OpenAI
- "Attention Is All You Need" (Vaswani et al., 2017)
- The Flash Attention implementation
- Hugging Face's tokenizers and datasets libraries

## License

MIT