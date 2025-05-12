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
This project is structured to guide you from the foundational concepts of Large Language Models (LLMs) to training, evaluating, and using your own GPT-style model from scratch. We recommend navigating through the documentation modules in the following order:

1.  **[Foundational Concepts & Further Reading](./PAPERS.md)**: (Optional) Understand the key research behind LLMs.
2.  **[Setting Up Your Data](./data/README.md)**: Learn how to acquire and prepare the datasets.
3.  **[Understanding Tokenization](./tokenization/README.md)**: Discover how text is converted into a format models can understand and how to train/use a tokenizer.
4.  **[The Model Architecture](./models/README.md)**: Dive into the components of our Transformer model (defined in `models/transformer_setup/`).
5.  **[Training the Model](./models/README.md#training-the-model)**: Step-by-step guide to train your LLM (using `models/gpt.py`).
6.  **[Using Your Trained Model - Inference](./models/README.md#inference)**: Generate text with your trained model (using `models/inference.py`).
7.  **[Evaluating Model Performance](./evaluation/README.md)**: Assess the quality of your trained LLM.
8.  **[Working with Hugging Face](./hugging_face/README.md)**: Make your model compatible with and share it on the Hugging Face Hub.

Each linked `README.md` file (or section) within the respective directories serves as a chapter, explaining the purpose and usage of the code within that module.

## Requirements

1.  Install `uv` from [here](https://docs.astral.sh/uv/getting-started/installation/).
2.  Create a virtual environment and sync dependencies:
    ```bash
    uv venv # Create .venv
    # Activate the environment
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows (PowerShell):
    # .\.venv\Scripts\Activate.ps1
    # On Windows (CMD):
    # .\.venv\Scripts\activate.bat
    uv sync
    ```
3.  If `flash-attn` did not install correctly via `uv sync` (common on some platforms), you might need to install it separately. Ensure you have the compatible NVIDIA CUDA toolkit installed and that your GPU supports FlashAttention (Ampere architecture or newer).
    ```bash
    # Example, adjust for your PyTorch and CUDA version if necessary
    uv pip install flash-attn --no-build-isolation
    ```
4.  Login to Hugging Face CLI (required for downloading some datasets like FineWeb-Edu and for uploading models):
    ```bash
    huggingface-cli login
    ```

## Project Structure Overview

This project is organized into several key directories:

-   **[`data/`](./data/README.md)**: Modules for acquiring, preprocessing, and loading datasets.
    -   `preprocessing/`: Scripts for cleaning text data.
-   **[`tokenization/`](./tokenization/README.md)**: BPE tokenizer implementation, training scripts (for a fully custom tokenizer), and configuration.
-   **[`models/`](./models/README.md)**: Core model architecture (`transformer_setup/`), main training script (`gpt_training.py`), and inference script (`inference.py`).
-   **[`evaluation/`](./evaluation/README.md)**: Scripts for evaluating trained models.
-   **[`hugging_face/`](./hugging_face/README.md)**: Modules for Hugging Face Hub integration.
-   **[`PAPERS.md`](./PAPERS.md)**: Curated list of influential research papers.
-   `.gitignore`, `LICENSE`, `pyproject.toml`, `requirements.txt`, `STRUCTURE.md`: Standard project and configuration files.


## Quick Usage
Check out the READMEs above to get a detailed usage reference 

### Training

To train the model from scratch:

```bash
cd models/
# For background training and logging:
nohup python3 -u gpt_training.py > train.log 2>&1 &
# For foreground training:
python3 gpt_training.py
# In a separate terminal:
tail -f train.log
# Or use the provided script from the project root:
./print_res.sh
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