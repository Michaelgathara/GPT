# The Transformer Model: Architecture, Training, and Inference

This directory is central to the LLM project, containing:
* The core model definition in the `transformer_setup/` subdirectory.
* The main script for training the model: `gpt_training.py`.
* The script for generating text with a trained model: `inference.py`.
* A utility to summarize the model architecture: `model_summary.py`.

This document will guide you through understanding each of these components.

## 1. Model Architecture (`transformer_setup/`)

The heart of this project is a GPT-style Transformer model, a decoder-only architecture designed for next-token prediction. This architecture is defined in `transformer_setup/transformer.py` and configured via `transformer_setup/params.py`.

### Introduction to the GPT-style Transformer
Our model follows the principles of Generative Pre-trained Transformers. It processes input sequences and learns to predict the subsequent token at each position. This is achieved through multiple layers of self-attention and feed-forward networks.

### Key Components (as defined in `transformer_setup/transformer.py`)

* **Embeddings:**
    * **Token Embeddings (`nn.Embedding`):** Maps input token IDs (from the tokenizer) to dense vector representations. The size of this embedding is `vocab_size x n_embd`.
    * **Positional Embeddings (`nn.Embedding`):** Since self-attention is permutation-invariant, these embeddings provide the model with information about the position of each token in the sequence. A learned embedding is used for each position up to `block_size` (context length).
    * The final input representation for a token is the sum of its token embedding and positional embedding.

* **Transformer Block (`Block` class):** The model consists of `n_layer` identical blocks stacked on top of each other. Each block contains:
    * **Multi-Head Self-Attention (MHSA) (`MultiHead` class):**
        * This mechanism allows the model to weigh the importance of different tokens in the input sequence when computing the representation for each token.
        * **Projections:** Input token representations are linearly projected into Query (Q), Key (K), and Value (V) vectors for each attention head.
        * **Scaled Dot-Product Attention:** Computes attention scores. A causal mask (triangular mask) is applied to ensure that a position can only attend to previous positions (and itself), which is crucial for autoregressive generation.
        * **FlashAttention:** If available (`HAS_FLASH_ATTN = True` in `transformer.py`) and configured (`use_flash_attn = True` in `params.py`), an optimized version of attention (`FlashAttentionHead`) is used. This significantly speeds up computation and reduces memory usage by minimizing reads/writes to GPU global memory, especially beneficial for long sequences. If FlashAttention is not available or disabled, a standard PyTorch implementation of attention (`Head`) is used as a fallback.
        * **Multiple Heads:** The attention mechanism is parallelized into `n_head` heads. The outputs of these heads are concatenated and linearly projected back to the embedding dimension (`n_embd`).
    * **Feed-Forward Network (FFN) (`FeedForward` class):**
        * Each block also contains a position-wise feed-forward network applied independently to each token representation.
        * This FFN uses a **SwiGLU (Swish Gated Linear Unit)** activation function, which typically offers better performance than traditional ReLU or GELU in Transformers. The architecture is: `Linear -> Swish -> Element-wise product with (Linear -> Gate) -> Linear`.
    * **Layer Normalization (`nn.LayerNorm`):** Applied *before* the self-attention and FFN sub-layers within each block (this is known as Pre-Norm). Pre-Norm generally leads to more stable training for deep Transformers. A final LayerNorm is applied after the last block.
    * **Residual Connections:** The input to each sub-layer (MHSA and FFN) is added to its output (output = `sub_layer(norm(input)) + input`). This helps with gradient flow and enables training deeper models.

* **Output Layer (`lm_head`):**
    * A final linear layer maps the output token representations from the last Transformer block to logits over the entire vocabulary. The dimensions are `n_embd x vocab_size`.
    * During training, these logits are used with a Cross-Entropy loss function to compare against the target tokens.

### Configuration (`transformer_setup/params.py`)
The `ModelConfig` class in `transformer_setup/params.py` centralizes all hyperparameters for the model architecture and training setup. Key architectural parameters include:

* `n_layer`: Number of Transformer blocks (e.g., 24).
* `n_embd`: Embedding dimension (e.g., 1536).
* `n_head`: Number of attention heads (e.g., 12). The dimension of each head will be `n_embd / n_head`.
* `block_size`: The maximum sequence length (context window) the model can process (e.g., 1024).
* `dropout`: Dropout rate applied in various layers during training (e.g., 0.1).
* `vocab_size`: (Though not directly in `params.py`, it's passed to the model during initialization and should match the tokenizer's vocabulary size).
* Flags:
    * `gradient_checkpointing`: Boolean to enable/disable gradient checkpointing in the Transformer blocks. This trades compute for memory, allowing larger models or batch sizes to be trained.
    * `use_flash_attn`: Boolean to enable/disable the use of FlashAttention (if available).

### Model Initialization and Weights
The `TransformerModel` class includes an `_init_weights` method that applies a specific initialization strategy (typically normal distribution with a small standard deviation for linear and embedding layers, and zeros/ones for LayerNorm biases/weights) to help with training stability.

### Model Summary (`model_summary.py`)
You can use the `model_summary.py` script to get a detailed look at the model's architecture, layer dimensions, parameter counts, and estimated memory usage. This is useful for verifying your configuration before starting a long training run.
```bash
# From the project root (gpt/)
python3 models/model_summary.py

# From this directory
python3 model_summary.py
```