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
This script uses `torchinfo` to print the summary. Ensure `torchinfo` is installed (uv pip install `torchinfo`).

## 2. Training the Model (`gpt_training.py`)
The primary script for training the Transformer model from scratch is `models/gpt_training.py`.

### Overview of the Training Process
The goal of training is to adjust the model's parameters (weights) so that it becomes proficient at predicting the next token in a sequence. This is achieved by feeding it large amounts of text data and minimizing a loss function (Cross-Entropy) that measures the difference between the model's predictions and the actual next tokens.

## **The gpt_training.py Script Details**

### Setup

- **Device Selection**: Automatically uses CUDA (GPU) if available, otherwise falls back to CPU (which will be very slow for training).
- **Seeding**: `torch.manual_seed` and `numpy.random.seed` are used for reproducibility, controlled by `config.seed` from `params.py`.

### Data Handling

- **Dataset Loading**: Typically loads a large dataset like FineWeb-Edu in streaming mode using functions from `data/` (e.g., `get_fineweb_data`).
- **Tokenizer**: Uses the pre-trained "gpt2" tokenizer from Hugging Face transformers by default (`AutoTokenizer.from_pretrained("gpt2")`). The `pad_token` is set to `eos_token` if not already defined.
- **create_token_generator Function**: This crucial function takes the raw tokenized stream from the dataset and yields `(x, y)` pairs suitable for training.
  - `x`: A chunk of `block_size` input token IDs.
  - `y`: The target token IDs, which are `x` shifted by one position (i.e., `y[i] = x[i+1]`). The last target token is typically a padding token.

### Model Initialization

- An instance of `TransformerModel` (from `transformer_setup.transformer`) is created using the hyperparameters defined in `ModelConfig` (`transformer_setup.params.py`) and the tokenizer's vocabulary size.

### Optimizer

- **AdamW** (`torch.optim.AdamW`): Used for updating model weights. It's a variant of the Adam optimizer with improved weight decay handling.
- **Configurable parameters** (from `params.py`): `learning_rate`, `weight_decay`, `beta1`, `beta2`.

### Learning Rate Scheduler (CosineWarmupScheduler)

This custom scheduler implements a learning rate schedule that is common for training Transformers:
- **Warmup Phase**: Linearly increases the learning rate from a small value (or 0) to the target `learning_rate` over `config.warmup_iters`. This helps stabilize training in the initial epochs.
- **Cosine Decay Phase**: After warmup, the learning rate gradually decays following a cosine curve down to a minimum value (or 0) by `config.max_iters`.
- The scheduler is stepped after each optimizer update.

### Training Loop

- Iterates for `config.max_iters`.
- **Gradient Accumulation**: Updates model weights only after processing `config.accumulation_steps` micro-batches. This allows for a larger effective batch size without requiring proportionally more GPU memory. The loss from each micro-batch is normalized by the accumulation steps.
- **Mixed Precision Training** (`torch.amp`):
  - Uses `torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast` (if CUDA is available) to perform computations in lower precision (like FP16 or BF16) where possible, while maintaining high precision for critical parts. This speeds up training and reduces memory usage.
- **Forward Pass**: For each micro-batch, input `x` and targets `y` are passed to the model to get logits and the loss.
- **Loss Calculation**: Cross-Entropy loss is computed between the model's logits and the target token IDs.
- **Backward Pass**: Gradients are computed. The GradScaler scales the loss to prevent underflow of gradients in mixed precision.
- **Gradient Clipping** (`torch.nn.utils.clip_grad_norm_`): Gradients are clipped to a maximum norm (e.g., 1.0) to prevent exploding gradients, which can destabilize training. This happens after unscaling by the GradScaler.
- **Optimizer Step**: The GradScaler tells the optimizer to update the model weights.
- **Scaler Update**: The GradScaler updates its scale factor for the next iteration.

### Evaluation (estimate_loss function)

- Periodically (every `config.eval_interval` iterations), the model's performance is evaluated on a validation set (if available, e.g., a subset of the training data or a dedicated validation split).
- The model is set to `eval()` mode (disabling dropout).
- Average validation loss is computed over `config.eval_iters` batches.

### Checkpointing

- The script saves checkpoints containing the model's `state_dict`, optimizer's `state_dict`, GradScaler's `state_dict`, the current iteration number, the best validation loss achieved so far, and the model configuration.
- Checkpoints are saved to the directory specified by `config.checkpoint_dir` (e.g., `checkpoints_1B/`).
- The "best" model (based on validation loss) is typically saved as `best_model.pt`. Periodic checkpoints might also be saved (e.g., `checkpoint_{iter_num}.pt`).
- **Resuming Training**: If a checkpoint exists, training can be resumed from that point, loading the model weights, optimizer state, scaler state, and iteration number. The learning rate scheduler is also advanced to the correct step.

### Logging

- Progress, loss, learning rate, and other metrics are logged to the console and to a file (e.g., `training_single_gpu.log` or `train.log` when using nohup).
- The `models/train.log` file in the repository shows an example of training output.

## **How to Run Training**

> [! TIP]
> Typically you would want to run this model on a GPU (we've only tested it on H100, H200, A100, V100, but it should work on a various set of Nvidia GPUs if you adjust the model size). We put together a guide on how to rent high-end GPUs in the `RentingGPUs.pdf` file on the main directory. 

1. Navigate to the models directory:
    ```bash
    cd models
    ```

2. Ensure your environment is activated and dependencies installed.
3. Start the training script:
    * For training in the foreground (output directly to console):
    ```bash
    python gpt_training.py
    ```
    * For training in the background and logging output to a file (recommended for long runs):
    ```bash
    nohup python3 -u gpt_training.py > train.log 2>&1 &
    ```
    * The `-u` flag ensures unbuffered output, so `train.log` updates in real-time.

4. Monitor Training (if running in background):
    * You should already have access to your terminal (if not, just press `enter`)
    * Use `tail` to view the log file:
        ```bash
        tail -f train.log
        ```
    * Alternatively, you can run the provided shell script:
        ```bash
        chmod +x print_res.sh
        ./print_res.sh
        # or if you do not have chmod access
        bash print_res.sh
        ```

## **Inference: Generating Text (`inference.py`)**
Once you have a trained model checkpoint, you can use `models/inference.py` to generate text.

### Overview
Inference is the process of using the trained model to make predictions on new inputs. For an LLM, this typically means providing a "prompt" (an initial sequence of text) and having the model generate a continuation.


* **Loading the Trained Model (**`load_trained_model` function):
   * This function handles loading the specified checkpoint file (`.pt`).
   * It retrieves the model configuration (`config`) saved within the checkpoint.
   * It loads the pre-trained "gpt2" tokenizer (`AutoTokenizer.from_pretrained("gpt2")`), ensuring `pad_token` is set.
   * It instantiates the `TransformerModel` using the loaded configuration and tokenizer's vocabulary size.
   * It loads the saved model weights (`model_state_dict`) into the instantiated model. It also handles unwrapping state dict keys if the model was saved with DistributedDataParallel (`module.` prefix).
   * The model is set to evaluation mode (`model.eval()`) and moved to the appropriate device (CUDA or CPU).
   * Optionally, `torch.compile(model, mode="reduce-overhead")` is attempted for potential inference speedup.
* **Text Generation Process (`generate_text` function and `model.generate()`)**:
   * The user-provided prompt is tokenized using the loaded tokenizer.
   * The tokenized prompt (input IDs) is fed into the `model.generate()` method. This method, defined within the `TransformerModel` class, performs autoregressive decoding:
      1. It takes the current sequence of tokens.
      2. It passes this sequence through the model to get logits for the next token.
      3. The logits for the very last position are processed (e.g., temperature scaling, top-k filtering).
      4. A new token is sampled from the resulting probability distribution.
      5. This new token is appended to the sequence.
      6. Steps 1-5 are repeated until `max_new_tokens` are generated or an End-Of-Sequence token is produced.
   * The generated sequence of token IDs is then decoded back into human-readable text using `tokenizer.decode()`.
   * Mixed precision (`torch.amp.autocast`) can be used during generation on CUDA for potential speed benefits.
* **Key Generation Parameters (Command-line arguments for `inference.py`)**:
   * `checkpoint_path` (required): Path to the model checkpoint file.
   * `--prompt` (optional): The initial text to seed the generation. If not provided, the script enters interactive mode.
   * `--max_new_tokens` (default: 200): Maximum number of new tokens to generate after the prompt.
   * `--temperature` (default: 0.8): Controls the randomness of the output.
      * Lower values (e.g., &lt; 0.8) make the output more deterministic and focused, picking more likely words.
      * Higher values (e.g., > 1.0) make the output more random and creative, potentially less coherent.
      * A value of 1.0 means sampling according to the model's learned probabilities without modification.
   * `--top_k` (default: 50): Restricts sampling to the `k` most likely next tokens. If set to 0, top-k filtering is disabled. This can help prevent the model from generating very unlikely or bizarre tokens.
   * `--device` (optional): Specify 'cuda' or 'cpu'. Auto-detects if None.
   * `--seed` (optional): Set a random seed for reproducible generation (if other parameters are also fixed).
   * `--verbose`: Print detailed loading information.

### How to Run Inference
1. **Navigate to the `models` directory**:
    ```bash
    cd models
    ```
2. **Run the script with your checkpoint and desired options**:
    * With a specific prompt:
        ```bash
        python inference.py ../checkpoints_1B/best_model.pt --prompt "Once upon a time, in a land far away, " --max_new_tokens 250 --temperature 0.7
        ```
    * In interactive mode:
        ```bash
        python inference.py ../checkpoints_1B/best_model.pt
        ```
        This script will enter a more interactive mode where you will be prompted for multiple queries until you exit. Type `exit` to stop.