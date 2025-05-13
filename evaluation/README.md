# Evaluating Model Performance

After training a language model, it's crucial to evaluate its performance to understand how well it has learned the underlying patterns of the language. This directory contains scripts for assessing the quality of your trained models. Currently, it includes a script to calculate **perplexity**, a standard metric for language model evaluation.

## Why Evaluate?

Evaluation helps answer questions like:
* How well does the model predict unseen text?
* Did the changes made during training (e.g., different hyperparameters, data) improve the model?
* How does the model compare to other baseline models?

While subjective evaluation (reading generated text) is useful, quantitative metrics provide objective measures of performance.

## Perplexity Explained

**Perplexity (PPL)** is one of the most common metrics for evaluating language models. Intuitively, it measures how "surprised" the model is by a sequence of text it hasn't seen before (e.g., a test set).

* **Lower perplexity is better.** A lower PPL indicates the model is less surprised, meaning its probability distribution for predicting the next token is closer to the actual distribution of the test set.
* Mathematically, perplexity is the exponential of the average cross-entropy loss over the test set: $ PPL = \exp(\text{Average Cross-Entropy Loss}) $.
* A perplexity of $N$ roughly means that the model is as confused on average as if it had to choose uniformly among $N$ possibilities for each token.

## Calculating Perplexity (`eval_perplexity.py`)

The script `evaluation/eval_perplexity.py` calculates the perplexity of a trained model checkpoint on the **WikiText-103 test set**. I will change this to use the `FineWeb-EDU` dataset at a later date.

### Key Steps in the Script:

1.  **Load Tokenizer:** Loads the standard "gpt2" tokenizer (`AutoTokenizer.from_pretrained("gpt2")`) used during training. *(Note: If you trained with a custom tokenizer, this script would need modification)*.
2.  **Load Model:**
    * Takes the path to a model checkpoint (`.pt` file) as input.
    * Loads the model configuration and weights using the `load_trained_model` utility (similar to `inference.py`).
    * Sets the model to evaluation mode (`model.eval()`) and moves it to the appropriate device (GPU/CPU).
3.  **Load Test Data:** Fetches the WikiText-103 dataset (`get_wikitext_data()`) and specifically uses the `test` split.
4.  **Tokenize Test Data:** Encodes the entire WikiText-103 test set into a single sequence of token IDs.
5.  **Create DataLoader:**
    * Uses a custom `TokenizedDataset` class to create sliding windows of `(input_sequence, target_sequence)` pairs from the tokenized test data, based on the model's `block_size`.
    * Creates a `DataLoader` to efficiently iterate over these pairs in batches.
6.  **Calculate Loss:**
    * Iterates through the `DataLoader`.
    * For each batch, performs a forward pass (`model(x, y)`) to get the cross-entropy loss *without* computing gradients (`torch.no_grad()`).
    * Accumulates the total loss and the total number of tokens processed.
7.  **Calculate Perplexity:**
    * Computes the average cross-entropy loss per token (`total_loss / total_tokens`).
    * Calculates perplexity using the formula $ PPL = \exp(\text{Average Loss}) $.
8.  **Output:** Prints the calculated average loss and perplexity.

### How to Run the Evaluation Script

1.  **Navigate to the `evaluation` directory:**
    ```bash
    cd evaluation
    ```
2.  **Run the script, providing the path to your checkpoint:**
    ```bash
    python eval_perplexity.py --checkpoint ../models/checkpoints_1B/best_model.pt
    ```
    *(Adjust the path `../models/checkpoints_1B/best_model.pt` to point correctly to your desired checkpoint file relative to the `evaluation/` directory, or use an absolute path).*

### Interpreting the Output

The script will output something like:
```bash
Evaluation Results:
Average Loss: 3.5123
Perplexity: 33.53
```

* **Average Loss:** The average cross-entropy loss per token on the test set. Lower is better.
* **Perplexity:** The exponentiated average loss. Lower is better. You can use this value to compare different checkpoints or model configurations trained under similar conditions.

## Limitations of Perplexity

While perplexity is a useful intrinsic metric, it has limitations:
* It doesn't always perfectly correlate with performance on downstream tasks (e.g., translation, summarization).
* It's sensitive to the tokenization scheme used. Comparing PPL across models with different tokenizers can be misleading.
* It doesn't capture aspects like factual accuracy, coherence over long ranges, or fairness.

Therefore, it's best used in conjunction with other evaluation methods, including qualitative analysis of generated text and performance on specific downstream tasks like coding arena.
