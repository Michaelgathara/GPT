# Understanding Tokenization for LLMs

Tokenization is a fundamental first step in any Natural Language Processing (NLP) pipeline, especially for Large Language Models (LLMs). It's the process of converting raw text into a sequence of smaller units called "tokens," which are then mapped to numerical IDs that the model can process.

## Default Approach: Using the Pre-trained GPT-2 Tokenizer

For simplicity, robustness, and to get started quickly, this project's main training script (`models/gpt_training.py`) uses the standard, pre-trained **GPT-2 tokenizer** from the Hugging Face `transformers` library.

* **How it's loaded in `models/gpt_training.py`:**
    ```python
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Ensure a padding token is set if not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # logger.info(f"Set tokenizer pad_token to eos_token: {tokenizer.pad_token}")
    ```
* **Key Properties of the GPT-2 Tokenizer:**
    * **Vocabulary Size:** 50,257. This includes tokens for common words, subwords, punctuation, and special control tokens.
    * **Algorithm:** It's based on Byte Pair Encoding (BPE), which efficiently handles a large vocabulary and out-of-vocabulary words by breaking them down into subword units.
    * **Special Tokens:**
        * `eos_token` (End Of Sequence): Marks the end of a text sequence. ID: 50256.
        * `bos_token` (Beginning Of Sequence): Marks the beginning of a text sequence. ID: 50256 (GPT-2 often uses the same ID for BOS and EOS).
        * `pad_token`: Used for padding shorter sequences in a batch to the same length. In this project, it's set to `eos_token` if not already defined.
* **Advantages of Using the GPT-2 Tokenizer:**
    * **Well-Tested and Robust:** It has been trained on a massive and diverse corpus.
    * **Widely Compatible:** Many pre-trained models and resources use it.
    * **Good Performance:** Generally provides a strong baseline for English language tasks.
    * **No Training Required:** You can use it directly without any setup.

When `models/gpt_training.py` processes data, it takes the raw text from the dataset (e.g., FineWeb-Edu), applies this GPT-2 tokenizer to convert the text into `input_ids`, and then prepares these IDs for the model.

## (Optional / Advanced) Training Your Own Custom BPE Tokenizer

This repository also includes a complete module for training your own Byte Pair Encoding (BPE) tokenizer from scratch, located in the `tokenization/custom_tokenizer/` directory.

* **Why train a custom tokenizer?**
    * **Domain-Specific Vocabulary:** If you're working with highly specialized text (e.g., medical, legal, specific coding languages) where common words or subwords differ significantly from general web text.
    * **Different Languages:** The standard GPT-2 tokenizer is optimized for English. For other languages, a custom tokenizer trained on that language's corpus is usually necessary.
    * **Research & Experimentation:** To explore different vocabulary sizes, BPE merge strategies, or special token configurations.
    * **Efficiency:** Potentially create a smaller or more efficient tokenizer for a specific task.

* **Overview of the Custom Tokenizer Module (`tokenization/custom_tokenizer/`):**
    * **Scripts:**
        * `custom_tokenizer/config.py`: Defines parameters like `VOCAB_SIZE`, `MIN_FREQUENCY`, and `SPECIAL_TOKENS` for your custom tokenizer.
        * `custom_tokenizer/data_processing.py`: Contains functions to load a dataset (e.g., WikiText-103 using `data.get_wikitext_data()`) and apply text cleaning (using `data.clean_textdata()`) to prepare a raw text corpus for tokenizer training.
        * `custom_tokenizer/trainer.py`: Implements the logic for creating and training the BPE tokenizer using the `tokenizers` library. The `load_tokenizer()` function here can also initiate training if a tokenizer file isn't found.
        * `custom_tokenizer/train_tokenizer.py`: A simple script to run the full custom tokenizer training process.
        * `tokenization/custom_tokenizer.py` (top-level in `tokenization/`): A script that orchestrates the cleaning of WikiText-103 and then trains and saves a BPE tokenizer.
    * **Process:**
        1.  A raw text corpus is prepared (e.g., `wikitext-103-train.txt`).
        2.  The `BpeTrainer` from the `tokenizers` library is configured with your desired vocabulary size, minimum token frequency, and special tokens.
        3.  The tokenizer is trained on the prepared corpus.
        4.  The trained tokenizer (including its vocabulary and merge rules) is saved to a JSON file (e.g., `custom_tokenizer.json`).
    * **To train a custom tokenizer (example using the top-level script):**
        ```bash
        # Ensure you are in the root directory of the project (gpt/)
        python tokenization/custom_tokenizer.py
        ```
        This will process WikiText-103 and save `custom_tokenizer.json` in the `tokenization/` directory.
    * **Using a Custom Tokenizer:**
        If you train a custom tokenizer and want to use it with the main model:
        1.  You would need to modify `models/gpt_training.py` to load your `custom_tokenizer.json` file using `Tokenizer.from_file("path/to/your/custom_tokenizer.json")` from the `tokenizers` library.
        2.  **Crucially**, you must update the `vocab_size` parameter in `models/transformer_setup/params.py` to match the vocabulary size of your custom tokenizer. The model's embedding layer and output layer dimensions depend on this.
        3.  Ensure the same custom tokenizer is used for both training and inference.

* **Important Note:** While the tools are provided, the main training pipeline (`models/gpt_training.py`) is pre-configured to use the standard "gpt2" tokenizer for simplicity. Switching to a custom tokenizer is an advanced step.

## Key Takeaways

* For most users, relying on the **default pre-trained GPT-2 tokenizer** is the recommended and easiest path.
* This project provides the necessary tools for users who wish to go even deeper and **train a custom BPE tokenizer** for specific needs, understanding that this requires model configuration adjustments.