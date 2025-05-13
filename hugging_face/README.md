# WORK IN PROGRESS


# Working with Hugging Face

The Hugging Face ecosystem (libraries like `transformers`, `datasets`, and the Model Hub) provides powerful tools for sharing, loading, and using pre-trained models. This module explains how to make your custom-trained Transformer model compatible with this ecosystem, upload it to the Hugging Face Hub, and test the uploaded model.

The scripts in this directory (`hugging_face/`) facilitate this process.

## Benefits of Hugging Face Integration

* **Easy Sharing:** Upload your trained model to the Hub, making it accessible to others or to yourself across different environments.
* **Standardized Loading:** Use the familiar `AutoModelForCausalLM.from_pretrained("your-username/your-model-name")` to load your custom model just like any standard Hugging Face model.
* **Inference Pipelines:** Potentially leverage Hugging Face `pipeline` objects for simplified text generation (though this might require ensuring full compatibility).
* **Community:** Become part of the wider ML community by sharing your work.

## 1. Making the Model Hugging Face Compatible (`hgtransformer.py`)

The core model defined in `models/transformer_setup/transformer.py` needs to be wrapped in classes that inherit from Hugging Face's `PreTrainedModel` and `PretrainedConfig` to be compatible with their ecosystem.

* **Script:** `hugging_face/hgtransformer.py`
* **Purpose:** This script re-defines the Transformer architecture (`Block`, `MultiHead`, `FeedForward`, etc.) but within the structure required by Hugging Face `transformers`.
* **Key Classes:**
    * **`CustomTransformerConfig(PretrainedConfig)`:** Defines the configuration class for your model. It inherits from `PretrainedConfig` and stores parameters like `vocab_size`, `embed_dim`, `num_heads`, `num_layers`, special token IDs, etc. It includes necessary attributes and methods required by `transformers`.
    * **`CustomTransformerModel(nn.Module)`:** Represents the core transformer logic, similar to `models/transformer_setup/transformer.py`, but potentially adapted to align better with HF's internal structure (e.g., handling `attention_mask`, standardized forward method signature).
    * **`CustomTransformerPreTrainedModel(PreTrainedModel)`:** A base class that handles weight initialization and provides common methods (`_init_weights`, `_set_gradient_checkpointing`) expected by Hugging Face.
    * **`CustomTransformerForCausalLM(CustomTransformerPreTrainedModel)`:** The main model class you'll interact with via `transformers`. It wraps `CustomTransformerModel`, includes the language modeling head (`lm_head`), and implements the `forward` method to return outputs in the expected format (e.g., `CausalLMOutputWithCrossAttentions`). It also defines methods needed for generation like `prepare_inputs_for_generation`.

* **Note:** The code in `hgtransformer.py` largely mirrors `models/transformer_setup/transformer.py` but is structured specifically for Hugging Face compatibility. Ensure any significant architectural changes in one are reflected in the other if you want consistent behavior. *(The current `huggingface_upload.py` script seems to use a slightly different approach where it defines simpler wrapper classes directly within the upload script and assigns the trained `TransformerModel` instance to the wrapper's attribute. The separate `hgtransformer.py` might be intended for a more robust, reusable integration or development purposes).*

## 2. Converting and Uploading the Model (`huggingface_upload.py`)

This script takes your trained checkpoint (`.pt` file) and converts it into the Hugging Face format, then optionally uploads it to the Hub.

* **Script:** `hugging_face/huggingface_upload.py`
* **Purpose:** To bridge the gap between your custom training checkpoint and a shareable Hugging Face model repository.
* **Key Steps:**
    1.  **Load Trained Model:** Uses `load_trained_model` (from `models.inference`) to load your checkpoint, tokenizer (standard "gpt2"), and original training configuration.
    2.  **Define HF Configuration:** Creates an instance of `CustomTransformerConfig` (defined within the script or imported if using `hgtransformer.py`'s definition) based on the parameters loaded from the checkpoint's config dictionary. Adds necessary fields like `architectures` and `model_type`.
    3.  **Define HF Model Wrapper:** Defines a simple wrapper class `CustomTransformerModelForCausalLM(PreTrainedModel)` that holds your trained `TransformerModel` instance. *(This is the simpler approach seen in the script, directly using the trained model instance).*
    4.  **Save Files Locally:**
        * Saves the configuration (`config.json`).
        * Saves the model weights using `hf_model.save_pretrained(output_path)`, which saves `pytorch_model.bin` (or `.safetensors`).
        * Saves the tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.) using `tokenizer.save_pretrained(output_path)`.
        * Creates a basic `README.md` file for the Hugging Face Hub repository with model details and usage examples.
    5.  **Push to Hub (Optional):**
        * If the `--push` flag is provided and a valid Hugging Face authentication token (`--token`) is given (or user is logged in via CLI), it uses `huggingface_hub` library functions (`login`, `HfApi`) to:
            * Create a new repository on the Hub (or use an existing one).
            * Upload the contents of the local save directory (`output_path`) to the specified repository (`model_name`).

### How to Run the Upload Script

1.  **Ensure you are logged in:**
    ```bash
    huggingface-cli login
    ```
2.  **Navigate to the `hugging_face` directory:**
    ```bash
    cd hugging_face
    ```
3.  **Run the script:**
    ```bash
    python huggingface_upload.py \
        --checkpoint ../models/checkpoints_1B/best_model.pt \
        --output_dir ./hf_model_output \
        --model_name "your-hf-username/your-model-name" \
        --push \
        --token "hf_YOUR_HUGGINGFACE_TOKEN" # Optional if already logged in
    ```
    * Replace `--checkpoint` path with your actual checkpoint.
    * Replace `--model_name` with the desired name on the Hugging Face Hub (e.g., `purelyfunctionalai/gibberishgpt`).
    * Use `--push` to upload; omit it to only save locally in `--output_dir`.
    * Provide your Hugging Face token if needed (usually required for writing/uploading unless login credentials are cached effectively).

## 3. Testing the Hugging Face Model (`test_hf_model.py`)

After converting or uploading your model, you should test it using the standard Hugging Face `transformers` loading mechanism to ensure everything works correctly.

* **Script:** `hugging_face/test_hf_model.py`
* **Purpose:** To load a model saved in the Hugging Face format (either locally or from the Hub) and perform a simple text generation test.
* **Key Steps:**
    1.  **Load Tokenizer:** Uses `AutoTokenizer.from_pretrained(model_path)`.
    2.  **Load Model:** Uses `AutoModelForCausalLM.from_pretrained(model_path)`. `model_path` can be a local directory (like `./hf_model_output`) or a Hugging Face Hub repository name (`your-hf-username/your-model-name`).
    3.  **Generation:** Takes a prompt, tokenizes it, and uses the standard `model.generate()` method (from the Hugging Face `PreTrainedModel` base class) to generate text, applying temperature and top-k sampling.
    4.  **Decoding:** Decodes the generated token IDs back into text.

### How to Run the Test Script

1.  **Navigate to the `hugging_face` directory:**
    ```bash
    cd hugging_face
    ```
2.  **Run the script, pointing to the HF model location:**
    * **Testing a local save:**
        ```bash
        python test_hf_model.py \
            --model_path ./hf_model_output \
            --prompt "Testing the locally saved HF model:"
        ```
    * **Testing an uploaded Hub model:**
        ```bash
        python test_hf_model.py \
            --model_path "your-hf-username/your-model-name" \
            --prompt "Testing the uploaded HF Hub model:" \
            --temperature 0.7 \
            --max_tokens 50
        ```
    * Adjust arguments (`--prompt`, `--max_tokens`, `--temperature`, `--top_k`) as needed.