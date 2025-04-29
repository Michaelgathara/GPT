import os
import sys
import torch
import argparse
from pathlib import Path
from huggingface_hub import login, HfApi

base_folder = os.path.abspath("..")
sys.path.append(base_folder)

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import PreTrainedModel, PretrainedConfig

from transformer_setup import ModelConfig, TransformerModel
from inference import load_trained_model

class CustomTransformerConfig(PretrainedConfig):
    model_type = "Custom_Flash_Attention_Transformer"
    
    def __init__(
        self, 
        vocab_size=50257,
        embed_dim=1536,
        num_heads=12,
        num_layers=24,
        max_seq_len=1024,
        dropout_prob=0.1,
        use_gradient_checkpoint=True,
        use_flash_attn=False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,         
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout_prob = dropout_prob
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.use_flash_attn = use_flash_attn
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

class CustomTransformerModelForCausalLM(PreTrainedModel):
    config_class = CustomTransformerConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        self.transformer = TransformerModel(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            dropout_prob=config.dropout_prob,
            use_gradient_checkpoint=config.use_gradient_checkpoint,
            use_flash_attn=config.use_flash_attn
        )
        
    def forward(self, input_ids, labels=None, **kwargs):
        logits, loss = self.transformer(input_ids, labels)
        if loss is not None:
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50, **kwargs):
        return self.transformer.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            max_seq_len=self.config.max_seq_len,
            temperature=temperature,
            top_k=top_k
        )

def save_for_huggingface(checkpoint_path, output_dir, model_name, push_to_hub=False, auth_token=None):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config_dict = load_trained_model(checkpoint_path, device, verbose=True)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to output directory: {output_path}")
    
    hf_config = CustomTransformerConfig(
        vocab_size=config_dict.get('vocab_size', tokenizer.vocab_size),
        embed_dim=config_dict.get('n_embd', 1536),
        num_heads=config_dict.get('n_head', 12),
        num_layers=config_dict.get('n_layer', 24),
        max_seq_len=config_dict.get('block_size', 1024),
        dropout_prob=config_dict.get('dropout', 0.1),
        use_gradient_checkpoint=config_dict.get('gradient_checkpointing', False),
        use_flash_attn=config_dict.get('use_flash_attn', False),
        
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        
        architectures=["CustomTransformerModelForCausalLM"],
        model_type="custom_transformer",
    )
    
    hf_config.save_pretrained(output_path)
    print("Saved config.json")
    
    hf_model = CustomTransformerModelForCausalLM(hf_config)
    hf_model.transformer = model
    
    hf_model.save_pretrained(output_path)
    print("Saved model weights")
    
    tokenizer.save_pretrained(output_path)
    print("Saved tokenizer")
    
    # Trying to write a README file here
    with open(output_path / "README.md", "w") as f:
        f.write(f"# {model_name}\n\n")
        f.write("Custom Transformer model for causal language modeling.\n\n")
        f.write("## Model Details\n\n")
        f.write(f"- Architecture: CustomTransformerModelForCausalLM\n")
        f.write(f"- Layers: {config_dict.get('n_layer', 24)}\n")
        f.write(f"- Hidden Size: {config_dict.get('n_embd', 1536)}\n")
        f.write(f"- Attention Heads: {config_dict.get('n_head', 12)}\n")
        f.write(f"- Context Length: {config_dict.get('block_size', 1024)}\n")
        f.write(f"- Parameters: {model.get_num_params():,}\n\n")
        f.write("## Usage\n\n")
        f.write("```python\n")
        f.write("from transformers import AutoModelForCausalLM, AutoTokenizer\n\n")
        f.write(f"model_name = \"{model_name}\"\n")
        f.write("tokenizer = AutoTokenizer.from_pretrained(model_name)\n")
        f.write("model = AutoModelForCausalLM.from_pretrained(model_name)\n")
        f.write("```\n\n")
        f.write("## Generation Example\n\n")
        f.write("```python\n")
        f.write("prompt = \"Your prompt here\"\n")
        f.write("inputs = tokenizer(prompt, return_tensors=\"pt\")\n")
        f.write("outputs = model.generate(inputs[\"input_ids\"], max_new_tokens=100, temperature=0.8, top_k=50)\n")
        f.write("print(tokenizer.decode(outputs[0]))\n")
        f.write("```\n")
    print("Created README.md")
    
    if push_to_hub:
        if not auth_token:
            raise ValueError("Authentication token required to push to HuggingFace Hub")
        
        login(auth_token)
        
        api = HfApi()
        api.create_repo(repo_id=model_name, exist_ok=True)
        
        print(f"Pushing to HuggingFace Hub: {model_name}")
        api.upload_folder(
            folder_path=str(output_path),
            repo_id=model_name,
            commit_message="Upload custom transformer model"
        )
        print(f"Successfully pushed to: https://huggingface.co/{model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and upload custom transformer model to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to your trained model checkpoint (.pt file)")
    parser.add_argument("--output_dir", type=str, default="./hf_model", help="Directory to save the converted model")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the model on HuggingFace Hub (e.g., 'username/model-name')")
    parser.add_argument("--push", action="store_true", help="Push the model to HuggingFace Hub")
    
    args = parser.parse_args()
    
    save_for_huggingface(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_name=args.model_name,
        push_to_hub=args.push,
        auth_token=args.token
    )