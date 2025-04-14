import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any 

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("Flash Attention is available!")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention is not available, falling back to standard attention")

base_folder = os.path.abspath("..")
print(f"Your base folder is: {base_folder}")
sys.path.append(base_folder)

from tokenization import load_tokenizer # Adjusted import based on training script structure
tokenizer = load_tokenizer()
from transformer_setup import ModelConfig, TransformerModel # Keep these imports

def load_model(checkpoint_path: str, device: torch.device) -> tuple[TransformerModel, Dict[str, Any]]:
    """Loads the model checkpoint and its configuration."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}...")
    try:
        # Load checkpoint safely, map_location moves tensors to the correct device BEFORE loading state dict
        # weights_only=False is needed because we load the 'config' dictionary too
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint file: {e}")

    # --- Configuration Loading ---
    if 'config' not in checkpoint:
        raise KeyError("Checkpoint does not contain the 'config' dictionary.")
    config_dict = checkpoint['config']
    print(f"DEBUG: n_embd loaded from checkpoint config: {config_dict.get('n_embd', 'Not Found')}")
    print("Checkpoint loaded. Model configuration found:")
    for key, val in config_dict.items():
        print(f"  {key}: {val}")

    # --- Get Vocab Size ---
    if 'vocab_size' in config_dict:
        model_vocab_size = config_dict['vocab_size']
        print(f"Using vocab_size from checkpoint config: {model_vocab_size}")
        # Optional: Check consistency with loaded tokenizer
        if hasattr(tokenizer, 'get_vocab_size') and tokenizer.get_vocab_size() != model_vocab_size:
             print(f"Warning: Tokenizer vocab size ({tokenizer.get_vocab_size()}) differs from checkpoint vocab size ({model_vocab_size}). Using checkpoint value.")
    else:
         print("Warning: 'vocab_size' not found in checkpoint config. Getting from loaded tokenizer.")
         if not hasattr(tokenizer, 'get_vocab_size'):
              raise RuntimeError("Cannot determine vocab_size: Not in checkpoint config and tokenizer cannot provide it.")
         model_vocab_size = tokenizer.get_vocab_size()
         print(f"Using vocab_size from tokenizer: {model_vocab_size}")


    # --- Model Instantiation ---
    try:
        model = TransformerModel(
            vocab_size=model_vocab_size, 
            embed_dim=config_dict['n_embd'],
            num_heads=config_dict['n_head'],
            num_layers=config_dict['n_layer'],
            max_seq_len=config_dict['block_size'],  
            dropout_prob=config_dict['dropout'],
            latent_dim=config_dict.get('latent_dim', 64),
            n_latent_vec=config_dict.get('n_latent_vec', 16),
            use_gradient_checkpoint=config_dict.get('gradient_checkpointing', False)
        )
    except KeyError as e:
         raise KeyError(f"Missing required key in checkpoint config: {e}")


    # --- Load State Dict ---
    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    try:
        # Load the weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state dictionary loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This might indicate a mismatch between the loaded config and the saved weights,")
        print("or that the checkpoint is corrupted.")
        raise e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during state_dict loading: {e}")

    # Move model to the specified device (redundant if map_location worked, but safe)
    model.to(device)

    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    return model, config_dict

def main():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    checkpoint_dir = "checkpoints"
    try:
        available_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not available_checkpoints:
            print(f"Error: No '.pt' checkpoint files found in '{checkpoint_dir}'.")
            return
        print("\nAvailable checkpoints:")
        for i, fname in enumerate(available_checkpoints):
            print(f"  {i+1}: {fname}")

        while True:
            choice = input(f"Select checkpoint number or name (e.g., '1' or '{available_checkpoints[0]}'): ")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available_checkpoints):
                    model_choice = available_checkpoints[idx]
                    break
                else:
                    print("Invalid number.")
            except ValueError:
                # Interpret as filename
                if choice in available_checkpoints:
                    model_choice = choice
                    break
                elif f"{choice}.pt" in available_checkpoints:
                     model_choice = f"{choice}.pt"
                     break
                else:
                    print(f"Checkpoint '{choice}' not found.")

        checkpoint_path = os.path.join(checkpoint_dir, model_choice)

    except FileNotFoundError:
        print(f"Error: Checkpoint directory '{checkpoint_dir}' not found.")
        return
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return


    # --- Load Model ---
    try:
        model, loaded_config = load_model(checkpoint_path, device)
        model_block_size = loaded_config['block_size']
        print(f"Model block size (max_seq_len): {model_block_size}")
    except (FileNotFoundError, KeyError, RuntimeError, Exception) as e:
        print(f"\nError loading model: {e}")
        return 

    # --- Generation Loop ---
    print("\nEnter your prompt below. Type 'exit' to quit.")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower().strip() == "exit":
            break

        try:
            # Encode prompt
            prompt_ids = tokenizer.encode(prompt).ids
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # Check if prompt is too long
            if input_tensor.shape[1] >= model_block_size:
                print(f"\nWarning: Prompt length ({input_tensor.shape[1]}) is >= model's max sequence length ({model_block_size}).")
                print("Truncating prompt...")
                input_tensor = input_tensor[:, :model_block_size - 1] # Truncate leaving space for at least one new token
                if input_tensor.shape[1] == 0:
                     print("Error: Prompt empty after truncation.")
                     continue

            print("\nGenerating...")
            # --- Generate ---
            with torch.no_grad(): 
                generated_tensor = model.generate(
                    idx=input_tensor, # Pass input tensor (might need to rename arg in generate)
                    max_new_tokens=300, # How many new tokens to generate
                    max_seq_len=model_block_size, # Use loaded block size
                    temperature=0.8, # adjust as needed
                    top_k=20        # adjust as needed
                )

            generated_ids = generated_tensor[0].tolist() 
            generated_text = tokenizer.decode(generated_ids)

            print("\n--- Generated Text ---")
            print(generated_text)
            print("----------------------")

        except AttributeError as e:
             print(f"\nError: Problem calling model.generate. Ensure the method exists and accepts expected arguments. Details: {e}")
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")
            # Continue the loop or break depending on severity
            # break # Optional: exit loop on error

if __name__ == "__main__":
    main()