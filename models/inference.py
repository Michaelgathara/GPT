# inference_gpt2_model.py
import os
import sys
import torch
import argparse
import time
import math
from contextlib import nullcontext # For conditionally using torch.amp

#  Add base folder to sys.path 
# Assumes transformer_setup.py is in the parent directory
try:
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if base_folder not in sys.path:
        sys.path.append(base_folder)
    print(f"Added base folder to path: {base_folder}")
    from transformer_setup import ModelConfig, TransformerModel
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'transformers' is installed (`pip install transformers`)")
    print("And that 'transformer_setup.py' is accessible (e.g., in the parent directory).")
    sys.exit(1)

#  Flash Attention Check 
try:
    # Note: Flash Attention >= 2.0 requires compatible hardware (Ampere, Ada, Hopper)
    # and correctly installed packages.
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("Flash Attention library found.")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention library not found, falling back to standard attention.")

def load_trained_model(checkpoint_path, device, verbose=True):
    """Loads the tokenizer and trained Transformer model from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    #  1. Load Tokenizer (Must match training) 
    if verbose: print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Set padding token if needed (as done in training)
    if tokenizer.pad_token is None:
        if verbose: print("Setting tokenizer pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    if verbose: print(f"Tokenizer loaded. Vocab size: {vocab_size}, PAD ID: {tokenizer.pad_token_id}")

    #  2. Load Checkpoint 
    if verbose: print(f"Loading checkpoint from: {checkpoint_path}...")
    # map_location ensures weights load correctly regardless of original device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    #  3. Load Configuration from Checkpoint 
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain the 'config' dictionary. Cannot determine model parameters.")

    saved_config_dict = checkpoint['config']
    if verbose:
        print("Model configuration loaded from checkpoint:")
        for key, val in saved_config_dict.items():
            print(f"  {key}: {val}")

    #  4. Initialize Model using Saved Configuration 
    # Create a ModelConfig object or directly use dict values
    # Ensure all necessary keys exist in the saved config
    required_keys = ['n_embd', 'n_head', 'n_layer', 'block_size', 'dropout', 'vocab_size']
    for key in required_keys:
        if key not in saved_config_dict:
             # Handle potential discrepancy if vocab_size wasn't saved explicitly in config
             if key == 'vocab_size' and 'vocab_size' not in saved_config_dict:
                 print(f"Warning: 'vocab_size' not found in saved config. Using tokenizer's vocab_size: {vocab_size}")
                 saved_config_dict['vocab_size'] = vocab_size
             else:
                 raise KeyError(f"Saved config dictionary is missing required key: '{key}'")

    # Check if vocab size matches tokenizer (important!)
    if saved_config_dict['vocab_size'] != vocab_size:
        print(f"WARNING: Mismatch between saved config vocab_size ({saved_config_dict['vocab_size']}) and tokenizer vocab_size ({vocab_size}).")
        print("Using tokenizer's vocab_size for model initialization, but this might indicate an issue.")
        # Decide: Raise error or proceed with caution? Let's use tokenizer's size but warn.
        model_vocab_size = vocab_size
    else:
        model_vocab_size = saved_config_dict['vocab_size']

    # Check Flash Attention setting from config if available
    use_flash_attn_config = saved_config_dict.get('use_flash_attn', False) # Default to False if not saved
    use_flash_attn = HAS_FLASH_ATTN and use_flash_attn_config
    if verbose: print(f"Using Flash Attention: {use_flash_attn} (Available: {HAS_FLASH_ATTN}, Configured: {use_flash_attn_config})")

    model = TransformerModel(
        vocab_size=model_vocab_size,
        embed_dim=saved_config_dict['n_embd'],
        num_heads=saved_config_dict['n_head'],
        num_layers=saved_config_dict['n_layer'],
        max_seq_len=saved_config_dict['block_size'],
        dropout_prob=saved_config_dict['dropout'], # Dropout is typically disabled by model.eval() anyway
        use_gradient_checkpoint=saved_config_dict.get('gradient_checkpointing', False), # From training config
        use_flash_attn=use_flash_attn # Use based on availability and config
    )

    #  5. Load Model State Dict 
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")

    state_dict = checkpoint['model_state_dict']

    # Handle potential issues if model was saved with DDP (DistributedDataParallel)
    # DDP prepends 'module.' to layer names. We need to remove it for single-GPU/CPU loading.
    unwrapped_state_dict = {}
    needs_unwrap = False
    for k, v in state_dict.items():
        if k.startswith('module.'):
            needs_unwrap = True
            unwrapped_state_dict[k[len('module.'):]] = v
        else:
            unwrapped_state_dict[k] = v
    if needs_unwrap:
        if verbose: print("Unwrapping state_dict keys (removed 'module.' prefix).")
        state_dict = unwrapped_state_dict

    # Load the weights, ensuring dimensions match
    try:
        model.load_state_dict(state_dict, strict=True)
        if verbose: print("Model state_dict loaded successfully.")
    except RuntimeError as e:
        print("\n ERROR LOADING STATE DICT ")
        print(e)
        print("\nThis usually means the model architecture defined in 'transformer_setup.py'")
        print("does not exactly match the architecture saved in the checkpoint.")
        print("Common issues:")
        print(" - Vocab size mismatch (Check tokenizer vs saved config)")
        print(" - Different n_embd, n_layer, n_head, block_size")
        print(" - Changes inside the Block or FeedForward definitions.")
        print(f"\nAttempted to load state_dict with keys: {state_dict.keys()}")
        print(f"Model expects keys: {model.state_dict().keys()}")
        # Example check: Compare shapes for a specific layer if error persists
        # print(f"Checkpoint lm_head shape: {state_dict.get('lm_head.weight').shape if 'lm_head.weight' in state_dict else 'Not Found'}")
        # print(f"Current model lm_head shape: {model.lm_head.weight.shape}")
        raise e

    #  6. Final Steps 
    model.to(device)
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    if verbose: print(f"Model loaded onto device: {device} and set to evaluation mode.")
    
    try:
        model = torch.compile(model, mode="reduce-overhead")
        if verbose: print("Model compiled successfully for inference.")
    except Exception as e:
        if verbose: print(f"torch.compile failed during inference setup: {e}. Proceeding without compiling.")

    # Print parameter count
    num_params = sum(p.numel() for p in model.parameters())
    if verbose: print(f"Model parameter count: {num_params:,}")

    return model, tokenizer, saved_config_dict # Return config for max_seq_len


def generate_text(model, tokenizer, config, prompt, device, max_new_tokens, temperature, top_k):
    """Generates text using the loaded model."""
    model.eval() # Ensure model is in eval mode

    # TODO: Look into how system prompting is implemented, might need post-training because this does not work 
    # Encode the prompt
    # user_prompt = prompt
    # system_prompt = "You are a knowledgeable and helpful AI assistant. Your primary strength lies in explaining concepts clearly and providing accurate, well-structured information. Aim for clarity, neutrality, and helpfulness in your responses. "
    # prompt = f"{system_prompt} {user_prompt}"
    prompt_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Determine max_seq_len from the loaded config
    max_seq_len = config['block_size']

    print(f"\nGenerating text (max_new_tokens={max_new_tokens}, temp={temperature}, top_k={top_k})...")
    print(f"Prompt: \"{prompt}\"")
    print("-" * 20)

    start_time = time.time()

    # Use autocast for potential speedup on CUDA (matches training setup)
    # Use nullcontext if on CPU or if autocast causes issues
    ctx = nullcontext()
    if device.type == 'cuda':
        # Check if model uses flash attention, as autocast might interfere differently
        # Depending on the flash_attn implementation details. For now, assume it's fine.
         ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
         print("Using torch.amp.autocast for CUDA generation.")


    with torch.no_grad(): # Essential for inference
        with ctx: # Apply autocast context
            generated_tensor = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                max_seq_len=max_seq_len,
                temperature=temperature,
                top_k=top_k
            )

    end_time = time.time()
    generation_time = end_time - start_time

    # Decode the generated sequence
    generated_ids = generated_tensor[0].tolist() # Get list of IDs from the tensor
    generated_text = tokenizer.decode(generated_ids)

    print("\n\n\nGenerated Text:")
    print(generated_text)
    print("-" * 20)
    print(f"Generation took {generation_time:.2f} seconds.")

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Inference script for the trained Transformer model.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file (.pt).")
    parser.add_argument("--prompt", type=str, default=None, help="Initial prompt text. If None, enters interactive mode.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (e.g., 1.0=no change, <1.0=less random, >1.0=more random).")
    parser.add_argument("--top_k", type=int, default=50, help="Sample from the top K most likely tokens (0 to disable).")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu'). Auto-detects if None.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--verbose", action='store_true', help="Print detailed loading information.")


    args = parser.parse_args()

    #  Setup Device 
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # MPS check could be added here if relevant for MacOS
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    #  Set Seed 
    if args.seed is not None:
        print(f"Setting random seed to: {args.seed}")
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed) # if using multi-GPU
        # Potentially add numpy seed if numpy is used elsewhere: np.random.seed(args.seed)

    #  Load Model 
    try:
        model, tokenizer, loaded_config = load_trained_model(args.checkpoint_path, device, args.verbose)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)

    #  Generation 
    if args.prompt is not None:
        # Single prompt generation from argument
        generate_text(
            model=model,
            tokenizer=tokenizer,
            config=loaded_config,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                prompt = input("\nEnter prompt: ")
                if prompt.lower().strip() in ["exit", "quit"]:
                    break
                if not prompt:
                    continue

                generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    config=loaded_config,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
            except EOFError:
                 break
            except KeyboardInterrupt: 
                 break
        print("\nExiting interactive mode.")


if __name__ == "__main__":
    main()