import os
import sys
import torch
import numpy as np
from tqdm import tqdm

base_folder = os.path.abspath("..")
print(f"Your base folder is: {base_folder}")
sys.path.append(base_folder)

from tokenization.custom_tokenizer.trainer import load_tokenizer
from models.transformer_setup import ModelConfig, TransformerModel

def load_model(checkpoint_path, device, tokenizer):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint['config']
    print("Checkpoint loaded. Model configuration:")
    for key, val in config_dict.items():
        print(f"  {key}: {val}")


    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=config_dict['n_embd'],
        num_heads=config_dict['n_head'],
        num_layers=config_dict['n_layer'],
        max_seq_len=config_dict['block_size'],
        dropout_prob=config_dict['dropout'],
        latent_dim=config_dict.get('latent_dim', 64),
        n_latent_vec=config_dict.get('n_latent_vec', 16),
        use_gradient_checkpoint=config_dict.get('gradient_checkpointing', False)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def calculate_perplexity(model, dataloader, device):
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Calculating perplexity"):
            x, y = x.to(device), y.to(device)
            
            # Get model predictions
            logits, loss = model(x, y)
            
            # Calculate loss per token
            batch_size, seq_len = x.shape
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens

    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss

def main():
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_choice = input("Which model checkpoint would you like to evaluate? (best_model.pt): ")

    # Default case
    if not model_choice:
        model_choice = "best_model.pt"

    checkpoint_path = os.path.join("../models/checkpoints", model_choice)
    model = load_model(checkpoint_path, device, tokenizer)
    config = ModelConfig()
    
    # Load test data
    from data import get_wikitext_data
    dataset = get_wikitext_data()
    test_data = dataset["test"]
    
    # Tokenize test data
    test_tokens = []
    for text in tqdm(test_data["text"], desc="Tokenizing test data"):
        tokens = tokenizer.encode(text).ids
        test_tokens.extend(tokens)
    
    # Create dataset and dataloader
    from torch.utils.data import Dataset, DataLoader
    
    class TokenizedDataset(Dataset):
        def __init__(self, tokens, block_size):
            self.tokens = tokens
            self.block_size = block_size
            
        def __len__(self):
            return len(self.tokens) - self.block_size
            
        def __getitem__(self, idx):
            x = torch.tensor(self.tokens[idx:idx + self.block_size], dtype=torch.long)
            y = torch.tensor(self.tokens[idx + 1:idx + 1 + self.block_size], dtype=torch.long)
            return x, y
    
    block_size = config.block_size
    test_dataset = TokenizedDataset(test_tokens, block_size)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Calculate perplexity
    perplexity, avg_loss = calculate_perplexity(model, test_loader, device)
    
    print(f"\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main() 
