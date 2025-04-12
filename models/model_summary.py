from torchinfo import summary
import torch
import sys
import os

base_folder = os.path.abspath("..")
sys.path.append(base_folder)

from transformer_setup import ModelConfig
from transformer_setup import TransformerModel as MLATransformerModel

config = ModelConfig()
model = MLATransformerModel(
    vocab_size=32000,
    embed_dim=config.n_embd,
    num_heads=config.n_head,
    num_layers=config.n_layer,
    max_seq_len=config.block_size,
    dropout_prob=config.dropout,
    use_gradient_checkpoint=config.gradient_checkpointing,
    latent_dim=config.latent_dim,
    n_latent_vec=config.n_latent_vec
)

batch_size = config.batch_size
seq_length = config.block_size

summary(
    model=model,
    input_size=(config.batch_size, config.block_size),  # (batch_size, seq_len)
    dtypes=[torch.long],  # This is critical - must be long for embedding layers
    col_names=["input_size", "output_size", "num_params", "trainable", "mult_adds"],
    depth=3,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Optionally, add a sample forward pass to verify everything works
# with torch.no_grad():
#     dummy_input = torch.randint(0, 32000, (batch_size, seq_length), dtype=torch.long).to(
#         torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     )
#     logits, loss = model(dummy_input, dummy_input)
#     print(f"Output logits shape: {logits.shape}")