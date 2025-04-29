from torchinfo import summary
import torch
from transformer_setup import ModelConfig, TransformerModel

config = ModelConfig()

model = TransformerModel(
    vocab_size=25000, 
    embed_dim=config.n_embd,
    num_heads=config.n_head,
    num_layers=config.n_layer,
    max_seq_len=config.block_size,
    dropout_prob=config.dropout,
    use_gradient_checkpoint=config.gradient_checkpointing,
    use_flash_attn=config.use_flash_attn
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

# summary(
#     model, 
#     input_size=(batch_size, seq_length),
#     dtypes=[torch.long],
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     verbose=2  
# )