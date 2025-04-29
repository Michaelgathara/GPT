from .params import ModelConfig

# from .mla_transformer import LatentAttentionHead, MultiHeadLatentAttention, FeedForward, Block, TransformerModel 

# __all__ = ['ModelConfig', 'LatentAttentionHead', 'MultiHeadLatentAttention', 'FeedForward', 'Block', 'TransformerModel']

from .transformer import TransformerModel, FeedForward, Block
__all__ = ['ModelConfig', 'TransformerModel', 'FeedForward', 'Block']