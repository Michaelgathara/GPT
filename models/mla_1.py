# keys are very low intrinsic dimension but maybe not values
# curse of dimensionality applies here -> high dimensional space just do not make euclidean sense
'''
whole point of attention is to dot product queries and keys to get different values so some have high values (attend to) and some have low values (ignore these tokens)

Softmax is just a scale so everything adds to 1

Having full length vectors (512) --> almost all attention scores are very similar

Keys and queries need to be low dimensional (<10)

Attention - query from the keys but reality is need positional embeddings (embed each word to dense vector and add embedding)

RoPE (Rotary position Embeddings) very popular (e.g. Llama) -> don't directly modify token embeddings in layers, right before you do attention you apply rotation matrix (matrix multiply) -> number of degrees you rotate depends on token position --> same for keys so you rotate both

    The only thing that matters in terms of how diff they are is how far apart query the key is (if 1 apart -> rotate almost same), (if far, amount of rotation very different) -> transformer learns to measure rotation differences if it wants to know how far the query and key are --> this is how it knows how some word came 5 before current word (or something like that)

    Do this right before you do query and key dot product multiplication
        Storying both in cache causes issue of keys looking very different

MLA solution is to separately do rotation/RoPE info and do base keys
    Vanilla rope rotates actual key
    MLA leaves the actual key alone --> extra rotation info and concatenate it --> model wants to learn how far a token is it just uses the last part that has the position info --> content of word uses left hand side (content portion)

Can say first x bit is meaning and next x bit is rotation part

'''

import sys
import os
import json
import time
import logging
import math
import multiprocessing
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# other imports
from tqdm import tqdm
from functools import partial
from typing import Optional

base_folder = os.path.abspath("..") #needed for accessing custom modules
print(f"Your base folder is: {base_folder}")
sys.path.append(base_folder)

# custom data module imports
from data import get_wikitext_data, clean_textdata, get_fineweb_data

# GPT tokenizer imports
'''
commented out for now to test custom BPE first
'''
# from tokenization import get_tiktoken_tokenizer
# from tokenizers import Tokenizer

# TOKENIZER_PATH = f"{base_folder}/tokenization/custom_tokenizer.json"
# tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# custom tokenizer
from tokenization.custom_tokenizer.trainer import load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctim)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHander("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("transformer_training")

# try to import flash_attn, this doesn't always work so default to standard attention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("Flash Attention is available!")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention is not available, falling back to standard attention")

# transformer imports
from transformer_setup import ModelConfig, FlashAttentionHead, MultiHead, Head, FeedForward, Block, TransformerModel

config = ModelConfig() # hyperparameters

# cosine learning rate scheduler with warmup
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_iters, max_iters):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.current_iter = 0
    
    def step(self):
        # linear warmup
        if self.current_iter < self.warmup_iters:
            lr_scale = min(1.0, float(self.current_iter + 1) / self.warmup_iters) 
        # cosine decay
        else:
            progress = float(self.current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            lr_scale = 0.5 * (1.0 * math.cos(math.pi * progress))
        
        # apply scale to all param groups in optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale
        
        self.current_iter += 1
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr'] # lr of first parameter group

