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
