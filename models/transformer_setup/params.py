class ModelConfig:
    def __init__(self):
        # model architecture
        self.batch_size = 64                # Batch size per GPU
        self.block_size = 512               # Context size
        self.n_embd = 512                   # Embedding dimension
        self.n_head = 8                    # Number of attention heads
        self.n_layer = 8                   # Number of transformer layers
        self.dropout = 0.1                  # Dropout rate
        
        # training parameters
        self.max_iters = 10000               # Number of iterations
        self.eval_interval = 100            # Evaluation interval
        self.learning_rate = 5e-3           # Learning rate
        self.eval_iters = 5                 # Evaluation iterations
        self.accumulation_steps = 4         # Gradient accumulation steps
        self.warmup_iters = 500             # Learning rate warmup iterations
        
        # Optimizer Settings
        self.weight_decay = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.95
        
        # Optimization flags
        self.gradient_checkpointing = False  # Use gradient checkpointing
        # Above does not work
        self.use_flash_attn = True          # Use Flash Attention if available
        
        self.checkpoint_dir = 'checkpoints' # Directory to save checkpoints
        self.log_dir = 'logs'               # Directory to save logs
        self.seed = 1337                    # Random seed