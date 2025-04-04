class ModelConfig:
    def __init__(self):
        # model architecture
        self.batch_size = 72                # Batch size per GPU
        self.block_size = 512               # Context size
        self.n_embd = 768                   # Embedding dimension
        self.n_head = 12                    # Number of attention heads
        self.n_layer = 12                   # Number of transformer layers
        self.dropout = 0.1                  # Dropout rate
        
        # training parameters
        self.max_iters = 15000               # Number of iterations
        self.eval_interval = 100            # Evaluation interval
        self.learning_rate = 1e-3          # Learning rate
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

        # MLA parameters
        self.latent_dem = 10                # Dimensionality of each latent query vector
        # TODO: fix value later
        self.num_latent = 10                # Number of learnable latent query vectors