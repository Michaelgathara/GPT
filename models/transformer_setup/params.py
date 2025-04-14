class ModelConfig:
    def __init__(self):
        # model architecture
        self.batch_size = 128               # Batch size per GPU - Increased for H100
        self.block_size = 1024              # Context size - Can increase for longer context
        self.n_embd = 1536                  # Embedding dimension - Increased model capacity
        self.n_head = 24                    # Number of attention heads - Keep it divisible by n_embd
        self.n_layer = 24                   # Number of transformer layers - Deeper model
        self.dropout = 0.1                  # Dropout rate - Keep it for regularization
        
        # training parameters
        self.max_iters = 10000               # Number of iterations - Train longer with larger model
        self.eval_interval = 200            # Evaluation interval - Adjust accordingly
        self.learning_rate = 3e-4          # Learning rate - Tune this carefully
        self.eval_iters = 10                 # Evaluation iterations - More for better estimate
        self.accumulation_steps = 2         # Gradient accumulation steps - Tradeoff memory/compute
        self.warmup_iters = 1000             # Learning rate warmup iterations - Longer for stability
        
        # Optimizer Settings
        self.weight_decay = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.95
        
        # Optimization flags
        self.gradient_checkpointing = True   # Use gradient checkpointing - Crucial for large models
        self.use_flash_attn = True          # Use Flash Attention if available - Mandatory for H100
        
        self.checkpoint_dir = 'checkpoints' # Directory to save checkpoints
        self.log_dir = 'logs'               # Directory to save logs
        self.seed = 1337                    # Random seed

        # MLA parameters
        self.latent_dim = 128               # Dimensionality of each latent query vector
        self.n_latent_vec = 256             # Number of learnable latent query vectors