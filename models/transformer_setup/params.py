class ModelConfig:
    def __init__(self):
        #  Target ~1B Model Architecture 
        self.n_layer = 24                   # Number of transformer layers
        self.n_embd = 1536                  # Embedding dimension
        self.n_head = 12                    # Number of attention heads (1536 / 12 = 128 head dim)
        self.block_size = 1024              # Context size (keep as is)
        self.dropout = 0.1                  # Dropout rate (keep as is)
        # Calculated ~1.06 Billion parameters with vocab=50257

        #  Training Parameters 
        self.batch_size = 32                # Micro-batch size per GPU (keep as is)
        self.accumulation_steps = 8         # Gradient accumulation steps (keep as is)
        # Effective batch size = 32 * 8 = 256

        # iterations for ~2.6 epochs on sample-10BT (10B tokens / (256*1024) tokens/iter)
        # 1 epoch ~ 38k iters. Let's aim for ~2-3 epochs.
        self.max_iters = 100000             # Increased iterations (~2.6 epochs on sample)
        self.eval_interval = 1000           # Evaluation interval
        self.eval_iters = 100               # Eval iterations
        self.warmup_iters = 1500            # Warmup iterations (~1.5% of max_iters)

        #  Optimizer Settings 
        self.learning_rate = 6e-4           # Adjusted learning rate (e.g., 0.0006)
        self.weight_decay = 0.1             # Weight decay
        self.beta1 = 0.9
        self.beta2 = 0.95                   # Keep AdamW betas

        #  Optimization Flags 
        self.gradient_checkpointing = True  # Use gradient checkpointing
        self.use_flash_attn = True          # Use Flash Attention if available

        #  Logistics 
        self.checkpoint_dir = 'checkpoints_1B' # Use a different dir for new model size
        self.log_dir = 'logs_1B'               # Use a different dir for new model size
        self.seed = 1337                    # Random seed