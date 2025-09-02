class Config:
    # Model
    vocab_size = 128  # character-level for now
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    block_size = 64

    # Training
    batch_size = 32
    lr = 3e-4
    weight_decay = 1e-2
    max_iters = 500
    eval_interval = 50

    # Optimization
    warmup_iters = 50
    grad_clip = 1.0

    # Checkpoint
    ckpt_dir = "checkpoints"
