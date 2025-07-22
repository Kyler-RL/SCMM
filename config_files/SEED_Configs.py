# SEED → SEED-IV model configs (3-category).
class Config(object):
    def __init__(self):
        # Model structure configs.
        self.in_channels = 62
        self.fea_length = 5
        self.encoder_out_dim = 128
        self.out_dim = 128
        self.num_classes = 3
        self.target_num_classes = 3

        # Training configs.
        self.pretrain_epoch = 200
        self.finetune_epoch = 50
        self.source_batch_size = 256
        self.target_batch_size = 128

        # Optimizer configs.
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.learning_rate = 5e-4
        self.weight_decay = 3e-4

        # Masking configs.
        self.mask_ratio = 0.5
        self.mask_num = 1
        self.threshold = 0.1
        self.mask_mode = "hybrid_mask"

        # Soft CL configs.
        self.upper_bound = 0.75
        self.sharpness = 0.5
        self.dist_metric = "cosine"
        self.temperature = 0.5
