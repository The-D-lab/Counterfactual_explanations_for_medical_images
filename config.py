# config.py
import os

class Config:
    def __init__(self):
        self.fold_no = 1
        self.batch_size = 16
        self.num_epochs = 100
        self.gpu_use = True
        self.patience_early_stopping = 30

        # VAE 
        self.input_channels = 3
        self.latent_dimension = 256
        self.patch_size = 224
        self.channels = (16, 32, 64, 128, 256, 512)
        self.strides = (1, 2, 2, 2, 2, 2)
        self.checkpoint_dir = r'.\\checkpoint'

        # Paths
        self.train_data_path = r"D:\\Xian\\VAE\\Train"
        self.test_data_path = r"D:\\Xian\\VAE\\Test"
        self.base_dir = r'Z:\\Xian\\VAE'

        #MLP parameters
        self.mlp_patience_early_stopping = 10
        self.mlp_lr = 1e-4
        self.mlp_weight_decay = 0.1
        self.mlp_batch_size = 16
        self.mlp_num_epochs = 200
        self.mlp_scheduler_patience = 10
        self.mlp_min_lr = 10e-6
        self.mlp_model_save_dir = os.path.join(self.base_dir, "Models")
        self.mlp_model_name = "MLP_image_only"

        # counterfactual saving directory
        self.vae_checkpoint_fold_no = 0
        self.mlp_checkpoint_fold_no = 0
        self.counterfactual_save_path = r'D:\\Xian\\VAE\\counterfactual_pred'
        self.generate_counterfactual_with_probability = 0.75
options = Config()