# model.py
import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import VarAutoEncoder


class ModTrainModel(torch.nn.Module):
    def __init__(self, config):
        super(ModTrainModel, self).__init__()

        # Model parameters from the config class
        self.config = config
        self.in_shape = (self.config.input_channels, self.config.patch_size, self.config.patch_size)
        self.out_channels = self.config.input_channels
        self.latent_size = self.config.latent_dimension
        self.channels = self.config.channels
        self.strides = self.config.strides

        # The sequential model
        self.cls_sq = nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Linear(64, 2)
        )

        # The VarAutoEncoder model
        self.train_model = VarAutoEncoder(
            dimensions=2,
            in_shape=self.in_shape,  # image spatial shape
            out_channels=self.out_channels,
            latent_size=self.latent_size,
            channels=self.channels,
            strides=self.strides
        )

    def forward(self, x):
        recon_x, mu, logvar, t_x = self.train_model(x)
        preds = torch.FloatTensor([0])  # assuming you wanted a tensor of 0
        return recon_x, mu, logvar, t_x, preds

    def decode_forward(self, t_x):
        return self.train_model.decode_forward(t_x)

class MLP_only_image(torch.nn.Module):  
    def __init__(self):
        super(MLP_only_image,self).__init__()    
        self.fc1 = torch.nn.Linear(256, 32)
        torch.nn.init.zeros_(self.fc1.weight)
        
        self.fc2 = torch.nn.Linear(64, 32)
        torch.nn.init.zeros_(self.fc2.weight)

        self.fc3 = torch.nn.Linear(32,16)
        torch.nn.init.zeros_(self.fc3.weight)
        
        self.fc4 = torch.nn.Linear(32,2)  
        torch.nn.init.zeros_(self.fc4.weight)
        
    def forward(self,in_values):
        output = F.relu(self.fc1(in_values))         
        output = self.fc4(output)  
        
        return output
