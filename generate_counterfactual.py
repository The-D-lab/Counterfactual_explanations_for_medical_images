# imports
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch
from PIL import Image
import os
import numpy as np

# other class imports
from config import options
from model import ModTrainModel
from model import MLP_only_image
from utils import load_and_preprocess_image, compute_bounds, scale_logits

flag = 1

# Loading VAE model
train_model = ModTrainModel(options)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
train_model = nn.DataParallel(train_model, device_ids = [0])
train_model = train_model.to(device)
train_model.load_state_dict(torch.load(options.checkpoint_dir + str(options.vae_checkpoint_fold_no) + ".pt"))

# classifier MLP
checkpoint_train = os.path.join(options.mlp_model_save_dir, options.mlp_model_name,'checkpoints', 'checkpoint'+ str(options.mlp_checkpoint_fold_no) +'.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
model = MLP_only_image()
model = nn.DataParallel(model, device_ids = [0]) 
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_train))

# setting into the evaluation mode
model.eval()
train_model.eval()

# loading image path for generating predictions and counterfactuals
test_imgs = [os.path.join(options.prediction_img_folder, x) for x in os.listdir(options.prediction_img_folder)]

for path in test_imgs:

    # loading the image tensor
    image = load_and_preprocess_image(path)
    image = np.expand_dims(np.transpose(image, (2,1,0)), axis=0)
    image = torch.Tensor(image)
    x = image.to(device)
    x = x.type('torch.cuda.FloatTensor')


    # infer the current batch 
    recon_x, mu, logvar, t_x,_ = train_model(x)
    val_img = recon_x.detach().cpu().numpy()   

    # generate mlp prediction
    preds_v = model(t_x)          
    pred = preds_v.argmax(1, keepdim=True)
    preds_sf = F.softmax(preds_v)[:,1]
        

            
    pred_t = preds_sf[0]
    dzdxp = torch.autograd.grad((pred_t), t_x)[0]
     
    lb_v, rb_v = compute_bounds(model, t_x, dzdxp, flag=1, lb_v=0, rb_v=0)
        
        
    for thresh_lam in range(lb_v,rb_v, 5):
        t_x_n = t_x + dzdxp*thresh_lam
        preds_v_n = model(t_x_n) 
        preds_sf_n = F.softmax(scale_logits(preds_v_n))[:,1]
        np_prediction = preds_sf_n.cpu().detach().numpy()                
        crop_t_x_n = t_x_n[:,:256]
        counterfactual = train_model.module.train_model.decode_forward(crop_t_x_n) 

        if abs(np_prediction - options.generate_counterfactual_with_probability) < 0.01:
            im = Image.fromarray((np.transpose(np.squeeze(counterfactual.cpu().detach().numpy()), (1,2,0))*255).astype("uint8"))
            im.save(os.path.join(options.counterfactual_save_path, path.split("\\")[-1].split(".")[0] + str(options.generate_counterfactual_with_probability) + ".png"))
            break
            
