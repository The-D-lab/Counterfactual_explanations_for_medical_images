# other imports
import numpy as np
import os
import nrrd 

# sklearn imports
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# torch imports
import torch
from torch import nn


#image processing imports
import cv2

# other class imports
from utils import process_and_split_data, load_and_preprocess_image
from config import options
from model import ModTrainModel



def infer_latent(train_model, img_paths, device, options):
    # dict to store latent vectors
    latent_dict = {}

    # set the model into eval mode
    train_model.eval()
    for img_path in img_paths:
        image = load_and_preprocess_image(img_path, options)

        image1 = np.transpose(image, (2,1,0))
        image = np.expand_dims(image1, axis=0)

        image = torch.Tensor(image)
        x = image.to(device)

        # type casting w.r.t the model weights
        x = x.type('torch.cuda.FloatTensor')

        # infer the current batch 
        with torch.no_grad():
            # infer the current batch 
            recon_x, mu, logvar, t_x,_ = train_model(x)

            latent_dict[img_path] = t_x.cpu().numpy()  # don't forget to move tensor to cpu before converting to numpy

    return latent_dict


# getting the latent space representation
train_model = ModTrainModel(options)

# allocating the right GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
train_model = nn.DataParallel(train_model, device_ids = [0])

# send the model to the device
train_model = train_model.to(device)

for fold_no in range(5):
    options.fold_no = fold_no
    train_imgs, val_imgs, test_imgs, label_train, label_val, label_test, LSM_train, LSM_val, LSM_test = process_and_split_data(options)

    # loading the model
    train_model.load_state_dict(torch.load(options.checkpoint_dir + str(fold_no) + ".pt"))

    # infer the latent space
    train_fold_latent_representation = infer_latent(train_model, train_imgs, device, options)
    val_fold_latent_representation = infer_latent(train_model, val_imgs, device, options)

    np.save(os.path.join(options.checkpoint_dir, "train_dict_fold" + str(fold_no) + ".npy", train_fold_latent_representation))
    np.save(os.path.join(options.checkpoint_dir, "val_dict_fold" + str(fold_no) + ".npy", train_fold_latent_representation))