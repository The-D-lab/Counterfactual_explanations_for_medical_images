# utils.py
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import glob
import os
import cv2
import nrrd

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def process_and_split_data(options):
    base_dir = options.base_dir
    train_data_path = options.train_data_path
    test_data_path = options.test_data_path
    FOLD_NUM = options.fold_no
    
    test_all = glob.glob(os.path.join(test_data_path,"*", "*.nrrd"))

    train_labels_patientid =np.array(pd.read_csv(os.path.join(base_dir, 'label_train.csv'))['ID'])
    train_labels = np.array(pd.read_csv(os.path.join(base_dir, 'label_train.csv'))['Label'])

    test_labels_patientid = np.array(pd.read_csv(os.path.join(base_dir, 'label_test.csv'))['ID'])
    test_labels = np.array(pd.read_csv(os.path.join(base_dir, 'label_test.csv'))['Label'])

    train_LSM = pd.read_csv(os.path.join(base_dir, 'label_train.csv')).to_numpy()[:,3:]#.loc[:, ["INR", "CSPH", "FLR/TLV"]].to_numpy()
    test_LSM = pd.read_csv(os.path.join(base_dir, 'label_test.csv')).to_numpy()[:,3:]#.loc[:, ["INR", "CSPH", "FLR/TLV"]].to_numpy()

    # patient wise images
    all_train_patients = np.array(os.listdir(train_data_path))
    all_test_patients = np.array(os.listdir(test_data_path))

    # Split into folds
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    idx_split = [x for x in kf.split(all_train_patients, train_labels)]

    # splitting the patients
    train = all_train_patients[idx_split[FOLD_NUM][0]]
    val = all_train_patients[idx_split[FOLD_NUM][1]]
    test = all_test_patients

    train = [os.path.join(train_data_path, x) for x in train]
    val = [os.path.join(train_data_path, x) for x in val]
    test = [os.path.join(test_data_path, x) for x in test]

    # generating individual images for train
    train = [glob.glob(os.path.join(x, "*.nrrd")) for x in train]
    train = [item for sublist in train for item in sublist]

    # generating individual images for validation
    val = [glob.glob(os.path.join(x, "*.nrrd")) for x in val]
    val = [item for sublist in val for item in sublist]

    #generating individual images for test
    test = [glob.glob(os.path.join(x, "*.nrrd")) for x in test]
    test = [item for sublist in test for item in sublist]

    ## Train images for autoencoder
    train_imgs = []
    train_imgs.extend(train)

    ## Val images for autoencoder
    val_imgs = []
    val_imgs.extend(val)

    ## Test images
    test_imgs = []
    test_imgs.extend(test)

    #getting labels
    label_train = train_labels[np.squeeze(np.vstack([np.where(train_labels_patientid == int(x.split("\\")[-2]))[0] for x in train]))]
    label_val = train_labels[np.squeeze(np.vstack([np.where(train_labels_patientid == int(x.split("\\")[-2]))[0] for x in val]))]
    label_test = test_labels[np.squeeze(np.vstack([np.where(test_labels_patientid == int(x.split("\\")[-2]))[0] for x in test_all]))]

    # getting LSM
    LSM_train = train_LSM[np.squeeze(np.vstack([np.where(train_labels_patientid == int(x.split("\\")[-2]))[0] for x in train]))]
    LSM_val = train_LSM[np.squeeze(np.vstack([np.where(train_labels_patientid == int(x.split("\\")[-2]))[0] for x in val]))]
    LSM_test = test_LSM[np.squeeze(np.vstack([np.where(test_labels_patientid == int(x.split("\\")[-2]))[0] for x in test_all]))]

    #printing the quantity of train, val and test imgs
    print("Train Images: ", len(train_imgs))
    print("Validation Images: ", len(val_imgs))
    print("Test Images: ", len(test_all))
    print("\n")

    print("###################################")
    print("Train labels:",len(label_train))
    print("Validation labels:",len(label_val))
    print("Test labels:",len(label_test))
    print("\n")

    print("###################################")
    print("Train LSM:",len(LSM_train))
    print("Validation LSM:",len(LSM_val))
    print("Test LSM:",len(LSM_test))

    return train_imgs, val_imgs, test_imgs, label_train, label_val, label_test, LSM_train, LSM_val, LSM_test


def preprocess_image(image, options):
    min_p = np.min(image)
    max_p = np.max(image)
    image = (image - min_p) / (max_p - min_p)
    image = cv2.resize(image, (options["patch_size"], options["patch_size"]))
    return image

def load_and_preprocess_image(img_path, options):
    image = nrrd.read(img_path)[0]
    image = preprocess_image(image, options)
    return image

def scale_logits(logits_p, temp_t = 1):
    logits_n = logits_p/temp_t
    return logits_n

def compute_bounds(model, t_x, dzdxp, flag=1, lb_v=0, rb_v=0):
    while flag:
        lb_v = lb_v - 400
        t_x_n = t_x + dzdxp*lb_v
        preds_v_n = model(t_x_n)
        preds_sf_n = F.softmax(scale_logits(preds_v_n))[:,1]
        np_prediction = preds_sf_n.cpu().detach().numpy()
        if np_prediction < 0.1:
            break

    while True:
        rb_v = rb_v + 400
        t_x_n = t_x + dzdxp*rb_v
        preds_v_n = model(t_x_n)
        preds_sf_n = F.softmax(scale_logits(preds_v_n))[:,1]
        np_prediction = preds_sf_n.cpu().detach().numpy()
        if np_prediction > 0.95:
            break
    return lb_v, rb_v