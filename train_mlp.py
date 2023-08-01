# other imports
import numpy as np
import os
from barbar import Bar

# sklearn imports
from sklearn import metrics

# torch imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

# other class imports
from utils import EarlyStopping, process_and_split_data
from config import options
from model import MLP_only_image
from dataset import ClassificationDataset_only_image

# training the MLP model 
model_dir = options.mlp_model_save_dir
model_name = options.mlp_model_name


# mkdir for stored models
if not os.path.exists(os.path.join(model_dir)):
    os.makedirs(os.path.join(model_dir))
if not os.path.exists(os.path.join(model_dir, model_name)):
    os.makedirs(os.path.join(model_dir, model_name))
if not os.path.exists(os.path.join(os.path.join(model_dir, model_name),'models')):
    os.makedirs(os.path.join(os.path.join(model_dir, model_name),'models'))
if not os.path.exists(os.path.join(os.path.join(model_dir, model_name),'checkpoints')):
    os.makedirs(os.path.join(os.path.join(model_dir, model_name),'checkpoints'))


for fold_no in range(5):
    writer = SummaryWriter(os.path.join(os.getcwd(),'runs', model_name + '_fold'+str(fold_no)))

    # getting the latent data
    train_data = np.load(os.path.join(options.checkpoint_dir, "train_dict_fold" + str(fold_no) + ".npy"))
    val_data = np.load(os.path.join(options.checkpoint_dir, "train_dict_fold" + str(fold_no) + ".npy"))
    train_list = train_data.tolist()
    val_list = val_data.tolist()
    train_values = [value for key,value in train_list.items()]
    val_values = [value for key,value in val_list.items()]

    # getting the corresponding labels
    options.fold_no = fold_no
    train_imgs, val_imgs, test_imgs, label_train, label_val, label_test, LSM_train, LSM_val, LSM_test = process_and_split_data(options)


    # setting training params
    early_stopping = EarlyStopping(patience=options.mlp_patience_early_stopping, verbose=True)

    training_dataset=ClassificationDataset_only_image(train_values, label_train)
    training_dataloader = DataLoader(training_dataset, batch_size= options.mlp_batch_size, shuffle=True, num_workers=0)
 
    validation_dataset = ClassificationDataset_only_image(val_values, label_val)
    validation_dataloader = DataLoader(validation_dataset, batch_size= options.mlp_batch_size,shuffle=False, num_workers=0)


    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    # instantiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
    model = MLP_only_image()
    model = nn.DataParallel(model, device_ids = [0]) #用多个GPU来加速训练
    model = model.to(device)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=options.mlp_lr, weight_decay=options.mlp_weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience = options.mlp_scheduler_patience, min_lr= options.mlp_min_lr)
    criterion = nn.CrossEntropyLoss()

    # training loop
    training = True
    epoch = 1

    try:
        while training:

            # epoch specific metrics
            train_loss = 0
            mask_loss = 0
            train_accuracy = 0
            val_loss = 0
            val_accuracy = 0
            total_loss = 0

            proba_t = []
            true_t = []
            proba_v = []
            true_v = []
            # -----------------------------
            # training samples
            # -----------------------------

            # set the model into train mode

            model.train()
            for b, batch in enumerate(Bar(training_dataloader)):


                    x = batch['value'].to(device)
                    y = batch['label'].to(device)
                    
                    x = torch.cat((x,z), 1)
                    
                    # Type Casting
                    x = x.type('torch.cuda.FloatTensor')
                 
                    # clear gradients
                    optimizer.zero_grad()

                    # infer the current batch 
                    preds = model(x)

                    # compute the loss. 
                    loss = criterion(preds,y)
                    train_loss += loss.item()

                    # backward loss and next step
                    loss.backward()
                    optimizer.step()

                    # compute the accuracy
                    pred = preds.max(1, keepdim=True)[1]
                    batch_accuracy = pred.eq(y.view_as(pred).long())
                    train_accuracy += (batch_accuracy.sum().item() / np.prod(y.shape))

                    for i in range(len(y)):
                            proba_t.append(np.squeeze(F.softmax(preds[i]).cpu().detach().numpy()))
                            true_t.append(batch['label'].detach().numpy()[i])

            fpr, tpr, _ = metrics.roc_curve(np.array(true_t), np.array(proba_t)[:,1])
            train_auc = metrics.auc(fpr, tpr)


            # -----------------------------
            # validation samples
            # -----------------------------

            # set the model into train mode
            model.eval()
            for b, batch in enumerate(Bar(validation_dataloader)):


                    x = batch['value'].to(device)
                    y = batch['label'].to(device)
                    z = batch["lsm"].to(device)

                    #z = z[:,None]
                    
                    x = torch.cat((x,z), 1)
                    x = x.type('torch.cuda.FloatTensor')

                    # infer the current batch 
                    with torch.no_grad():
                        preds = model(x)
                        loss = criterion(preds, y)
                        val_loss += loss.item()

                        # compute the accuracy 
                        pred = preds.max(1, keepdim=True)[1]
                        batch_accuracy = pred.eq(y.view_as(pred).long())
                        val_accuracy += batch_accuracy.sum().item() / np.prod(y.shape) 
                        for i in range(len(y)):
                            proba_v.append(np.squeeze(F.softmax(preds[i]).cpu().detach().numpy()))
                            true_v.append(batch['label'].detach().numpy()[i])
                            


            fpr, tpr, _ = metrics.roc_curve(np.array(true_v), np.array(proba_v)[:,1])
            val_auc = metrics.auc(fpr, tpr) 
            scheduler.step(val_loss)


            # compute mean metrics
            train_loss /= (len(training_dataloader))
            train_accuracy /= (len(training_dataloader))
            val_loss /= (len(validation_dataloader))
            val_accuracy /= (len(validation_dataloader))
            early_stopping(val_loss, model, str(fold_no))

            train_loss_all.append(train_loss)
            train_acc_all.append(train_accuracy)
            val_loss_all.append(val_loss)
            val_acc_all.append(val_accuracy)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            print('Epoch {:d} train_loss {:.4f} train_acc {:.4f} train_auc {:.4f} val_loss {:.4f} val_acc {:.4f} val_auc {:.4f}'.format(
                    epoch, 
                    train_loss, 
                    train_accuracy,
                    train_auc,
                    val_loss,
                    val_accuracy,
                    val_auc))

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_accuracy', train_accuracy, epoch)
            writer.add_scalar('train_auc', train_auc, epoch)
            writer.add_scalar('val_losss', val_loss, epoch)
            writer.add_scalar('val_accuracy', val_accuracy, epoch)
            writer.add_scalar('val_auc', val_auc, epoch)

            # save weights
            if not os.path.exists(os.path.join(model_dir, model_name,'models',str(fold_no))):
                os.makedirs(os.path.join(model_dir, model_name,'models',str(fold_no)))
            #torch.save(model.state_dict(), 
            #          os.path.join(r'Z:\Xian\Models',model_name,'models', str(FOLD_NUM),'model' + str(epoch) + '.pth'))
            
            if epoch >= options.mlp_num_epochs:
                training = False

            # update epochs    
            epoch += 1
        print('********************************************************************************')
        writer.close()

    except KeyboardInterrupt:
        pass


