# dataset.py
import torch
from torch.utils.data import Dataset

class ClassificationDataset_only_image(Dataset):
 
    def __init__(self, value_names, labels):
        self.value_names = value_names
        self.label = labels

    def __len__(self):
        return len(self.value_names)
 
    def __getitem__(self, idx):
        value = self.value_names[idx][0]
        label = torch.tensor(self.label[idx])
        label = label.long()
        sample = {'value': value, 'label': label}
        return sample