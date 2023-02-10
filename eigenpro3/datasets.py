import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.functional import one_hot
from os.path import join as pjoin


class CustomDataset(Dataset):

    def __init__(self, X, y, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self,idx):
        return (self.X[idx], self.y[idx])


def makedataloaders(X, y, devices=[torch.device('cpu')]):

        device = devices
        samples_per_device = X.shape[0] // len(devices)
        trainloaders = []
        
        for i, g in enumerate(devices):
            trainloaders.append(
                CustomDataset(
                    X[i * samples_per_device : (i+1) * samples_per_device].to(g), 
                    y[i * samples_per_device : (i+1) * samples_per_device].to(g) 
                )
            )

        return trainloaders
