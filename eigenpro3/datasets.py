import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import one_hot
from os.path import join as pjoin

def makedataloaders( X,y,devices=[torch.device('cpu')]):



        device = devices
        X_train = torch.tensor(X)
        y_train = torch.tensor(y)

        batches_in_1gpu = X_train.shape[0] // len(devices)
        X_train_all = []
        y_train_all = []
        for ind, g in enumerate(devices):
            if ind < len(devices) - 1:
                X_train_all.append(X_train[ind * batches_in_1gpu:(ind + 1) * batches_in_1gpu].to(g))
                y_train_all.append(y_train[ind * batches_in_1gpu:(ind + 1) * batches_in_1gpu].to(g))
            else:
                X_train_all.append(X_train[ind * batches_in_1gpu:].to(g))
                y_train_all.append(y_train[ind * batches_in_1gpu:].to(g))


        trainloader = []
        for ind in range(len(devices)):
            trainloader.append(dataset_custom(X_train_all[ind],y_train_all[ind]))
        return trainloader

class dataset_custom(Dataset):
    def __init__(self, X,Y,
                 **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = Y



# import torch
# from .svd import nystrom_kernel_svd
# class Dataset:
#
#     def __init__(self, data, kernel_fn, nystrom_size=10000, top_q=None, precondition=False):
#         self.data = data
#         self.n_samples = len(data)
#         self.kernel = kernel_fn
#         self.nystrom_size = nystrom_size
#         self.top_q = nystrom_size//10 if top_q is None else top_q
#
#         if precondition:
#             self.setup_preconditioner()
#         else:
#             self.correction = lambda x: x
#
#
#     def setup_preconditioner(self):
#         self.nystrom_ids = torch.randperm(self.n_samples)[:self.nystrom_size]
#         eigvals, eigvecs, self.beta = nystrom_kernel_svd(
#             self.data[self.nystrom_ids].cpu(), self.kernel, self.top_q+1)
#         self.scaled_eigvecs = eigvecs[:, 1:]*((1-eigvals[0]/eigvals[1:])*1/eigvals[1:]).sqrt()
#         self.tail_eigval = eigvals[0]
#
#
#     def corrector(self, g, batch_ids):
#         kmat_g = self.kernel(self.data[self.nystrom_ids], self.data[batch_ids]) @ g
#         return self.scaled_eigvecs @ (self.scaled_eigvecs.T @ kmat_g)
