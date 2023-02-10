from .svd import nystrom_kernel_svd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# def get_optimal_params(mem_gb):
#     raise NotImplementedError
#     return bs, eta, top_q


def Yaccu(y,method = 'argmax'):

    if type(y) != type(torch.tensor([0])):
        y = torch.tensor(y)

    if y.size()[1] == 1:
        y_s = torch.zeros_like(y)
        y_s[torch.where(y > 0)[0]] = 1
        y_s[torch.where(y < 0)[0]] = -1
    elif method == 'argmax':
        y_s = torch.argmax(y, dim=1)
    elif method == 'top5':
        (values, y_s) = torch.topk(y,5,dim=1)

    return y_s

def accuracy(alpha, centers, dataloader, kernel_fn, device=torch.device('cpu')):
    alpha= alpha.to(device)
    accu = 0
    cnt = 0
    for (X_batch,y_batch) in dataloader:
        X_batch = X_batch.to(device)
        kxbatchz = kernel_fn(X_batch,centers)
        y_batch = y_batch.to(device)

        cnt += X_batch.shape[0]
        yhat_test = kxbatchz@alpha
        yhat_test_sign = Yaccu(yhat_test)
        accu += sum(yhat_test_sign == y_batch)

        del X_batch, y_batch
        torch.cuda.empty_cache()

    accu = accu / cnt

    return accu


def fmm(k, theta, y,device):
    grad = (k @ theta)
    return grad.to(device)

def get_precondioner(centers,nystrom_samples,kernel_fn, data_preconditioner_level):
    Lam_x, E_x, beta = nystrom_kernel_svd(
        nystrom_samples,
        kernel_fn, data_preconditioner_level
    )

    nystrom_size = nystrom_samples.shape[0]

    tail_eig_x = Lam_x[data_preconditioner_level-1]
    Lam_x = Lam_x[:data_preconditioner_level-1]
    E_x = E_x[:, :data_preconditioner_level-1]
    D_x = (1 - tail_eig_x / Lam_x) / Lam_x / nystrom_size

    batch_size = int(1 / tail_eig_x)
    if batch_size < beta / tail_eig_x + 1:
        lr = batch_size / beta / (2)
    else:
        lr = learning_rate_prefactor * batch_size / (beta + (batch_size - 1) * tail_eig_x)

    Kmat_xs_z = kernel_fn(nystrom_samples.cpu(), centers)
    preconditioner_matrix = Kmat_xs_z.T @ (D_x * E_x)
    del Kmat_xs_z
    return preconditioner_matrix,E_x,batch_size,lr


def float_x(data):
    '''Set data array precision.'''
    if torch.is_tensor(data):
        data = data.float()
    else:
        data = np.float32(data)
    return data


class CustomDataset(Dataset):

    def __init__(self, X,y,
                 **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self,idx):
        return (
            self.X[idx],
            self.y[idx]
            )

def divide_to_gpus(somelist,chunck_size,devices):
    somelist_replica = torch.cuda.comm.broadcast(somelist, devices)
    somelist_all = []
    for i in range(len(devices)):
        if i < len(devices) - 1:
            somelist_all.append(somelist_replica[i][i * chunck_size // len(devices):
                                                           (i + 1) * chunck_size // len(devices), :])
        else:
            somelist_all.append(somelist_replica[i][i * chunck_size // len(devices):, :])

    return somelist_all
