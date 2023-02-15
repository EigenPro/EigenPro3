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
