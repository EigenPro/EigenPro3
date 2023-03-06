from .svd import nystrom_kernel_svd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import ipdb
import wandb
import os
from .printing import midrule, bottomrule
from torch.cuda.comm import broadcast
from concurrent.futures import ThreadPoolExecutor

import time
# def get_optimal_params(mem_gb):
#     raise NotImplementedError
#     return bs, eta, top_q

# def mse(y, y_hat):
#     loss_mse = torch.nn.MSELoss(reduction='mean')
#     return loss_mse(y, y_hat).item()





def update_wandb_config(mydict, run_config, key_prefix=''):
    for key, value in mydict.items():
        if isinstance(value, dict):
            update_wandb_config(value, run_config, key_prefix=key_prefix + str(key) + '.')
        else:
            run_config.update({f'{key_prefix}{key}': f'{value}'})


def setup_wandb(setup_dict):
    print("wandb setup starts...")

    os.environ["WANDB_API_KEY"] = setup_dict['key']
    os.environ["WANDB_MODE"] = setup_dict['mode']  # online or offline

    run = wandb.init(project=setup_dict['project_name'], \
                     entity=setup_dict['org'])
    # we'll use the run_id as the name and correlate it with filesystem organization
    run.name = setup_dict['name']
    run.save()
    # for now we are overriding wandb config with our config
    # but this can also go the other way around if it's easier
    update_wandb_config(setup_dict, run.config)

    return run



def mse(alpha, centers, dataloader, kernel_fn, devices=[torch.device('cpu')]):
    # alpha= alpha.to(device)

    center0_all = divide_to_gpus(centers[0],
                                 centers[0].shape[0],
                                 devices)
    center2_all = divide_to_gpus(centers[2],
                                 centers[2].shape[0],
                                 devices)

    alpha0_all = divide_to_gpus(alpha[0],
                                alpha[0].shape[0],
                                devices)
    alpha2_all = divide_to_gpus(alpha[2],
                                alpha[2].shape[0],
                                devices)
    loss = 0
    cnt = 0
    loss_mse = torch.nn.MSELoss(reduction='sum')
    for (X_batch,y_batch) in dataloader:
        # X_batch = X_batch.to(device)
        # kxbatchz0 = kernel_fn(X_batch,centers[0])
        # kxbatchz1 = kernel_fn(X_batch, centers[1])
        # yhat_test = kxbatchz0 @ alpha[0].to(device) + kxbatchz1 @ alpha[1].to(device)
        # if len(alpha[2])>0:
        # kxbatchz2 = kernel_fn(X_batch, centers[2])
        # yhat_test+=kxbatchz2 @ alpha[2].to(device)
        X_batch_all = broadcast(X_batch, devices)
        predict_0 = multi_gpu_grad(X_batch_all, center0_all, alpha0_all,kernel_fn,devices[0])
        predict_1 = kernel_fn(X_batch.to(devices[0]), centers[1].to(devices[0]))@alpha[1]
        predict_2 = multi_gpu_grad(X_batch_all, center2_all, alpha2_all,kernel_fn,devices[0])
        yhat_test = predict_0.to(devices[0]) + predict_1.to(devices[0]) + predict_2.to(devices[0])

        y_batch = y_batch.to(devices[0])
        cnt += X_batch.shape[0]

        loss += loss_mse(yhat_test,y_batch)

        del X_batch, y_batch
        torch.cuda.empty_cache()

    loss = loss / cnt

    return loss

def mse_bu(alpha, centers, dataloader, kernel_fn, device=torch.device('cpu')):
    alpha= alpha.to(device)
    loss = 0
    cnt = 0
    loss_mse = torch.nn.MSELoss(reduction='sum')
    for (X_batch,y_batch) in dataloader:
        X_batch = X_batch.to(device)
        kxbatchz = kernel_fn(X_batch,centers)
        yhat_test = kxbatchz @ alpha


        y_batch = y_batch.to(device)
        cnt += X_batch.shape[0]

        loss += loss_mse(yhat_test,y_batch)

        del X_batch, y_batch
        torch.cuda.empty_cache()

    loss = loss / cnt

    return loss


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


def accuracy(alpha, centers, dataloader, kernel_fn, devices=[torch.device('cpu')]):
    # alpha= alpha.to(device)

    center0_all = divide_to_gpus(centers[0],
                                 centers[0].shape[0],
                                 devices)
    center2_all = divide_to_gpus(centers[2],
                                 centers[2].shape[0],
                                 devices)

    alpha0_all = divide_to_gpus(alpha[0],
                                 alpha[0].shape[0],
                                 devices)
    alpha2_all = divide_to_gpus(alpha[2],
                                 alpha[2].shape[0],
                                 devices)
    accu = 0
    cnt = 0
    for (X_batch,y_batch) in dataloader:
        # X_batch = X_batch.to(device)
        X_batch_all = broadcast(X_batch, devices)
        # kxbatchz0 = kernel_fn(X_batch,centers0)
        # kxbatchz1 = kernel_fn(X_batch, centers1)
        # yhat_test = kxbatchz0 @ alpha[0].to(device) + kxbatchz1 @ alpha[1].to(device)
        # kxbatchz2 = kernel_fn(X_batch, centers[2])
        # yhat_test+=kxbatchz2 @ alpha[2].to(device)
        predict_0 = multi_gpu_grad(X_batch_all, center0_all, alpha0_all,kernel_fn,devices[0])
        predict_1 = kernel_fn(X_batch.to(devices[0]), centers[1].to(devices[0]))@alpha[1]
        predict_2 = multi_gpu_grad(X_batch_all, center2_all, alpha2_all,kernel_fn,devices[0])

        yhat_test = predict_0.to(devices[0]) + predict_1.to(devices[0]) + predict_2.to(devices[0])

        y_batch = y_batch.to(devices[0]).argmax(-1)
        cnt += X_batch.shape[0]

        yhat_test_sign = Yaccu(yhat_test)
        accu += sum(yhat_test_sign == y_batch)

        del X_batch, y_batch
        torch.cuda.empty_cache()

    accu = accu / cnt

    return accu

def accuracy_bu(alpha, centers, dataloader, kernel_fn, device=torch.device('cpu')):
    alpha= alpha.to(device)
    accu = 0
    cnt = 0
    for (X_batch,y_batch) in dataloader:
        X_batch = X_batch.to(device)
        kxbatchz = kernel_fn(X_batch,centers)
        yhat_test = kxbatchz @ alpha


        y_batch = y_batch.to(device).argmax(-1)
        cnt += X_batch.shape[0]

        yhat_test_sign = Yaccu(yhat_test)
        accu += sum(yhat_test_sign == y_batch)

        del X_batch, y_batch
        torch.cuda.empty_cache()

    accu = accu / cnt

    return accu


def fmm(k, theta, device):
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

def log_performance(weights,centers,val_loader,kernel, devices,wandb_run,t,time_start,name='Train'):
    
    elapsed_time = time.time() - time_start
    accu = accuracy(weights, centers, val_loader, kernel, devices)
    mse_out = mse(weights, centers, val_loader, kernel, devices)
    print(midrule)
    print(f'Step {t + 1:4d}        {name} accuracy: {accu * 100.:5.2f}%')
    print(f'Step {t + 1:4d}        {name} mse: {mse_out:5.2f}')
    print(bottomrule)

    wandb_run.define_metric(f'accu_{name}', step_metric='t')
    wandb_run.define_metric(f'mse_{name}', step_metric='t')

    result_dict = {f'accu_{name}': accu * 100,
                   f'mse_{name}': mse_out,
                   't':t,
                   'elapsed time':elapsed_time
                   }

    wandb_run.log(result_dict)

def multi_gpu_grad(X_batch_all,centers_all,weights_all,kernel,device_base,Kz_xbatch_chunk_return=False):

    with ThreadPoolExecutor() as executor:
        Kz_xbatch_chunk = [executor.submit(kernel, inputs[0], inputs[1]) for inputs
                           in zip(*[X_batch_all, centers_all])]

    kxbatchz_all = [i.result() for i in Kz_xbatch_chunk]

    ######## gradient calculation parallel on GPUs
    with ThreadPoolExecutor() as executor:
        gradients = [executor.submit(fmm, inputs[0], inputs[1], device_base) for inputs
                     in zip(*[kxbatchz_all, weights_all])]

    del kxbatchz_all

    grad = 0
    ##### summing gradients over GPUs
    for r in gradients:
        grad += r.result()
    del gradients

    if Kz_xbatch_chunk_return:
        return grad,Kz_xbatch_chunk
    else:
        return grad