import torch
from eigenpro3.utils import load_dataset
from eigenpro3.utils import accuracy
from eigenpro3.utils import setup_wandb
from eigenpro3.datasets import CustomDataset
from eigenpro3.models import KernelModel
from eigenpro3.kernels import laplacian, ntk_relu
from torch.nn.functional import one_hot
import os

import argparse


import numpy as np
import ipdb

from eigenpro3.utils.data_utils import Cifar5mmobilenetDataset

os.environ['DATA_DIR'] = '/expanse/lustre/projects/csd716/amirhesam/data/' #####EDIT THIS LINE TO MATCH YOUR DATA LOCATION
if torch.cuda.is_available():
    DEVICE_LIST = (torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count()))
else:
    DEVICE_LIST = [torch.device(f'cpu')]

torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument(f'--T', default=1, type=int)
args = parser.parse_args()


kernel_fn = lambda x, z: laplacian(x, z, bandwidth=20.0)
# kernel_fn = lambda x, z: ntk_relu(x, z, depth=10)

# p = 1_000 # model size
# n_classes, (X_train, y_train), (X_test, y_test) = load_dataset('cifar10')
# centers = X_train[torch.randperm(X_train.shape[0])[:p]]

cifa5m = Cifar5mmobilenetDataset(subsample=2_000_000,n_test=100000, num_knots=1_000_000)
n_classes=10
X_train, y_train = cifa5m.X_train,one_hot(cifa5m.y_train,n_classes)
X_test, y_test  = cifa5m.X_test,one_hot(cifa5m.y_test,n_classes)
centers  = cifa5m.knots_x


train_eval_inices = np.random.choice(X_train.shape[0],50_000,replace=False)
test_eval_inices = np.random.choice(X_test.shape[0],50_000,replace=False)

testloader = torch.utils.data.DataLoader(
    CustomDataset(X_test[test_eval_inices], y_test[test_eval_inices]), batch_size=8192,
    shuffle=False, pin_memory=True)

trainloader = torch.utils.data.DataLoader(
    CustomDataset(X_train[train_eval_inices], y_train[train_eval_inices]), batch_size=8192,
    shuffle=False, pin_memory=True)

wandb_init = {}
wandb_init ['project_name'] ='EP3_speedup'
wandb_init['name'] = f'n={X_train.shape[0]}, p={centers.shape[0]}'
wandb_init["key"] = "d6a313a1acc41247c5261111fade60242038f3fd"
wandb_init["mode"]= "online"
wandb_init['org'] = "belkinlab"

wandb_run = setup_wandb(wandb_init)


print(f'number of training set:{X_train.shape[0]}')
print(f'number of centers:{centers.shape[0]}')

model = KernelModel(n_classes, centers, kernel_fn, everyTProject=args.T,
                    X=X_train,y=y_train,devices = DEVICE_LIST,wandb_run=wandb_run)

model.fit(model.train_loaders, [trainloader,testloader], score_fn=accuracy, epochs=50)
