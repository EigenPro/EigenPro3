import os, torch
from eigenpro3.common_datasets import load_cifar10_data
from eigenpro3.utils import CustomDataset, score
from eigenpro3.models import KernelModel
from eigenpro3.kernels import laplacian
from torch.nn.functional import one_hot

# os.environ['DATA_DIR'] = '/scratch/bbjr/abedsol1/'
kernel_fn = lambda x, y: laplacian(x, y, bandwidth=20.0)

cifar10 = load_cifar10_data()

X_train,y_train = cifar10[1]
y_train = one_hot(y_train)
X_train = (X_train/255.0).reshape(-1,32*32*3)
centers_ids = torch.randperm(X_train.shape[0])[:5_000]
centers = torch.tensor(X_train[centers_ids])


X_test, y_test = cifar10[2]
X_test = (X_test / 255.0).reshape(-1, 32 * 32 * 3)
testloader = torch.utils.data.DataLoader(CustomDataset(X_test,y_test), batch_size=512,
                                             shuffle=False, num_workers=16,pin_memory=True)


model = KernelModel(y_train, centers, kernel_fn, X=X_train,
    devices =[torch.device('cuda:0'), torch.device('cuda:1')], 
    multi_gpu=True)
model.fit(model.train_loaders, testloader, score_fn=score)
