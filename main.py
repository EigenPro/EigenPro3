import os, torch
from eigenpro3.data_utils import load_dataset
from eigenpro3.utils import CustomDataset, accuracy
from eigenpro3.models import KernelModel
from eigenpro3.kernels import laplacian, ntk_relu
from torch.nn.functional import one_hot

p = 5000 # model size

#kernel_fn = lambda x, z: laplacian(x, z, bandwidth=20.0)
kernel_fn = lambda x, z: ntk_relu(x, z, depth=2)

n_classes, (X_train, y_train), (X_test, y_test) = load_dataset('cifar10')

centers = X_train[torch.randperm(X_train.shape[0])[:p]]

testloader = torch.utils.data.DataLoader(
    CustomDataset(X_test, y_test.argmax(-1)), batch_size=512,
    shuffle=False, num_workers=16,pin_memory=True)


model = KernelModel(y_train, centers, kernel_fn, X=X_train,
    devices =[torch.device('cuda:0'), torch.device('cuda:1')], 
    multi_gpu=True)

model.fit(model.train_loaders, testloader, score_fn=accuracy)
