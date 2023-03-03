import torch
from eigenpro3.utils import load_dataset
from eigenpro3.utils import accuracy
from eigenpro3.datasets import CustomDataset
from eigenpro3.models import KernelModel
from eigenpro3.kernels import laplacian, ntk_relu
from torch.nn.functional import one_hot
import os

os.environ['DATA_DIR'] = ''  ##### add your data directory 
if torch.cuda.is_available():
    DEVICE_LIST = (torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count()))
else:
    DEVICE_LIST = [torch.device(f'cpu')]

p = 5000 # model size

kernel_fn = lambda x, z: laplacian(x, z, bandwidth=20.0)
# kernel_fn = lambda x, z: ntk_relu(x, z, depth=10)

n_classes, (X_train, y_train), (X_test, y_test) = load_dataset('cifar10')

centers = X_train[torch.randperm(X_train.shape[0])[:p]]

testloader = torch.utils.data.DataLoader(
    CustomDataset(X_test, y_test.argmax(-1)), batch_size=512,
    shuffle=False, pin_memory=True)


model = KernelModel(n_classes, centers, kernel_fn, X=X_train,y=y_train,devices = DEVICE_LIST)

model.fit(model.train_loaders, testloader, score_fn=accuracy, epochs=20)
