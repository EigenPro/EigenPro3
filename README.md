# EigenPro3
EigenPro (short for Eigenspace Projections) is an algorithm for training general kernel models of the form
$$f(x)=\sum_{i=1}^p \alpha_i K(x,z_i)$$
where $z_i$ are $p$ model centers. The model centers can be arbitrary, i.e., do not need to be a subset of the training data. The algorithm requires only $O(p)$ memory, and takes advantage of multiple GPUs. 

The EigenPro3 algorithm is based on Projected dual-preconditioned Stochastic Gradient Descent. If fully decouples the model and training
A complete derivation for the training algorithm is given in the following paper  
**Title:** [Toward Large Kernel Models](https://arxiv.org/abs/2302.02605) (2023)  
**Authors:** Amirhesam Abedsoltan, Mikhail Belkin, Parthe Pandit.

# Installation
```
pip install git+https://github.com/EigenPro/EigenPro3.git@testing
```
Tested on:
- pytorch >= 1.13 (not installed along with this package)

## Demo on CIFAR-10 dataset
Set an environment variable `DATA_DIR='/path/to/dataset/'` where the file `cifar-10-batches-py` can be found. If you would like to download the data, see instructions below the following code-snippet.
```python
import torch
from eigenpro3.utils import accuracy, load_dataset
from eigenpro3.datasets import CustomDataset
from eigenpro3.models import KernelModel
from eigenpro3.kernels import laplacian, ntk_relu

p = 5000 # model size

if torch.cuda.is_available():
    DEVICES = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
else:
    DEVICES = [torch.device('cpu')]

kernel_fn = lambda x, z: laplacian(x, z, bandwidth=20.0)
# kernel_fn = lambda x, z: ntk_relu(x, z, depth=2)

n_classes, (X_train, y_train), (X_test, y_test) = load_dataset('cifar10')

centers = X_train[torch.randperm(X_train.shape[0])[:p]]

testloader = torch.utils.data.DataLoader(
    CustomDataset(X_test, y_test.argmax(-1)), batch_size=512,
    shuffle=False, num_workers=16,pin_memory=True)


model = KernelModel(y_train[p:], centers, kernel_fn, X=X_train[p:],
    devices = DEVICES, 
    multi_gpu=True)

model.fit(model.train_loaders, testloader, score_fn=accuracy)
```
### Downloading Data
```python
from torchvision.datasets import CIFAR10
import os
CIFAR10(os.environ['DATA_DIR'], train=True, download=True)
```

## Limitations of EigenPro 2.0
EigenPro 2.0 can only train models of the form $$f(x)=\sum_{i=1}^n \alpha_i K(x,x_i)$$ where $x_i$ are $n$ training samples.

## Algorithm
**EigenPro 3.0** applies a dual preconditioner, one for the model and one for the data. It applies a projected-preconditioned SGD
$$f^{t+1}=\mathrm{proj}(f^t - \eta\mathcal{P}(\nabla L(f^t)))$$
where $\nabla L$ is a Fréchet derivative, $\mathcal{P}$ is a preconditioner, and $\textrm{proj}$ is a projection operator onto the model space.

## COMING SOON
The code is scheduled for deployment on or before February 15, 2023. Stay tuned!!
