# EigenPro3: Fast training of large kernel models

*General kernel models* are predictors of the form
$$f(x)=\sum_{i=1}^p \alpha_i K(x,z_i)$$
where $z_i$ are model centers, which can be arbitrary, i.e., they need not be a subset of the training data. EigenPro3 requires only $O(p)$ memory, and uses all availale GPUs, by default.

The algorithm is based on Projected dual-preconditioned Stochastic Gradient Descent. A complete derivation for the training algorithm is given in the following paper  
**Title:** [Toward Large Kernel Models](https://arxiv.org/abs/2302.02605) (2023)  
**Authors:** Amirhesam Abedsoltan, Mikhail Belkin, Parthe Pandit.  
**TL;DR:** Recent studies indicate that kernel machines can perform similarly or better than deep neural networks (DNNs) on small datasets. The interest in kernel machines has been additionally bolstered by the discovery of their equivalence to wide neural networks in certain regimes. 
However, a key feature of DNNs is their ability to scale the model size and training data size independently, whereas in traditional kernel machines model size is tied to data size. EigenPro 3.0, an algorithm that provides a way forward for constructing large-scale *general kernel models* that decouple the model and training data.

# Installation
```
pip install git+https://github.com/EigenPro/EigenPro3.git
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
    shuffle=False, pin_memory=True)

model = KernelModel(n_classes, centers, kernel_fn, X=X_train,y=y_train,devices = DEVICES)
model.fit(model.train_loaders, testloader, score_fn=accuracy, epochs=20)
```
### Downloading Data
```python
from torchvision.datasets import CIFAR10
import os
CIFAR10(os.environ['DATA_DIR'], train=True, download=True)
```
## Tutorial to train in a batched manner
Refer to the [Fashion_MNIST_batched.ipynb](https://github.com/EigenPro/EigenPro3/blob/testing/demos/Custom_dataloader.ipynb) tutorial where you can use your own dataloader.
# Limitations of EigenPro 2.0
EigenPro 2.0 can only train models of the form $$f(x)=\sum_{i=1}^n \alpha_i K(x,x_i)$$ where $x_i$ are $n$ training samples.

## Algorithm details
**EigenPro 3.0** solves the optimization problem,
$$\underset{f\in\mathcal{H}}{\text{argmin}}\quad \sum_{i=1}^n (f(x_i)-y_i)^2 \quad \text{s.t.}\quad f(x)=\sum_{i=1}^p \alpha_i K(x,z_i)\qquad\forall x$$
    
**EigenPro 3.0** applies a dual preconditioner, one for the model and one for the data. It applies a projected-preconditioned SGD
$$f^{t+1}=\textrm{proj}_C(f^t - \eta\mathcal{P}(\nabla L(f^t)))$$
where $\nabla L$ is a Fréchet derivative, $\mathcal{P}$ is a preconditioner, and $\textrm{proj}_C$ is a projection operator onto the model space.
