# EigenPro
EigenPro (short for Eigenspace Projections) is an algorithm for training general kernel models of the form
$$f(x)=\sum_{i=1}^p \alpha_i K(x,z_i)$$
where $z_i$ are $p$ model centers.

# Installation
```
pip install git+https://github.com/EigenPro/EigenPro3.git@testing
```
Requirements:
- pytorch >= 1.13 (not installed along with this package)

## Testing installation
```python
import torch
from eigenpro3.utils import accuracy, load_dataset
from eigenpro3.datasets import CustomDataset
from eigenpro3.models import KernelModel
from eigenpro3.kernels import laplacian, ntk_relu

p = 5000 # model size

kernel_fn = lambda x, z: laplacian(x, z, bandwidth=20.0)
# kernel_fn = lambda x, z: ntk_relu(x, z, depth=2)

n_classes, (X_train, y_train), (X_test, y_test) = load_dataset('cifar10')

centers = X_train[torch.randperm(X_train.shape[0])[:p]]

testloader = torch.utils.data.DataLoader(
    CustomDataset(X_test, y_test.argmax(-1)), batch_size=512,
    shuffle=False, num_workers=16,pin_memory=True)


model = KernelModel(y_train[p:], centers, kernel_fn, X=X_train[p:],
    devices = [torch.device('cuda:i') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [torch.device('cpu')], 
    multi_gpu=True)

model.fit(model.train_loaders, testloader, score_fn=accuracy)
```

## Limitations of EigenPro 2.0
EigenPro 2.0 can only train models of the form $$f(x)=\sum_{i=1}^n \alpha_i K(x,x_i)$$ where $x_i$ are $n$ training samples.

## Algorithm
**EigenPro 3.0** applies a dual preconditioner, one for the model and one for the data. It applies a projected-preconditioned SGD
$$f^{t+1}=\mathrm{proj}(f^t - \eta\mathcal{P}(\nabla L(f^t)))$$
where $\nabla L$ is a Fréchet derivative, $\mathcal{P}$ is a preconditioner, and $\textrm{proj}$ is a projection operator onto the model space.

## COMING SOON
The code is scheduled for deployment on or before February 15, 2023. Stay tuned!!
