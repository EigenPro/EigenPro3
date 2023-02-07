# EigenPro
EigenPro (short for Eigenspace Projections) is an algorithm for training general kernel models of the form
$$f(x)=\sum_{i=1}^p \alpha_i K(x,z_i)$$
where $z_i$ are $p$ model centers.

# Installation
```
pip install git+https://github.com/EigenPro/EigenPro3.git
```
Requirements:
- pytorch >= 1.13 (not installed along with this package)

## Testing installation
```python
import torch
from eigenpro3.kernels import laplacian
from eigenpro3 import KernelModel, Dataset

n, p, d, c = 1000, 100, 10, 3
bw = 1.

samples = torch.randn(n, d)
centers = torch.randn(p, d)
labels = torch.randn(n, c)

kernel_fn = lambda x, z: laplacian(x, z, bandwidth=1.)

data = Dataset(samples, kernel_fn=kernel_fn, precondition=True, top_q=10)

model = KernelModel(centers=centers, kernel_fn=kernel_fn)

model.fit(data, labels, batch_size=12)
```

## Limitations of EigenPro 2.0
EigenPro 2.0 can only train models of the form $$f(x)=\sum_{i=1}^n \alpha_i K(x,x_i)$$ where $x_i$ are $n$ training samples.

## Algorithm
**EigenPro 3.0** applies a dual preconditioner, one for the model and one for the data. It applies a projected-preconditioned SGD
$$f^{t+1}=\mathrm{proj}(f^t - \eta\mathcal{P}(\nabla L(f^t)))$$
where $\nabla L$ is a Fréchet derivative, $\mathcal{P}$ is a preconditioner, and $\textrm{proj}$ is a projection operator onto the model space.

## COMING SOON
The code is scheduled for deployment on or before February 15, 2023. Stay tuned!!
