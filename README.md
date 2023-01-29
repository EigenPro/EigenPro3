# EigenPro3
EigenPro (short for Eigenspace Projections) is an algorithm for training general kernel models of the form
$$f(x)=\sum_{i=1}^p \alpha_i K(x,z_i)$$
where $z_i$ are $p$ model centers.

# Limitations of EigenPro2.0
EigenPro 2.0 can only train models of the form $$f(x)=\sum_{i=1}^n \alpha_i K(x,x_i)$$ where $x_i$ are $n$ training samples.

# Algorithm
EigenPro 3.0 applies a dual preconditioner, one for the model and one for the data. It applies a projected-preconditioned SGD
$$f^{t+1}=\mathrm{proj}(f^t - \eta\mathcal{P}(\nabla L(f^t)))$$
where $\nabla L$ is a Fréchet derivative, $\mathcal{P}$ is a preconditioner, and $\textrm{proj}$ is a projection operator
