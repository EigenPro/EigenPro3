'''Implementation of kernel functions.'''

import torch

eps = 1e-12

def euclidean(samples, centers, squared=True):
    '''Calculate the pointwise distance.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        squared: boolean.

    Returns:
        pointwise distances (n_sample, n_center).
    '''
    samples_norm = torch.sum(samples**2, dim=1, keepdim=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = torch.sum(centers**2, dim=1, keepdim=True)
    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(torch.t(centers))
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)
    if not squared:
        distances.clamp_(min=0)        
        distances.sqrt_()

    return distances


def gaussian(samples, centers, bandwidth):
    '''Gaussian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers)
    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


def laplacian(samples, centers, bandwidth):
    '''Laplacian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


def dispersal(samples, centers, bandwidth, gamma):
    '''Dispersal kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.
        gamma: dispersal factor.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers)
    kernel_mat.pow_(gamma / 2.)
    kernel_mat.mul_(-1. / bandwidth)
    kernel_mat.exp_()
    return kernel_mat


def ntk_relu(X, Z, depth=1, bias=0.):
    """
    Returns the evaluation of nngp and ntk kernels
    for fully connected neural networks
    with ReLU nonlinearity.
    
    depth  (int): number of layers of the network
    bias (float): (default=0.)
    """
    from torch import acos, pi
    kappa_0 = lambda u: (1-acos(u)/pi)
    kappa_1 = lambda u: u*kappa_0(u) + (1-u.pow(2)).sqrt()/pi
    Z = Z if Z is not None else X
    norm_x = X.norm(dim=-1)[:, None].clip(min=eps)
    norm_z = Z.norm(dim=-1)[None, :].clip(min=eps)
    S = X @ Z.T
    N = S + bias**2
    for k in range(1, depth):
        in_ = (S/norm_x/norm_z).clip(min=-1+eps,max=1-eps)
        S = norm_x*norm_z*kappa_1(in_)
        N = N * kappa_0(in_) + S + bias**2
    return N

def ntk_relu_unit_sphere(X, Z, depth=1, bias=0.):
    """
    Returns the evaluation of nngp and ntk kernels
    for fully connected neural networks
    with ReLU nonlinearity.
    Assumes inputs are normalized to unit norm.
    
    depth  (int): number of layers of the network
    bias (float): (default=0.)
    """
    from torch import acos, pi
    kappa_0 = lambda u: (1-acos(u)/pi)
    kappa_1 = lambda u: u*kappa_0(u) + (1-u.pow(2)).sqrt()/pi
    Z = Z if Z is not None else X
    S = X @ Z.T
    N = S + bias**2
    for k in range(1, depth):
        in_ = (S).clip(min=-1+eps,max=1-eps)
        S = kappa_1(in_)
        N = N * kappa_0(in_) + S + bias**2
    return N

if __name__ == "__main__":
    import torch
    from torch.nn.functional import normalize
    n, m, d = 1000, 800, 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.randn(n, d, device=DEVICE)
    X_ = normalize(X, dim=-1)
    Z = torch.randn(m, d, device=DEVICE)
    Z_ = normalize(Z, dim=-1)
    KXZ_ntk = ntk_relu(X, Z, 64, bias=1.)
    KXZ_ntk_ = ntk_relu_unit_sphere(X_, Z_, 64, bias=1.)
    print(
        KXZ_ntk.diag().max().item(), 
        KXZ_ntk_.diag().max().item()
    )
