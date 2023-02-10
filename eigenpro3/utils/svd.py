'''Utility functions for performing fast SVD.'''
import scipy.linalg as linalg
import torch

def nystrom_kernel_svd(samples, kernel_fn, top_q):
    """Compute top eigensystem of kernel matrix using Nystrom method.
    Arguments:
        samples: data matrix of shape (n_sample, n_feature).
        kernel_fn: tensor function k(X, Y) that returns kernel matrix.
        top_q: top-q eigensystem.
    Returns:
        eigvals: top eigenvalues of shape (top_q).
        eigvecs: (rescaled) top eigenvectors of shape (n_sample, top_q).
    """
    # samples /= torch.max(samples).item()
    n_sample, _ = samples.shape
    kmat = kernel_fn(samples, samples).cpu().data.numpy()
    scaled_kmat = kmat / n_sample
    vals, vecs = linalg.eigh(scaled_kmat,
                             eigvals=(n_sample - top_q, n_sample - 1))
    eigvals = torch.from_numpy(vals).flip(0)[:top_q]
    eigvecs = torch.from_numpy(vecs).flip(1)[:, :top_q]
    beta = torch.from_numpy(kmat).diag().max()

    return eigvals, eigvecs, beta
