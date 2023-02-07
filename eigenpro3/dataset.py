import torch
from .svd import nystrom_kernel_svd


class Dataset:

    def __init__(self, data, kernel_fn, nystrom_size=10000, top_q=None, precondition=False, device=None):
        self.device = device
        self.data = data.to(self.device)
        self.n_samples = len(data)
        self.kernel = kernel_fn
        self.nystrom_size = nystrom_size
        self.top_q = nystrom_size//10 if top_q is None else top_q

        if precondition:
            self.setup_preconditioner()
        else:
            self.correction = lambda x: x


    def setup_preconditioner(self):
        self.nystrom_ids = torch.randperm(self.n_samples)[:self.nystrom_size].to(self.device)
        eigvals, eigvecs, self.beta = nystrom_kernel_svd(
            self.data[self.nystrom_ids].cpu(), self.kernel, self.top_q+1)
        self.scaled_eigvecs = (eigvecs[:, 1:]*((1-eigvals[0]/eigvals[1:])*1/eigvals[1:]).sqrt()).to(self.device)
        self.tail_eigval = eigvals[0].to(self.device)


    def corrector(self, g, batch_ids):
        kmat_g = self.kernel(self.data[self.nystrom_ids], self.data[batch_ids]) @ g
        return self.scaled_eigvecs @ (self.scaled_eigvecs.T @ kmat_g)
