import torch
from .utils import get_optimal_params

class KernelModel:

    def __init__(self, centers, kernel_fn, dev_mem=16, device=None):
        self.centers = centers
        self.device = device
        self.n_centers = len(centers)
        self.kernel = kernel_fn
        self.weight = None
        self.kzz_inv = torch.linalg.inv(self.kernel(self.centers, self.centers))


    def predict(self, X, return_kmat=False):
        kmat = self.kernel(X, self.centers)
        preds = kmat @ self.weight
        if return_kmat:
            return preds, kmat
        else:
            return preds


    def score(self, X, y, metric='mse'):
        if metric == 'mse':
            return (self.predict.sub_(y)).pow_(2).mean()
        elif metric == 'mse':
            return (self.predict.argmax(-1)==y.argmax(-1)).sum()/len(y)


    def fit(self, X, y, epochs=1, batch_size=None, lr=0.01):
        self.weight = torch.zeros(self.n_centers, y.shape[-1], device=self.device)
        batch_size = X.n_samples if batch_size is None else batch_size
        batches = torch.randperm(X.n_samples).split(batch_size)
        self.Kzxs = self.kernel(self.centers, X.data[X.nystrom_ids])
        for t in range(epochs):
            self.fit_epoch_(X.data, y, batches, X.corrector, lr=lr)


    def fit_epoch_(self, X, y, batches, data_corrector=None, lr=None):
        for batch_ids in batches:
            self.weight.sub_(lr * self.fit_batch_(X[batch_ids], y[batch_ids], batch_ids, data_corrector))


    def fit_batch_(self, X, y, batch_ids, data_corrector=None):
        grad = self.preconditioned_gradient_(X, y, batch_ids, data_corrector)
        return self.projector_(grad)


    def preconditioned_gradient_(self, X, y, batch_ids, data_corrector=None):
        preds, kmat = self.predict(X, return_kmat=True)
        grad = preds - y
        return kmat.T @ grad - self.Kzxs @ data_corrector(grad, batch_ids)


    def projector_(self, grad):
        return self.kzz_inv @ grad
