'''Construct kernel model with EigenPro optimizer.'''
import collections, time, torch, concurrent.futures, torch.nn as nn
from .utils.svd import nystrom_kernel_svd
from .utils import midrule, bottomrule
from timeit import default_timer as timer
import math


def asm_eigenpro_fn(samples, map_fn, top_q, bs_gpu, alpha, min_q=5, seed=1):
    """Prepare gradient map for EigenPro and calculate
    scale factor for learning ratesuch that the update rule,
        p <- p - eta * g
    becomes,
        p <- p - scale * eta * (g - eigenpro_fn(g))

    Arguments:
        samples:	matrix of shape (n_sample, n_feature).
        map_fn:    	kernel k(samples, centers) where centers are specified.
        top_q:  	top-q eigensystem for constructing eigenpro iteration/kernel.
        bs_gpu:     maxinum batch size corresponding to GPU memory.
        alpha:  	exponential factor (<= 1) for eigenvalue rescaling due to approximation.
        min_q:  	minimum value of q when q (if None) is calculated automatically.
        seed:   	seed for random number generation.

    Returns:
        eigenpro_fn:	tensor function.
        scale:  		factor that rescales learning rate.
        top_eigval:  	largest eigenvalue.
        beta:   		largest k(x, x) for the EigenPro kernel.
    """
    start = time.time()
    n_sample, _ = samples.shape

    if top_q is None:
        svd_q = min(n_sample - 1, 1000)
    else:
        svd_q = top_q

    eigvals, eigvecs, beta = nystrom_kernel_svd(samples, map_fn, svd_q)
    eigvecs /= math.sqrt(n_sample)

    # Choose k such that the batch size is bounded by
    #   the subsample size and the memory size.
    #   Keep the original k if it is pre-specified.
    if top_q is None:
        max_bs = min(max(n_sample / 5, bs_gpu), n_sample)
        top_q = (torch.pow(1 / eigvals, alpha) < max_bs).sum().data - 1
        top_q = max(top_q, min_q)

    eigvals, tail_eigval = eigvals[:top_q - 1], eigvals[top_q - 1]
    eigvecs = eigvecs[:, :top_q - 1]

    device = samples.device
    eigvals_t = eigvals.to(device)
    eigvecs_t = eigvecs.to(device)
    tail_eigval_t = tail_eigval.float().to(device)

    scale = torch.pow(eigvals[0] / tail_eigval, alpha)
    diag_t = (1 - torch.pow(tail_eigval_t / eigvals_t, alpha)) / eigvals_t

    def eigenpro_fn(grad, kmat):
        '''Function to apply EigenPro preconditioner.'''
        return torch.mm(eigvecs_t * diag_t,
                        torch.t(torch.mm(torch.mm(torch.t(grad),
                                                  kmat),
                                         eigvecs_t)))

    print("Projection SVD_time: %.2f s, top_q: %d, top_eigval: %.2f, new top_eigval: %.2e" %
          (time.time() - start, top_q, eigvals[0], eigvals[0] / scale))


    return eigenpro_fn, scale, eigvals[0], beta, eigvals, eigvecs


class HilbertProjection(nn.Module):
    '''Fast Kernel Regression using EigenPro iteration.'''

    def __init__(self, kernel_fn, centers, y_dim, devices=["cuda"],
                 weight_init=None,wandb=None,kzz_gpu=False,multi_gpu=False):
        super().__init__()
        self.multi_gpu = multi_gpu
        self.kernel_fn = kernel_fn
        self.n_centers, self.x_dim = centers.shape

        self.devices = devices
        self.device = devices[0]
        self.pinned_list = []
        self.eigenpro_f = None
        self.precond_verbose = True

        self.wandb_run = wandb
        self.centers = self.tensor(centers, release=True)

        if multi_gpu:
            self.centers_replica = torch.cuda.comm.broadcast(self.centers, devices)
            self.centers_all = []
            for i in range(len(devices)):
                if i < len(devices) - 1:
                    self.centers_all.append(self.centers_replica[i][i * self.n_centers // len(devices):
                                                                    (i + 1) * self.n_centers // len(devices), :])
                else:
                    self.centers_all.append(self.centers_replica[i][i * self.n_centers // len(devices):, :])
        else:
            self.centers_all  = [self.centers.to(self.device)]
            self.centers_replica = self.centers_all


        self.fmmv = lambda x, y,theta: (self.kernel_fn(x,y)@theta)

        self.kzz_gpu = kzz_gpu
        if self.kzz_gpu:
            self.kzz = self.kernel_fn(self.centers,self.centers).to(self.device)

        self.mse_error = torch.tensor(100_000, device=self.device)


        if weight_init is not None:
            self.weight = self.tensor(weight_init, release=True)
        else:
            self.weight = self.tensor(torch.zeros(
                self.n_centers, y_dim), release=True)
        self.weight_decay = None

        if multi_gpu:
            self.weight_replica = torch.cuda.comm.broadcast(self.weight, devices)
            self.weight_all = []
            for i in range(len(devices)):
                if i < len(devices) - 1:
                    self.weight_all.append(self.weight_replica[i][i * self.n_centers // len(devices):
                                                                  (i + 1) * self.n_centers // len(devices), :])
                else:
                    self.weight_all.append(self.weight_replica[i][i * self.n_centers // len(devices):, :])
        else:
            self.weight_all = [self.weight.to(self.device)]


    def sync_gpu(self):
        for i in self.devices:
            torch.cuda.synchronize(i)

    def __del__(self):
        for pinned in self.pinned_list:
            _ = pinned.to("cpu")
        torch.cuda.empty_cache()

    def tensor(self, data, dtype=None, release=True):
        if torch.is_tensor(data):
            tensor = data.detach().clone().to(dtype=dtype, device=self.device)
        else:
            tensor = torch.tensor(data, dtype=dtype,
                              requires_grad=False).to(self.device)
        if release:
            self.pinned_list.append(tensor)
        return tensor



    def forward(self, samples_all):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            res = [executor.submit(self.fmmv, input[0], input[1], input[2]) for input
                   in zip(*[samples_all, self.centers_all, self.weight_all])]
        pred = 0
        for i in range(len(self.devices)):
            pred += res[i].result().to(self.device)
        return pred


    def primal_gradient(self, samples_all, labels):
        pred = self.forward(samples_all)
        # ipdb.set_trace()
        grad = pred - labels
        return grad

    @staticmethod
    def _compute_opt_params(bs, bs_gpu, beta, top_eigval):
        if bs is None:
            bs = min(int(beta / top_eigval + 1), bs_gpu)

        if bs < beta / top_eigval + 1:
            eta = bs / beta /2
        else:
            eta = 0.99 * 1 * bs / (beta + (bs - 1) * top_eigval)
        return bs, eta

    def eigenpro_iterate(self, z_batch_all, gz_batch, eta, batch_ids):
        grad = self.primal_gradient(z_batch_all, gz_batch)
        self.weight.index_add_(0, batch_ids, -eta * grad)

        # update fixed coordinate block (for EigenPro)
        kmat = self.kernel_fn(z_batch_all[0], self.nystrom_samples)
        correction = self.eigenpro_f(grad, kmat)
        self.weight.index_add_(0, self.nystrom_ids, eta * correction)
        self.weight.mul_(1 - eta * self.weight_decay)

        if self.multi_gpu:
            for i in range(len(self.devices)):
                if i<len(self.devices)-1:
                    self.weight_all[i] = self.weight[i*self.n_centers//len(self.devices):
                                                                    (i+1)*self.n_centers//len(self.devices),:].to(self.devices[i])
                else:
                    self.weight_all[i] = self.weight[i*self.n_centers//len(self.devices):,:].to(self.devices[i])
        else:
            self.weight_all = [self.weight.to(self.device)]

        return

    def evaluate(self, z_eval, y_eval, bs,
                 metrics=('mse', 'multiclass-acc')):

        p_eval = self.forward(z_eval)


        eval_metrics = collections.OrderedDict()
        if 'mse' in metrics:
            eval_metrics['mse'] = torch.mean(torch.square(p_eval - y_eval))
        return eval_metrics


    def setup_preconditioner(self, *args):

        (self.eigenpro_f, self.gap, self.top_eigval,
         self.beta, self.eigvals, self.eigvecs) = asm_eigenpro_fn(*args)
        self.new_top_eigval = self.top_eigval / self.gap

    def fit_batch(self, z_batch_all, gz_batch, eta, batch_ids):
        self.eigenpro_iterate(z_batch_all, gz_batch, eta, batch_ids)


    def fit_hilbert_projection(
        self,  gz_train,mem_gb=12, weight_decay=None,
        n_nystrom_subsamples=None, top_q=None, bs=None, eta=None, scale=1,metrics=['mse']
    ):

        self.weight_decay = 0.0 if weight_decay is None else weight_decay
        self.weight = self.weight * 0
        n_samples, n_labels = gz_train.shape

        # Calculate batch size / learning rate for improved EigenPro iteration.
        if self.eigenpro_f is None:

            if n_nystrom_subsamples is None:
                if n_samples < 100000:
                    n_nystrom_subsamples = min(n_samples, 2000)
                else:
                    n_nystrom_subsamples = 10000


            mem_bytes = (mem_gb - 1) * 1024 ** 3  # preserve 1GB
            bsizes = torch.arange(n_samples)
            mem_usages = ((self.x_dim + 3 * n_labels + bsizes + 1)
                          * self.n_centers + n_nystrom_subsamples * 1000) * 4
            bs_gpu = torch.sum(mem_usages < mem_bytes)  # device-dependent batch size

            sample_ids = torch.randperm(n_samples)[:n_nystrom_subsamples]
            self.nystrom_ids = self.tensor(sample_ids).long()
            self.nystrom_samples = self.centers[self.nystrom_ids]
            self.setup_preconditioner(self.nystrom_samples, self.kernel_fn, top_q, bs_gpu, .95)
            if eta is None:
                self.bs, self.eta = self._compute_opt_params(
                    bs, bs_gpu, self.beta, self.new_top_eigval)
            else:
                self.bs, _ = self._compute_opt_params(bs, bs_gpu, self.beta, self.new_top_eigval)

            if self.precond_verbose:
                print("Projection setup: Nystrom size=%d, bs_gpu=%d, eta=%.2f, bs=%d, top_eigval=%.2e, beta=%.2f" %
                      (n_nystrom_subsamples, bs_gpu, self.eta, self.bs, self.top_eigval, self.beta))
                print(bottomrule)

            self.bs_gpu = bs_gpu.item()
            self.eta = self.tensor(scale * self.eta / self.bs, dtype=torch.double)

        z_train_eval, gz_train_eval = self.centers[0:1000], gz_train[0:1000]
        if self.multi_gpu:
            z_batch_eval_all = torch.cuda.comm.broadcast(z_train_eval, self.devices)
        else:
            z_batch_eval_all = [z_train_eval.to(self.device)]
        epoch = 0
        self.mse_error = 10000
        while epoch<2 and self.mse_error>10**-6:
            if torch.is_tensor(self.bs):
                final_step = n_samples // self.bs.item()
            else:
                final_step = n_samples // self.bs

            permutation = torch.randperm(self.centers.size()[0],device = self.device)
            step = 0

            for i in range(0,self.centers.size()[0], int(self.bs)):

                batch_ids = permutation[i:i + int(self.bs)]
                z_batch_all = []
                for j in range(len(self.devices)):
                    z_batch_all.append(self.centers_replica[j][batch_ids.cpu(),:])


                self.fit_batch(
                    z_batch_all, gz_train[batch_ids], self.eta, batch_ids
                )
                if self.multi_gpu:
                    self.sync_gpu()
                if step % 5==0 or step == final_step:
                    tr_score = self.evaluate(
                        z_batch_eval_all, gz_train_eval, self.bs, metrics=metrics
                    )
                    self.mse_error = tr_score["mse"]
                step += 1
                if self.mse_error<10**-6:
                    break
            epoch = epoch + 1
        return self.weight
