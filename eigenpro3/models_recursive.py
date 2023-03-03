import torch, time
from .utils import fmm, accuracy, divide_to_gpus, bottomrule, midrule, mse, log_performance
from .utils.svd import nystrom_kernel_svd
from .datasets import makedataloaders
from .projection import HilbertProjection
from torch.cuda.comm import broadcast
from concurrent.futures import ThreadPoolExecutor

import ipdb


class KernelModel():

    def __init__(self, n_classes, centers, kernel_fn, y=None, X=None, devices=[torch.device('cpu')],
                 make_dataloader=True,
                 nystrom_samples=None, n_nystrom_samples=5_000,
                 data_preconditioner_level=500, everyTProject=1, wandb_run=None):
        self.wandb_run = wandb_run

        self.T = everyTProject
        self.devices = tuple(devices)
        self.device_base = self.devices[0]

        self.n_classes = n_classes
        self.make_dataloader = make_dataloader

        self.centers = centers
        self.n_centers = len(centers)
        self.kernel = kernel_fn
        self.weights = torch.zeros(self.n_centers, n_classes)

        self.tmp_centers = []
        self.f_tmp_centers = []
        self.preconditoners_grad = []
        self.corrected_gz_scaled = 0

        self.multi_gpu = len(self.devices) > 1

        if self.multi_gpu:
            ######## dsitributing the weights over all avalibale GPUs
            self.weights_all = divide_to_gpus(self.weights, self.n_centers, self.devices)

            ######## dsitributing the centers over all avalibale GPUs
            self.centers_all = divide_to_gpus(self.centers, self.n_centers, self.devices)

        else:
            self.weights_all = [self.weights.to(self.device_base)]
            self.centers_all = [self.centers.to(self.device_base)]

        ###### DATA Preconditioner
        ##### note that batch size and learning rate will be determined in this stage #########
        if nystrom_samples == None:
            ####### randomly select nystrom samples from X
            nystrom_ids = torch.randperm(X.shape[0])[:n_nystrom_samples]
            self.nystrom_samples = X[nystrom_ids]
        else:
            self.nystrom_samples = nystrom_samples
        print('Setting up data preconditioner')
        start_time = time.time()
        self.data_preconditioner_matrix, self.eigenvectors_data, self.D, self.batch_size, self.lr \
            = self.get_preconditioner(data_preconditioner_level + 1)
        print(f'Setup time = {time.time() - start_time:5.2f} s')
        print("Done.\n" + bottomrule)

        self.centers = self.centers.to(self.device_base)
        self.weights = self.weights.to(self.device_base)
        self.nystrom_samples = self.nystrom_samples.to(self.device_base)
        self.data_preconditioner_matrix = self.data_preconditioner_matrix.to(self.device_base)
        self.eigenvectors_data = self.eigenvectors_data.to(self.device_base)
        self.D = self.D.to(self.device_base)
        ###### DATA Preconditioner #########

        #####log into wandb#####
        self.wandb_run.summary['learning rate'] = self.lr
        self.wandb_run.summary['T'] = self.T
        self.wandb_run.summary['batch_size'] = self.batch_size

        if make_dataloader:
            ###### Distribute all equally over all available GPUs #####
            self.train_loaders = makedataloaders(X, y, self.batch_size, self.devices)

        ###### Initilize inexact projection #########
        print('Setting up inexact projector')
        self.inexact_projector = HilbertProjection(
            self.kernel, self.centers, self.n_classes,
            devices=self.devices, multi_gpu=self.multi_gpu)
        print('Done.\n' + midrule)

    def fit(self, train_loaders, val_loader=None, epochs=10, score_fn=None):
        self.val_loader = val_loader
        self.step = 0
        for t in range(epochs):
            self.epoch = t
            self.fit_epoch(train_loaders)
            # if val_loader!=None and score_fn!=None:
            #     log_performance(self.weights,self.centers,val_loader[0],
            #                     self.kernel,self.device_base,self.wandb_run,t,name='Train')
            #     log_performance(self.weights, self.centers, val_loader[1],
            #                     self.kernel, self.device_base,self.wandb_run,t, name='Test')

    def fit_epoch(self, train_loaders):
        batch_num = 0
        for trainloader_ind, train_loader in enumerate(train_loaders):
            for (X_batch, y_batch) in train_loader:
                ###### fitting the batch
                if (batch_num + 1) % self.T == 0 or batch_num + 1 == len(train_loader):
                    project = True
                else:
                    project = False

                self.fit_batch(X_batch, y_batch, project=project)

                if self.multi_gpu:
                    self.sync_gpus()

                torch.cuda.empty_cache()

                if (batch_num + 1) % 1 == 0:
                    print(f'epoch {self.epoch: 4d}\t batch {batch_num + 1 :4d}')

                batch_num += 1
                del X_batch, y_batch

    def fit_batch(self, X_batch, y_batch, project=True):

        if self.multi_gpu:
            ##### Putting a copy of the batch in all available GPUs
            X_batch_all = broadcast(X_batch, self.devices)
            y_batch_all = broadcast(y_batch, self.devices)
        else:
            X_batch_all = [X_batch.to(self.device_base)]
            y_batch_all = [y_batch.to(self.device_base)]

        preconditoner_grad = self.precondition_correction(X_batch_all[0])
        self.preconditoners_grad.append(preconditoner_grad)

        gz, grad = self.get_gradient(X_batch_all, y_batch_all)

        self.f_tmp_centers.append(grad)
        self.tmp_centers.append(X_batch_all[0])

        preconditon_grad = self.data_preconditioner_matrix @ (preconditoner_grad @ grad)
        corrected_gz = gz - preconditon_grad

        del X_batch_all, y_batch_all
        if self.multi_gpu:
            self.sync_gpus()

        torch.cuda.empty_cache()
        self.corrected_gz_scaled += corrected_gz.to(self.device_base)

        if self.step % self.T == 0:
            print('Projection...')
            self.update_weights()
            self.tmp_centers = []
            self.f_tmp_centers = []
            self.preconditoners_grad = []
            self.corrected_gz_scaled = 0
            print('Projection done.')
        log_performance(self.weights, self.centers, self.val_loader[0],
                        self.kernel, self.device_base, self.wandb_run, self.step, name='Train')
        log_performance(self.weights, self.centers, self.val_loader[1],
                        self.kernel, self.device_base, self.wandb_run, self.step, name='Test')
        self.step += 1

    def get_gradient(self, X_batch_all, y_batch_all):

        ####### Kz_xbatch calculation parallel on GPUs
        with ThreadPoolExecutor() as executor:
            Kz_xbatch_chunk = [executor.submit(self.kernel, inputs[0], inputs[1]) for inputs
                               in zip(*[X_batch_all, self.centers_all])]

        kxbatchz_all = [i.result() for i in Kz_xbatch_chunk]

        ######## gradient calculation parallel on GPUs
        with ThreadPoolExecutor() as executor:
            gradients = [executor.submit(fmm, inputs[0], inputs[1], self.device_base) for inputs
                         in zip(*[kxbatchz_all, self.weights_all])]

        del kxbatchz_all

        grad = 0
        out = []
        ##### summing gradients over GPUs
        for r in gradients:
            grad += r.result()
        del gradients

        ###### complete gradient
        grad = grad - y_batch_all[0]

        correction_newcenters = self.get_newcenters_corrections(X_batch_all[0])
        grad = (grad - (self.lr / (self.batch_size)) * correction_newcenters)

        ##### gradients calculated in centers
        for r in Kz_xbatch_chunk:
            kgrad = r.result().T @ grad.to(r.result().device)
            out.append(kgrad.to(self.device_base))
        del Kz_xbatch_chunk, kgrad

        if self.multi_gpu:
            self.sync_gpus()
        torch.cuda.empty_cache()
        gz = torch.cat(out, dim=0)

        return gz, grad

    def get_newcenters_corrections(self, X_batch):
        if len(self.tmp_centers) == 0:
            return 0
        else:
            Kxbatch_newcenters = self.kernel(X_batch, torch.cat(self.tmp_centers))
            f_all_newcenters = torch.cat(self.f_tmp_centers)
            # ipdb.set_trace()
            return Kxbatch_newcenters @ f_all_newcenters - (self.D * self.preconditoners_grad[-1].T) \
                   @ (torch.cat(self.preconditoners_grad[:-1], dim=1) @ f_all_newcenters)

    def get_preconditioner(self, data_preconditioner_level):
        Lam_x, E_x, beta = nystrom_kernel_svd(
            self.nystrom_samples,
            self.kernel, data_preconditioner_level
        )

        nystrom_size = self.nystrom_samples.shape[0]

        tail_eig_x = Lam_x[data_preconditioner_level - 1]
        Lam_x = Lam_x[:data_preconditioner_level - 1]
        E_x = E_x[:, :data_preconditioner_level - 1]
        D_x = (1 - tail_eig_x / Lam_x) / Lam_x / nystrom_size

        batch_size = int(beta / tail_eig_x)
        if batch_size < beta / tail_eig_x + 1:
            lr = batch_size / beta / (2)
        else:
            lr = learning_rate_prefactor * batch_size / (beta + (batch_size - 1) * tail_eig_x)

        print(f'Data: learning rate: {lr:.2f}, batch size={batch_size:5d}, top eigenvalue:{Lam_x[0]:.2f},'
              f' new top eigenvalue:{tail_eig_x:.2e}, beta:{beta:.2f}')
        print("Data preconditioner is ready.")

        Kmat_xs_z = self.kernel(self.nystrom_samples.cpu(), self.centers)
        preconditioner_matrix = Kmat_xs_z.T @ (D_x * E_x)
        del Kmat_xs_z
        return preconditioner_matrix, E_x, D_x, batch_size, lr

    def precondition_correction(self, X_batch):
        Kmat_xs_xbatch = self.kernel(self.nystrom_samples, X_batch)
        # preconditoner_grad = self.data_preconditioner_matrix @ (self.eigenvectors_data.T @ (Kmat_xs_xbatch))
        preconditoner_grad = self.eigenvectors_data.T @ (Kmat_xs_xbatch)
        del Kmat_xs_xbatch
        return preconditoner_grad

    def update_weights(self):

        gz_projection = self.corrected_gz_scaled.to(self.device_base)
        self.theta2= self.inexact_projector.fit_hilbert_projection(
            gz_projection, mem_gb=25)
        # self.theta2 = torch.inverse(self.kernel(self.centers, self.centers)) @ gz_projection

        # print(self.theta2)
        new_weights = self.weights.to(self.device_base) - (self.lr / (self.batch_size)) * self.theta2.to(
            self.device_base)

        self.weights = new_weights.to(self.device_base)

        if self.multi_gpu:
            for i in range(len(self.devices)):
                if i < len(self.devices) - 1:
                    self.weights_all[i] = self.weights[i * self.n_centers // len(self.devices):
                                                       (i + 1) * self.n_centers // len(self.devices), :].to(
                        self.devices[i])
                else:
                    self.weights_all[i] = self.weights[i * self.n_centers // len(self.devices):, :].to(self.devices[i])
        else:
            self.weights_all = [self.weights]

    def sync_gpus(self):
        for i in self.devices:
            torch.cuda.synchronize(i)
