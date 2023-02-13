import torch, time
from .utils import fmm, accuracy, divide_to_gpus, bottomrule, midrule 
from .utils.svd import nystrom_kernel_svd
from .datasets import makedataloaders
from .projection import HilbertProjection
from torch.cuda.comm import broadcast
from concurrent.futures import ThreadPoolExecutor


class KernelModel():

    def __init__(self, n_classes ,centers, kernel_fn,y=None,X=None, devices =[torch.device('cpu')], make_dataloader=True,
                 nystrom_samples=None, n_nystrom_samples=5_000, data_preconditioner_level=500):

        self.devices = tuple(devices)
        self.device_base = self.devices[0]

        self.n_classes = n_classes
        self.make_dataloader = make_dataloader

        self.centers = centers
        self.n_centers = len(centers)
        self.kernel = kernel_fn
        self.weights = torch.zeros(self.n_centers, n_classes)

        
        self.multi_gpu = len(self.devices) > 1

        if self.multi_gpu:
            ######## dsitributing the weights over all avalibale GPUs
            self.weights_all = divide_to_gpus(self.weights,self.n_centers, self.devices)

            ######## dsitributing the centers over all avalibale GPUs
            self.centers_all = divide_to_gpus(self.centers,self.n_centers, self.devices)

        else:
            self.weights_all = [self.weights.to(self.device_base)]
            self.centers_all = [self.centers.to(self.device_base)]

        
        ###### DATA Preconditioner
        ##### note that batch size and learning rate will be determined in this stage #########
        if nystrom_samples==None:
            ####### randomly select nystrom samples from X
            nystrom_ids = torch.randperm(X.shape[0])[:n_nystrom_samples]
            self.nystrom_samples = X[nystrom_ids]
        else:
            self.nystrom_samples = nystrom_samples
        print('Setting up data preconditioner')
        start_time = time.time()
        self.data_preconditioner_matrix, self.eigenvectors_data, self.batch_size,self.lr \
            = self.get_preconditioner(data_preconditioner_level + 1)
        print(f'Setup time = {time.time()-start_time:5.2f} s')
        print("Done.\n" + bottomrule)

        self.centers = self.centers.to(self.device_base)
        self.weights = self.weights.to(self.device_base)
        self.nystrom_samples = self.nystrom_samples.to(self.device_base)
        self.data_preconditioner_matrix = self.data_preconditioner_matrix.to(self.device_base)
        self.eigenvectors_data = self.eigenvectors_data.to(self.device_base)
        ###### DATA Preconditioner #########



        if make_dataloader:
        ###### Distribute all equally over all available GPUs #####
            self.train_loaders = makedataloaders(X,y,self.batch_size,self.devices)

        ###### Initilize inexact projection #########
        print('Setting up inexact projector')
        self.inexact_projector = HilbertProjection(
            self.kernel, self.centers, self.n_classes, 
            devices=self.devices, multi_gpu=self.multi_gpu)
        print('Done.\n' + midrule)

    def fit(self, train_loaders, val_loader=None,epochs=10,score_fn=None):
        for t in range(epochs):
            self.epoch = t
            self.fit_epoch(train_loaders)
            if val_loader!=None and score_fn!=None:
                accu = score_fn(self.weights,self.centers,val_loader,self.kernel,self.device_base)
                print(midrule)
                print(f'epoch {t+1:4d}        validation accuracy: {accu*100.:5.2f}%')
                print(bottomrule)



    def fit_epoch(self, train_loaders):
        batch_num = 0
        for trainloader_ind, train_loader in enumerate(train_loaders):
            for (X_batch, y_batch) in train_loader:
                ###### fitting the batch
                self.fit_batch(X_batch, y_batch)

                if self.multi_gpu:
                    self.sync_gpus()

                torch.cuda.empty_cache()

                if (batch_num + 1)%2==0:
                    print(f'epoch {self.epoch: 4d}\t batch {batch_num+1 :4d}')

                batch_num += 1
                del  X_batch, y_batch


    def fit_batch(self, X_batch, y_batch):

        if self.multi_gpu:
            ##### Putting a copy of the batch in all available GPUs
            X_batch_all = broadcast(X_batch, self.devices)
            y_batch_all = broadcast(y_batch, self.devices)
        else:
            X_batch_all = [X_batch.to(self.device_base)]
            y_batch_all = [y_batch.to(self.device_base)]

        gz, grad = self.get_gradient(X_batch_all, y_batch_all)


        preconditon_grad = self.precondition_correction(X_batch_all[0], grad)
        corrected_gz = gz - preconditon_grad

        del X_batch_all, y_batch_all
        if self.multi_gpu:
            self.sync_gpus()

        torch.cuda.empty_cache()
        self.corrected_gz_scaled = corrected_gz.to(self.device_base)
        self.update_weights()


    def get_gradient(self, X_batch_all, y_batch_all):

        ####### Kz_xbatch calculation parallel on GPUs
        with ThreadPoolExecutor() as executor:
            Kz_xbatch_chunk = [executor.submit(self.kernel, inputs[0], inputs[1]) for inputs
                    in zip(*[X_batch_all, self.centers_all])]

        kxbatchz_all = [i.result() for i in Kz_xbatch_chunk]

        ######## gradient calculation parallel on GPUs
        with ThreadPoolExecutor() as executor:
            gradients = [executor.submit(fmm, inputs[0], inputs[1], inputs[2],self.device_base) for inputs
                    in zip(*[kxbatchz_all, self.weights_all, y_batch_all])]

        del kxbatchz_all

        grad = 0
        out = []
        ##### summing gradients over GPUs
        for r in gradients:
            grad += r.result()
        del gradients
        
        ###### complete gradient
        grad = grad - y_batch_all[0]

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


    def get_preconditioner(self, data_preconditioner_level):
        Lam_x, E_x, beta = nystrom_kernel_svd(
            self.nystrom_samples,
            self.kernel, data_preconditioner_level
        )

        nystrom_size = self.nystrom_samples.shape[0]

        tail_eig_x = Lam_x[data_preconditioner_level-1]
        Lam_x = Lam_x[:data_preconditioner_level-1]
        E_x = E_x[:, :data_preconditioner_level-1]
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
        return preconditioner_matrix, E_x, batch_size, lr

    
    def precondition_correction(self, X_batch, grad):
        Kmat_xs_xbatch = self.kernel(self.nystrom_samples, X_batch)
        preconditon_grad = self.data_preconditioner_matrix @ (self.eigenvectors_data.T @ (Kmat_xs_xbatch @ grad))
        del Kmat_xs_xbatch
        return preconditon_grad


    def update_weights(self):

        gz_projection = self.corrected_gz_scaled.to(self.device_base)
        self.theta2= self.inexact_projector.fit_hilbert_projection(
            gz_projection, mem_gb=20)

        new_weights = self.weights.to(self.device_base) - (self.lr / (self.batch_size)) * self.theta2.to(self.device_base)
        
        self.weights = new_weights.to(self.device_base)
        
        if self.multi_gpu:
            for i in range(len(self.devices)):
                if i < len(self.devices) - 1:
                    self.weights_all[i] = self.weights[i * self.n_centers // len(self.devices):
                                                      (i + 1) * self.n_centers // len(self.devices), :].to(self.devices[i])
                else:
                    self.weights_all[i] = self.weights[i * self.n_centers // len(self.devices):, :].to(self.devices[i])
        else:
            self.weights_all = [self.weights]
    
    
    def sync_gpus(self):
        for i in self.devices:
            torch.cuda.synchronize(i)
