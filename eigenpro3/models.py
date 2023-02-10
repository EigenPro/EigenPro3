import torch
from .utils import fmm, get_preconditioner, accuracy, divide_to_gpus
from .datasets import makedataloaders
from .projection import HilbertProjection
from torch.cuda.comm import broadcast
from concurrent.futures import ThreadPoolExecutor


class KernelModel():

    def __init__(self, y, centers, kernel_fn,X=None, devices =[torch.device('cpu')], make_dataloader=True,
                 nystrom_samples=None, n_nystrom_samples=5_000, data_preconditioner_level=500,multi_gpu=False):

        self.devices = tuple(devices)
        self.device_base = self.devices[0]

        self.n_classes = y.shape[-1]

        if make_dataloader:
        ###### Distribute all equally over all available GPUs #####
            self.train_loaders = makedataloaders(X,y,self.devices)

        self.centers = centers
        self.n_centers = len(centers)
        self.kernel = kernel_fn
        self.weights = torch.zeros(self.n_centers, y.shape[-1])

        self.multi_gpu = multi_gpu
        if multi_gpu:
            ######## dsitributing the weights over all avalibale GPUs
            self.weights_all = divide_to_gpus(self.weights,self.n_centers, self.devices)

            ######## dsitributing the centers over all avalibale GPUs
            self.centers_all = divide_to_gpus(self.centers,self.n_centers, self.devices)

        else:
            self.weights_all = [self.weights.to(self.device_base)]
            self.centers_all = [self.centers.to(self.device_base)]

        ###### Initilize inexact projection #########
        self.InexactProjector = HilbertProjection(
            self.kernel, self.centers, self.n_classes, 
            devices=self.devices, multi_gpu=self.multi_gpu)


        ###### DATA Preconditioner
        ##### note that batch size and learning rate will be determined in this stage #########
        if nystrom_samples==None:
            ####### randomly select nystrom samples from X
            nystrom_ids = torch.randperm(X.shape[0])[:n_nystrom_samples]
            self.nystrom_samples = X[nystrom_ids]
        else:
            self.nystrom_samples = nystrom_samples

        self.data_preconditioner_matrix,self.eigenvectors_data,self.batch_size,self.lr \
            = get_preconditioner( self.centers, self.nystrom_samples,self.kernel, data_preconditioner_level + 1)
        print("Data preconditioner is ready.")

        self.centers = self.centers.to(self.device_base)
        self.weights = self.weights.to(self.device_base)
        self.nystrom_samples = self.nystrom_samples.to(self.device_base)
        self.data_preconditioner_matrix = self.data_preconditioner_matrix.to(self.device_base)
        self.eigenvectors_data = self.eigenvectors_data.to(self.device_base)
        ###### DATA Preconditioner #########


    def fit(self, train_loaders, val_loader=None,epochs=10,score_fn=None):
        for t in range(epochs):
            self.epoch = t
            self.fit_epoch(train_loaders)
            if val_loader!=None and score_fn!=None:
                accu = score_fn(self.weights,self.centers,val_loader,self.kernel,self.device_base)
                print('-'*40)
                print(f'epoch {t+1:4d}        validation accuracy: {accu*100.:5.2f}%')
                print('='*40)



    def fit_epoch(self, train_loaders):
        batch_num = 0
        for trainloader_ind, train_loader in enumerate(train_loaders):
            permutation = torch.randperm(train_loader.X.size()[0], device=train_loader.X.device)
            for i in range(0, train_loader.X.size()[0], int(self.batch_size)):

                batch_ids = permutation[i: i + int(self.batch_size)]
                # self.corrected_gz_scaled = 0

                X_batch = train_loader.X[batch_ids]
                y_batch = train_loader.y[batch_ids]

                ###### fitting the batch
                self.fit_batch(X_batch, y_batch)

                if self.multi_gpu:
                    self.sync_gpus()

                torch.cuda.empty_cache()

                if (batch_num + 1)%2==0:
                    print(f'epoch {self.epoch: 4d}\t batch {batch_num+1 :4d}')

                batch_num += 1
                del batch_ids, X_batch, y_batch


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
        # self.corrected_gz_scaled += corrected_gz.to(self.device_base)
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

    
    def precondition_correction(self, X_batch, grad):
        Kmat_xs_xbatch = self.kernel(self.nystrom_samples, X_batch)
        preconditon_grad = self.data_preconditioner_matrix @ (self.eigenvectors_data.T @ (Kmat_xs_xbatch @ grad))
        del Kmat_xs_xbatch
        return preconditon_grad


    def update_weights(self):

        gz_projection = self.corrected_gz_scaled.to(self.device_base)
        self.theta2, _ = self.InexactProjector.fit_hilbert_projection(
            None,
            gz_projection, mem_gb=12,
            return_log=False
        )

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
