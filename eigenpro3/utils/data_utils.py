import torch, os
from torchvision.datasets import MNIST, EMNIST, FashionMNIST, KMNIST, CIFAR10
from torch.nn.functional import one_hot
from .printing import midrule

from torch.utils.data import Dataset
from os.path import join as pjoin
import numpy as np

def unit_range_normalize(samples):
    samples -= samples.min(dim=0, keepdim=True).values
    return samples / samples.max(dim=1, keepdim=True).values


def load_cifar10_data(**kwargs):
    train_data = CIFAR10(os.environ['DATA_DIR'], train=True,download=True)
    test_data = CIFAR10(os.environ['DATA_DIR'], train=False,download=True)
    n_class = len(train_data.classes)
    return (
        n_class,
        (torch.from_numpy(train_data.data), torch.LongTensor(train_data.targets)),
        (torch.from_numpy(test_data.data), torch.LongTensor(test_data.targets)),
    )


def load_mnist_data(**kwargs):
    train_data = MNIST(os.environ['DATA_DIR'], train=True)
    test_data = MNIST(os.environ['DATA_DIR'], train=False)
    n_class = len(train_data.classes)
    return (
        n_class,
        (train_data.data, train_data.targets),
        (test_data.data, test_data.targets),
    )


def load_emnist_data(**kwargs):
    train_data = EMNIST(os.environ['DATA_DIR'], train=True, **kwargs)
    test_data = EMNIST(os.environ['DATA_DIR'], train=False, **kwargs)
    n_class = len(train_data.classes)
    return (
        n_class,
        (train_data.data, train_data.targets),
        (test_data.data, test_data.targets),
    )


def load_fmnist_data(**kwargs):
    train_data = FashionMNIST(os.environ['DATA_DIR'], train=True)
    test_data = FashionMNIST(os.environ['DATA_DIR'], train=False)
    n_class = len(train_data.classes)
    return (
        n_class,
        (train_data.data, train_data.targets),
        (test_data.data, test_data.targets),
    )


def load_kmnist_data(**kwargs):
    train_data = KMNIST(os.environ['DATA_DIR'], train=True)
    test_data = KMNIST(os.environ['DATA_DIR'], train=False)
    n_class = len(train_data.classes)
    return (
        n_class,
        (train_data.data, train_data.targets),
        (test_data.data, test_data.targets),
    )


def load_dataset(dataset='mnist', DEVICE=torch.device('cpu'), **kwargs):
    n_class, (x_train, y_train), (x_test, y_test) = eval(f'load_{dataset}_data')(**kwargs)

    x_train = x_train.reshape(x_train.shape[0], -1).to(DEVICE).float()
    x_test = x_test.reshape(x_test.shape[0], -1).to(DEVICE).float()
    
    x_train = unit_range_normalize(x_train)
    x_test = unit_range_normalize(x_test)
    
    y_train = one_hot(y_train, n_class).to(DEVICE).float()
    y_test = one_hot(y_test, n_class).to(DEVICE).float()
    
    print(f"Loaded {dataset.upper()} dataset to {DEVICE}")
    print(f"{n_class} classes")
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(midrule)

    return n_class, (x_train, y_train), (x_test, y_test)


class Cifar5mmobilenetDataset(Dataset):

    def __init__(self,
                 DATADIR='/expanse/lustre/projects/csd716/amirhesam/data/cifar5m_mobilenet',
                 parts=4,
                 device=torch.device('cpu'), subsample=100_000,
                 n_test=100000, num_knots=5_000,
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        print('Loading cifar5mmobilenet train set...')
        for ind in range(parts + 1):
            print(f'part={ind}')
            # z = np.load(pjoin(DATADIR, f'part{i}.npz'))
            self.X_train.append(
                torch.load(pjoin(DATADIR, f'ciar5m_mobilenetv2_100_feature_train_{ind}.pt'), torch.device('cpu')))
            self.y_train.append(
                torch.load(pjoin(DATADIR, f'ciar5m_mobilenetv2_100_y_train_{ind}.pt'), torch.device('cpu')))
            # print(f'Loaded part {i + 1}/6')
        print("Loading cifar5mmobilenet test set...")
        # z = np.load(pjoin(DATADIR, 'part5.npz'))
        self.X_test.append(
            torch.load(pjoin(DATADIR, f'ciar5m_mobilenetv2_100_feature_test.pt')))
        self.y_test.append(torch.load(pjoin(DATADIR, f'ciar5m_mobilenetv2_100_y_test.pt')))

        self.X_train = torch.cat(self.X_train)
        self.y_train = torch.cat(self.y_train).long()
        self.X_test = torch.cat(self.X_test)
        self.y_test = torch.cat(self.y_test).long()

        test_ind = np.random.choice(self.X_test.shape[0], size=n_test, replace=False)
        self.X_test = self.X_test[test_ind]
        self.y_test = self.y_test[test_ind]

        randomind_knots = np.random.choice(
            range(self.y_train.shape[0]), size=num_knots, replace=False)
        self.knots_x = self.X_train[randomind_knots]
        self.knots_y = self.y_train[randomind_knots]


        diff_set = set(range(self.y_train.shape[0])) - set(randomind_knots)
        diff_set = np.array(list(diff_set))
        randomind = np.random.choice(
            diff_set, size=subsample, replace=False)
        self.X_train = self.X_train[randomind]
        self.y_train = self.y_train[randomind]

