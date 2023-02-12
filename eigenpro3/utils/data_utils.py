import torch, os
from torchvision.datasets import MNIST, EMNIST, FashionMNIST, KMNIST, CIFAR10
from torch.nn.functional import one_hot
from .printing import midrule

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
