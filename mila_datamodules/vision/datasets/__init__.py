import torchvision.datasets

from .utils import adapt_dataset

MNIST = adapt_dataset(torchvision.datasets.MNIST)
CIFAR10 = adapt_dataset(torchvision.datasets.CIFAR10)
CIFAR100 = adapt_dataset(torchvision.datasets.CIFAR100)
