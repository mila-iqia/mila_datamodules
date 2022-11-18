"""Wrappers around the torchvision datasets, adapted to the current cluster.

These wrappers mostly just use a good default value for the 'root' argument based
on the current cluster.

They might also silently first copy the dataset to SLURM_TMPDIR before reading it, depending on the
size of the dataset.

NOTE: These wrappers don't do anything when this is ran outside a SLURM cluster.
"""
from __future__ import annotations

import pl_bolts.datasets
import torchvision.datasets

from .adapted_datasets import adapt_dataset
from .bch_coco import BchCocoCaptions, BchCocoDetection
from .bch_imagenet import BchImageNet, BchUnlabeledImagenet
from .benzina_coco import BenzinaCocoCaptions, BenzinaCocoDetection
from .benzina_imagenet import BenzinaImageNet
from .binary_mnist import BinaryEMNIST, BinaryMNIST
from .caltech101 import Caltech101

MNIST = adapt_dataset(torchvision.datasets.MNIST)
CIFAR10 = adapt_dataset(torchvision.datasets.CIFAR10)
CIFAR100 = adapt_dataset(torchvision.datasets.CIFAR100)
FashionMNIST = adapt_dataset(torchvision.datasets.FashionMNIST)

Caltech101 = adapt_dataset(Caltech101)
# Caltech101 = adapt_dataset(torchvision.datasets.Caltech101)

Caltech256 = adapt_dataset(torchvision.datasets.Caltech256)
CelebA = adapt_dataset(torchvision.datasets.CelebA)
Cityscapes = adapt_dataset(torchvision.datasets.Cityscapes)
INaturalist = adapt_dataset(torchvision.datasets.INaturalist)
Places365 = adapt_dataset(torchvision.datasets.Places365)
STL10 = adapt_dataset(torchvision.datasets.STL10)
SVHN = adapt_dataset(torchvision.datasets.SVHN)
# TODO: For some of these datasets that have other required arguments, perhaps we could add "good"
# default values for some of them?
# e.g. EMNIST: split="mnist"
#     CocoCaptions: annFile=/wherever_its_stored/annotations/captions_train2017.json
CocoDetection = adapt_dataset(torchvision.datasets.CocoDetection)
CocoCaptions = adapt_dataset(torchvision.datasets.CocoCaptions)

EMNIST = adapt_dataset(torchvision.datasets.EMNIST)

BinaryMNIST = adapt_dataset(BinaryMNIST)
BinaryEMNIST = adapt_dataset(BinaryEMNIST)
# BinaryMNIST = adapt_dataset(pl_bolts.datasets.BinaryMNIST)
# BinaryEMNIST = adapt_dataset(pl_bolts.datasets.BinaryEMNIST)

BchCocoCaptions
BchCocoDetection
BchImageNet
BchUnlabeledImagenet
BenzinaCocoCaptions
BenzinaCocoDetection
BenzinaImageNet

# todo: Add the other datasets here.
