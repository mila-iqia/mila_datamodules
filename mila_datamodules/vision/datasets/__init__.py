"""Wrappers around the torchvision datasets, adapted to the current cluster.

These wrappers mostly just use a good default value for the 'root' argument based
on the current cluster.

They might also silently first copy the dataset to SLURM_TMPDIR before reading it, depending on the
size of the dataset.

NOTE: These wrappers don't do anything when outside a SLURM cluster.

TODO: For some of these datasets that have other required arguments, perhaps we could add "good"
default values for some of them? (Currently doing that in the VisionDataModule associated with
this dataset).
"""
from __future__ import annotations

import torchvision.datasets

from ._binary_mnist import _PatchedBinaryEMNIST, _PatchedBinaryMNIST
from ._mnist import MNIST as _PatchedMnist
from .adapted_datasets import AdaptedDataset
from .prepare_dataset import (
    make_symlinks_to_archives_in_tempdir,
    read_from_datasets_directory,
)
from .prepare_imagenet import prepare_imagenet_dataset


class ReadFromDatasetsDirectory(AdaptedDataset):
    prepare_dataset = read_from_datasets_directory


class LoadFromLinkedArchivesInSlurmTmpdir(AdaptedDataset):
    prepare_dataset = make_symlinks_to_archives_in_tempdir


class ImageNet(AdaptedDataset, torchvision.datasets.ImageNet):
    prepare_dataset = prepare_imagenet_dataset


class MNIST(ReadFromDatasetsDirectory, _PatchedMnist):
    pass


class FashionMNIST(ReadFromDatasetsDirectory, torchvision.datasets.FashionMNIST):
    pass


class CIFAR10(ReadFromDatasetsDirectory, torchvision.datasets.CIFAR10):
    pass


class CIFAR100(ReadFromDatasetsDirectory, torchvision.datasets.CIFAR100):
    pass


class Caltech101(LoadFromLinkedArchivesInSlurmTmpdir, torchvision.datasets.Caltech101):
    pass


class Caltech256(LoadFromLinkedArchivesInSlurmTmpdir, torchvision.datasets.Caltech256):
    pass


class CelebA(LoadFromLinkedArchivesInSlurmTmpdir, torchvision.datasets.CelebA):
    pass


class Cityscapes(LoadFromLinkedArchivesInSlurmTmpdir, torchvision.datasets.Cityscapes):
    pass


class INaturalist(LoadFromLinkedArchivesInSlurmTmpdir, torchvision.datasets.INaturalist):
    pass


class Places365(LoadFromLinkedArchivesInSlurmTmpdir, torchvision.datasets.Places365):
    pass


class STL10(AdaptedDataset, torchvision.datasets.STL10):
    # TODO
    pass


class SVHN(AdaptedDataset, torchvision.datasets.SVHN):
    # TODO
    pass


class EMNIST(ReadFromDatasetsDirectory, torchvision.datasets.EMNIST):
    pass


class CocoDetection(LoadFromLinkedArchivesInSlurmTmpdir, torchvision.datasets.CocoDetection):
    pass


class CocoCaptions(LoadFromLinkedArchivesInSlurmTmpdir, torchvision.datasets.CocoCaptions):
    pass


class BinaryMNIST(ReadFromDatasetsDirectory, _PatchedBinaryMNIST):
    pass


class BinaryEMNIST(ReadFromDatasetsDirectory, _PatchedBinaryEMNIST):
    pass


# todo: Add the other datasets here.
