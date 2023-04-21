from __future__ import annotations

from pathlib import Path

import torchvision.datasets as tvd

from mila_datamodules.cli.torchvision.blocks import (
    CallDatasetConstructor,
    Compose,
    ExtractArchives,
    MakeSymlinksToDatasetFiles,
    MoveFiles,
    PrepareVisionDataset,
    StopOnSucess,
)
from mila_datamodules.cli.torchvision.coco import (
    CocoDetectionArgs,
    PrepareCocoCaptions,
    PrepareCocoDetection,
)
from mila_datamodules.cli.torchvision.dataset_args import DatasetArguments
from mila_datamodules.clusters.cluster import Cluster

from ._types import VD

# NOTE: For some datasets, we have datasets stored in folders with the same structure. This here is
# only really used to prevent repeating a bit of code in the dictionary below.
# TODO: Find an exception to this rule and design this dict with that in mind.
standardized_torchvision_datasets_dir = {
    Cluster.Mila: Path("/network/datasets"),
    Cluster.Beluga: Path("~/project/rpp-bengioy/data/curated").expanduser().resolve(),
}

# NOTE: Repeating the clusters for each dataset might look a bit tedious and repetitive, but I
# think that's fine for now.


# TODO: Improve code reuse below a bit, since lots of datasets have the same operations.
def PrepareSimpleDataset(torchvision_datasets_dir: str, dataset_type: type[VD], dataset_name: str):
    return Compose(
        StopOnSucess(dataset_type, exceptions=[RuntimeError]),
        MakeSymlinksToDatasetFiles(f"{torchvision_datasets_dir}/{dataset_name}"),
        # Torchvision will look into a caltech101 directory to
        # preprocess the dataset
        MoveFiles({"*": f"{dataset_name}/*"}),
        CallDatasetConstructor(dataset_type),
    )


prepare_torchvision_datasets: dict[type, dict[Cluster, PrepareVisionDataset]] = {
    tvd.Caltech101: {
        cluster: Compose(
            StopOnSucess(tvd.Caltech101, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{Cluster.torchvision_datasets_dir}/caltech101"),
            # Torchvision will look into a caltech101 directory to
            # preprocess the dataset
            MoveFiles({"*": "caltech101/*"}),
            CallDatasetConstructor(tvd.Caltech101),
        )
        for cluster in [Cluster.Mila, Cluster.Beluga]
    },
    tvd.Caltech256: {
        cluster: Compose(
            StopOnSucess(tvd.Caltech256, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{Cluster.torchvision_datasets_dir}/caltech256"),
            # Torchvision will look into a caltech256 directory to
            # preprocess the dataset
            MoveFiles({"*": "caltech256/*"}),
            CallDatasetConstructor(tvd.Caltech256),
        )
        for cluster in [Cluster.Mila, Cluster.Beluga]
    },
    tvd.CelebA: {
        cluster: Compose(
            StopOnSucess(tvd.CelebA, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{cluster.torchvision_datasets_dir}/celeba"),
            # Torchvision will look into a celeba directory to preprocess
            # the dataset
            MoveFiles(
                {
                    "Anno/**/*": "celeba/*",
                    "Eval/**/*": "celeba/*",
                    "Img/**/*": "celeba/*",
                }
            ),
            CallDatasetConstructor(tvd.CelebA),
        )
        for cluster in [Cluster.Mila, Cluster.Beluga]
    },
    tvd.CIFAR10: {
        cluster: Compose(
            StopOnSucess(tvd.CIFAR10, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(
                {
                    "cifar-10-python.tar.gz": (
                        f"{cluster.torchvision_datasets_dir}/cifar10/cifar-10-python.tar.gz"
                    )
                }
            ),
            CallDatasetConstructor(tvd.CIFAR10),
        )
        for cluster in [Cluster.Mila, Cluster.Beluga]
    },
    tvd.CIFAR100: {
        cluster: Compose(
            StopOnSucess(tvd.CIFAR100, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(
                {
                    "cifar-100-python.tar.gz": (
                        f"{cluster.torchvision_datasets_dir}/cifar100/cifar-100-python.tar.gz"
                    )
                }
            ),
            CallDatasetConstructor(tvd.CIFAR100),
        )
        for cluster in [Cluster.Mila, Cluster.Beluga]
    },
    tvd.Cityscapes: {
        cluster: Compose(
            StopOnSucess(tvd.Cityscapes, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{cluster.torchvision_datasets_dir}/cityscapes"),
            CallDatasetConstructor(tvd.Cityscapes),
        )
        for cluster in [Cluster.Mila, Cluster.Beluga]
    },
    tvd.FashionMNIST: {
        cluster: Compose(
            StopOnSucess(tvd.FashionMNIST, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{cluster.torchvision_datasets_dir}/fashionmnist"),
            # Torchvision will look into a FashionMNIST/raw directory to
            # preprocess the dataset
            MoveFiles({"*": "FashionMNIST/raw/*"}),
            CallDatasetConstructor(tvd.FashionMNIST),
        )
        for cluster in [Cluster.Mila, Cluster.Beluga]
    },
    tvd.INaturalist: {
        cluster: Compose(
            StopOnSucess(tvd.INaturalist, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/inat"),
            # Torchvision will look for those files to preprocess the
            # dataset
            MoveFiles(
                {
                    "train.tar.gz": "2021_train.tgz",
                    "train_mini.tar.gz": "2021_train_mini.tgz",
                    "val.tar.gz": "2021_valid.tgz",
                }
            ),
            CallDatasetConstructor(tvd.INaturalist),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.ImageNet: {
        # TODO: Write a customized `PrepareVisionDataset` for ImageNet that uses Olexa's magic tar
        # command.
        cluster: Compose(
            StopOnSucess(tvd.ImageNet, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/imagenet"),
            CallDatasetConstructor(tvd.ImageNet),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.KMNIST: {
        cluster: Compose(
            StopOnSucess(tvd.KMNIST, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/kmnist"),
            # Torchvision will look into a KMNIST/raw directory to
            # preprocess the dataset
            MoveFiles({"*": "KMNIST/raw/*"}),
            CallDatasetConstructor(tvd.KMNIST),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.MNIST: {
        # On the Mila and Beluga cluster we have archives which are extracted
        # into 4 "raw" binary files. We do need to match the expected directory
        # structure of the torchvision MNIST dataset though.  NOTE: On Beluga,
        # we also have the MNIST 'raw' files in
        # /project/rpp-bengioy/data/MNIST/raw, no archives.
        cluster: Compose(
            StopOnSucess(tvd.MNIST, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/mnist"),
            # Torchvision will look into a raw directory to preprocess the
            # dataset
            MoveFiles({"*": "raw/*"}),
            CallDatasetConstructor(tvd.MNIST),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Places365: {
        cluster: Compose(
            StopOnSucess(tvd.Places365, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/places365"),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/places365.var/places365_challenge"),
            MoveFiles({"256/*.tar": "./*", "large/*.tar": "./*"}),
            CallDatasetConstructor(tvd.Places365),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.QMNIST: {
        cluster: Compose(
            StopOnSucess(tvd.QMNIST, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/qmnist"),
            # Torchvision will look into a QMNIST/raw directory to
            # preprocess the dataset
            MoveFiles({"*": "QMNIST/raw/*"}),
            CallDatasetConstructor(tvd.QMNIST),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.STL10: {
        cluster: Compose(
            StopOnSucess(tvd.STL10, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{datasets_folder}/stl10"),
            CallDatasetConstructor(tvd.STL10),
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dir.items()
    },
    tvd.SVHN: {
        cluster: Compose(
            StopOnSucess(tvd.SVHN, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/svhn"),
            CallDatasetConstructor(tvd.SVHN),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.UCF101: {
        cluster: Compose(
            StopOnSucess(tvd.UCF101, exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/ucf101"),
            ExtractArchives(
                {
                    "UCF101.rar": ".",
                    "UCF101TrainTestSplits-RecognitionTask.zip": ".",
                }
            ),
            CallDatasetConstructor(tvd.UCF101),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
}
"""Dataset preparation functions per dataset type, per cluster."""


# Coco is special enough to warrant its own module. We add its entries here.

prepare_torchvision_datasets[tvd.CocoDetection] = {
    cluster: PrepareCocoDetection(cluster.torchvision_datasets_dir)
    for cluster in [Cluster.Mila, Cluster.Beluga]
}

prepare_torchvision_datasets[tvd.CocoCaptions] = {
    cluster: PrepareCocoCaptions(cluster.torchvision_datasets_dir)
    for cluster in [Cluster.Mila, Cluster.Beluga]
}


command_line_args_for_dataset: dict[
    type[tvd.VisionDataset], DatasetArguments | type[DatasetArguments]
] = {
    tvd.CocoDetection: CocoDetectionArgs(variant="stuff"),
    tvd.CocoCaptions: CocoDetectionArgs(variant="captions"),
}
