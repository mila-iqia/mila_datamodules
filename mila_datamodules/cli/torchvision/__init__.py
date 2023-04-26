from __future__ import annotations

from pathlib import Path
from typing import Callable

import torchvision.datasets as tvd
from typing_extensions import Concatenate

from mila_datamodules.cli.blocks import (
    CallDatasetConstructor,
    Compose,
    ExtractArchives,
    MakeSymlinksToDatasetFiles,
    MoveFiles,
    PrepareVisionDataset,
    StopOnSucess,
)
from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.torchvision.coco import (
    CocoDetectionArgs,
    PrepareCocoCaptions,
    PrepareCocoDetection,
)
from mila_datamodules.cli.types import P
from mila_datamodules.clusters.cluster import Cluster

from ..types import VD

# NOTE: For some datasets, we have datasets stored in folders with the same structure. This here is
# only really used to prevent repeating a bit of code in the dictionary below.
# TODO: Find an exception to this rule and design this dict with that in mind.
standardized_torchvision_datasets_dir = {
    Cluster.Mila: Path("/network/datasets"),
    Cluster.Beluga: Path("~/project/rpp-bengioy/data/curated").expanduser().resolve(),
}

# NOTE: Repeating the clusters for each dataset might look a bit tedious and repetitive, but I
# think that's fine for now.


def ComposeWithChecks(
    dataset_type: Callable[Concatenate[str, P], VD], *callables: PrepareVisionDataset[VD, P]
):
    """Adds checks before and after running `callables` when preparing `dataset_type`.

    - Adds a check before, to bypass `callables` if the dataset is already setup.
    - Adds a check after `callables` to check that the dataset is correctly setup.

    Parameters
    ----------
    dataset_type:
        A dataset type or callable.

    Returns
    -------
    A `Compose` with the added checks.
    """
    return Compose(
        # Try calling the dataset constructor. If there are no errors, skip the other steps.
        # If there is a RuntimeError, continue.
        StopOnSucess(
            CallDatasetConstructor(dataset_type, verify=True, get_index=0),
            continue_if_raised=RuntimeError,
        ),
        *callables,
        # Call the dataset constructor at the end, to make sure everything was setup correctly.
        # TODO: This `verify` argument is a bit confusing. Here we want to use `download=True` here
        # so that the md5 checksums of the archives are verified and so the files are extracted,
        # but we don't want to download anything from the internet!
        CallDatasetConstructor(dataset_type, verify=False, get_index=0),
    )


prepare_torchvision_datasets: dict[type, dict[Cluster, PrepareVisionDataset]] = {
    tvd.Caltech101: {
        cluster: ComposeWithChecks(
            tvd.Caltech101,
            MakeSymlinksToDatasetFiles(
                {
                    p: f"{datasets_dir}/caltech101/{p}"
                    for p in [
                        "101_ObjectCategories.tar.gz",
                        "Annotations.tar",
                    ]
                }
            ),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Caltech256: {
        cluster: ComposeWithChecks(
            tvd.Caltech256,
            MakeSymlinksToDatasetFiles(
                {p: f"{datasets_dir}/caltech256/{p}" for p in ["256_ObjectCategories.tar"]}
            ),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CelebA: {
        cluster: ComposeWithChecks(
            tvd.CelebA,
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/celeba"),
            # Torchvision will look into a celeba directory to preprocess
            # the dataset, so we move the files a new directory.
            MoveFiles(
                {
                    "Anno/*": "celeba/*",
                    "Eval/*": "celeba/*",
                    "Img/*": "celeba/*",
                }
            ),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CIFAR10: {
        cluster: ComposeWithChecks(
            tvd.CIFAR10,
            MakeSymlinksToDatasetFiles(
                {"cifar-10-python.tar.gz": (f"{datasets_dir}/cifar10/cifar-10-python.tar.gz")}
            ),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CIFAR100: {
        cluster: ComposeWithChecks(
            tvd.CIFAR100,
            MakeSymlinksToDatasetFiles(
                {"cifar-100-python.tar.gz": (f"{datasets_dir}/cifar100/cifar-100-python.tar.gz")}
            ),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Cityscapes: {
        cluster: ComposeWithChecks(
            tvd.Cityscapes,
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/cityscapes"),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.FashionMNIST: {
        cluster: ComposeWithChecks(
            tvd.FashionMNIST,
            # Make symlinks + rename in one step:
            MakeSymlinksToDatasetFiles(
                {
                    f"FashionMNIST/raw/{filename}": f"{datasets_dir}/fashionmnist/{filename}"
                    for filename in [
                        "t10k-images-idx3-ubyte.gz",
                        "t10k-labels-idx1-ubyte.gz",
                        "train-images-idx3-ubyte.gz",
                        "train-labels-idx1-ubyte.gz",
                    ]
                }
            ),
        )
        # )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.INaturalist: {
        cluster: ComposeWithChecks(
            tvd.INaturalist,
            # Make symlinks + rename in one step:
            MakeSymlinksToDatasetFiles(
                {
                    tv_expects_name: f"{datasets_dir}/inat/{our_archive_name}"
                    for tv_expects_name, our_archive_name in {
                        "2021_train.tgz": "train.tar.gz",
                        "2021_train_mini.tgz": "train_mini.tar.gz",
                        "2021_valid.tgz": "val.tar.gz",
                    }.items()
                }
            ),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.ImageNet: {
        # TODO: Write a customized `PrepareVisionDataset` for ImageNet that uses Olexa's magic tar
        # command.
        cluster: ComposeWithChecks(
            tvd.ImageNet,
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/imagenet"),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.KMNIST: {
        cluster: ComposeWithChecks(
            tvd.KMNIST,
            MakeSymlinksToDatasetFiles(datasets_dir / "kmnist"),
            # Torchvision will look into a KMNIST/raw directory to
            # preprocess the dataset
            MoveFiles({"*": "KMNIST/raw/*"}),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.MNIST: {
        cluster: ComposeWithChecks(
            tvd.MNIST,
            MakeSymlinksToDatasetFiles(
                {
                    f"raw/{filename}": f"{datasets_dir}/mnist/{filename}"
                    for filename in [
                        "t10k-images-idx3-ubyte.gz",
                        "t10k-labels-idx1-ubyte.gz",
                        "train-images-idx3-ubyte.gz",
                        "train-labels-idx1-ubyte.gz",
                    ]
                }
            ),
            # Torchvision will look into a raw directory to preprocess the
            # dataset
            # MoveFiles({"*": "raw/*"}),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Places365: {
        cluster: ComposeWithChecks(
            tvd.Places365,
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/places365"),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/places365.var/places365_challenge"),
            MoveFiles({"256/*.tar": "./*", "large/*.tar": "./*"}),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.QMNIST: {
        cluster: ComposeWithChecks(
            tvd.QMNIST,
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/qmnist"),
            # Torchvision will look into a QMNIST/raw directory to
            # preprocess the dataset
            MoveFiles({"*": "QMNIST/raw/*"}),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.STL10: {
        cluster: ComposeWithChecks(
            tvd.STL10,
            MakeSymlinksToDatasetFiles(
                {p: f"{datasets_folder}/stl10/{p}" for p in ["stl10_binary.tar.gz"]}
            ),
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dir.items()
    },
    tvd.SVHN: {
        cluster: ComposeWithChecks(
            tvd.SVHN,
            MakeSymlinksToDatasetFiles(
                {
                    p: f"{datasets_dir}/svhn/{p}"
                    for p in [
                        "extra_32x32.mat",
                        "extra.tar.gz",
                        "test_32x32.mat",
                        "test.tar.gz",
                        "train_32x32.mat",
                        "train.tar.gz",
                    ]
                },
            ),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    # TODO: This dataset requires some constructor arguments: `annotation_path`, `frames_per_clip`.
    tvd.UCF101: {
        cluster: ComposeWithChecks(
            tvd.UCF101,
            MakeSymlinksToDatasetFiles(
                {
                    p: f"{datasets_dir}/ucf101/{p}"
                    for p in [
                        "UCF101_STIP_Part1.rar"
                        "UCF101TrainTestSplits-DetectionTask.zip"
                        "UCF101.rar"
                        "UCF101_STIP_Part2.rar"
                        "UCF101TrainTestSplits-RecognitionTask.zip"
                    ]
                }
            ),
            ExtractArchives(
                {
                    "UCF101.rar": ".",
                    "UCF101TrainTestSplits-RecognitionTask.zip": ".",
                }
            ),
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
