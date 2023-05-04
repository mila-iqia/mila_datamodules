from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import torchvision.datasets as tvd

from mila_datamodules.cli.blocks import (
    CallDatasetConstructor,
    Compose,
    CopyFiles,
    ExtractArchives,
    MakeSymlinksToDatasetFiles,
    MoveFiles,
    PrepareDatasetFn,
    StopOnSuccess,
)
from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.torchvision.coco import (
    CocoDetectionArgs,
    PrepareCocoCaptions,
    PrepareCocoDetection,
)
from mila_datamodules.cli.torchvision.places365 import Places365Args, prepare_places365
from mila_datamodules.cli.types import VD
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_slurm_tmpdir

# NOTE: For some datasets, we have datasets stored in folders with the same structure. This here is
# only really used to prevent repeating a bit of code in the dictionary below.
# TODO: Find an exception to this rule and design this dict with that in mind.
standardized_torchvision_datasets_dir = {
    Cluster.Mila: Path("/network/datasets"),
    Cluster.Beluga: Path("~/project/rpp-bengioy/data/curated").expanduser().resolve(),
}

# NOTE: Repeating the clusters for each dataset might look a bit tedious and repetitive, but I
# think that's fine for now.

prepare_torchvision_datasets: dict[type, dict[Cluster, PrepareDatasetFn]] = {
    tvd.Caltech101: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.Caltech101)),
            MakeSymlinksToDatasetFiles(
                {
                    f"caltech101/{p}": f"{datasets_dir}/caltech101/{p}"
                    for p in [
                        "101_ObjectCategories.tar.gz",
                        "Annotations.tar",
                    ]
                }
            ),
            CallDatasetConstructor(tvd.Caltech101, extract_and_verify_archives=True),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Caltech256: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.Caltech256)),
            MakeSymlinksToDatasetFiles(
                {
                    f"caltech256/{p}": f"{datasets_dir}/caltech256/{p}"
                    for p in ["256_ObjectCategories.tar"]
                }
            ),
            CallDatasetConstructor(tvd.Caltech256, extract_and_verify_archives=True),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CelebA: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.CelebA)),
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
            CallDatasetConstructor(tvd.CelebA, extract_and_verify_archives=True),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CIFAR10: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.CIFAR10)),
            MakeSymlinksToDatasetFiles(
                {"cifar-10-python.tar.gz": (f"{datasets_dir}/cifar10/cifar-10-python.tar.gz")}
            ),
            CallDatasetConstructor(tvd.CIFAR10, extract_and_verify_archives=True),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CIFAR100: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.CIFAR100)),
            MakeSymlinksToDatasetFiles(
                {"cifar-100-python.tar.gz": (f"{datasets_dir}/cifar100/cifar-100-python.tar.gz")}
            ),
            CallDatasetConstructor(tvd.CIFAR100, extract_and_verify_archives=True),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Cityscapes: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.Cityscapes)),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/cityscapes"),
            CallDatasetConstructor(tvd.Cityscapes),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.FashionMNIST: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.FashionMNIST)),
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
            CallDatasetConstructor(tvd.FashionMNIST, extract_and_verify_archives=True),
        )
        # )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    # TODO: We could further speed things up by only preparing the version that is required by the
    # user (within '2017', '2018', '2019', '2021_train', '2021_train_mini', '2021_valid')
    tvd.INaturalist: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.INaturalist)),
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
            CallDatasetConstructor(tvd.INaturalist, extract_and_verify_archives=True),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.ImageNet: {
        # TODO: Write a customized `PrepareVisionDataset` for ImageNet that uses Olexa's magic tar
        # command.
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.ImageNet)),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/imagenet"),
            CallDatasetConstructor(tvd.ImageNet),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.KMNIST: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.KMNIST)),
            MakeSymlinksToDatasetFiles(datasets_dir / "kmnist"),
            # Torchvision will look into a KMNIST/raw directory to
            # preprocess the dataset
            MoveFiles({"*": "KMNIST/raw/*"}),
            CallDatasetConstructor(tvd.KMNIST, extract_and_verify_archives=True),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.MNIST: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.MNIST)),
            MakeSymlinksToDatasetFiles(
                {
                    f"MNIST/raw/{filename}": f"{datasets_dir}/mnist/{filename}"
                    for filename in [
                        "t10k-images-idx3-ubyte.gz",
                        "t10k-labels-idx1-ubyte.gz",
                        "train-images-idx3-ubyte.gz",
                        "train-labels-idx1-ubyte.gz",
                    ]
                }
            ),
            CallDatasetConstructor(tvd.MNIST, extract_and_verify_archives=True),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.QMNIST: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.QMNIST)),
            MakeSymlinksToDatasetFiles(
                {
                    f"QMNIST/raw/{p}": f"/network/datasets/qmnist/{p}"
                    for p in [
                        "qmnist-test-images-idx3-ubyte.gz",
                        "qmnist-test-labels-idx2-int.gz",
                        "qmnist-test-labels.tsv.gz",
                        "qmnist-train-images-idx3-ubyte.gz",
                        "qmnist-train-labels-idx2-int.gz",
                        "qmnist-train-labels.tsv.gz",
                        "xnist-images-idx3-ubyte.xz",
                        "xnist-labels-idx2-int.xz",
                    ]
                }
            ),
            CallDatasetConstructor(tvd.QMNIST, extract_and_verify_archives=True),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.STL10: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.STL10)),
            MakeSymlinksToDatasetFiles(
                {p: f"{datasets_folder}/stl10/{p}" for p in ["stl10_binary.tar.gz"]}
            ),
            CallDatasetConstructor(tvd.STL10, extract_and_verify_archives=True),
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dir.items()
    },
    tvd.SVHN: {
        cluster: Compose(
            StopOnSuccess(CallDatasetConstructor(tvd.SVHN)),
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
            CallDatasetConstructor(tvd.SVHN, extract_and_verify_archives=True),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    # NOTE: This dataset requires a `pip install av`, otherwise we get:
    # ImportError: PyAV is not installed, and is necessary for the video operations in torchvision.
    # See https://github.com/mikeboers/PyAV#installation for instructions on how to
    # install PyAV on your system.
    tvd.UCF101: {
        cluster: Compose(
            StopOnSuccess(
                lambda root, *args, **kwargs: CallDatasetConstructor(tvd.UCF101)(
                    str(Path(root) / "UCF-101"), *args, **kwargs
                ),
                # CallDatasetConstructor(tvd.UCF101),
                continue_if_raised=(FileNotFoundError,),
            ),
            MakeSymlinksToDatasetFiles(
                {
                    p: f"{datasets_dir}/ucf101/{p}"
                    for p in [
                        "UCF101_STIP_Part1.rar",
                        "UCF101TrainTestSplits-DetectionTask.zip",
                        "UCF101.rar",
                        "UCF101_STIP_Part2.rar",
                        "UCF101TrainTestSplits-RecognitionTask.zip",
                    ]
                }
            ),
            CopyFiles({"UCF-101": f"{datasets_dir}/ucf101.var/ucf101_torchvision/UCF-101"}),
            # TODO: Need to move all the folders from the UCF-101 folder to the root.
            # MoveFiles({"UCF-101/*": "."}),
            ExtractArchives(
                {
                    # "UCF101.rar": ".",  # todo: support .rar files if possible
                    "UCF101TrainTestSplits-RecognitionTask.zip": ".",
                }
            ),
            # TODO: Weird, but we need to add this "UCF-101" folder to the "root" argument for it
            # to work.
            # lambda root, *args, **kwargs: str(Path(root) / "UCF-101"),
            lambda root, *args, **kwargs: CallDatasetConstructor(tvd.UCF101)(
                str(Path(root) / "UCF-101"), *args, **kwargs
            ),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
}
"""Dataset preparation functions per dataset type, per cluster."""


# Coco is special enough to warrant its own module. We add its entries here.

# TODO: Support more splits and variants of Coco
# FIXME: Fix this strangeness. We need to be able to just go
# `mila_datamodules prepare cocodetection --variant=stuff --split=val` and it should just work.
prepare_torchvision_datasets[tvd.CocoDetection] = {
    cluster: PrepareCocoDetection(cluster.torchvision_datasets_dir, variant="stuff", split="train")
    for cluster in [Cluster.Mila, Cluster.Beluga]
}

prepare_torchvision_datasets[tvd.CocoCaptions] = {
    cluster: PrepareCocoCaptions(
        cluster.torchvision_datasets_dir, variant="captions", split="train"
    )
    for cluster in [Cluster.Mila, Cluster.Beluga]
}


@dataclass
class VisionDatasetArgs(DatasetArguments[VD]):
    root: Path = field(default_factory=lambda: get_slurm_tmpdir() / "datasets")


command_line_args_for_dataset: dict[
    type[tvd.VisionDataset], DatasetArguments | type[DatasetArguments]
] = {
    tvd.CocoDetection: CocoDetectionArgs(variant="stuff"),
    tvd.CocoCaptions: CocoDetectionArgs(variant="captions"),
}


prepare_torchvision_datasets[tvd.Places365] = {
    cluster: prepare_places365(cluster.torchvision_datasets_dir) for cluster in [Cluster.Mila]
}
command_line_args_for_dataset[tvd.Places365] = Places365Args


@dataclass
class UCF101Args(DatasetArguments[tvd.UCF101]):
    frames_per_clip: int  # number of frames in a clip.

    root: Path = field(default_factory=lambda: get_slurm_tmpdir() / "datasets")

    annotation_path: str = "ucfTrainTestlist"
    """path to the folder containing the split files; see docstring above for download instructions
    of these files."""

    def to_dataset_kwargs(self) -> dict:
        dataset_kwargs = dataclasses.asdict(self)
        dataset_kwargs["annotation_path"] = str(self.root / dataset_kwargs["annotation_path"])
        return dataset_kwargs


command_line_args_for_dataset[tvd.UCF101] = UCF101Args
command_line_args_for_dataset[tvd.UCF101] = UCF101Args
