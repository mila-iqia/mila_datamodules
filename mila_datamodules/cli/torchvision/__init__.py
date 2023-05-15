from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torchvision.datasets as tvd
from typing_extensions import Concatenate, Literal

from mila_datamodules.blocks import (
    AddToPreparedDatasetsFile,
    CallDatasetFn,
    Compose,
    CopyFiles,
    ExtractArchives,
    MakePreparedDatasetUsableByOthersOnSameNode,
    MakeSymlinksToDatasetFiles,
    MoveFiles,
    PrepareDatasetFn,
    ReuseAlreadyPreparedDatasetOnSameNode,
    SkipRestIfThisWorks,
)
from mila_datamodules.blocks.reuse import (
    SkipIfAlreadyPrepared,
)
from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.torchvision.base import VisionDatasetArgs
from mila_datamodules.cli.torchvision.coco import (
    CocoCaptionArgs,
    CocoDetectionArgs,
    PrepareCocoCaptions,
    PrepareCocoDetection,
)
from mila_datamodules.cli.torchvision.places365 import Places365Args, prepare_places365
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_slurm_tmpdir
from mila_datamodules.types import D, P


def skip_if_already_prepared(
    step: PrepareDatasetFn[D, P],
    dataset_fn: Callable[Concatenate[str, P], D] | None = None,
) -> Compose[D, P]:
    dataset_fn = dataset_fn or step.dataset_fn
    if not dataset_fn:
        raise RuntimeError("Need to pass a dataset_fn or a step with a dataset_fn attribute.")
    return Compose[D, P](
        SkipIfAlreadyPrepared(dataset_fn),
        step,
        AddToPreparedDatasetsFile(dataset_fn),
    )


def reuse_across_nodes(
    step: PrepareDatasetFn[D, P],
    prepared_files_or_dirs: list[str],
    extra_files_depending_on_kwargs: dict[str, dict[Any, str | list[str]]] | None = None,
    dataset_fn: Callable[Concatenate[str, P], D] | None = None,
) -> Compose[D, P]:
    dataset_fn = dataset_fn or step.dataset_fn
    if not dataset_fn:
        raise RuntimeError("Need to pass a dataset_fn or a step with a dataset_fn attribute.")
    return Compose(
        # TODO: This looks in other SLURM_TMPDIRs after checking if the dataset was listed in the
        # prepared datasets file, but before checking if it can be loaded from the
        # current job's SLURM_TMPDIR with the dataset function. Can this cause any issues? In what
        # cases could the user have the dataset already prepared, but the entry isn't in the
        # prepared datasets file? (perhaps this saves us from the case where the preparation is
        # interrupted, since perhaps ImageFolder datasets could load correctly, even though the
        # dataset isn't 100% preprocessed correctly?)
        SkipRestIfThisWorks(
            ReuseAlreadyPreparedDatasetOnSameNode(
                dataset_fn,
                prepared_files_or_dirs=prepared_files_or_dirs,
                extra_files_depending_on_kwargs=extra_files_depending_on_kwargs,
            )
        ),
        step,
        MakePreparedDatasetUsableByOthersOnSameNode(
            dataset_fn,
            prepared_files_or_dirs=prepared_files_or_dirs,
            extra_files_depending_on_kwargs=extra_files_depending_on_kwargs,
        ),
    )


def prepare_vision_dataset(
    preparation_step: PrepareDatasetFn[D, P],
    prepared_files_or_dirs: list[str],
    extra_files_depending_on_kwargs: dict[str, dict[Any, str | list[str]]] | None = None,
    dataset_fn: Callable[Concatenate[str, P], D] | None = None,
) -> Compose[D, P]:
    dataset_fn = dataset_fn or preparation_step.dataset_fn
    if not dataset_fn:
        raise RuntimeError("Need to pass a dataset_type or a step with a dataset_type attribute.")
    return skip_if_already_prepared(
        reuse_across_nodes(
            preparation_step,
            prepared_files_or_dirs=prepared_files_or_dirs,
            extra_files_depending_on_kwargs=extra_files_depending_on_kwargs,
            dataset_fn=dataset_fn,
        ),
        dataset_fn=dataset_fn,
    )


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
        cluster: prepare_vision_dataset(
            Compose(
                # Try reading from SLURM_TMPDIR, skip the rest if that works.
                SkipRestIfThisWorks(CallDatasetFn(tvd.Caltech101)),
                MakeSymlinksToDatasetFiles(
                    {
                        f"caltech101/{p}": f"{datasets_dir}/caltech101/{p}"
                        for p in [
                            "101_ObjectCategories.tar.gz",
                            "Annotations.tar",
                        ]
                    }
                ),
                CallDatasetFn(tvd.Caltech101, extract_and_verify_archives=True),
            ),
            prepared_files_or_dirs=["caltech101"],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Caltech256: {
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.Caltech256)),
                MakeSymlinksToDatasetFiles(
                    {
                        f"caltech256/{p}": f"{datasets_dir}/caltech256/{p}"
                        for p in ["256_ObjectCategories.tar"]
                    }
                ),
                CallDatasetFn(tvd.Caltech256, extract_and_verify_archives=True),
            ),
            prepared_files_or_dirs=["caltech256"],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CelebA: {
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.CelebA)),
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
                CallDatasetFn(tvd.CelebA, extract_and_verify_archives=True),
            ),
            prepared_files_or_dirs=["celeba"],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CIFAR10: {
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.CIFAR10)),
                MakeSymlinksToDatasetFiles(
                    {"cifar-10-python.tar.gz": (f"{datasets_dir}/cifar10/cifar-10-python.tar.gz")}
                ),
                CallDatasetFn(tvd.CIFAR10, extract_and_verify_archives=True),
            ),
            prepared_files_or_dirs=["cifar-10-batches-py", "cifar-10-python.tar.gz"],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CIFAR100: {
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.CIFAR100)),
                MakeSymlinksToDatasetFiles(
                    {
                        "cifar-100-python.tar.gz": (
                            f"{datasets_dir}/cifar100/cifar-100-python.tar.gz"
                        )
                    }
                ),
                CallDatasetFn(tvd.CIFAR100, extract_and_verify_archives=True),
            ),
            prepared_files_or_dirs=["cifar-100-batches-py"],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Cityscapes: {
        cluster: prepare_vision_dataset(
            Compose(
                # SkipRestIfThisWorks(CallDatasetFn(tvd.Cityscapes)),
                # TODO: List the files to share for Cityscapes.
                # TODO: Also possibly add a subdirectory for it, because it creates lots of folders
                # SkipRestIfThisWorks(ReuseAlreadyPreparedDatasetOnSameNode(tvd.Cityscapes, ...)),
                MakeSymlinksToDatasetFiles(
                    {
                        file: f"{datasets_dir}/cityscapes/{file}"
                        for file in [
                            "camera_trainextra.zip",
                            "gtBbox_cityPersons_trainval.zip",
                            "leftImg8bit_blurred.zip",
                            "leftImg8bit_trainvaltest.zip",
                            "camera_trainvaltest.zip",
                            "gtCoarse.zip",
                            "leftImg8bit_demoVideo.zip",
                            "vehicle_trainextra.zip",
                            "extra/leftImg8bit_sequence_trainvaltest.zip",
                            "gtFine_trainvaltest.zip",
                            "leftImg8bit_trainextra.zip",
                            "vehicle_trainvaltest.zip",
                        ]
                    }
                ),
                CallDatasetFn(tvd.Cityscapes),
            ),
            prepared_files_or_dirs=[
                "camera_trainextra.zip",
                "gtBbox_cityPersons_trainval.zip",
                "leftImg8bit_blurred.zip",
                "leftImg8bit_trainvaltest.zip",
                "camera_trainvaltest.zip",
                "gtCoarse.zip",
                "leftImg8bit_demoVideo.zip",
                "vehicle_trainextra.zip",
                "extra/leftImg8bit_sequence_trainvaltest.zip",
                "gtFine_trainvaltest.zip",
                "leftImg8bit_trainextra.zip",
                "vehicle_trainvaltest.zip",
            ],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.FashionMNIST: {
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.FashionMNIST)),
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
                CallDatasetFn(tvd.FashionMNIST, extract_and_verify_archives=True),
            ),
            prepared_files_or_dirs=["FashionMNIST"],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    # TODO: We could further speed things up by only preparing the version that is required by the
    # user (within '2017', '2018', '2019', '2021_train', '2021_train_mini', '2021_valid')
    tvd.INaturalist: {
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.INaturalist)),
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
                CallDatasetFn(tvd.INaturalist, extract_and_verify_archives=True),
            ),
            prepared_files_or_dirs=[
                "2021_train_mini.tgz",
                "2021_train.tgz",
                "2021_valid.tgz",
            ],
            # TODO: Figure out the additional files based on the version.
            extra_files_depending_on_kwargs={
                "version": {
                    "2021_train": "2021_train",
                    "2021_train_mini": "2021_train_mini",
                    "2021_valid": "2021_valid",
                    None: "2021_train",
                }
            },
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.ImageNet: {
        # TODO: Write a customized function for ImageNet that uses Olexa's magic tar
        # commands (preferably in Python form).
        # """
        # mkdir -p $SLURM_TMPDIR/imagenet/val
        # cd       $SLURM_TMPDIR/imagenet/val
        # tar  -xf /network/scratch/b/bilaniuo/ILSVRC2012_img_val.tar
        # mkdir -p $SLURM_TMPDIR/imagenet/train
        # cd       $SLURM_TMPDIR/imagenet/train
        # tar  -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar \
        #      --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'
        # """
        cluster: prepare_vision_dataset(
            Compose(
                MakeSymlinksToDatasetFiles(
                    {
                        archive: f"{datasets_dir}/imagenet/{archive}"
                        for archive in [
                            "ILSVRC2012_devkit_t12.tar.gz",
                            "ILSVRC2012_img_train.tar",
                            "ILSVRC2012_img_val.tar",
                            "md5sums",
                        ]
                    }
                ),
                # Call the constructor to verify the checksums and extract the archives in `root`.
                CallDatasetFn(tvd.ImageNet),
            ),
            prepared_files_or_dirs=[
                "ILSVRC2012_devkit_t12.tar.gz",
                "ILSVRC2012_img_train.tar",
                "ILSVRC2012_img_val.tar",
                "md5sums",
                "meta.bin",
            ],
            extra_files_depending_on_kwargs={
                "split": {"train": "train", "val": "val", None: "train"}
            },
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.KMNIST: {
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.KMNIST)),
                MakeSymlinksToDatasetFiles(
                    {
                        f"KMNIST/raw/{file}": f"{datasets_dir}/kmnist/{file}"
                        for file in [
                            "k49_classmap.csv",
                            "k49-train-imgs.npz",
                            "kmnist-test-imgs.npz",
                            "kmnist-train-labels.npz",
                            "train-images-idx3-ubyte.gz",
                            "k49-test-imgs.npz",
                            "k49-train-labels.npz",
                            "kmnist-test-labels.npz",
                            "t10k-images-idx3-ubyte.gz",
                            "train-labels-idx1-ubyte.gz",
                            "k49-test-labels.npz",
                            "kmnist_classmap.csv",
                            "kmnist-train-imgs.npz",
                            "t10k-labels-idx1-ubyte.gz",
                        ]
                    }
                ),
                CallDatasetFn(tvd.KMNIST, extract_and_verify_archives=True),
            ),
            prepared_files_or_dirs=["KMNIST/raw"],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.MNIST: {
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.MNIST)),
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
                CallDatasetFn(tvd.MNIST, extract_and_verify_archives=True),
            ),
            prepared_files_or_dirs=[
                f"MNIST/raw/{filename}"
                for filename in [
                    "t10k-images-idx3-ubyte.gz",
                    "t10k-labels-idx1-ubyte.gz",
                    "train-images-idx3-ubyte.gz",
                    "train-labels-idx1-ubyte.gz",
                ]
            ],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.QMNIST: {
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.QMNIST)),
                MakeSymlinksToDatasetFiles(
                    {
                        f"QMNIST/raw/{p}": f"{datasets_dir}/qmnist/{p}"
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
                CallDatasetFn(tvd.QMNIST, extract_and_verify_archives=True),
            ),
            # TODO: Also enumerate the files here so we can check more explicitly.
            prepared_files_or_dirs=["QMNIST/raw"],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.STL10: {
        # TODO: Add args for the split here?
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.STL10)),
                MakeSymlinksToDatasetFiles(
                    {p: f"{datasets_folder}/stl10/{p}" for p in ["stl10_binary.tar.gz"]}
                ),
                CallDatasetFn(tvd.STL10, extract_and_verify_archives=True),
            ),
            # TODO: Double-check this.
            prepared_files_or_dirs=["stl10_binary.tar.gz", "stl10_binary"],
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dir.items()
    },
    tvd.SVHN: {
        cluster: prepare_vision_dataset(
            Compose(
                SkipRestIfThisWorks(CallDatasetFn(tvd.SVHN)),
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
                CallDatasetFn(tvd.SVHN, extract_and_verify_archives=True),
            ),
            prepared_files_or_dirs=[
                "extra_32x32.mat",
                "extra.tar.gz",
                "test_32x32.mat",
                "test.tar.gz",
                "train_32x32.mat",
                "train.tar.gz",
            ],
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    # NOTE: This dataset requires a `pip install av`, otherwise we get:
    # ImportError: PyAV is not installed, and is necessary for the video operations in torchvision.
    # See https://github.com/mikeboers/PyAV#installation for instructions on how to
    # install PyAV on your system.
    tvd.UCF101: {
        cluster: Compose(
            SkipIfAlreadyPrepared(tvd.UCF101),
            Compose(
                # TODO: Weird, but we need to add this "UCF-101" folder to the "root" argument for
                # it to work...
                SkipRestIfThisWorks(
                    lambda root, *args, **kwargs: CallDatasetFn(tvd.UCF101)(
                        str(Path(root) / "UCF-101"), *args, **kwargs
                    ),
                    # CallDatasetConstructor(tvd.UCF101),
                    continue_if_raised=(FileNotFoundError,),
                ),
                # TODO: Debug this step here.
                SkipRestIfThisWorks(
                    ReuseAlreadyPreparedDatasetOnSameNode(
                        tvd.UCF101,
                        [
                            "UCF-101",
                            "UCF101_STIP_Part1.rar",
                            "UCF101TrainTestSplits-DetectionTask.zip",
                            "UCF101.rar",
                            "UCF101_STIP_Part2.rar",
                            "UCF101TrainTestSplits-RecognitionTask.zip",
                        ],
                    ),
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
                # TODO: Weird, but we need to add this "UCF-101" folder to the "root" argument for
                # it to work.
                lambda root, *args, **kwargs: CallDatasetFn(tvd.UCF101)(
                    str(Path(root) / "UCF-101"), *args, **kwargs
                ),
            ),
            MakePreparedDatasetUsableByOthersOnSameNode(
                tvd.UCF101,
                [
                    "UCF-101",
                    "UCF101_STIP_Part1.rar",
                    "UCF101TrainTestSplits-DetectionTask.zip",
                    "UCF101.rar",
                    "UCF101_STIP_Part2.rar",
                    "UCF101TrainTestSplits-RecognitionTask.zip",
                ],
            ),
            AddToPreparedDatasetsFile(tvd.UCF101),
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


command_line_args_for_dataset: dict[
    type[tvd.VisionDataset], DatasetArguments | type[DatasetArguments]
] = {
    tvd.CocoDetection: CocoDetectionArgs,
    tvd.CocoCaptions: CocoCaptionArgs,
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


@dataclass
class INaturalistArgs(VisionDatasetArgs):
    version: Literal[
        "2017", "2018", "2019", "2021_train", "2021_train_mini", "2021_valid"
    ] = "2021_train"
    """Which version of the dataset to prepare.

    Note, only the 2021 versions appear to be supported on the Mila cluster atm.
    """


command_line_args_for_dataset[tvd.INaturalist] = INaturalistArgs


@dataclass
class ImageNetArgs(VisionDatasetArgs):
    """Prepare the ImageNet dataset."""

    split: Literal["train", "val"] = "train"
    """Which split to use."""


command_line_args_for_dataset[tvd.ImageNet] = ImageNetArgs
