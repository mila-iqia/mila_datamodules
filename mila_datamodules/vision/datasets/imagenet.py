"""Adapter for the torchvision.datasets.ImageNet class, optimized for SLURM clusters."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from logging import getLogger as get_logger
from pathlib import Path
from typing import Literal, TypedDict

from torchvision.datasets import ImageNet
from typing_extensions import NotRequired

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_slurm_tmpdir

from .adapted_datasets import prepare_dataset

logger = get_logger(__name__)


class ImageNetFiles(TypedDict):
    train_archive: str
    """Path to the "ILSVRC2012_img_train.tar"-like archive containing the training set."""

    val_archive: NotRequired[str]
    """Path to the "ILSVRC2012_img_val.tar" archive containing the test set.

    One of `val_archive` or `val_folder` must be provided.
    """

    val_folder: NotRequired[str]
    """Path to the 'val' folder containing the test set.

    One of `val_archive` or `val_folder` must be provided.
    """

    devkit: str
    """Path to the ILSVRC2012_devkit_t12.tar.gz file."""


imagenet_file_locations: dict[Cluster, ImageNetFiles] = {
    Cluster.Mila: {
        "train_archive": "/network/datasets/imagenet/ILSVRC2012_img_train.tar",
        "val_folder": "/network/datasets/imagenet.var/imagenet_torchvision/val",
        "devkit": "/network/datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz",
    },
    # TODO: Need help with filling these:
    Cluster.Beluga: {
        "train_archive": "/project/rpp-bengioy/data/curated/imagenet/ILSVRC2012_img_train.tar",
        "val_archive": "/project/rpp-bengioy/data/curated/imagenet/ILSVRC2012_img_val.tar",
        "devkit": "/project/rpp-bengioy/data/curated/imagenet/ILSVRC2012_devkit_t12.tar.gz",
    },
    # ClusterType.CEDAR: {},
    # ClusterType.GRAHAM: {},
    # ...
}
"""A map that shows where to retrieve the imagenet files for each SLURM cluster."""


@prepare_dataset.register(ImageNet)
def prepare_imagenet_dataset(
    dataset: ImageNet,
    root: str | None = None,
    split: Literal["train", "val"] = "train",
    *args,
    **kwargs,
) -> str | None:
    """Optimized setup for ImageNet."""
    cluster = Cluster.current_or_error()

    fast_data_dir = get_slurm_tmpdir() / "data" / type(dataset).__name__
    fast_data_dir.parent.mkdir(exist_ok=True)
    fast_data_dir.mkdir(exist_ok=True)
    # TODO: Could also only do the split that is required here.
    copy_imagenet_to_dest(destination_dir=fast_data_dir)
    return str(fast_data_dir)


def copy_imagenet_to_dest(
    destination_dir: str | Path,
) -> None:
    """Copies/extracts the ImageNet dataset into the destination folder (ideally in `slurm_tmpdir`)

    NOTE: This is the transcribed wisdom of @obilaniu (Olexa Bilaniuk).
    See [this Slack thread](https://mila-umontreal.slack.com/archives/CFAS8455H/p1652168938773169?thread_ts=1652126891.083229&cid=CFAS8455H)
    for more info.
    """
    # TODO: Make this work for other clusters if possible.
    assert CURRENT_CLUSTER is not None, "Only works on a SLURM cluster!"
    paths = imagenet_file_locations[CURRENT_CLUSTER]
    train_archive = paths["train_archive"]
    devkit = paths["devkit"]
    if "val_folder" in paths:
        val_folder = paths["val_folder"]
        val_archive = None
    elif "val_archive" in paths:
        val_folder = None
        val_archive = paths["val_archive"]
    else:
        cluster = Cluster.current()
        raise RuntimeError(
            f"One of 'val_folder' or 'val_archive' must be set for cluster {cluster}!"
        )

    destination_dir = Path(destination_dir)
    raise_errors = False

    train_folder = destination_dir / "train"
    train_folder.mkdir(exist_ok=True, parents=True)

    val_done_file = destination_dir / "val_done.txt"
    if not val_done_file.exists():
        if val_folder is not None:
            print(f"Copying imagenet val dataset to {destination_dir}/val ...")
            subprocess.run(
                args=f"cp -r {val_folder} {destination_dir}",
                shell=True,
                check=raise_errors,
                stdout=sys.stdout,
            )
        else:
            assert val_archive is not None
            val_folder = destination_dir / "val"
            val_folder.mkdir(exist_ok=True, parents=True)
            raise NotImplementedError(
                f"Extract the validation set from {val_archive} to {destination_dir}/val"
            )
            # TODO: Probably something like this, but need to double-check.
            with temporarily_chdir(val_folder):
                subprocess.run(
                    args=(
                        f"tar -xf {val_archive} "
                        "--to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'"
                    ),
                    shell=True,
                    check=raise_errors,
                    stdout=sys.stdout,
                )
        val_done_file.touch()

    train_done_file = destination_dir / "train_done.txt"
    if not train_done_file.exists():
        print(f"Copying imagenet train dataset to {train_folder} ...")
        print("(NOTE: This should take no more than 10 minutes.)")
        with temporarily_chdir(train_folder):
            subprocess.run(
                args=(
                    f"tar -xf {train_archive} "
                    "--to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'"
                ),
                shell=True,
                check=raise_errors,
                stdout=sys.stdout,
            )
        train_done_file.touch()

    devkit_dest = destination_dir / "ILSVRC2012_devkit_t12.tar.gz"
    if not devkit_dest.exists():
        print("Copying the devkit file...")
        shutil.copyfile(devkit, devkit_dest)
    print("DONE!")


@contextmanager
def temporarily_chdir(new_dir: Path):
    """Temporarily navigate to the given directory."""
    start_dir = Path.cwd()
    try:
        os.chdir(new_dir)
        yield
    except OSError:
        raise
    finally:
        os.chdir(start_dir)
