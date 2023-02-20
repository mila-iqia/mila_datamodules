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
from torchvision.datasets.imagenet import parse_val_archive
from typing_extensions import NotRequired

from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_slurm_tmpdir
from mila_datamodules.vision.datasets.adapted_datasets import prepare_dataset

logger = get_logger(__name__)


class ImageNetFiles(TypedDict):
    train_archive: str
    """Path to the "ILSVRC2012_img_train.tar"-like archive containing the training set."""

    val_archive: NotRequired[str]
    """Path to the "ILSVRC2012_img_val.tar" archive containing the test set.

    One of `val_archive` or `val_folder` must be provided.
    """

    val_folder: NotRequired[str]
    """Path to the extracted 'val' folder containing the test set.

    One of `val_archive` or `val_folder` must be provided.
    """

    devkit: str
    """Path to the ILSVRC2012_devkit_t12.tar.gz file."""


imagenet_file_locations: dict[Cluster, ImageNetFiles] = {
    Cluster.Mila: {
        "train_archive": "/network/datasets/imagenet/ILSVRC2012_img_train.tar",
        "val_archive": "/network/datasets/imagenet/ILSVRC2012_img_val.tar",
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
    """Optimized setup for the ImageNet dataset on a SLURM cluster. (although mostly based on the
    Mila cluster)

    Copies/extracts the ImageNet dataset into the destination folder (ideally in `slurm_tmpdir`)

    NOTE: This is the transcribed wisdom of @obilaniu (Olexa Bilaniuk).
    See [this Slack thread](https://mila-umontreal.slack.com/archives/CFAS8455H/p1652168938773169?thread_ts=1652126891.083229&cid=CFAS8455H)
    for more info.
    """
    destination_dir = get_slurm_tmpdir() / "data" / type(dataset).__name__
    destination_dir.parent.mkdir(exist_ok=True)
    destination_dir.mkdir(exist_ok=True)

    cluster = Cluster.current_or_error()
    paths = imagenet_file_locations[cluster]
    train_archive = Path(paths["train_archive"])
    devkit = paths["devkit"]

    destination_dir = Path(destination_dir)

    extracted_train_folder = destination_dir / "train"
    extracted_train_folder.mkdir(exist_ok=True, parents=True)

    extracted_val_folder = destination_dir / "val"
    extracted_val_folder.mkdir(exist_ok=True, parents=True)

    devkit_dest = destination_dir / "ILSVRC2012_devkit_t12.tar.gz"
    if not devkit_dest.exists():
        devkit_dest.symlink_to(target=devkit)

    if not all(
        (d / "meta.bin").exists()
        for d in (destination_dir, extracted_train_folder, extracted_val_folder)
    ):
        generate_meta_bins(destination_dir)

    if split in ["val", "both"]:
        val_archive: Path | None = None
        val_folder: Path | None = None
        if "val_archive" in paths:
            val_archive = Path(paths["val_archive"])
        elif "val_folder" in paths:
            val_folder = Path(paths["val_folder"])
        else:
            raise RuntimeError(
                f"One of 'val_folder' or 'val_archive' must be set for cluster {cluster}!"
            )

        val_done_file = extracted_val_folder / "val_done.txt"
        if not val_done_file.exists():
            if val_archive:
                # Create a symlink to the val archive in SLURM_TMPDIR, then use the torchvision
                # functions to unpack it.
                val_archive_in_slurm_tmpdir = destination_dir / val_archive.name
                if not val_archive_in_slurm_tmpdir.exists():
                    val_archive_in_slurm_tmpdir.symlink_to(target=val_archive)
                print("Parsing val archive ...")
                parse_val_archive(str(destination_dir), file=val_archive.name)
            else:
                assert val_folder
                copy_val_folder(val_folder, extracted_val_folder)
            val_done_file.touch()

    if split in ["train", "both"]:
        train_done_file = destination_dir / "train_done.txt"

        train_archive_in_slurm_tmpdir = destination_dir / Path(train_archive).name
        if not train_archive_in_slurm_tmpdir.exists():
            train_archive_in_slurm_tmpdir.symlink_to(target=train_archive)

        if not train_done_file.exists():
            print(f"Extracting imagenet train archive to {extracted_train_folder} ...")
            print("(NOTE: This should take ~7 minutes.)")
            with temporarily_chdir(extracted_train_folder):
                print(
                    "> tar -xf {train_archive} "
                    "--to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'"
                )
                raise_errors = False
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

    return str(destination_dir)


def generate_meta_bins(data_dir: str | Path) -> None:
    """Generates the meta.bin file required by the PL imagenet datamodule, and copies it in the
    train and val directories."""
    from pl_bolts.datasets.imagenet_dataset import UnlabeledImagenet

    UnlabeledImagenet.generate_meta_bins(str(data_dir))
    data_dir = Path(data_dir)
    meta_bin_file = data_dir / "meta.bin"
    assert meta_bin_file.exists() and meta_bin_file.is_file()
    (data_dir / "train").mkdir(parents=False, exist_ok=True)
    (data_dir / "val").mkdir(parents=False, exist_ok=True)
    shutil.copyfile(meta_bin_file, data_dir / "train" / "meta.bin")
    shutil.copyfile(meta_bin_file, data_dir / "val" / "meta.bin")


def copy_val_folder(val_folder: Path, extracted_val_folder: Path):
    print(f"Copying imagenet val dataset to {extracted_val_folder} ...")
    shutil.copytree(val_folder, extracted_val_folder)


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


def debug():
    from mila_datamodules.vision.datasets import ImageNet

    val_dataset = ImageNet(split="val")
    print(val_dataset)

    train_dataset = ImageNet(split="train")
    print(train_dataset)


if __name__ == "__main__":
    debug()
