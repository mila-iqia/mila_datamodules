from __future__ import annotations

import contextlib
import os
import tarfile
from logging import getLogger as get_logger
from pathlib import Path

import tqdm
from typing_extensions import Literal

from mila_datamodules.cli.utils import pbar as pbar

logger = get_logger(__name__)


@contextlib.contextmanager
def change_directory(path: Path):
    curdir = Path.cwd()
    os.chdir(path)
    yield
    os.chdir(curdir)


def prepare_imagenet(
    root: str | Path,
    split: Literal["train", "val"] = "train",
    *args,
    **kwargs,
):
    logger.info("Extracting the ImageNet archives using Olexa's tar magic...")
    if args or kwargs:
        logger.debug(f"Ignoring {args} and {kwargs}.")

    root = Path(root)
    # Create a symlink to the archive in $SLURM_TMPDIR, because torchvision expects it to be
    # there.

    network_imagenet_dir = Path("/network/datasets/imagenet")
    val_archive_file_name = "ILSVRC2012_img_val.tar"
    train_archive_file_name = "ILSVRC2012_img_train.tar"
    devkit_file_name = "ILSVRC2012_devkit_t12.tar.gz"
    md5sums_file_name = "md5sums"

    def _symlink_if_needed(filename: str, network_imagenet_dir: Path):
        symlink = root / filename
        if not symlink.exists():
            symlink.symlink_to(network_imagenet_dir / filename)

    _symlink_if_needed(train_archive_file_name, network_imagenet_dir)
    _symlink_if_needed(val_archive_file_name, network_imagenet_dir)
    _symlink_if_needed(devkit_file_name, network_imagenet_dir)
    _symlink_if_needed(md5sums_file_name, network_imagenet_dir)

    if split == "train":
        train_dir = root / "train"
        train_dir.mkdir(exist_ok=True, parents=True)

        # The ImageNet train archive is a tarfile of tarfiles (one for each class).
        with tarfile.open(network_imagenet_dir / train_archive_file_name) as train_tarfile:
            for member in tqdm.tqdm(
                train_tarfile,
                total=1000,  # hard-coded here, since we know there are 1000 folders.
                desc="Extracting train archive",
                unit="Directories",
                position=0,
            ):
                buffer = train_tarfile.extractfile(member)
                assert buffer is not None
                subdir = train_dir / member.name.replace(".tar", "")
                subdir.mkdir(mode=0o755, parents=True, exist_ok=True)
                files_in_subdir = set(p.name for p in subdir.iterdir())
                with tarfile.open(fileobj=buffer, mode="r|*") as sub_tarfile:
                    # n_extracted = 0
                    for tarinfo in sub_tarfile:
                        if tarinfo.name in files_in_subdir:
                            # Image file is already in the directory.
                            continue
                        sub_tarfile.extract(tarinfo, subdir)
                        # n_extracted += 1
                    # if n_extracted == 0:
                    #     logger.debug(f"Subdirectory {subdir} already has all required files.")
                    # else:
                    #     logger.debug(f"Extracted {n_extracted} files in {subdir}")

        # NOTE: Equivalent bash command:
        # with change_directory(train_dir):
        #     subprocess.check_output(
        #         f"tar  -xf {network_imagenet_dir}/{train_archive_file_name} "
        #         + "--to-command='mkdir -p ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'",
        #         shell=True,
        #     )
    else:
        val_dir = root / "val"
        val_dir.mkdir(exist_ok=True, parents=True)
        with tarfile.open(network_imagenet_dir / val_archive_file_name) as val_tarfile:
            val_tarfile.extractall(val_dir)
