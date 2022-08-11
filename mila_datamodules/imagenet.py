""" ImageNet datamodule adapted to the Mila cluster.

Can be used either with a PyTorch-Lightning Trainer, or by itself to easily get efficient
dataloaders for the ImageNet dataset.

Requirements (these are the versions I'm using, but this can probably be loosened a bit).
- pytorch-lightning==1.6.0
- lightning-bolts==0.5
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, NewType
from pytorch_lightning import Trainer

from pl_bolts.datamodules.imagenet_datamodule import (
    ImagenetDataModule as _ImagenetDataModule,
)
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datasets import UnlabeledImagenet
from torch import nn
from torch.utils.data import DataLoader

from .utils import get_cpus_on_node, get_slurm_tmpdir

C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)


class ImagenetDataModule(_ImagenetDataModule):
    """Imagenet DataModule adapted to the Mila cluster.

    - Copies/Extracts the datasets to the `$SLURM_TMPDIR/imagenet` directory.
    - Uses the right number of workers, depending on the SLURM configuration.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        meta_dir: str | None = None,
        num_imgs_per_val_class: int = 50,
        image_size: int = 224,
        num_workers: int | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Callable | nn.Module | None = None,
        val_transforms: Callable | nn.Module | None = None,
        test_transforms: Callable | nn.Module | None = None,
    ) -> None:
        slurm_tmpdir = get_slurm_tmpdir()
        assert slurm_tmpdir
        fixed_data_dir = str(slurm_tmpdir / "imagenet")
        if data_dir is not None and data_dir != fixed_data_dir:
            warnings.warn(
                RuntimeWarning(
                    f"Ignoring passed data_dir ({data_dir}), using {fixed_data_dir} instead."
                )
            )
        data_dir = fixed_data_dir

        super().__init__(
            data_dir=data_dir,
            meta_dir=meta_dir,
            num_imgs_per_val_class=num_imgs_per_val_class,
            image_size=image_size,
            num_workers=num_workers or get_cpus_on_node(),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self.trainer: Trainer | None = None

    def prepare_data(self) -> None:
        """Prepares the data, copying the dataset to the SLURM temporary directory.

        NOTE: When using this datamodule without the PyTorch-Lightning Trainer, make sure to call
        prepare_data() before calling train/val/test_dataloader().
        """
        copy_imagenet_to_dest(self.data_dir)
        _generate_meta_bins(self.data_dir)
        super().prepare_data()

    def train_dataloader(self) -> DataLoader:
        return super().train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return super().val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return super().test_dataloader()

    @property
    def train_transforms(self) -> nn.Module | Callable | None:
        return self._train_transforms

    @train_transforms.setter
    def train_transforms(self, value: nn.Module | Callable | None):
        self._train_transforms = value

    @property
    def val_transforms(self) -> nn.Module | Callable | None:
        return self._val_transforms

    @val_transforms.setter
    def val_transforms(self, value: nn.Module | Callable | None):
        self._val_transforms = value

    @property
    def test_transforms(self) -> nn.Module | Callable | None:
        return self._test_transforms

    @test_transforms.setter
    def test_transforms(self, value: nn.Module | Callable | None):
        self._test_transforms = value

    @property
    def dims(self) -> tuple[C, H, W]:
        """A tuple describing the shape of your data. Extra functionality exposed in ``size``.

        .. deprecated:: v1.5     Will be removed in v1.7.0.
        """
        return self._dims

    @dims.setter
    def dims(self, v: tuple[C, H, W]):
        self._dims = v


def copy_imagenet_to_dest(destination: str | Path) -> None:
    """Copies/extracts the ImageNet dataset into the destination folder (ideally in `slurm_tmpdir`)

    NOTE: This is the transcribed wisdom of @obilaniu (Olexa Bilaniuk).
    See [this Slack thread](https://mila-umontreal.slack.com/archives/CFAS8455H/p1652168938773169?thread_ts=1652126891.083229&cid=CFAS8455H)
    for more info.
    """
    destination = Path(destination)
    raise_errors = False

    train_folder = destination / "train"
    train_folder.mkdir(exist_ok=True, parents=True)

    val_done_file = destination / "val_done.txt"
    if not val_done_file.exists():
        print(f"Copying imagenet val dataset to {destination}/val ...")
        subprocess.run(
            args=f"cp -r /network/datasets/imagenet.var/imagenet_torchvision/val {destination}",
            shell=True,
            check=raise_errors,
            stdout=sys.stdout,
        )
        val_done_file.touch()

    train_done_file = destination / "train_done.txt"
    if not train_done_file.exists():
        print(f"Copying imagenet train dataset to {destination}/train ...")
        print("(NOTE: This should take approximately 10 minutes.)")
        with temporarily_chdir(destination / "train"):
            subprocess.run(
                args=(
                    "tar -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar "
                    "--to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'"
                ),
                shell=True,
                check=raise_errors,
                stdout=sys.stdout,
            )
        train_done_file.touch()

    devkit_file = Path("/network/datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz")
    devkit_dest = destination / "ILSVRC2012_devkit_t12.tar.gz"
    if not devkit_dest.exists():
        print(f"Copying the devkit file...")
        shutil.copyfile(devkit_file, devkit_dest)
    print("DONE!")


def _generate_meta_bins(data_dir: str | Path) -> None:
    """Generates the meta.bin file required by the PL imagenet datamodule, and copies it in the
    train and val directories.
    """
    UnlabeledImagenet.generate_meta_bins(str(data_dir))
    data_dir = Path(data_dir)
    meta_bin_file = data_dir / "meta.bin"
    assert meta_bin_file.exists() and meta_bin_file.is_file()
    (data_dir / "train").mkdir(parents=False, exist_ok=True)
    (data_dir / "val").mkdir(parents=False, exist_ok=True)
    shutil.copyfile(meta_bin_file, data_dir / "train" / "meta.bin")
    shutil.copyfile(meta_bin_file, data_dir / "val" / "meta.bin")


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
