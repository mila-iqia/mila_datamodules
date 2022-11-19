"""ImageNet datamodule adapted to the Mila cluster.

Can be used either with a PyTorch-Lightning Trainer, or by itself to easily get efficient
dataloaders for the ImageNet dataset.

Requirements (these are the versions I'm using, but this can probably be loosened a bit).
- pytorch-lightning==1.6.0
- lightning-bolts==0.5
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, TypedDict

from torch import nn
from torch.utils.data import DataLoader

from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.utils import get_cpus_on_node
from mila_datamodules.vision.datasets import BchUnlabeledImagenet

from .imagenet import ImagenetDataModule

class ImageNetFiles(TypedDict):
    image: str
    """ Path to the "ilsvrc2012.img" image containing the torchvision structured files. """


imagenet_file_locations: dict[Cluster, ImageNetFiles] = {
    Cluster.Mila: {
        "image": "/network/datasets/imagenet.var/imagenet_bcachefs/ilsvrc2012.img",
    },
    # TODO: Need help with filling these:
    Cluster.Beluga: {
        "image": "/project/rpp-bengioy/data/curated/imagenet.var/imagenet_bcachefs/ilsvrc2012.img",
    },
}
""" A map that shows where to retrieve the "ilsvrc2012.img" for each SLURM cluster. """


class ImagenetBchDataModule(ImagenetDataModule):
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
        warnings.warn("Bcachefs has not yet been battle tested. Please use with care.")

        if meta_dir is not None:
            warnings.warn(
                RuntimeWarning(
                    f"Ignoring passed meta_dir ({meta_dir})."
                )
            )
        meta_dir = None

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
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
        )

    def prepare_data(self) -> None:
        """Prepares the data, copying the dataset to the SLURM temporary directory.

        NOTE: When using this datamodule without the PyTorch-Lightning Trainer, make sure to call
        prepare_data() before calling train/val/test_dataloader().
        """
        copy_imagenet_to_dest(self.data_dir)

    def setup(self):
        train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        # Uses the train split of imagenet2012 and puts away a portion of it for
        # the validation split.
        self.dataset_train = BchUnlabeledImagenet(
            os.path.join(self.data_dir, "ilsvrc2012.img"),
            num_imgs_per_class=-1,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            meta_dir="/",
            split="train",
            transform=train_transforms,
        )

        val_transforms = self.val_transform() if self.val_transforms is None else self.val_transforms
        # Uses the part of the train split of imagenet2012  that was not used for training via
        # `num_imgs_per_val_class`
        self.dataset_val = BchUnlabeledImagenet(
            os.path.join(self.data_dir, "ilsvrc2012.img"),
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            meta_dir="/",
            split="val",
            transform=val_transforms,
        )

        test_transforms = self.val_transform() if self.test_transforms is None else self.test_transforms
        # Uses the validation split of imagenet2012 for testing.
        self.dataset_test = BchUnlabeledImagenet(
            os.path.join(self.data_dir, "ilsvrc2012.img"),
            num_imgs_per_class=-1,
            meta_dir="/",
            split="test",
            transform=test_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_test, shuffle=False)

    def _data_loader(self, dataset, shuffle: bool = False, **kwargs: Any) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            **kwargs
        )


def copy_imagenet_to_dest(
    destination_dir: str | Path,
) -> None:
    """Copies the ImageNet Bcachefs dataset into the destination folder (ideally in `slurm_tmpdir`)
    """
    paths = imagenet_file_locations[Cluster.current()]
    image = paths["image"]

    destination_dir = Path(destination_dir)
    raise_errors = False

    destination_dir.mkdir(exist_ok=True, parents=True)

    done_file = destination_dir / "bch_done.txt"
    if not done_file.exists():
        print(f"Copying imagenet bcachefs image to {destination_dir}/ ...")
        subprocess.run(
            args=["cp", "-aLt", destination_dir, image],
            check=raise_errors,
            stdout=sys.stdout,
        )
        done_file.touch()
    print("DONE!")
