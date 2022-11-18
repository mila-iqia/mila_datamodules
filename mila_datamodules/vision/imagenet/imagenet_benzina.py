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
from typing import Callable, TypedDict

import benzina.torch as bz
import benzina.torch.operations as ops
import numpy as np
from torch import nn

from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.utils import get_cpus_on_node
from mila_datamodules.vision.datasets import BenzinaImageNet

from .imagenet_bch import ImagenetBchDataModule

# FIXME: I don't like hard-coded values.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


class ImageNetFiles(TypedDict):
    image: str
    """ Path to the "ilsvrc2012.img" image containing the torchvision structured files. """


imagenet_file_locations: dict[Cluster, ImageNetFiles] = {
    Cluster.Mila: {
        "image": "/network/datasets/imagenet.var/imagenet_benzina/ilsvrc2012.img",
    },
    # TODO: Need help with filling these:
    Cluster.Beluga: {
        "image": "/project/rpp-bengioy/data/curated/imagenet.var/imagenet_benzina/ilsvrc2012.img",
    },
}
""" A map that shows where to retrieve the "ilsvrc2012.img" for each SLURM cluster. """


class ImagenetBenzinaDataModule(ImagenetBchDataModule):
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
        if num_imgs_per_val_class is not None:
            warnings.warn(
                RuntimeWarning(
                    f"Ignoring passed num_imgs_per_val_class ({num_imgs_per_val_class})."
                )
            )
        num_imgs_per_val_class = 0
        if num_workers is not None:
            warnings.warn(
                RuntimeWarning(
                    f"Ignoring passed num_workers ({num_workers})."
                )
            )
        num_workers = None
        if pin_memory is not None:
            warnings.warn(
                RuntimeWarning(
                    f"Ignoring passed pin_memory ({pin_memory})."
                )
            )
        pin_memory = None

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
        self.dataset_train = BenzinaImageNet(
            os.path.join(self.data_dir, "ilsvrc2012.img"),
            split="train",
        )
        self.dataset_val = BenzinaImageNet(
            os.path.join(self.data_dir, "ilsvrc2012.img"),
            split="val",
        )
        self.dataset_test = BenzinaImageNet(
            os.path.join(self.data_dir, "ilsvrc2012.img"),
            split="test",
        )

    def train_dataloader(self) -> bz.DataLoader:
        """The train dataloader."""
        transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        return self._data_loader(self.dataset_train, shuffle=self.shuffle, **transforms)

    def val_dataloader(self) -> bz.DataLoader:
        """The val dataloader."""
        transforms = self.val_transform() if self.val_transforms is None else self.val_transforms
        return self._data_loader(self.dataset_val, **transforms)

    def test_dataloader(self) -> bz.DataLoader:
        """The test dataloader."""
        transforms = self.val_transform() if self.test_transforms is None else self.test_transforms
        return self._data_loader(self.dataset_test, **transforms)

    def _data_loader(self, dataset: bz.dataset.BenzinaDatasetMixin, shuffle: bool = False, **kwargs: Any) -> bz.DataLoader:
        return bz.DataLoader(
            dataset,
            path=os.path.join(self.data_dir, "ilsvrc2012.img"),
            shape=256,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            **kwargs
        )

    def train_transform(self) -> Callable:
        """The standard imagenet transforms.

        .. code-block:: python
            {
                "bias_transform": ops.ConstantBiasTransform(bias=IMAGENET_MEAN),
                "norm_transform": ops.ConstantNormTransform(norm=IMAGENET_STD),
                "warp_transform": ops.SimilarityTransform(
                    scale=(0.08, 1.0),
                    ratio=(3./4., 4./3.),
                    flip_h=0.5,
                    random_crop=True,
                ),
            }
        """
        return {
            "bias_transform": ops.ConstantBiasTransform(bias=IMAGENET_MEAN),
            "norm_transform": ops.ConstantNormTransform(norm=IMAGENET_STD),
            "warp_transform": ops.SimilarityTransform(
                scale=(0.08, 1.0),
                ratio=(3./4., 4./3.),
                flip_h=0.5,
                random_crop=True,
            ),
        }

    def val_transform(self) -> Callable:
        """The standard imagenet transforms for validation.

        .. code-block:: python
            {
                "bias_transform": ops.ConstantBiasTransform(bias=IMAGENET_MEAN),
                "norm_transform": ops.ConstantNormTransform(norm=IMAGENET_STD),
                "warp_transform": ops.CenterResizedCrop(),
            }
        """

        return {
            "bias_transform": ops.ConstantBiasTransform(bias=IMAGENET_MEAN),
            "norm_transform": ops.ConstantNormTransform(norm=IMAGENET_STD),
            "warp_transform": ops.CenterResizedCrop(),
        }


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

    done_file = destination_dir / "benzina_done.txt"
    if not done_file.exists():
        print(f"Copying imagenet bcachefs image to {destination_dir}/ ...")
        subprocess.run(
            args=["cp", "-aLt", destination_dir, image],
            check=raise_errors,
            stdout=sys.stdout,
        )
        done_file.touch()
    print("DONE!")
