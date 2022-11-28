from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import bcachefs as bch
import benzina.torch as bz
import benzina.torch.operations as ops
import numpy as np
import torch
from torch import nn
from torchvision.datasets import DatasetFolder

from mila_datamodules.clusters import Cluster
from mila_datamodules.vision.datasets import BchCocoCaptions
from .coco_bch import CocoCaptionsBchDataModule

# FIXME: I don't like hard-coded values.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

class CocoFiles(TypedDict):
    image: str
    """ Path to the "ilsvrc2012.img" image containing the torchvision structured files. """


coco_file_locations: dict[Cluster, CocoFiles] = {
    Cluster.Mila: {
        "image": "/network/datasets/coco.var/coco_benzina/coco.img",
    },
    # TODO: Need help with filling these:
    Cluster.Beluga: {
        "image": "/project/rpp-bengioy/data/curated/coco.var/coco_benzina/coco.img",
    },
}
""" A map that shows where to retrieve the "ilsvrc2012.img" for each SLURM cluster. """


class CocoCaptionsBenzinaDataModule(CocoCaptionsBchDataModule):
    """Datamodule for the COCO image caption dataset.

    Raw dataset items are tuples of the form (image, captions), where `image` is a PIL image, and
    `captions`
    varying shape

    TODO: Images don't have a fixed dimensionality, and I don't know what the 'standard crop'
    would be for this dataset.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        val_split: int | float = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 1,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Callable | nn.Module | None = None,
        val_transforms: Callable | nn.Module | None = None,
        test_transforms: Callable | nn.Module | None = None,
        **kwargs,
    ) -> None:
        # Benzina format for Coco is currently unsupported as the aveerage image
        # size in Coco is 640x480 and Benzina currently forces a downscale to
        # 512x512
        raise NotImplementedError("Benzina currently doesn't support the Coco dataset")

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
            data_dir,
            val_split,
            num_workers,
            normalize,
            batch_size,
            seed,
            shuffle,
            pin_memory,
            drop_last,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            **kwargs,
        )

    def prepare_data(self) -> None:
        """Prepares the data, copying the dataset to the SLURM temporary directory.

        NOTE: When using this datamodule without the PyTorch-Lightning Trainer, make sure to call
        prepare_data() before calling train/val/test_dataloader().
        """
        copy_coco_to_dest(self.data_dir)

    def setup(self):
        # NOTE: It's a bit too complicated to try to reuse the stuff from the base class.
        train_root = "train2017"
        test_root = "val2017"
        predict_root = "test2017"

        train_ann_file = "/annotations/captions_train2017.json"
        test_ann_file = "/annotations/captions_val2017.json"

        with bch.mount(os.path.join(self.data_dir, "coco.img")) as bchfs:
            self.dataset_test = BchCocoCaptions(
                cursor=bchfs.cd(test_root), annFile=test_ann_file
            )
            dataset_trainval = BchCocoCaptions(
                cursor=bchfs.cd(train_root), annFile=train_ann_file
            )
            self.dataset_predict = UnsupervisedImageDataset(
                cursor=bchfs.cd(predict_root), transforms=self.test_transforms
            )

        # Fork the RNG just to be 100% sure we don't have any effect on the global RNG state.
        with torch.random.fork_rng():
            # NOTE: Inside `super()._split_dataset`, the RNG is seeded with `self.seed`. So this is
            # supposed to be fine.
            self.dataset_train = self._split_dataset(dataset_trainval, train=True)
            self.dataset_val = self._split_dataset(dataset_trainval, train=False)

    def default_transforms(self) -> Callable:
        """Default transformations to use when none are provided."""

        transforms = {}
        if self.normalize:
            # TODO: Seems like people use the normalization from ImageNet on COCO? Not 100% sure.
            transforms = {
                **transforms,
                "bias_transform": ops.ConstantBiasTransform(bias=IMAGENET_MEAN),
                "norm_transform": ops.ConstantNormTransform(norm=IMAGENET_STD),
            }
        return transforms

    def train_dataloader(self) -> bz.DataLoader:
        """The train dataloader."""
        transforms = self.train_transforms or self.default_transforms()
        return self._data_loader(self.dataset_train, shuffle=self.shuffle, **transforms)

    def val_dataloader(self) -> bz.DataLoader:
        """The val dataloader."""
        transforms = self.val_transforms or self.default_transforms()
        return self._data_loader(self.dataset_val, **transforms)

    def test_dataloader(self) -> bz.DataLoader:
        """The test dataloader."""
        transforms = self.test_transforms or self.default_transforms()
        return self._data_loader(self.dataset_test, **transforms)

    def predict_dataloader(self) -> bz.DataLoader:
        transforms = self.test_transforms or self.default_transforms()
        return self._data_loader(self.dataset_predict, **transforms)

    def _data_loader(self, dataset: bz.dataset.BenzinaDatasetMixin, shuffle: bool = False, **kwargs: Any) -> bz.DataLoader:
        return bz.DataLoader(
            dataset,
            path=os.path.join(self.data_dir, "coco.img"),
            shape=512,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            **kwargs
        )


class UnsupervisedImageDataset(bz.dataset.BenzinaDatasetMixin, bz.dataset.ClassificationDatasetMixin, DatasetFolder):
    """Simple dataset for a folder containing images."""

    def __init__(self, cursor: bch.Cursor):
        bz.dataset.BenzinaDatasetMixin.__init__(cursor)
        DatasetFolder.__init__(cursor.filename, loader=None, extensions=("mp4",))

    def __getitem__(self, index: int):
        item = bz.dataset.BenzinaDatasetMixin.__getitem__(index)
        # remove target
        item.aux = tuple()
        return item

    def find_classes(self, _directory: str) -> Tuple[List[str], Dict[str, int]]:
        return ["."], {".": -1}


def copy_coco_to_dest(
    destination_dir: str | Path,
) -> None:
    """Copies the Coco Bcachefs dataset into the destination folder (ideally in `slurm_tmpdir`)
    """
    paths = coco_file_locations[Cluster.current()]
    image = paths["image"]

    destination_dir = Path(destination_dir)
    raise_errors = False

    destination_dir.mkdir(exist_ok=True, parents=True)

    done_file = destination_dir / "benzina_done.txt"
    if not done_file.exists():
        print(f"Copying coco bcachefs image to {destination_dir}/ ...")
        subprocess.run(
            args=["cp", "-atL", destination_dir, image],
            check=raise_errors,
            stdout=sys.stdout,
        )
        done_file.touch()
    print("DONE!")
