from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Callable, TypedDict

import bcachefs as bch
import torch
from PIL import Image
from pl_bolts.datamodules.imagenet_datamodule import imagenet_normalization
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor

from mila_datamodules.clusters import Cluster
from mila_datamodules.vision.datasets import BchCocoCaptions
from .coco import CocoCaptionsDataModule

class CocoFiles(TypedDict):
    image: str
    """ Path to the "ilsvrc2012.img" image containing the torchvision structured files. """


coco_file_locations: dict[Cluster, CocoFiles] = {
    Cluster.Mila: {
        "image": "/network/datasets/coco.var/coco_bcachefs/coco.img",
    },
    # TODO: Need help with filling these:
    Cluster.Beluga: {
        "image": "/project/rpp-bengioy/data/curated/coco.var/coco_bcachefs/coco.img",
    },
}
""" A map that shows where to retrieve the "ilsvrc2012.img" for each SLURM cluster. """


class CocoCaptionsBchDataModule(CocoCaptionsDataModule):
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
        warnings.warn("Bcachefs has not yet been battle tested. Please use with care.")

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
            test_transforms = self.test_transforms or self.default_transforms()
            self.dataset_test = BchCocoCaptions(
                cursor=bchfs.cd(test_root), annFile=test_ann_file, transform=test_transforms
            )
            # NOTE: In order to get the transforms right, we'll have to instantiate this twice, with
            # the train transforms, and once again with the val transforms.
            # TODO: If we create a DataModule for COCODetection based on this one, we should pass
            # transforms= (with an s) to the CocoCaptions constructor, so the transforms are synced
            # between the image and the segmentation mask.
            with bchfs.cd(train_root) as cursor:
                train_transforms = self.train_transforms or self.default_transforms()
                dataset_trainval = BchCocoCaptions(
                    cursor=cursor, annFile=train_ann_file, transform=train_transforms
                )
                val_transforms = self.val_transforms or self.default_transforms()
                dataset_trainval_val = BchCocoCaptions(
                    cursor=cursor, annFile=train_ann_file, transform=val_transforms
                )

            self.dataset_predict = UnsupervisedImageDataset(
                cursor=bchfs.cd(predict_root), transforms=self.test_transforms
            )

        # Fork the RNG just to be 100% sure we don't have any effect on the global RNG state.
        with torch.random.fork_rng():
            # NOTE: Inside `super()._split_dataset`, the RNG is seeded with `self.seed`. So this is
            # supposed to be fine.
            self.dataset_train = self._split_dataset(dataset_trainval, train=True)
            self.dataset_val = self._split_dataset(dataset_trainval_val, train=False)

    def default_transforms(self) -> Callable:
        """Default transformations to use when none are provided."""

        transforms: list[Callable] = [ToTensor()]
        if self.normalize:
            # TODO: Seems like people use the normalization from ImageNet on COCO? Not 100% sure.
            transforms.append(imagenet_normalization())
        return Compose(transforms)

    def predict_dataloader(self) -> DataLoader:
        assert self.dataset_predict is not None
        return self._data_loader(self.dataset_predict)


class UnsupervisedImageDataset(Dataset):
    """Simple dataset for a folder containing images."""

    def __init__(self, cursor: bch.Cursor, transforms: Callable | None = None, extension="jpg"):
        if cursor.closed:
            with cursor:
                self._cursor = cursor.cd(cursor.pwd)
        else:
            self._cursor = cursor.cd(cursor.pwd)
        self.files = sorted((ent for ent in cursor if ent.name.endswith(f"*.{extension}")), key=lambda _ent: _ent.name)
        self.transforms = transforms

    def __getitem__(self, idx: int):
        ent = self.files[idx]
        with self._cursor.open(ent.inode) as _f:
            image = Image.open(_f).convert("RGB")
        if self.transforms:
            return self.transforms(image)
        return image

    def __len__(self):
        return len(self.files)


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

    done_file = destination_dir / "bch_done.txt"
    if not done_file.exists():
        print(f"Copying coco bcachefs image to {destination_dir}/ ...")
        subprocess.run(
            args=["cp", "-aLt", destination_dir, image],
            check=raise_errors,
            stdout=sys.stdout,
        )
        done_file.touch()
    print("DONE!")
