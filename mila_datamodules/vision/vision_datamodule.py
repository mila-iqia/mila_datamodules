from __future__ import annotations

import typing

try:
    import cv2  # noqa (Has to be done before any ffcv/torch-related imports).
except ImportError:
    pass
import pl_bolts
import pytorch_lightning
from pl_bolts.datamodules import CIFAR10DataModule as _CIFAR10DataModule
from pl_bolts.datamodules import MNISTDataModule as _MNISTDataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule as _VisionDataModule

if typing.TYPE_CHECKING:
    from typing import Callable

    from torch import nn


class VisionDataModule(_VisionDataModule):
    """Fixes the fact that all the vision datamodules were broken by the lack of coordination
    between PyTorch-Lightning and pl-bolts releases 1.7.0 and 0.5.0, respectively.

    The transforms arguments were removed from LightningDataModule, while they were still necessary
    for the VisionDataModule to work. This was fixed on the master branch of lightning-bolts, and a
    future release of pl_bolts will include this fix (likely v0.6.0).
    """

    def __init__(
        self,
        *args,
        train_transforms: Callable | nn.Module | None = None,
        val_transforms: Callable | nn.Module | None = None,
        test_transforms: Callable | nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
