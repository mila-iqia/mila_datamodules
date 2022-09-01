from __future__ import annotations

import typing
from typing import ClassVar

import pl_bolts
import pytorch_lightning
from pl_bolts.datamodules.vision_datamodule import VisionDataModule as _VisionDataModule
from torchvision.datasets import VisionDataset

if typing.TYPE_CHECKING:
    from typing import Callable

    from torch import nn


class _TransformsFix(pytorch_lightning.LightningDataModule):
    """Fixes the fact that all the vision datamodules were broken by the lack of coordination
    between PyTorch-Lightning and pl-bolts releases 1.7.0 and 0.5.0, respectively.

    The transforms arguments were removed from LightningDataModule, while they were still necessary
    for the VisionDataModules to work. This was fixed on the master branch of lightning-bolts, and
    a future release of pl_bolts will include this fix (likely v0.6.0).
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


_pl_version = tuple(map(int, pytorch_lightning.__version__.split(".")))
_bolts_version = tuple(map(int, pl_bolts.__version__.split(".")))


class VisionDataModule(
    _VisionDataModule,
    *((_TransformsFix,) if _pl_version >= (1, 7, 0) and _bolts_version <= (0, 5, 0) else ()),
):
    dataset_cls: ClassVar[type[VisionDataset]]

    # NOTE: Could also override this `prepare_data` to some custom copying/etc that we want.
    # However, we have to be careful that this also works with multiple workers / GPUS, where only
    # the first worker would call the `prepare_data` method. Therefore, the current approach
    # (changing `self.dataset_cls`) might be best for now.
    def prepare_data(self) -> None:
        """Saves files to data_dir."""
        super().prepare_data()
        # NOTE: Does the following:
        # self.dataset_cls(self.data_dir, train=True, download=True)
        # self.dataset_cls(self.data_dir, train=False, download=True)
