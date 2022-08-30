from __future__ import annotations

import typing

try:
    import cv2  # noqa (Has to be done before any ffcv/torch-related imports).
except ImportError:
    pass
import pl_bolts
import pytorch_lightning
from pl_bolts.datamodules.vision_datamodule import VisionDataModule as _VisionDataModule

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


_bolts_version = tuple(map(int, pl_bolts.__version__.split(".")))


class VisionDataModule(
    _VisionDataModule, *((_TransformsFix,) if _bolts_version <= (0, 5, 0) else ())
):
    ...

    # TODO: Could also apply the changes here in the `prepare_data` method!
    # This could also change the value of self.data_dir, using the same logic as in
    # `adapt_dataset`.
    # OR (perhaps better): we could have a fully qualified constructor and change the
    # self.data_dir attribute? Or even add a `data_dir` property?
    # However, have to be careful that this also works with multiple workers / GPUS, where only
    # the first worker would have `prepare_data` be called. --> The current approach (changing
    # `self.dataset_cls`) might be best for now.
    def prepare_data(self) -> None:
        """Saves files to data_dir."""
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)
