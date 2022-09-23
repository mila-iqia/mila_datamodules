from __future__ import annotations

import torchvision.datasets as tvd
from pl_bolts.datamodules import STL10DataModule as _STL10DataModule

from mila_datamodules.registry import get_dataset_root

from .vision_datamodule import _TransformsFix

# Get the data directory, if possible.
default_data_dir = get_dataset_root(tvd.STL10)


# NOTE: This prevents a useless download by setting the `data_dir` to a good default value.
class STL10DataModule(_TransformsFix, _STL10DataModule):
    # NOTE: This doesn't subclass VisionDataModule, or set a `dataset_cls` attribute, so it's a
    # little bit harder to optimize this datamodule. However, it's probably fine in this case to
    # just load the data from the dataset dir, since it's in-memory numpy arrays anyway.
    def __init__(
        self,
        data_dir: str | None = default_data_dir,
        unlabeled_val_split: int = 5000,
        train_val_split: int = 500,
        num_workers: int = 0,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            unlabeled_val_split=unlabeled_val_split,
            train_val_split=train_val_split,
            num_workers=num_workers,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )
