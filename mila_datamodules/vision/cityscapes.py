from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from pl_bolts.datamodules import CityscapesDataModule as _CityscapesDataModule
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from mila_datamodules.clusters.cluster_enum import ClusterType
from mila_datamodules.clusters.utils import SLURM_TMPDIR, replace_kwargs

from .pl_bolts_fix import VisionDataModule

cityscapes_dir_locations: dict[ClusterType, str] = {
    ClusterType.MILA: "/network/datasets/cityscapes.var/cityscapes_torchvision",
}
cityscapes_dir_location = cityscapes_dir_locations.get(ClusterType.current())


class CityscapesDataModule(_CityscapesDataModule):
    def __init__(
        self,
        data_dir: str | None = None,
        quality_mode: str = "fine",
        target_type: str = "instance",
        num_workers: int = 0,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Callable | nn.Module | None = None,
        val_transforms: Callable | nn.Module | None = None,
        test_transforms: Callable | nn.Module | None = None,
    ) -> None:
        data_dir = str(SLURM_TMPDIR / "cityscapes")
        super().__init__(
            data_dir=data_dir,
            quality_mode=quality_mode,
            target_type=target_type,
            num_workers=num_workers,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def prepare_data(self):
        if cityscapes_dir_location is None:
            raise NotImplementedError(
                f"Don't know where cityscapes data is located in cluster {ClusterType.current()}"
            )

        done_file = Path(self.data_dir) / "done.txt"
        if not done_file.exists():
            print(f"Copying cityscapes data from {cityscapes_dir_location} to {self.data_dir}")
            shutil.copytree(cityscapes_dir_location, self.data_dir)
            done_file.touch()
        super().prepare_data()
