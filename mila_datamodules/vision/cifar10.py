from __future__ import annotations

from pl_bolts.datamodules import CIFAR10DataModule as _CIFAR10DataModule

from mila_datamodules.vision.datasets import CIFAR10

from .vision_datamodule import VisionDataModule


class CIFAR10DataModule(_CIFAR10DataModule, VisionDataModule):
    dataset_cls = CIFAR10

    def setup(self, stage: str | None = None):
        # Fix a bug in the base class, where `stage="validate"` is not handled properly.
        if stage == "validate":
            stage = "fit"
        super().setup(stage)
