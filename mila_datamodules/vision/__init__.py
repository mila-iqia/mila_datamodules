from __future__ import annotations

from typing import Union

from typing_extensions import Literal

try:
    import cv2  # noqa (Has to be done before any ffcv/torch-related imports).
except ImportError:
    pass

from mila_datamodules.clusters import CURRENT_CLUSTER

if CURRENT_CLUSTER is None:
    from pl_bolts.datamodules import (
        BinaryEMNISTDataModule,
        BinaryMNISTDataModule,
        CIFAR10DataModule,
        CityscapesDataModule,
        EMNISTDataModule,
        FashionMNISTDataModule,
        ImagenetDataModule,
        MNISTDataModule,
        STL10DataModule,
    )
    from pl_bolts.datamodules.vision_datamodule import VisionDataModule
else:
    from .cifar10 import CIFAR10DataModule
    from .cityscapes import CityscapesDataModule
    from .fashion_mnist import FashionMNISTDataModule
    from .imagenet import ImagenetDataModule
    from .imagenet_ffcv import ImagenetFfcvDataModule
    from .mnist import (
        BinaryEMNISTDataModule,
        BinaryMNISTDataModule,
        EMNISTDataModule,
        MNISTDataModule,
    )
    from .stl10 import STL10DataModule
    from .vision_datamodule import VisionDataModule


# TODO: Re-introduce the CIFAR100DataModule.
