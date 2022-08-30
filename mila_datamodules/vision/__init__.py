from __future__ import annotations

try:
    import cv2  # noqa (Has to be done before any ffcv/torch-related imports).
except ImportError:
    pass

from .cifar10 import CIFAR10DataModule
from .cityscapes import CityscapesDataModule
from .fashion_mnist import FashionMNISTDataModule
from .imagenet import ImagenetDataModule
from .imagenet_ffcv import ImagenetFfcvDataModule
from .mnist import MNISTDataModule
from .vision_datamodule import VisionDataModule

# TODO: Re-introduce the CIFAR100DataModule.
