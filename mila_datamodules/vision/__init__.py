from __future__ import annotations

import typing

import cv2  # noqa (Has to be done before any ffcv/torch-related imports).
import pl_bolts
import pytorch_lightning
from pl_bolts.datamodules.vision_datamodule import VisionDataModule

if typing.TYPE_CHECKING:
    from typing import Callable
    from torch import nn

from .cifar10 import CIFAR10DataModule
from .imagenet import ImagenetDataModule

try:
    from .imagenet_ffcv import ImagenetFfcvDataModule
except ImportError:
    pass
