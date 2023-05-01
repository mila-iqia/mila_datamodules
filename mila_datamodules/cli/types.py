from __future__ import annotations

import typing
from typing import Any, Callable

import torchvision.datasets as tvd
from typing_extensions import ParamSpec, TypeVar

if typing.TYPE_CHECKING:
    P = ParamSpec("P", default=...)
else:
    P = ParamSpec("P", default=Any)

VD = TypeVar("VD", bound=tvd.VisionDataset, default=tvd.VisionDataset)
VD_co = TypeVar("VD_co", bound=tvd.VisionDataset, default=tvd.VisionDataset, covariant=True)
C = TypeVar("C", bound=Callable)
T = TypeVar("T")
