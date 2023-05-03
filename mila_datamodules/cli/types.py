from __future__ import annotations

import dataclasses
import typing
from typing import Any, Callable

import torchvision.datasets as tvd
from typing_extensions import ParamSpec, Protocol, TypeVar

if typing.TYPE_CHECKING:
    P = ParamSpec("P", default=...)
else:
    P = ParamSpec("P", default=Any)

VD = TypeVar("VD", bound=tvd.VisionDataset, default=tvd.VisionDataset)
VD_co = TypeVar("VD_co", bound=tvd.VisionDataset, default=tvd.VisionDataset, covariant=True)
D = TypeVar("D")
D_co = TypeVar("D_co", covariant=True)

C = TypeVar("C", bound=Callable)
T = TypeVar("T")


class Dataclass(Protocol):
    __dataclass_fields__: dict[str, dataclasses.Field]
