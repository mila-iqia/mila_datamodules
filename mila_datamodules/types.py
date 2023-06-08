from __future__ import annotations

import dataclasses
import typing
from typing import Callable, TypeVar, Union

from typing_extensions import Concatenate, ParamSpec, Protocol

if typing.TYPE_CHECKING:
    import torchvision.datasets as tvd

P = ParamSpec("P")
VD = TypeVar("VD", bound="tvd.VisionDataset")
VD_co = TypeVar("VD_co", bound="tvd.VisionDataset", covariant=True)
D = TypeVar("D")
D_co = TypeVar("D_co", covariant=True)

C = TypeVar("C", bound=Callable)
T = TypeVar("T")

DatasetFn = Union[type[D_co], Callable[P, D_co]]
DatasetFnWithStrArg = Union[type[D_co], Callable[Concatenate[str, P], D_co]]


class Dataclass(Protocol):
    __dataclass_fields__: dict[str, dataclasses.Field]
