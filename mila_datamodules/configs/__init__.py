"""Auto-generated Configuration dataclasses for use with Hydra or SimpleParsing."""
from __future__ import annotations

import functools
import typing
from typing import Callable

import hydra_zen
from hydra.core.config_store import ConfigStore
from typing_extensions import ParamSpec

if typing.TYPE_CHECKING:
    from hydra_zen.typing._implementations import BuildsWithSig

from mila_datamodules.vision import (
    BinaryEMNISTDataModule,
    BinaryMNISTDataModule,
    CIFAR10DataModule,
    CityscapesDataModule,
    EMNISTDataModule,
    FashionMNISTDataModule,
    ImagenetDataModule,
    ImagenetFfcvDataModule,
    MNISTDataModule,
    STL10DataModule,
)

T = typing.TypeVar("T")
P = ParamSpec("P")
C = typing.TypeVar("C", bound=Callable)


_cs = ConfigStore.instance()


def _cache(fn: C) -> C:
    # This is here so we don't lose any type information from the signature of the
    # wrapped callable.
    return functools.lru_cache(maxsize=None)(fn)  # type: ignore


@_cache
def _builds(datamodule_class: Callable[P, T]) -> type[BuildsWithSig[type[T], P]]:
    """creates a Config dataclass for the given datamodule type using hydra-zen, and registers it
    in the config store."""
    name = datamodule_class.__qualname__ + "Config"
    config_class = hydra_zen.builds(
        datamodule_class, populate_full_signature=True, dataclass_name=name
    )
    name = datamodule_class.__qualname__.replace("DataModule", "").lower()
    _cs.store(group="datamodule", provider="mila_datamodules", name=name, node=config_class)
    return config_class


# Create and declare the config classes for each datmodule.
# NOTE: These are really nicely typed, thanks to `hydra_zen`!
BinaryEMNISTDataModuleConfig = _builds(BinaryEMNISTDataModule)
BinaryMNISTDataModuleConfig = _builds(BinaryMNISTDataModule)
CIFAR10DataModuleConfig = _builds(CIFAR10DataModule)
CityscapesDataModuleConfig = _builds(CityscapesDataModule)
EMNISTDataModuleConfig = _builds(EMNISTDataModule)
FashionMNISTDataModuleConfig = _builds(FashionMNISTDataModule)
ImagenetDataModuleConfig = _builds(ImagenetDataModule)
ImagenetFfcvDataModuleConfig = _builds(ImagenetFfcvDataModule)
MNISTDataModuleConfig = _builds(MNISTDataModule)
STL10DataModuleConfig = _builds(STL10DataModule)
# IDEA: Dynamically create configs for any remaining/newer datamodules that aren't yet hard-coded?


def register_configs(group: str = "datamodule") -> None:
    _cs = ConfigStore.instance()
    provider = "mila_datamodules"
    kwargs = dict(group=group, provider=provider)
    _cs.store(**kwargs, name="binaryemnist", node=BinaryEMNISTDataModuleConfig)
    _cs.store(**kwargs, name="binarymnist", node=BinaryMNISTDataModuleConfig)
    _cs.store(**kwargs, name="cifar10", node=CIFAR10DataModuleConfig)
    _cs.store(**kwargs, name="cityscapes", node=CityscapesDataModuleConfig)
    _cs.store(**kwargs, name="emnist", node=EMNISTDataModuleConfig)
    _cs.store(**kwargs, name="fashionmnist", node=FashionMNISTDataModuleConfig)
    _cs.store(**kwargs, name="imagenet", node=ImagenetDataModuleConfig)
    _cs.store(**kwargs, name="imagenetffcv", node=ImagenetFfcvDataModuleConfig)
    _cs.store(**kwargs, name="mnist", node=MNISTDataModuleConfig)
    _cs.store(**kwargs, name="stl10", node=STL10DataModuleConfig)


# NOTE: Could there be cases in which we don't want to register these configs?
# register_configs()
