"""Utilities for creating the Config dataclasses."""
from __future__ import annotations

import functools
import typing
from typing import Callable

import hydra_zen
from hydra.core.config_store import ConfigStore
from typing_extensions import ParamSpec

if typing.TYPE_CHECKING:
    from hydra_zen.typing._implementations import BuildsWithSig

T = typing.TypeVar("T")
P = ParamSpec("P")
C = typing.TypeVar("C", bound=Callable)


_cs = ConfigStore.instance()


def get_dynamic_config_for_name(name: str):
    import inspect

    from pytorch_lightning import LightningDataModule

    import mila_datamodules.vision

    from ._utils import builds

    datamodules = [
        v
        for v in vars(mila_datamodules.vision).values()
        if inspect.isclass(v) and issubclass(v, LightningDataModule)
    ]
    for datamodule in datamodules:
        if name == f"{datamodule.__name__}Config":
            print(f"Returning a dynamically created config class for datamodule {datamodule}.")
            config_dataclass_for_datamodule = builds(datamodule)
            return config_dataclass_for_datamodule


def _cache(fn: C) -> C:
    # This is here so we don't lose any type information from the signature of the
    # wrapped callable.
    return functools.lru_cache(maxsize=None)(fn)  # type: ignore


@_cache
def builds(
    datamodule_class: Callable[P, T], register: bool = True
) -> type[BuildsWithSig[type[T], P]]:
    """creates a Config dataclass for the given datamodule type using hydra-zen.

    NOTE: Can also register the configs in the config store if `register=True`.
    """
    name = datamodule_class.__qualname__ + "Config"
    config_class = hydra_zen.builds(
        datamodule_class,
        populate_full_signature=True,
        # dataclass_name=name,
        zen_dataclass={"cls_name": name},
    )
    # NOTE: Perhaps we shouldn't register these configs by default? e.g. if users already have their
    # own configs for `cifar10`, etc?
    if register:
        name = datamodule_class.__qualname__.replace("DataModule", "").lower()
        _cs.store(group="datamodule", provider="mila_datamodules", name=name, node=config_class)
    return config_class
