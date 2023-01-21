"""Auto-generated Configuration dataclasses for use with Hydra or SimpleParsing."""
from __future__ import annotations

from .datamodule import *  # noqa: F403
from mila_datamodules.configs.datamodule._utils import get_dynamic_config_for_name


def __getattr__(name: str):
    """This gets called when running `from mila_datamodules.configs import something`. and when
    this `something` is not defined.

    In this case, if `name` ends with `Config`, and there is a datamodule with that name, e.g.
    `"CocoCaptionsDataModuleConfig"`, then we return a dynamically created config dataclass for the
    `CocoCaptionsDataModule` class.
    """

    dynamic_datamodule_config_dataclass = get_dynamic_config_for_name(name)
    if dynamic_datamodule_config_dataclass:
        return dynamic_datamodule_config_dataclass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
