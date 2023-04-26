"""TODO: Write some dataset preparation functions for HuggingFace datasets.
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib
import os
import warnings
from dataclasses import asdict, dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Literal, Protocol, TypedDict

from simple_parsing import field

from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_scratch_dir

logger = get_logger(__name__)


class PrepareHfDatasetFn(Protocol):
    """A function that prepares a HuggingFace dataset, and returns the environment variables that
    should be set in the user job."""

    def __call__(self, *args, **kwargs) -> HfDatasetsEnvVariables:
        ...


@dataclass
class HfDatasetsEnvVariables:
    HF_HOME: str | Path = field(default_factory=lambda: get_scratch_dir() / "cache/huggingface")
    HF_DATASETS_CACHE: str | Path = field(
        default_factory=lambda: get_scratch_dir() / "cache/huggingface/datasets"
    )

    # When running on a cluster where compute nodes don't have internet access, we copy what we can
    # from $SCRATCH to $SLURM_TMPDIR, and set these variables to 1 to avoid attempting to
    # downloading anything that is missing:
    HF_DATASETS_OFFLINE: Literal[0, 1] = 0

    # # TODO: Seems to be used for model weights.
    HUGGINGFACE_HUB_CACHE: str | Path = field(
        default_factory=lambda: get_scratch_dir() / "cache/huggingface/hub"
    )
    TRANSFORMERS_OFFLINE: Literal[0, 1] = field(
        default_factory=lambda: 1
        if not Cluster.current_or_error().internet_access_on_compute_nodes
        else 0
    )

    @classmethod
    def under_dir(cls, hf_home_dir: str | Path):
        """Returns the environment variables to set so that HuggingFace looks for things in the
        given directory."""
        hf_home_dir = Path(hf_home_dir)
        return cls(
            HF_HOME=hf_home_dir,
            HF_DATASETS_CACHE=hf_home_dir / "datasets",
            HUGGINGFACE_HUB_CACHE=hf_home_dir / "hub",
        )

    @classmethod
    def in_scratch(cls):
        return cls.under_dir(hf_home_dir=get_scratch_dir() / "cache" / "huggingface")

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


class DatasetsConfigModuleAttributes(TypedDict):
    """typeddict of the attributes that are modified in the `datasets.config` module when using the
    `set_hf_variables` function."""

    HF_CACHE_HOME: str
    HF_DATASETS_CACHE: Path
    HF_DATASETS_OFFLINE: bool


def set_hf_variables(**hf_variables) -> None:
    """Sets the environment variables that control where huggingface datasets are stored.

    Also modifies the `datasets.config` module in-place to reflect the changes.
    """
    for key, value in hf_variables.items():
        os.environ[key] = str(value)

    _apply_changes_to_datasets_config_module()
    _apply_changes_to_hf_vars_in_global_scope()


def _apply_changes_to_datasets_config_module():
    """Reloads the `datasets.config` module to reflect the changes made to the env variables.

    NOTE: This will not update the values of variables that have already been imported from
    `datasets.config` into another module!
    For example, if in some module foo.py there is a line with
    `from datasets.config import HF_CACHE_HOME`, then calling this after that import will not
    change the value of that variable. `transformers.models.(...) module
    """
    import datasets.config

    importlib.reload(datasets.config)


def _apply_changes_to_hf_vars_in_global_scope():
    import datasets.config

    global_scope = globals()
    for variable_name, value in vars(datasets.config).items():
        if variable_name in global_scope and variable_name == variable_name.upper():
            warnings.warn(
                RuntimeWarning(
                    f"Found what looks like an imported variable from `datasets.config` in the "
                    f"global scope: {variable_name!r}. Changing its value to {value!r} to match "
                    f"other changes. Note that any other variables that depend on this value "
                    f"would need to be updated!"
                )
            )
            global_scope[variable_name] = value


@contextlib.contextmanager
def use_variables(
    hf_variables: HfDatasetsEnvVariables | None = None,
    **specific_variables_to_set: str | Path | int,
):
    assert bool(hf_variables) ^ bool(specific_variables_to_set), "Need an argument"

    if hf_variables:
        backup = {key: os.environ.get(key) for key in asdict(hf_variables)}
        variables_to_set = asdict(hf_variables)
    else:
        backup = {key: os.environ.get(key) for key in specific_variables_to_set}
        variables_to_set = specific_variables_to_set.copy()

    set_hf_variables(**variables_to_set)

    yield hf_variables

    for key, previous_value_or_none in backup.items():
        if previous_value_or_none is None:
            # The environment variable wasn't set before, so we remove it:
            os.environ.pop(key)
        else:
            os.environ[key] = previous_value_or_none

    # Reload *again*, to get back to what it was set to before.
    _apply_changes_to_datasets_config_module()
    _apply_changes_to_hf_vars_in_global_scope()
