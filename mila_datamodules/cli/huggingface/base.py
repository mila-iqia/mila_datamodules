"""TODO: Write some dataset preparation functions for HuggingFace datasets.
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib
import logging
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, TypedDict

from datasets import DownloadConfig, Version, load_dataset, load_dataset_builder
from simple_parsing import field
from typing_extensions import Concatenate, TypeVar

from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.shared_cache.setup import (
    setup_cache,
)
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_scratch_dir, get_slurm_tmpdir
from mila_datamodules.types import P
from mila_datamodules.utils import cpus_per_node

logger = get_logger(__name__)

_LoadDatasetFn = TypeVar("_LoadDatasetFn", bound=Callable)


@dataclass
class PrepareGenericDatasetArgs(DatasetArguments):
    """Options for preparing an arbitrary HuggingFace dataset."""

    path: str = field(positional=True)

    name: str | None = None

    # NOTE: These next args are available "for free" in our CLI because HF uses dataclasses!
    # data_dir: Optional[str] = None
    # data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]=None
    # cache_dir: Optional[str] = None
    # features: Optional[Features] = None
    # download_config: Optional[DownloadConfig] = None
    # download_mode: Optional[Union[DownloadMode, str]] = None
    revision: str | Version | None = None
    use_auth_token: bool | str | None = None
    # storage_options: Optional[dict] = None


class PrepareHfDatasetFn(Protocol):
    """A function that prepares a HuggingFace dataset, and returns the environment variables that
    should be set in the user job."""

    def __call__(
        self,
        root: str,
        name: str | None = None,
        /,
        _fn: Callable[Concatenate[str, str | None, P], Any] = load_dataset,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> HfDatasetsEnvVariables:
        ...


C = TypeVar("C", bound=Callable)


class WithEnvVars(PrepareHfDatasetFn):
    """A decorator that sets the environment variables before calling the function."""

    def __init__(self, env_vars: HfDatasetsEnvVariables, fn: PrepareHfDatasetFn):
        self.env_vars = env_vars
        self.fn = fn

    def __call__(
        self,
        root: str,
        name: str | None = None,
        _fn: Callable[Concatenate[str, str | None, P], Any] = load_dataset,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> HfDatasetsEnvVariables:
        with use_variables(self.env_vars):
            self.fn(root, name, _fn=_fn, *args, **kwargs)
        return self.env_vars


# def prepare_generic_v2(
#     path: str,
#     name: str | None = None,
#     __fn: Callable[Concatenate[str, str | None, P], Any] = load_dataset,
#     *args: P.args,
#     **kwargs: P.kwargs,
# ):
#     # TODO: Specify a subdirectory so that we only setup the huggingface stuff.
#     setup_cache(
#         user_cache_dir=user_cache_dir,
#         # subdirectory="huggingface",
#     )

#     dataset_constructor = functools.partial(__fn, path, name, *args, **kwargs)
#     hf_in_slurm_tmpdir = HfDatasetsEnvVariables(
#         HF_DATASETS_CACHE=get_slurm_tmpdir() / "cache/huggingface/datasets",
#     )
#     hf_in_scratch = HfDatasetsEnvVariables(
#         HF_DATASETS_CACHE=get_scratch_dir() / "cache/huggingface/datasets",
#     )
#     fn = Compose(
#         # Try to load the dataset from SLURM_TMPDIR/cache.
#         SkipRestIfThisWorks(WithEnvVars(hf_in_slurm_tmpdir, dataset_constructor)),
#         lambda root, *args, **kwargs: setup_cache(
#             user_cache_dir=get_scratch_dir() / "cache", subdirectory="huggingface"
#         ),
#         # Setup the dataset in $SCRATCH/cache/huggingface
#         WithEnvVars(hf_in_scratch, dataset_constructor),
#         # Make symlinks in $SLURM_TMPDIR for every file in $SCRATCH/cache/huggingface
#         lambda root, *args, **kwargs: setup_cache(
#             user_cache_dir=get_slurm_tmpdir() / "cache",
#             shared_cache_dir=get_scratch_dir() / "cache",
#             subdirectory="huggingface",
#             skip_modify_bash_aliases=True,
#         ),
#     )
#     return fn


def prepare_hf_dataset(
    path: str, name: str, **load_dataset_builder_kwargs
) -> HfDatasetsEnvVariables:
    """Prepare the a generic HuggingFace dataset."""
    cluster = Cluster.current_or_error()
    scratch = get_scratch_dir()
    slurm_tmpdir = get_slurm_tmpdir()
    from mila_datamodules.cli.shared_cache.setup import (
        DEFAULT_SHARED_CACHE_DIR,
    )
    from mila_datamodules.cli.shared_cache.setup import (
        logger as setup_cache_logger,
    )

    setup_cache_logger.setLevel(logging.INFO)
    setup_cache_logger.removeHandler(setup_cache_logger.handlers[0])

    # Load the dataset under $SCRATCH/cache/huggingface first, since we don't have a shared copy
    # on the cluster.
    scratch_cache_dir = scratch / "cache"
    shared_cache_dir = DEFAULT_SHARED_CACHE_DIR

    dataset_subdir = Path("huggingface/datasets") / path
    if name is not None:
        dataset_subdir = dataset_subdir / name

    logger.info(
        f"Making symlinks from {scratch_cache_dir/dataset_subdir} to "
        f"{shared_cache_dir/dataset_subdir}"
    )
    setup_cache(
        user_cache_dir=scratch_cache_dir,
        shared_cache_dir=shared_cache_dir,
        subdirectory=str(dataset_subdir),
    )

    # # TODO: On the second run, this will remove the files in $SLURM_TMPDIR that aren't symlinks
    # slurm_tmpdir_cache_dir = slurm_tmpdir / "cache"
    # logger.info(
    #     f"Making symlinks from {slurm_tmpdir_cache_dir/subdirectory} to "
    #     f"{scratch_cache_dir/subdirectory}"
    # )
    # # NOTE: Can reuse the same exact script to make the symlinks.
    # # because it thinks are duplicates. We don't want that.
    # setup_cache(
    #     user_cache_dir=scratch_cache_dir,
    #     shared_cache_dir=shared_cache_dir,
    #     subdirectory=subdirectory,
    # )

    # Number of processes to use for preparing the dataset. Note, since this is expected to be only
    # executed once per node, we can use all the CPUs
    num_proc = cpus_per_node()

    scratch_hf_datasets_cache = scratch_cache_dir / "huggingface/datasets"
    with use_variables(HF_DATASETS_CACHE=scratch_hf_datasets_cache):
        logger.info(f"Downloading and preparing the dataset in {scratch_hf_datasets_cache} ...")
        # load_dataset(path, name, num_proc=num_proc, **load_dataset_builder_kwargs)
        dataset_builder = load_dataset_builder(path, name, **load_dataset_builder_kwargs)
        # # TODO: There are tons of arguments that we could probably pass here.
        dataset_builder.download_and_prepare(num_proc=num_proc)

    # Copy the dataset from $SCRATCH to $SLURM_TMPDIR
    # TODO: if a `name` is passed, only copy that sub-subdirectory.
    dataset_dir = scratch_hf_datasets_cache / path
    if name is not None:
        dataset_dir = dataset_dir / name

    relative_path = dataset_dir.relative_to(scratch)
    logger.info(f"Copying dataset from {dataset_dir} -> {slurm_tmpdir / relative_path}")
    try:
        shutil.copytree(
            src=scratch / relative_path,
            dst=slurm_tmpdir / relative_path,
            symlinks=True,  # TODO: Keep symlinks in source dir as symlinks in destination dir?
            dirs_exist_ok=True,
        )
    except shutil.Error as err:
        _, _, messages = zip(*err.args[0])
        for message in messages:
            assert isinstance(message, str)
            if not message.startswith("[Errno 17] File exists"):
                raise IOError(message)

    logger.info("Checking that the dataset was copied correctly to SLURM_TMPDIR...")
    with use_variables(
        HfDatasetsEnvVariables(
            HF_DATASETS_CACHE=slurm_tmpdir / "cache/huggingface/datasets",
            HF_DATASETS_OFFLINE=1,  # Disabling internet just to be sure everything is setup.
        )
    ):
        download_config = DownloadConfig(local_files_only=True)
        load_dataset_builder_kwargs = load_dataset_builder_kwargs.copy()
        load_dataset_builder_kwargs.setdefault("download_config", download_config)
        load_dataset_builder_kwargs.setdefault("num_proc", num_proc)
        load_dataset(
            path,
            name,
            **load_dataset_builder_kwargs,
        )

    offline_bit = 0 if cluster.internet_access_on_compute_nodes else 1

    return HfDatasetsEnvVariables(
        HF_HOME=f"{scratch}/cache/huggingface",
        HF_DATASETS_CACHE=f"{slurm_tmpdir}/cache/huggingface/datasets",
        HF_DATASETS_OFFLINE=offline_bit,
    )


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
