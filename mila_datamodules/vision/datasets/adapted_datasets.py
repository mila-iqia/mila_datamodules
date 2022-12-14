"""Wrappers around the torchvision datasets, adapted to the current cluster."""
from __future__ import annotations

import functools
import inspect
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, TypeVar, cast

from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.clusters import CURRENT_CLUSTER, SCRATCH, SLURM_TMPDIR
from mila_datamodules.registry import (
    dataset_files,
    get_dataset_root,
    is_stored_on_cluster,
    too_large_for_slurm_tmpdir,
)
from mila_datamodules.utils import all_files_exist, copy_dataset_files

T = TypeVar("T", bound=type)
D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
VD = TypeVar("VD", bound=VisionDataset)
C = Callable[P, D]

logger = get_logger(__name__)


def _cache(fn: C) -> C:
    return functools.cache(fn)  # type: ignore


def __getattr__(name: str) -> type[VD]:
    import torchvision.datasets as tvd

    if hasattr(tvd, name):
        attribute = getattr(tvd, name)

        if inspect.isclass(attribute) and issubclass(attribute, VisionDataset):
            return adapt_dataset(attribute)
    raise AttributeError(name)


@_cache
def adapt_dataset(dataset_type: type[VD]) -> type[VD]:
    """When not running on a SLURM cluster, returns the given input.

    When running on a SLURM cluster, returns a subclass of the given dataset that has a
    modified/adapted constructor.

    This constructor does a few things, but basically works like a three-level caching system.
    1. /network/datasets/torchvision (read-only "cache")
    2. $SCRATCH/cache/torch (writeable "cache")
    3. $SLURM_TMPDIR/data (writeable "cache", fastest storage).

    If the dataset isn't found on the cluster, it will be downloaded in $SCRATCH
    If the dataset fits in $SLURM_TMPDIR, it will be copied from wherever it is, and placed
    there.
    The dataset is then read from SLURM_TMPDIR.
    """
    if CURRENT_CLUSTER is None:
        return dataset_type  # Do nothing, since we're not on a SLURM cluster.

    dataset_subclass = type(
        dataset_type.__name__, (dataset_type,), {"__init__": adapted_constructor(dataset_type)}
    )
    dataset_subclass = cast("type[VD]", dataset_subclass)
    return dataset_subclass


# TODO: Use some fancy typing stuff indicate that the 'root' argument becomes optional.


def adapted_constructor(
    dataset_cls: Callable[Concatenate[str, P], VD]
) -> Callable[Concatenate[str | None, P], None]:
    """Creates a constructor for the given dataset class that does the required preprocessing steps
    before instantiating the dataset."""

    if dataset_cls not in dataset_files:
        for dataset in dataset_files:
            if dataset.__name__ == dataset_cls.__name__:
                files = dataset_files[dataset]
                break
        else:
            github_issue_url = (
                f"https://github.com/lebrice/mila_datamodules/issues/new?"
                f"template=feature_request.md&"
                f"title=Feature%20request:%20{dataset_cls.__name__}%20files"
            )
            raise NotImplementedError(
                f"Don't know which files are associated with dataset type {dataset_cls}!"
                f"Consider adding the names of the files to the dataset_files dictionary, or "
                f"creating an issue for this at {github_issue_url}"
            )
        required_files = files
    else:
        required_files = dataset_files[dataset_cls]

    @functools.wraps(dataset_cls.__init__)
    def _custom_init(self, *args: P.args, **kwargs: P.kwargs):
        """A customized constructor that does some optimization steps before calling the
        original."""
        assert isinstance(dataset_cls, type)
        base_init = dataset_cls.__init__
        bound_args = inspect.signature(base_init).bind_partial(self, *args, **kwargs)
        fast_tmp_dir = SLURM_TMPDIR / "data"
        scratch_dir = SCRATCH / "data"

        # The original passed value for the 'root' argument (if any).
        original_root: str | None = bound_args.arguments.get("root")
        new_root: str | Path | None = None

        # TODO: In the case of EMNIST or other datasets which we don't have stored, we should
        # probably try to download them in $SCRATCH/data first, then copy them to SLURM_TMPDIR if
        # they fit.
        if is_stored_on_cluster(dataset_cls):
            dataset_root = get_dataset_root(dataset_cls)
        else:
            dataset_root = None

        # TODO: Double-check / review this logic here. It's a bit messy.
        # TODO: Make this `all_files_exist` also check for empty directories.

        if all_files_exist(required_files, fast_tmp_dir):
            logger.info("Dataset is already stored in SLURM_TMPDIR")
            new_root = fast_tmp_dir
        elif all_files_exist(required_files, scratch_dir):
            if dataset_cls in too_large_for_slurm_tmpdir:
                logger.info("Dataset is large, files will be read from SCRATCH.")
                new_root = scratch_dir
            else:
                logger.info("Copying files from SCRATCH to SLURM_TMPDIR")
                copy_dataset_files(required_files, scratch_dir, fast_tmp_dir)
                new_root = fast_tmp_dir
        elif dataset_root and all_files_exist(required_files, dataset_root):
            # We know where the dataset is stored already (dataset_root is not None), and all files
            # are already found in the dataset root.
            logger.info("Copying files from the torchvision dir to SLURM_TMPDIR")
            copy_dataset_files(required_files, dataset_root, fast_tmp_dir)
            new_root = dataset_root
        # TODO: Double-check these cases here, they are more difficult to handle.
        elif original_root and all_files_exist(required_files, original_root):
            # If all files exist in the originally passed root_dir, then we just load it from
            # there.
            new_root = original_root
        elif original_root is None and all_files_exist(required_files, Path.cwd()):
            # TODO: The dataset would be loaded from the current directory, most probably.
            # Do we do anything special in this case?
            pass
        else:
            # NOTE: For datasets that can be downloaded but that we don't have, we could try to download it
            # into SCRATCH for later use, before copying it to SLURM_TMPDIR.
            logger.warn(
                f"Dataset files not found in the usual directories for the current cluster.\n"
                f"Creating dataset in $SCRATCH/data instead of the passed value '{original_root}'."
            )
            new_root = scratch_dir

        if new_root is not None:
            new_root = str(new_root)
        bound_args.arguments["root"] = new_root or original_root

        args = bound_args.args  # type: ignore
        kwargs = bound_args.kwargs  # type: ignore
        base_init(*args, **kwargs)

    return _custom_init
