"""Wrappers around torchvision datasets that use a good default value for the 'root' argument based
on the current cluster."""
from __future__ import annotations

import functools
import inspect
import warnings
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, TypeVar, cast

from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.clusters import CURRENT_CLUSTER, SCRATCH, SLURM_TMPDIR
from mila_datamodules.clusters.cluster_enum import ClusterType
from mila_datamodules.utils import all_files_exist, copy_dataset_files, replace_kwargs
from mila_datamodules.vision.datasets.dataset_files import (
    dataset_files,
    dataset_roots_per_cluster,
    too_large_for_slurm_tmpdir,
)

P = ParamSpec("P")
D = TypeVar("D", bound=Dataset)
VD = TypeVar("VD", bound=VisionDataset)
C = Callable[P, D]

logger = get_logger(__name__)


def get_dataset_root(dataset_cls: type, cluster: ClusterType | None = None) -> str:
    """Gets the root directory to use to read the given dataset on the given cluster.

    Raises a NotImplementedError if the dataset location isn't known for that cluster.
    """
    cluster = cluster or ClusterType.current()
    dataset_roots = dataset_roots_per_cluster[cluster]
    if dataset_cls not in dataset_roots:
        raise NotImplementedError(
            f"Don't know yet where dataset `{dataset_cls}` is stored on cluster {cluster}!"
        )
    return str(dataset_roots[dataset_cls])


# TODO: Use types to also indicate that the 'root' argument becomes optional.


def make_dataset_fn(dataset_type: Callable[P, D]) -> Callable[P, D]:
    """Wraps a dataset into a function that constructs it efficiently.

    When invoked, the function first checks if the dataset is already downloaded in $SLURM_TMPDIR.
    If so, it is read and returned. If not, checks if the dataset is already stored somewhere in
    the cluster ($SCRATCH/data, then /network/datasets directory). If so, try to copy it over to
    SLURM_TMPDIR. If that works, read the dataset from SLURM_TMPDIR if the dataset isn't too large.
    If the dataset isn't found anywhere, then it is downloaded to SLURM_TMPDIR and read from there.

    NOTE: This is currently unused, as it was replaced with `adapted_constructor`. However it's
    cleaner, in a sense, since it's a stateless function. The only issue is that it returns a
    function rather than a class, and so it isn't 100% compatible with the original behaviour of
    the class it is wrapping.
    """

    if dataset_type not in dataset_files:
        warnings.warn(
            RuntimeWarning(
                f"Don't know which files are associated with dataset type {dataset_type}!"
                f"Consider adding the names of the files to the dataset_files dictionary."
            )
        )
        # NOTE: Perhaps we could also check the files in $SCRATCH/data, then attempt to download it
        # into $SCRATCH/data, get the list of newly created files in $SCRATCH/data, and if we
        # detect any new files, we then copy those into SLURM_TMPDIR, and then re-read the dataset
        # from SLURM_TMPDIR instead!
        return dataset_type

    required_files = dataset_files[dataset_type]
    fast_dir = SLURM_TMPDIR / "data"
    scratch_dir = SCRATCH / "data"
    dataset_type = cast(Callable[P, D], dataset_type)
    create_in_fast_dir = replace_kwargs(dataset_type, root=fast_dir)

    def copy_and_load_from(source_dir: Path):
        """Returns a function that copies the dataset files from `source_dir` to `fast_dir` and
        then loads the dataset from `fast_dir`."""

        @functools.wraps(dataset_type)
        def _copy_and_load(*args: P.args, **kwargs: P.kwargs) -> D:
            copy_dataset_files(required_files, source_dir, fast_dir)
            return create_in_fast_dir(*args, **kwargs)

        return _copy_and_load

    if all_files_exist(required_files, fast_dir):
        # The dataset is in $SLURM_TMPDIR/data.
        return create_in_fast_dir
    if all_files_exist(required_files, scratch_dir):
        # The dataset is stored in $SCRATCH/data.
        if dataset_type in too_large_for_slurm_tmpdir:
            return replace_kwargs(dataset_type, root=scratch_dir)
        return copy_and_load_from(scratch_dir)
    if all_files_exist(required_files, CURRENT_CLUSTER.torchvision_dir):
        if dataset_type in too_large_for_slurm_tmpdir:
            return replace_kwargs(dataset_type, root=CURRENT_CLUSTER.torchvision_dir)
        else:
            return copy_and_load_from(CURRENT_CLUSTER.torchvision_dir)

    # TODO: For datasets that can be downloaded but that we don't have, we could try to download it
    # into SCRATCH for later use, before copying it to SLURM_TMPDIR.

    def _try_slurm_tmpdir(*args: P.args, **kwargs: P.kwargs) -> D:
        try:
            return create_in_fast_dir(*args, **kwargs)
        except OSError:
            # Fallback to the normal callable (the passed in class)
            return dataset_type(*args, **kwargs)

    return _try_slurm_tmpdir


def adapted_constructor(
    dataset_cls: Callable[Concatenate[str, P], D]
) -> Callable[Concatenate[str | None, P], None]:
    """Creates a constructor for the given dataset class that does the required preprocessing
    steps."""
    if dataset_cls not in dataset_files:
        raise NotImplementedError(
            f"Don't know which files are associated with dataset type {dataset_cls}!"
            f"Consider adding the names of the files to the dataset_files dictionary."
        )
    required_files = dataset_files[dataset_cls]

    @functools.wraps(dataset_cls.__init__)
    def _custom_init(self, *args: P.args, **kwargs: P.kwargs):
        """A customized constructor that does the required steps before calling the original."""
        assert isinstance(dataset_cls, type)
        # base_init = super(cls, self).__init__
        base_init = dataset_cls.__init__
        bound_args = inspect.signature(base_init).bind_partial(self, *args, **kwargs)
        fast_tmp_dir = SLURM_TMPDIR / "data"
        scratch_dir = SCRATCH / "data"

        # The original passed value for the 'root' argument (if any).
        original_root: str | None = bound_args.arguments.get("root")
        new_root: str | Path | None = None

        if all_files_exist(required_files, fast_tmp_dir):
            logger.info("Dataset is already stored in SLURM_TMPDIR")
            new_root = fast_tmp_dir
        elif all_files_exist(required_files, scratch_dir):
            if dataset_cls not in too_large_for_slurm_tmpdir:
                logger.info("Copying files from SCRATCH to SLURM_TMPDIR")
                copy_dataset_files(required_files, scratch_dir, fast_tmp_dir)
                new_root = fast_tmp_dir
            else:
                logger.info("Dataset is large, files will be read from SCRATCH.")
                new_root = scratch_dir
        elif all_files_exist(required_files, CURRENT_CLUSTER.torchvision_dir):
            logger.info("Copying files from the torchvision dir to SLURM_TMPDIR")
            copy_dataset_files(required_files, CURRENT_CLUSTER.torchvision_dir, fast_tmp_dir)
            new_root = CURRENT_CLUSTER.torchvision_dir
        # TODO: Double-check these cases here, they are a more difficult to handle.
        elif original_root and all_files_exist(required_files, original_root):
            # If all files exist in the originally passed root_dir, then we just load it from
            # there.
            new_root = original_root
        elif original_root is None and all_files_exist(required_files, Path.cwd()):
            # TODO: The dataset would be loaded from the current directory, most probably.
            # Do we do anythin special in this case?
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
