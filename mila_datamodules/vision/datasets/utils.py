"""Wrappers around torchvision datasets that use a good default value for the 'root' argument based
on the current cluster."""
from __future__ import annotations

import functools
import inspect
import warnings
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, TypeVar, cast

import torchvision.datasets as tvd
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.clusters import CURRENT_CLUSTER, SCRATCH, SLURM_TMPDIR
from mila_datamodules.clusters.cluster_enum import ClusterType
from mila_datamodules.utils import all_files_exist, copy_dataset_files, replace_kwargs

P = ParamSpec("P")
D = TypeVar("D", bound=Dataset)
VD = TypeVar("VD", bound=VisionDataset)
C = Callable[P, D]

logger = get_logger(__name__)


def dataset_root(dataset_cls: type, cluster: ClusterType | None = None) -> Path:
    cluster = cluster or ClusterType.current()
    if issubclass(dataset_cls, tvd.VisionDataset):
        return cluster.torchvision_dir
    raise NotImplementedError


known_dataset_files = {
    tvd.MNIST: ["MNIST"],
    tvd.CIFAR10: ["cifar-10-batches-py"],
    tvd.CIFAR100: ["cifar-100-python"],
    tvd.FashionMNIST: ["FashionMNIST"],
    tvd.Caltech101: ["caltech101"],
    tvd.Caltech256: ["caltech256"],
    tvd.CelebA: ["celeba"],
    tvd.Cityscapes: ["cityscapes"],
    tvd.INaturalist: ["inat"],
    tvd.Places365: ["places365"],
    tvd.STL10: ["stl10"],
    tvd.SVHN: ["SVHN"],
    tvd.CocoDetection: ["annotations", "test2017", "train2017", "val2017"],
}
""" A map of the folder associated with each dataset type, relative to the `torchvision_dir` of a
cluster. This is also equivalent to the name of the folder that would be downloaded when using the
dataset with `download=True`.
"""

too_large_for_slurm_tmpdir: set[Callable] = set()
""" Set of datasets which are too large to store in $SLURM_TMPDIR.

NOTE: Unused atm.
"""

# TODO: Use types to also indicate that the 'root' argument becomes optional.


def make_dataset_fn(dataset_type: Callable[P, D] | type[D]) -> Callable[P, D]:
    """Wraps a dataset into a function that constructs it efficiently.

    When invoked, the function first checks if the dataset is already downloaded in $SLURM_TMPDIR.
    If so, it is read and returned. If not, checks if the dataset is already stored somewhere in
    the cluster ($SCRATCH/data, then /network/datasets directory). If so, try to copy it over to
    SLURM_TMPDIR. If that works, read the dataset from SLURM_TMPDIR if the dataset isn't too large.
    If the dataset isn't found anywhere, then it is downloaded to SLURM_TMPDIR and read from there.
    """

    if dataset_type not in known_dataset_files:
        warnings.warn(
            RuntimeWarning(
                f"Don't know which files are associated with dataset type {dataset_type}!"
                f"Consider adding the names of the files to the dataset_files dictionary."
            )
        )
        return dataset_type
    required_files = known_dataset_files[dataset_type]
    fast_dir = SLURM_TMPDIR / "data"
    scratch_dir = SCRATCH / "data"
    dataset_type = cast(Callable[P, D], dataset_type)
    load_from_fast_dir = replace_kwargs(dataset_type, root=fast_dir)

    def copy_and_load(source_dir: Path):
        """Returns a function that copies the dataset files from `source_dir` to `fast_dir` and
        then loads the dataset from `fast_dir`."""

        @functools.wraps(dataset_type)
        def _copy_and_load(*args: P.args, **kwargs: P.kwargs) -> D:
            copy_dataset_files(required_files, source_dir, fast_dir)
            return load_from_fast_dir(*args, **kwargs)

        return _copy_and_load

    if all_files_exist(required_files, fast_dir):
        return load_from_fast_dir
    if all_files_exist(required_files, scratch_dir):
        if dataset_type in too_large_for_slurm_tmpdir:
            return replace_kwargs(dataset_type, root=scratch_dir)
        else:
            return copy_and_load(scratch_dir)
    if all_files_exist(required_files, CURRENT_CLUSTER.torchvision_dir):
        if dataset_type in too_large_for_slurm_tmpdir:
            return replace_kwargs(dataset_type, root=CURRENT_CLUSTER.torchvision_dir)
        else:
            return copy_and_load(CURRENT_CLUSTER.torchvision_dir)

    def _try_slurm_tmpdir(*args: P.args, **kwargs: P.kwargs) -> D:
        try:
            return load_from_fast_dir(*args, **kwargs)
        except OSError:
            # Fallback to the normal callable (the passed in class)
            return dataset_type(*args, **kwargs)

    return _try_slurm_tmpdir


def adapted_constructor(
    cls: Callable[Concatenate[str, P], D]
) -> Callable[Concatenate[str | None, P], None]:
    dataset_type = cls
    if dataset_type not in known_dataset_files:
        raise NotImplementedError(
            f"Don't know which files are associated with dataset type {dataset_type}!"
            f"Consider adding the names of the files to the dataset_files dictionary."
        )
    required_files = known_dataset_files[dataset_type]

    def _custom_init(self, *args: P.args, **kwargs: P.kwargs):
        assert isinstance(cls, type)
        # base_init = super(cls, self).__init__
        base_init = cls.__init__
        bound_args = inspect.signature(base_init).bind_partial(self, *args, **kwargs)
        fast_tmp_dir = SLURM_TMPDIR / "data"
        scratch_dir = SCRATCH / "data"

        if all_files_exist(required_files, fast_tmp_dir):
            bound_args.arguments["root"] = fast_tmp_dir
        elif all_files_exist(required_files, scratch_dir):
            if dataset_type not in too_large_for_slurm_tmpdir:
                logger.info("Copying files from SCRATCH to SLURM_TMPDIR")
                copy_dataset_files(required_files, scratch_dir, fast_tmp_dir)
                bound_args.arguments["root"] = fast_tmp_dir
            else:
                logger.info("Dataset is large, files will be read from SCRATCH.")
                bound_args.arguments["root"] = scratch_dir
        elif all_files_exist(required_files, CURRENT_CLUSTER.torchvision_dir):
            logger.info("Copying files from the torchvision dir to SLURM_TMPDIR")
            copy_dataset_files(required_files, CURRENT_CLUSTER.torchvision_dir, fast_tmp_dir)
            bound_args.arguments["root"] = CURRENT_CLUSTER.torchvision_dir

        args = bound_args.args  # type: ignore
        kwargs = bound_args.kwargs  # type: ignore
        base_init(*args, **kwargs)

    return _custom_init
