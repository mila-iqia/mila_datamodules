"""Wrappers around torchvision datasets that use a good default value for the 'root' argument based
on the current cluster."""
from __future__ import annotations

import functools
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, TypeVar

import torchvision.datasets as tvd
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing_extensions import ParamSpec

from mila_datamodules.clusters import CURRENT_CLUSTER, SCRATCH, SLURM_TMPDIR
from mila_datamodules.clusters.utils import (
    all_files_exist,
    copy_dataset_files,
    replace_kwargs,
)

P = ParamSpec("P")
D = TypeVar("D", bound=Dataset)
VD = TypeVar("VD", bound=VisionDataset)
C = Callable[P, D]

logger = get_logger(__name__)


known_dataset_files = {
    tvd.MNIST: ["MNIST"],
    tvd.CIFAR10: ["cifar-10-batches-py"],
    tvd.CIFAR100: ["cifar-100-python"],
    tvd.FashionMNIST: ["FashionMNIST"],
}
""" A map of the files for each dataset type, relative to the `torchvision_dir`. """

too_large_for_slurm_tmpdir: set[Callable] = set()
""" Set of datasets which are too large to store in $SLURM_TMPDIR.

NOTE: Unused atm.
"""


def adapt_dataset(dataset_type: Callable[P, D]) -> Callable[P, D]:
    """Check if the dataset is already downloaded in $SLURM_TMPDIR.

    If so, read and return it. If not, check if the dataset is already stored somewhere in the
    cluster. If so, try to copy it over to SLURM_TMPDIR. If that works, read the dataset from
    SLURM_TMPDIR. If not, then download the dataset to the fast directory (if possible), and read
    it from there.
    """

    fastest_load_fn = replace_kwargs(dataset_type, root=SLURM_TMPDIR / "data")
    if dataset_type not in known_dataset_files:
        raise NotImplementedError(
            f"Don't know which files are associated with dataset type {dataset_type}!"
            f"Consider adding the names of the files to the dataset_files dictionary."
        )
    required_files = known_dataset_files[dataset_type]
    # raise NotImplementedError(
    #     "TODO: This shouldn't run anything at import time. Files / etc should only be copied at "
    #     "runtime."
    # )
    if all_files_exist(required_files, SLURM_TMPDIR / "data"):
        return fastest_load_fn
    if all_files_exist(required_files, SCRATCH / "data"):

        @functools.wraps(dataset_type)
        def _wrap(*args: P.args, **kwargs: P.kwargs) -> D:
            copy_dataset_files(required_files, SCRATCH / "data", SLURM_TMPDIR / "data")
            return fastest_load_fn(*args, **kwargs)

        return _wrap

    if all_files_exist(required_files, CURRENT_CLUSTER.torchvision_dir):

        @functools.wraps(dataset_type)
        def _wrap(*args: P.args, **kwargs: P.kwargs) -> D:
            copy_dataset_files(
                required_files, CURRENT_CLUSTER.torchvision_dir, SLURM_TMPDIR / "data"
            )
            return fastest_load_fn(*args, **kwargs)

        # TODO: Check if this is actually worth doing. (I'm thinking probably not.)
        # _copy_dataset_files(dataset_type, SLURM_TMPDIR, SCRATCH)
        return _wrap

    def _wrap(*args: P.args, **kwargs: P.kwargs) -> D:
        try:
            return fastest_load_fn(*args, **kwargs)
        except OSError:
            # Fallback to the normal callable (the passed in class)
            return dataset_type(*args, **kwargs)

    return _wrap
