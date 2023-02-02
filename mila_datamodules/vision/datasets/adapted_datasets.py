"""Wrappers around the torchvision datasets, adapted to the current cluster."""
from __future__ import annotations

import functools
import inspect
import os
import textwrap
import warnings
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, Type, TypeVar, cast

import torchvision.datasets as tvd
import torchvision.datasets.utils
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import calculate_md5
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.clusters import get_slurm_tmpdir
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.registry import (
    files_to_copy_for_dataset,
    is_stored_on_cluster,
    locate_dataset_root_on_cluster,
)

T = TypeVar("T", bound=type)
D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
C = Callable[P, D]

VD = TypeVar("VD", bound=VisionDataset)
VD_cov = TypeVar("VD_cov", bound=VisionDataset, covariant=True)
VD_cot = TypeVar("VD_cot", bound=VisionDataset, contravariant=True)

logger = get_logger(__name__)


def _cache(fn: C) -> C:
    return functools.cache(fn)  # type: ignore


# NOTE: Turning off this auto-generation of adapted datasets for now, just so we're forced to be
# explicit about which datasets are supported or not.
# TODO: Need to turn this back on so the adapted dataset classes can be pickled.


def __getattr__(name: str) -> type[AdaptedDataset | VisionDataset]:
    import torchvision.datasets as tvd

    if hasattr(tvd, name):
        attribute = getattr(tvd, name)

        if inspect.isclass(attribute) and issubclass(attribute, VisionDataset):
            dataset_class = attribute
            # note: Do nothing if we don't have this dataset on the cluster?
            if not is_stored_on_cluster(dataset_class):
                return dataset_class
            warnings.warn(
                UserWarning(
                    f"Dynamically creating an adapter for dataset class {dataset_class}, which "
                    f"is not explicitly supported on this cluster. "
                )
            )
            return adapt_dataset(dataset_class)
    raise AttributeError(name)


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    actual_md5 = calculate_md5(fpath, **kwargs)
    expected = md5
    if actual_md5 == expected:
        logger.debug(f"MD5 checksum of {fpath} matches expected value: {expected}")
        return True
    else:
        logger.debug(f"MD5 checksum of {fpath} does not match expected value: {expected}!")
        return False


def check_integrity(fpath: str, md5: str | None = None) -> bool:
    logger.debug(f"Using our patched version of `check_integrity` on path {fpath}")
    path = Path(fpath).resolve()
    while path.is_symlink():
        logger.debug(f"Following symlink for {path} instead of redownloading dataset!")
        path = path.readlink()
        logger.debug(f"Resolved path: {path}")
    if not os.path.isfile(fpath):
        logger.debug(f"{fpath} is still not real path?!")
        return False
    if md5 is None:
        logger.debug(f"no md5 check for {fpath}!")
        return True
    return check_md5(fpath, md5)


setattr(torchvision.datasets.utils, "check_integrity", check_integrity)
setattr(torchvision.datasets.utils, "check_md5", check_md5)


@functools.singledispatch
def prepare_dataset_before_init(
    dataset: AdaptedDataset[VD, P],
    root: str | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
):
    """Called before `dataset` is initialized. The constructor arguments are passed.

    This is where the dataset is copied/extracted to the optimal location on the cluster
    (usually a directory in $SLURM_TMPDIR).

    Returns the new/corrected value to use for the 'root' argument.

    NOTE: This shouldn't depend on any attributes to be set in `dataset`, since its __init__ hasn't
    been called yet.
    """
    example = textwrap.dedent(
        """\
        @prepare_dataset_before_init.register(MyDataset)
        def setup_my_dataset(dataset: MyDataset, root: str | None = None, *args, **kwargs) -> str:
            ''' prepares the optimal dataset folder and returns its location to be used as 'root'.

            This should ideally extract / copy / symlink stuff into SLURM_TMPDIR, so that loading
            the dataset can be very quick later.
            '''
        """
    )
    raise NotImplementedError(
        f"Don't have a registered way to prepare the dataset {type(dataset).__name__}!\n"
        f"Consider registering a function with @prepare_dataset_before_init.register, like so:\n\n"
        + example
    )


@prepare_dataset_before_init.register(tvd.CIFAR10)
@prepare_dataset_before_init.register(tvd.CIFAR100)
@prepare_dataset_before_init.register(tvd.MNIST)
@prepare_dataset_before_init.register(tvd.FashionMNIST)
def read_from_datasets_directory(
    dataset: AdaptedDataset[VD, P],
    root: str | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> str | None:
    cluster = Cluster.current_or_error()
    dataset_class = dataset.original_class
    downloaded_dataset_root = locate_dataset_root_on_cluster(
        dataset_cls=dataset_class, cluster=cluster
    )
    # TODO: Should we be guaranteed that the dataset is already downloaded on the cluster at this
    # point?
    assert is_stored_on_cluster(dataset_class, cluster=cluster)
    if downloaded_dataset_root is None:
        logger.warning(
            f"Unable to find a downloaded version of {dataset_class.__name__} on {cluster.name} cluster."
        )
        return root
    if kwargs.get("download"):
        warnings.warn(
            UserWarning(
                f"Not downloading the {dataset_class.__name__} dataset, since it is "
                f"already stored on the cluster at {downloaded_dataset_root}",
            )
        )
    logger.info(f"Dataset {dataset_class} will be read directly from {downloaded_dataset_root}.")
    return downloaded_dataset_root


@prepare_dataset_before_init.register(tvd.Places365)
@prepare_dataset_before_init.register(tvd.ImageFolder)
def make_symlinks_to_archives_in_tempdir(
    dataset: AdaptedDataset[VD, Concatenate[str, P]],
    root: str | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> str | None:
    """Prepare the dataset folder by creating symlinks to the archives inside $SLURM_TMPDIR.

    The dataset constructor should then extract the archives in $SLURM_TMPDIR and load the dataset.
    """
    dataset_class = dataset.original_class
    cluster = Cluster.current_or_error()

    fast_data_dir = get_slurm_tmpdir() / "data" / dataset_class.__name__
    fast_data_dir.parent.mkdir(exist_ok=True)
    fast_data_dir.mkdir(exist_ok=True)

    # FIXME: Fix this: hard-coded for Places365 while debugging.
    # files_required_for_dataset = files_to_copy(dataset_class=dataset_class, cluster=cluster)
    # assert False, files_required_for_dataset
    # TODO: Get the files and archives that are required for the dataset.
    files = files_to_copy_for_dataset(dataset_type=dataset_class, cluster=cluster, *args, **kwargs)

    # dataset_archives_root = Path("/network/datasets/places365")
    # extracted_dataset_root = Path("/network/datasets/places365.var/places365_torchvision")
    # # TODO: Need to copy all metadata files from the torchvision dir, but the archives are in the
    # # original root.

    # archives_to_link = list(glob_any(dataset_archives_root, ["*.tar", "*/*.tar"]))
    # archives_to_link = archives_in_dir(dataset_archives_root, recurse=False)

    # other_files_to_link = list(glob_any(extracted_dataset_root, metadata_file_formats))
    # # TODO: How to also programmatically include the text files and such?
    # files = archives_to_link + other_files_to_link

    for path_relative_to_storage_dir, file_in_network_storage in files:
        # TODO: Do we want to preserve the directory structure of the files? Or flatten things?
        # For now, I'll try a flat structure.
        dest = fast_data_dir / file_in_network_storage.name
        # dest = dest_dir / path_relative_to_storage_dir

        if dest.exists():
            continue
        dest.symlink_to(file_in_network_storage)
        logger.debug(f"Creating link to archive: {dest} -> {file_in_network_storage}")
        # shutil.copyfile(src=file, dst=dest, follow_symlinks=True)

    return str(fast_data_dir)
    # kwargs["download"] = False

    # d = tvd.Places365(root=str(fast_data_dir), *args, **kwargs)

    # raise NotImplementedError(
    #     f"TODO: Move/symlink the archives for dataset {dataset_class} in SLURM_TMPDIR, and "
    #     f"hopefully torchvision does the rest without downloading anything."
    # )


class AdaptedDataset(VisionDataset, Generic[VD_cov, P]):
    original_class: ClassVar[type[VisionDataset]]
    """The original dataset class that this adapted dataset 'wraps'."""

    def _pre_init_(self, root: str | None = None, *args: P.args, **kwargs: P.kwargs) -> str:
        """Called before the dataset is initialized. The constructor arguments are passed.

        This is where the dataset is copied/extracted to the optimal location on the cluster
        (usually a directory in $SLURM_TMPDIR).

        Returns the new/corrected value to use for the 'root' argument.

        NOTE: Also making it a class method here, so that users can more easily customize the
        pre-init behaviour for a particular dataset, by creating a subclass of the adapted dataset
        and overriding this method.
        """
        new_root = prepare_dataset_before_init(self, root=root, *args, **kwargs)
        return new_root

    def __init__(self, root: str | None = None, *args: P.args, **kwargs: P.kwargs) -> None:
        # Creates the dataset, but invoking the _pre_init_() method first.

        new_root = self._pre_init_(root=root, *args, **kwargs)
        logger.info(f"New root for {self.original_class.__name__}: {new_root} ({root=})")
        if root is not None and new_root != root:
            warnings.warn(
                RuntimeWarning(
                    f"Ignoring passed 'root' argument: {root}, using {new_root} instead."
                )
            )
        super().__init__(new_root, *args, **kwargs)


_dataset_adapters: dict[
    type[VisionDataset] | Callable[..., VisionDataset], type[AdaptedDataset]
] = {}


def adapt_dataset(dataset_class: Callable[Concatenate[str, P], VD]) -> type[AdaptedDataset[VD, P]]:
    """Creates an optimized version of the given dataset for the current SLURM cluster.

    Returns a subclass of the given dataset class and of an adapter class.
    This is basically equivalent to this:
    ```python
    adapter_class = _dataset_adapters.get(dataset_class, AdaptedDataset)

    class AdaptedVersion(adapter_class, dataset_class):
        original_class: ClassVar[type[VisionDataset]] = dataset_class

    return AdaptedVersion
    ```


    The default adapter


    This constructor does a few things, but basically works like a three-level caching system.
    1. /network/datasets/torchvision (read-only "cache")
    2. $SCRATCH/cache/torch (writeable "cache")
    3. $SLURM_TMPDIR/data (writeable "cache", fastest storage).

    If the dataset isn't found on the cluster, it will be downloaded in $SCRATCH
    If the dataset fits in $SLURM_TMPDIR, it will be copied from wherever it is, and placed
    there.
    The dataset is then read from SLURM_TMPDIR.
    """
    dataset_adapted_class: type[AdaptedDataset[VD, P]] = _dataset_adapters.get(
        dataset_class, AdaptedDataset
    )
    dataset_subclass = type(
        dataset_class.__name__,
        (
            dataset_adapted_class,
            dataset_class,
        ),
        {},  # dataset_type.__dict__,
        # {"__init__": adapted_constructor(dataset_type)},
    )
    dataset_subclass = cast(Type[AdaptedDataset[VD, P]], dataset_subclass)
    dataset_subclass.original_class = dataset_class  # type: ignore
    return dataset_subclass


# TODOS:
"""
- For datasets that fit in RAM, (or are loaded into RAM anyway) (e.g. cifar10/cifar100), we should
  just read them from /network/datasets/torchvision, and not bother copying them to SLURM_TMPDIR.
- For datasets that don't fit in RAM, extract the archive directly to $SLURM_TMPDIR.
  NOTE: Might need to also create a symlink of the archive in $SLURM_TMPDIR so that the tvd Dataset
  constructor doesn't re-download it to SLURM_TMPDIR.
- NOTE: No speedup reading from $SCRATCH or /network/datasets. Same filesystem
For ComputeCanada:
- Extract the archive from the datasets folder to $SLURM_TMPDIR without copying.

In general, for datasets that don't fit in SLURM_TMPDIR, we should use $SCRATCH as the
"SLURM_TMPDIR".
NOTE: setting --tmp=800G is a good idea if you're going to move a 600gb dataset to SLURM_TMPDIR.
"""
