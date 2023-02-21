from __future__ import annotations

import functools
import inspect
import textwrap
import warnings
from logging import getLogger as get_logger
from typing import Callable, TypeVar

import torchvision.datasets as tvd
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.clusters import get_slurm_tmpdir
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.registry import (
    files_to_use_for_dataset,
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
# TODO: It's a bit weird to require an un-initialized object as input, makes it hard to unit test.
# It might be better to instead accept the type of dataset as an input, but then we'd need to
# figure out how to dispatch.
# OR, we could just have a dictionary. But then we lose the benefits of the dispatching, e.g. to
# have one function for all the ImageFolder datasets, etc.


@functools.singledispatch
def prepare_dataset(
    dataset: Callable[Concatenate[str, P], VD] | VD,
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


@prepare_dataset.register(type)
def _dispatch_dataset_class(dataset_type: type[VisionDataset], *args, **kwargs):
    fake_instance = dataset_type.__new__(dataset_type)
    return prepare_dataset.dispatch(dataset_type)(fake_instance, *args, **kwargs)


@prepare_dataset.register(tvd.CIFAR10)
@prepare_dataset.register(tvd.CIFAR100)
@prepare_dataset.register(tvd.MNIST)
@prepare_dataset.register(tvd.FashionMNIST)
def read_from_datasets_directory(
    dataset: VisionDataset, root: str | None = None, *args, **kwargs
) -> str | None:
    dataset_class = type(dataset)
    cluster = Cluster.current_or_error()
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


@prepare_dataset.register(tvd.Places365)
@prepare_dataset.register(tvd.ImageFolder)
def make_symlinks_to_archives_in_tempdir(
    dataset: VD, root: str | None = None, *constructor_args, **constructor_kwargs
) -> str | None:
    """Prepare the dataset folder by creating symlinks to the archives inside $SLURM_TMPDIR.

    The dataset constructor should then extract the archives in $SLURM_TMPDIR and load the dataset.

    TODO: The dataset constructor often only extracts the archives if the `download` parameter is
    set to True!
    """
    dataset_class = type(dataset)
    cluster = Cluster.current_or_error()

    fast_data_dir = get_slurm_tmpdir() / "data" / dataset_class.__name__
    fast_data_dir.parent.mkdir(exist_ok=True)
    fast_data_dir.mkdir(exist_ok=True)

    files = files_to_use_for_dataset(
        dataset_class, cluster=cluster, *constructor_args, **constructor_kwargs
    )
    assert files
    for path_relative_to_storage_dir, file_in_network_storage in files.items():
        # TODO: Do we want to preserve the directory structure of the files? Or flatten things?
        # For now, I'll try a flat structure.
        dest = fast_data_dir / file_in_network_storage.name
        # dest = dest_dir / path_relative_to_storage_dir

        if dest.exists():
            continue
        dest.symlink_to(file_in_network_storage)
        logger.debug(f"{dest} -> {file_in_network_storage}")

    # TODO: We probably need to call the dataset constructor with download=True, so that it
    # actually extracts the dataset files from the archives.
    if "download" in inspect.signature(dataset_class.__init__).parameters:
        kwargs = constructor_kwargs.copy()
        kwargs["download"] = True
        logger.info(
            f"Calling {dataset_class.__name__}.__init__ with download=True once, to extract the "
            f"archives to {fast_data_dir}."
        )
        dataset_class(root=str(fast_data_dir), *constructor_args, **kwargs)

    return str(fast_data_dir)


from .prepare_imagenet import prepare_imagenet_dataset  # noqa
