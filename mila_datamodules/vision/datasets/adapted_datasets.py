"""Wrappers around the torchvision datasets, adapted to the current cluster."""
from __future__ import annotations

import functools
import warnings
from logging import getLogger as get_logger
from typing import Callable, Generic, Type, TypeVar, cast

from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.clusters.utils import on_slurm_cluster
from mila_datamodules.vision.datasets.prepare_dataset import prepare_dataset

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

---------

Previous approach:

This constructor does a few things, but basically works like a three-level caching system.
1. /network/datasets/torchvision (read-only "cache")
2. $SCRATCH/cache/torch (writeable "cache")
3. $SLURM_TMPDIR/data (writeable "cache", fastest storage).

If the dataset isn't found on the cluster, it will be downloaded in $SCRATCH
If the dataset fits in $SLURM_TMPDIR, it will be copied from wherever it is, and placed
there.
The dataset is then read from SLURM_TMPDIR.



"""


T = TypeVar("T", bound=type)
D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
C = Callable[P, D]

VD = TypeVar("VD", bound=VisionDataset)
VD_cov = TypeVar("VD_cov", bound=VisionDataset, covariant=True)
VD_cot = TypeVar("VD_cot", bound=VisionDataset, contravariant=True)

logger = get_logger(__name__)
# TODO: Proved not to be necessary in the end. Probably will be removed.
# from ._torchvision_checksum_patch import apply_patch
# apply_patch()


def _cache(fn: C) -> C:
    return functools.cache(fn)  # type: ignore


# NOTE: Turning off this auto-generation of adapted datasets for now, just so we're forced to be
# explicit about which datasets are supported or not.
# TODO: Need to turn this back on so the adapted dataset classes can be pickled.


# def __getattr__(name: str) -> type[AdaptedDataset | VisionDataset]:
#     pass

#     matching_dataset_class = getattr(torchvision.datasets, name, None)
#     if inspect.isclass(matching_dataset_class) and issubclass(
#         matching_dataset_class, VisionDataset
#     ):
#         matching_dataset_class = matching_dataset_class
#         current_cluster: Cluster | None = Cluster.current()
#         # note: Do nothing if we don't have this dataset on the cluster?
#         if not is_stored_on_cluster(matching_dataset_class):
#             warnings.warn(
#                 UserWarning(
#                     f"Not dynamically adapting dataset class {matching_dataset_class}, since "
#                     + (
#                         f"it is not detected as being stored on the {current_cluster.name} cluster. "
#                         if current_cluster
#                         else "this is apparently not being executed on a SLURM cluster. "
#                     )
#                     + "Returning the class from torchvision instead. If you believe this is a "
#                     "mistake, please make an issue on the mila_datamodules repo. at "
#                     + get_github_issue_url(
#                         dataset_name=matching_dataset_class.__name__,
#                         cluster_name=current_cluster.name if current_cluster else "local",
#                     )
#                 )
#             )
#             return matching_dataset_class
#         warnings.warn(
#             UserWarning(
#                 f"Dynamically creating an adapter for dataset class {matching_dataset_class}, which "
#                 f"is not explicitly supported on this cluster. Your mileage may vary."
#             )
#         )
#         return adapt_dataset(matching_dataset_class)
#     raise AttributeError(name)


# TODO: This 'VD' typevar here is actually not quite correct. We don't need to be adapting a
# VisionDataset per-se, all we care about is that it has `root` as a constructor argument.


class AdaptedDataset(VisionDataset, Generic[VD]):
    original_class: type[VD]
    """The original dataset class that this adapted dataset 'wraps'."""

    def prepare_dataset(self, root: str | None = None, *args, **kwargs) -> str:
        """Called before the original dataset constructor is called."""
        raise NotImplementedError
        # return prepare_dataset(self, root, *args, **kwargs)

    def __init__(self, root: str | None = None, *args, **kwargs) -> None:
        if not on_slurm_cluster():
            return super().__init__(root=root, *args, **kwargs)

        # Call the optimized dataset preparation routine (which may be as simple as just replacing
        # the `root` parameter) before actually calling __init__ with the original dataset.
        new_root = self.prepare_dataset(root=root, *args, **kwargs)
        assert new_root is not None
        assert new_root != "None"
        logger.info(f"New root for {type(self).__name__}: {new_root} ({root=})")
        if root is not None and new_root != root:
            warnings.warn(
                RuntimeWarning(
                    f"Ignoring passed 'root' argument: {root}, using {new_root} instead."
                )
            )
        super().__init__(new_root, *args, **kwargs)


def adapt_dataset(dataset_class: Callable[Concatenate[str, P], VD]) -> type[AdaptedDataset[VD]]:
    """Creates an optimized version of the given dataset for the current SLURM cluster.

    Returns a subclass of the given dataset class and of an adapter.

    This is basically equivalent to this:
    ```python
    class CIFAR10(AdaptedDataset, torchvision.datasets.CIFAR10):
        original_class: ClassVar[type[torchvision.datasets.CIFAR10]] = torchvision.datasets.CIFAR10
    ```
    """
    # Dynamically create a subclass of the adapter and the original class, so that the adapter's
    # __init__ takes precedence in the super().__init__ mro.
    dataset_subclass = type(
        dataset_class.__name__,
        (
            AdaptedDataset,
            dataset_class,
        ),
        {},
    )
    dataset_subclass = cast(Type[AdaptedDataset[VD]], dataset_subclass)
    dataset_subclass.original_class = dataset_class  # type: ignore
    return dataset_subclass


from mila_datamodules.vision.datasets.prepare_dataset import prepare_dataset  # noqa


@prepare_dataset.register(AdaptedDataset)
def _dispatch_adapted_dataset(dataset: AdaptedDataset[VD], *args, **kwargs):
    raise RuntimeError(
        "You should not pass wrapped dataset classes to `prepare_dataset`. "
        "Use the original dataset class instead."
    )
