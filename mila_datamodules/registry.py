"""A registry of where each dataset is stored on each cluster."""
from __future__ import annotations

import functools
import inspect
import textwrap
import warnings
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import pl_bolts.datasets
import torchvision.datasets as tvd
from torch.utils.data import Dataset
from typing_extensions import ParamSpec

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_scratch_dir
from mila_datamodules.errors import (
    DatasetNotFoundOnClusterError,
    UnsupportedDatasetError,
)
from mila_datamodules.utils import (
    _get_key_to_use_for_indexing,
    getitem_with_subclasscheck,
)
from mila_datamodules.vision.datasets._binary_mnist import _PatchedBinaryMNIST
from mila_datamodules.vision.datasets._utils import archives_in_dir

# TODO: Redesign this whole registry idea.

logger = get_logger(__name__)
_dataset_files = {
    tvd.MNIST: ["MNIST"],
    tvd.CIFAR10: ["cifar-10-batches-py"],
    tvd.CIFAR100: ["cifar-100-python"],
    tvd.FashionMNIST: ["FashionMNIST"],
    tvd.Caltech101: ["caltech101"],
    tvd.Caltech256: ["caltech256"],
    tvd.CelebA: ["celeba"],
    tvd.Cityscapes: ["leftImg8bit", "gtFine", "gtCoarse"],
    tvd.INaturalist: ["2021_train", "2021_train_mini", "2021_valid"],
    tvd.Places365: [
        "categories_places365.txt",
        "places365_train_standard.txt",
        "data_large_standard",
        "places365_test.txt",
        "places365_val.txt",
        "val_large",
    ],
    tvd.STL10: ["stl10_binary"],
    tvd.SVHN: ["train_32x32.mat", "test_32x32.mat", "extra_32x32.mat"],
    # NOTE: We're not currently using the `test` folder for the coco datasets, since there isn't
    # an annotation file with it. (this makes sense, you probably have to submit predictions to the
    # website or something).
    tvd.CocoDetection: ["annotations", "test2017", "train2017", "val2017"],
    tvd.CocoCaptions: ["annotations", "test2017", "train2017", "val2017"],
    tvd.EMNIST: ["EMNIST"],
    # NOTE: This isn't quite true for the original class.
    # pl_bolts.datasets.BinaryMNIST: ["MNIST"],
    _PatchedBinaryMNIST: ["MNIST"],
    pl_bolts.datasets.BinaryMNIST: ["MNIST"],
    pl_bolts.datasets.BinaryEMNIST: ["EMNIST"],
    tvd.FlyingChairs: ["FlyingChairs/data", "FlyingChairs/FlyingChairs_train_val.txt"],
    tvd.CLEVRClassification: ["clevr"],
    tvd.Country211: ["country211"],
    tvd.DTD: ["dtd/images", "dtd/labels", "dtd/imdb"],
    tvd.EuroSAT: ["eurosat"],
    tvd.FER2013: ["fer2013"],
    tvd.FGVCAircraft: ["fgvc-aircraft-2013b"],
    tvd.CarlaStereo: ["carla-highres"],
    tvd.Flickr8k: ["flickr8k"],
    tvd.Food101: ["food-101/images", "food-101/meta"],
    tvd.GTSRB: ["gtsrb"],
    # TODO: Double-check this one.
    tvd.HMDB51: ["hmdb51"],
    # TODO: Not sure if we want to actually include these here: They are datasets for which we
    # should have archives, and that generate these ugly generic names on extraction.
    tvd.ImageNet: ["train", "val"],
    tvd.Kinetics: ["split"],
    tvd.Kitti: ["Kitti/raw/training", "Kitti/raw/testing"],
    # TODO: Double-check this one.
    tvd.Kitti2012Stereo: ["Kitti2012/testing", "Kitti2012/training"],
    tvd.KittiFlow: ["KittiFlow/testing", "KittiFlow/training"],
    tvd.LFWPairs: ["lfw-py"],
    tvd.LFWPeople: ["lfw-py"],
    tvd.LSUN: ["*_lsun"],  # TODO: Double-check this one.
    tvd.LSUNClass: ["*_lsun"],  # TODO: Double-check this one.
    tvd.KMNIST: ["KMNIST"],
    tvd.QMNIST: ["QMNIST"],
    tvd.Omniglot: ["omniglot-py"],
    tvd.OxfordIIITPet: ["oxford-iiit-pet"],
    tvd.PCAM: ["pcam"],
    tvd.RenderedSST2: ["rendered-sst2"],
    tvd.SBDataset: ["img", "cls"],
    tvd.SBU: ["SBUCaptionedPhotoDataset.tar.gz"],
    # note: disagrees with the docstring, but seems correct based on the code.
    tvd.SEMEION: ["semeion.data"],
    tvd.SUN397: ["SUN397"],
    # TODO: Not sure what this downloads. Seems to be just ImageFolder with the root, so unclear what files are required in this case..
    tvd.UCF101: ["ucf101"],
}
"""A map of the folder/files associated with each dataset type, relative to the `root_dir`. This is
roughly the list of files/folders that would be downloaded when creating the dataset with
`download=True`.

NOTE: An entry being in this dict *does not* mean that this dataset is available on the cluster!
This is simply a list of the files that are expected to be present in the `root` directory of each
dataset type in order for it to work.

These files are copied over to $SCRATCH or $SLURM_TMPDIR before the dataset is read, depending on
the type of dataset.
"""


_mila_torchvision_dir = "/network/datasets/torchvision"
_beluga_curated_datasets_dir = "/project/rpp-bengioy/data/curated"

# TODO: Fill this in!
# TODO: Also allow configuring / updating these using a config file instead of having to hard-code
# things.
# Add the known dataset locations on the mila cluster.

dataset_roots_per_cluster = {
    Cluster.Mila: {
        tvd.MNIST: _mila_torchvision_dir,
        tvd.MNIST: _mila_torchvision_dir,
        tvd.CIFAR10: _mila_torchvision_dir,
        tvd.Caltech101: _mila_torchvision_dir,
        tvd.Caltech256: _mila_torchvision_dir,
        tvd.CelebA: _mila_torchvision_dir,
        tvd.Cityscapes: _mila_torchvision_dir,
        tvd.CocoCaptions: _mila_torchvision_dir,
        tvd.CocoDetection: _mila_torchvision_dir,
        tvd.CIFAR100: _mila_torchvision_dir,
        tvd.FashionMNIST: _mila_torchvision_dir,
        tvd.INaturalist: _mila_torchvision_dir,
        tvd.Places365: _mila_torchvision_dir,
        tvd.STL10: _mila_torchvision_dir,
        tvd.SVHN: _mila_torchvision_dir,
        # NOTE: BinaryMNIST from pl_bolts.datasets is buggy, and doesn't work out-of-the-box.
        # Our adapted version works though.
        # pl_bolts.datasets.BinaryMNIST,
        _PatchedBinaryMNIST: _mila_torchvision_dir,
    },
    Cluster.Beluga: {
        tvd.MNIST: "/project/rrp-bengioy/data",
        tvd.CIFAR10: _beluga_curated_datasets_dir,
        tvd.Cityscapes: _beluga_curated_datasets_dir,
        tvd.CocoCaptions: _beluga_curated_datasets_dir,
        tvd.CocoDetection: _beluga_curated_datasets_dir,
        _PatchedBinaryMNIST: "/project/rrp-bengioy/data",
    },
}
"""For each cluster, for each type of dataset, the value of `root` to use to load the dataset."""

P = ParamSpec("P")


def _rel_paths_and_paths(
    root: str | Path, fn: Callable[[Path], Iterable[Path]]
) -> Iterable[tuple[str, Path]]:
    root = Path(root)
    for path in fn(root):
        yield str(path.relative_to(root)), path


def list_archives_in(
    archive_dir: str | Path | list[str] | list[Path], recurse: bool = False
) -> Callable[[], Iterable[tuple[str, Path]]]:
    """Function that, when invoked, lists out the archives in the given directory."""
    if isinstance(archive_dir, (str, Path)):
        archive_dirs = [archive_dir]
    else:
        archive_dirs = archive_dir

    def _list_archives():
        for archive_dir in archive_dirs:
            yield from _rel_paths_and_paths(
                archive_dir, fn=functools.partial(archives_in_dir, recurse=recurse)
            )

    return _list_archives


_files_to_use_for_dataset: dict[type, dict[Cluster, Callable[[], Iterable[tuple[str, Path]]]]] = {
    tvd.Places365: {
        Cluster.Mila: list_archives_in(
            [
                "/network/datasets/places365",
                "/network/datasets/places365/large",
            ]
        )
    },
    tvd.ImageNet: {
        Cluster.Mila: list_archives_in("/network/datasets/imagenet"),
        Cluster.Beluga: list_archives_in("/project/rpp-bengioy/data/curated/imagenet/"),
    },
}
"""Dictionary that gives, for each dataset type, for each cluster, a function that returns all the
files and/or archives that should be copied over to the SLURM_TMPDIR before the dataset is read
from there.

This is so that torchvision then extracts the files from the archives and reads them from there.
"""


def files_to_use_for_dataset(
    dataset: Callable[P, Dataset],
    cluster: Cluster,
    *constructor_args: P.args,
    **constructor_kwargs: P.kwargs,
) -> dict[str, Path] | None:
    """Returns the files that should be moved for the given dataset type on the current cluster.

    Note: This is only really applicable to datasets that should be read from archives, e.g.
    ImageNet.
    """
    from mila_datamodules.vision.datasets.adapted_datasets import AdaptedDataset

    if inspect.isclass(dataset) and issubclass(dataset, AdaptedDataset):
        dataset = dataset.original_class
    assert inspect.isclass(dataset)
    files_per_cluster = getitem_with_subclasscheck(
        potential_classes=_files_to_use_for_dataset, key=dataset, default=None
    )
    if files_per_cluster is None:
        logger.debug(f"We don't know which files should be copied for {dataset} yet.")
        return None

    if cluster not in files_per_cluster:
        warnings.warn(
            RuntimeWarning(
                f"We don't know which files should be copied for {dataset} on {cluster} yet"
            )
        )
        return None
    return dict(files_per_cluster[cluster]())


# Probably remove this?
dataset_archives_locations_per_cluster: dict[Cluster, dict[type, list[str]]] = {}
#     Cluster.Mila: {
#         # tvd.Places365: [
#         #     "/network/datasets/places365/256/train_256_places365standard.tar",
#         #     "/network/datasets/places365/256/val_256.tar",
#         #     "/network/datasets/places365/256/test_256.tar",
#         # ],
#         tvd.ImageNet: [
#             "/network/datasets/imagenet/ILSVRC2012_img_train.tar",
#             "/network/datasets/imagenet/ILSVRC2012_img_val.tar",
#             "/network/datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz",
#         ],
#     },
#     Cluster.Beluga: {
#         tvd.CIFAR100: ["/project/rpp-bengioy/data/curated/cifar100/cifar-100-python.tar.gz"],
#         tvd.ImageNet: [
#             "/project/rpp-bengioy/data/curated/imagenet/ILSVRC2012_img_train.tar",
#             "/project/rpp-bengioy/data/curated/imagenet/ILSVRC2012_img_val.tar",
#             "/project/rpp-bengioy/data/curated/imagenet/ILSVRC2012_devkit_t12.tar.gz",
#         ],
#     },
# }
# """For each cluster, for each type of dataset, the list of archives needed to load the dataset."""


too_large_for_slurm_tmpdir: set[Callable] = set()
"""Set of datasets which are too large to store in $SLURM_TMPDIR."""

V = TypeVar("V")


def is_supported_dataset(dataset_type: type) -> bool:
    """Returns whether we 'support' this dataset: if we know which files are required to load it.

    NOTE: This doesn't check if we have a copy of those files on the current cluster! To do this,
    use `is_stored_on_cluster`.
    """

    try:
        _get_key_to_use_for_indexing(_dataset_files, key=dataset_type)
        return True
    except KeyError:
        return False


def files_needed(dataset_class: type, cluster: Cluster) -> list[Path] | None:
    """Returns the files that should be copied (or symlinked) to SLURM_TMPDIR to load this dataset.

    TODO: Should return 'live' paths, and prioritize using archives when available on the current
    cluster.
    """
    returned_files: list[Path] = []

    archives = _archives_required_for(dataset_class=dataset_class, cluster=cluster)
    if archives:
        returned_files.extend(archives)

    # TODO: rework the _archives_required and files_required dicts. It should just give back, for
    # a given cluster and dataset, what files (archives preferably) should be copied / symlinked
    # to SLURM_TMPDIR to create the dataset.

    root = locate_dataset_root_on_cluster(dataset_cls=dataset_class, cluster=cluster)
    if root is None:
        return archives
    files = files_required_for(dataset_type=dataset_class)
    for file in files:
        filepath = Path(root) / file
        returned_files.append(filepath)

    # double-check that we actually have all the files we're saying are needed.
    for file in returned_files:
        if not file.exists():
            # Create a better error (e.g. a RegistryMismatch error)?
            raise FileNotFoundError(
                f"Internal error: File {file} is supposed to exist according to our registry!"
            )
    return returned_files


def files_required_for(dataset_type: type) -> list[str]:
    # TODO: Would be nice if this actually returned 'live' Path objects for the current cluster!
    try:
        return getitem_with_subclasscheck(_dataset_files, key=dataset_type)
    except KeyError as exc:
        pass

    files_to_copy = files_to_use_for_dataset(dataset_type=dataset_type)
    if files_to_copy:
        return files_to_copy

    error = UnsupportedDatasetError(
        dataset=dataset_type,
    )

    folders = _get_folders_from_docstring(dataset_type)
    if folders:
        return folders
    raise error


def _archives_required_for(
    dataset_class: type, cluster: Cluster | None = Cluster.current()
) -> list[Path] | None:
    if cluster is None:
        raise NotImplementedError(
            f"Can't tell which archives are required for dataset {dataset_class} on local machines!"
        )
    try:
        archive_paths = getitem_with_subclasscheck(
            dataset_archives_locations_per_cluster[cluster], key=dataset_class
        )
        # TODO: Maybe return the actual archive type to use?
        return [Path(archive_path) for archive_path in archive_paths]
    except KeyError:
        logger.debug(
            f"Don't know what archives are present on cluster {cluster} for dataset {dataset_class}!"
        )
        return None


# TODO: How about we adopt a de-centralized kind of registry, a bit like gym?
# In each dataset module, we could have a `mila_datamodules.register(name, locations={Mila: ...})`?


# NOTE: Does it make sense to allow passing `cluster=None` here? (in the sense that we're not on a cluster?)


def is_stored_on_cluster(dataset_cls: type, cluster: Cluster | None = CURRENT_CLUSTER) -> bool:
    """Returns whether the given dataset is stored on this cluster.

    This first checks if the `dataset_cls` has an entry in either the
    `dataset_archives_per_cluster` or `dataset_roots_per_cluster` dictionaries.
    If so, returns True.
    If not, this dynamically checks if know which archives or files can be used to create the
    dataset, and whether they exist on the current cluster.

    NOTE: This may check for the existence of the files or archives on the current cluster.
    """
    # TODO: This needs a big rework. Either we centralize the registry, or we don't.
    # TODO: Should raise a custom warning type whenever the registry is wrong, and return False.
    # required_files = files_to_copy_for_dataset(dataset_cls, cluster=cluster)

    if cluster in dataset_roots_per_cluster and bool(
        getitem_with_subclasscheck(dataset_roots_per_cluster[cluster], dataset_cls, default=None)
    ):
        dataset_cls = _get_key_to_use_for_indexing(dataset_roots_per_cluster[cluster], dataset_cls)
        # There is a directory from where this dataset can be read directly.
        dataset_root = Path(dataset_roots_per_cluster[cluster][dataset_cls])
        if not dataset_root.exists() and dataset_root.is_dir():
            warnings.warn(
                RuntimeWarning(
                    f"Registry is wrong: it lists {dataset_root} as the root to use to load dataset "
                    f"{dataset_cls.__name__} but that directory doesn't exist!"
                )
            )
        return True

    if cluster is None:
        # No way to check if the dataset is stored on the local machine.
        return False

    files_to_copy = files_to_use_for_dataset(dataset_cls, cluster=cluster)
    if files_to_copy is None:
        logger.debug(
            f"Don't know what files to copy for dataset {dataset_cls} on cluster {cluster}. "
            f"Assuming that the dataset isn't stored."
        )
        return False
    if not files_to_copy:
        raise NotImplementedError(
            f"There supposedly aren't any files that need to be copied for dataset {dataset_cls} "
            f"on cluster {cluster}, which is hard to believe. This is probably a bug."
        )
    # TODO: What about checking for those files (or other files) inside $SCRATCH?
    # IDEA: This is interesting. Perhaps we could use SCRATCH before we use the dataset dir, that
    # way users could "overwrite" some of the files. e.g. use some of the files from SCRATCH and
    # the rest would come from /network/datasets?
    scratch_data_dir = get_scratch_dir() / "data"
    for relative_file_name, file_on_network in files_to_copy.items():
        file_on_scratch = scratch_data_dir / relative_file_name
        if file_on_network.exists():
            continue
        if file_on_scratch.exists():
            continue
        warnings.warn(
            RuntimeWarning(
                f"Registry is wrong: Says that in order to load the {dataset_cls}, we'd need "
                f"to copy {file_on_network} to the data directory, but that file doesn't "
                f"exist!\n"
                f"We also looked in $SCRATCH at {file_on_scratch}, but that doesn't exist either!"
            )
        )
        return False
    return True


def locate_dataset_root_on_cluster(
    dataset_cls: type | Callable[..., Dataset],
    cluster: Cluster | None = CURRENT_CLUSTER,
    default: str | None = None,
) -> str:
    """Gets the root directory to use to read the given dataset on the given cluster.

    If the dataset is not available on the cluster and `default` is set, then the default value is
    returned. Otherwise, if `default` is None, raises a NotImplementedError.
    """
    # TODO: This is specific to torchvision datasets at the moment. Either move this to the vision
    # folder, or make it actually more general.
    # TODO: This isn't exactly using anything about archives either!

    # TODO: Unclear what to do when not on a cluster. Should this just not be used?
    if cluster is None:
        if default is not None:
            return default
        raise RuntimeError(
            f"Was asked what root directory is appropriate for dataset type {dataset_cls} while "
            f"not on a SLURM cluster, and no default value was passed. "
            # NOTE: Now raising an error instead, just to avoid any surprises.
        )

    if not is_supported_dataset(dataset_cls):
        if default is not None:
            return default
        raise UnsupportedDatasetError(dataset=dataset_cls, cluster=cluster)

    if not is_stored_on_cluster(dataset_cls, cluster):
        if default is not None:
            return default
        # We don't know where this dataset is in this cluster.
        raise DatasetNotFoundOnClusterError(
            dataset=dataset_cls,
            cluster=cluster,
        )

    if cluster not in dataset_roots_per_cluster:
        raise DatasetNotFoundOnClusterError(dataset=dataset_cls, cluster=cluster)
        # f"Don't know where datasets are stored in cluster {cluster_name}! \n"
        # f"If you do know where it can be found on {cluster_name}, or on any other "
        # f"cluster, 🙏 please make an issue at {github_issue_url} to add it to the registry! 🙏"
        # )

    dataset_root = dataset_roots_per_cluster[cluster].get(dataset_cls)
    if dataset_cls is None:
        # Unsupported dataset?
        raise DatasetNotFoundOnClusterError(dataset=dataset_cls, cluster=cluster)
        # raise NotImplementedError(
        #     f"No known location for dataset {dataset_cls.__name__} on any of the clusters!\n"
        #     f"If you do know where it can be found on {cluster_name}, or on any other "
        #     f"cluster, 🙏 please make an issue at {github_issue_url} to add it to the registry! 🙏"
        # )
    return str(dataset_root)


def _get_folders_from_docstring(dataset_type: type) -> list[str] | None:
    # FIXME: Hacky AF. Tries to parse the docstring of the class to extract the 'structure' portion
    doc = dataset_type.__doc__
    if not doc:
        return

    doc_lines = doc.splitlines()
    structure_begin_line_index = [
        line_index for line_index, line in enumerate(doc_lines) if line.endswith("::")
    ]
    if len(structure_begin_line_index) != 1:
        return
    structure_begin_line_index = structure_begin_line_index[0] + 1

    if doc_lines[structure_begin_line_index] == "":
        structure_begin_line_index += 1

    structure_end_line_index = [
        line_index for line_index, line in enumerate(doc_lines) if line.strip().startswith("Args:")
    ]

    if len(structure_end_line_index) != 1:
        return
    structure_end_line_index = structure_end_line_index[0]

    structure_block = textwrap.dedent(
        "\n".join(doc_lines[structure_begin_line_index:structure_end_line_index])
    )
    if structure_block.startswith("root"):
        structure_begin_line_index += 1
        structure_block = textwrap.dedent(
            "\n".join(doc_lines[structure_begin_line_index:structure_end_line_index])
        )

    if not structure_block:
        return
    top_level_folders = [
        line for line in structure_block.splitlines() if line and not line[0].isspace()
    ]

    return top_level_folders
