from __future__ import annotations

import contextlib
import inspect
import shutil
import typing
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from shutil import unpack_archive
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
)
from zipfile import ZipFile

import torchvision.datasets as tvd
from typing_extensions import Concatenate, ParamSpec, TypeVar

from mila_datamodules.cli.utils import is_local_main, runs_on_local_main_process_first
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_slurm_tmpdir

logger = get_logger(__name__)
# from simple_parsing import ArgumentParser
SLURM_TMPDIR = get_slurm_tmpdir()
if typing.TYPE_CHECKING:
    P = ParamSpec("P", default=...)
else:
    P = ParamSpec("P", default=Any)

VD = TypeVar("VD", bound=tvd.VisionDataset, default=tvd.VisionDataset)
VD_co = TypeVar("VD_co", bound=tvd.VisionDataset, default=tvd.VisionDataset, covariant=True)
C = TypeVar("C", bound=Callable)

current_cluster = Cluster.current_or_error()


class PrepareVisionDataset(Protocol[VD_co, P]):
    def __call__(
        self,
        root: str | Path,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        raise NotImplementedError


class CallDatasetConstructor(PrepareVisionDataset[VD_co, P]):
    def __init__(self, dataset_type: Callable[Concatenate[str, P], VD_co], verify=False):
        self.dataset_type = dataset_type
        self.verify = verify

    @runs_on_local_main_process_first
    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        """Use the dataset constructor to prepare the dataset in the `root` directory.

        If the dataset has a `download` argument in its constructor, it will be set to `True` so
        the archives are extracted.

        NOTE: This should only really be called after the actual dataset preparation has been done
        in a subclass's `__call__` method.

        Returns `root` (as a string).
        """
        Path(root).mkdir(parents=True, exist_ok=True)

        dataset_kwargs = dataset_kwargs.copy()  # type: ignore
        if "download" in inspect.signature(self.dataset_type).parameters:
            dataset_kwargs["download"] = not self.verify

        logger.debug(
            f"Using dataset constructor: {self.dataset_type} with args {dataset_args}, and "
            f"kwargs {dataset_kwargs}"
        )
        dataset_instance = self.dataset_type(str(root), *dataset_args, **dataset_kwargs)
        if is_local_main():
            print(dataset_instance)
        return str(root)


def _recursive_list_files(root: Path, ignore_prefix: tuple[str, ...] = (".",)) -> Iterable[Path]:
    if not root.exists():
        return []

    for entry in root.iterdir():
        if entry.name.startswith(ignore_prefix):
            continue
        if entry.is_file():
            yield entry
        if entry.is_dir():
            # NOTE: The Path objects here will have the right prefix (including `root`). No need
            # to add it.
            yield from _recursive_list_files(entry, ignore_prefix=ignore_prefix)


def dataset_files_in_source_dir(
    source: str | Path, ignore_prefixes=(".", "scripts", "README")
) -> dict[str, Path]:
    source = Path(source).expanduser().resolve()
    return {
        str(file.relative_to(source)): file
        for file in _recursive_list_files(Path(source), ignore_prefix=ignore_prefixes)
    }


class MakeSymlinksToDatasetFiles(PrepareVisionDataset[VD, P]):
    """Creates symlinks to the datasets' files in the `root` directory."""

    def __init__(
        self,
        source_dir_or_relative_paths_to_files: str | Path | Mapping[str, str | Path],
    ):
        """
        Parameters
        ----------

        - source_or_relative_paths_to_files:
            Either a source directory, in which case all the files under that directory are
            symlinked, or a mapping from filenames (relative to the 'root' directory) where the
            symlink should be created, to the absolute path to the file on the cluster.
        """
        self.relative_paths_to_files: dict[str, Path]
        if isinstance(source_dir_or_relative_paths_to_files, (str, Path)):
            source = source_dir_or_relative_paths_to_files
            self.relative_paths_to_files = dataset_files_in_source_dir(source)
        else:
            self.relative_paths_to_files = {
                str(k): Path(v) for k, v in source_dir_or_relative_paths_to_files.items()
            }

    @runs_on_local_main_process_first
    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        for relative_path, dataset_file in self.relative_paths_to_files.items():
            assert dataset_file.exists()
            # Make a symlink in the local scratch directory to the archive on the network.
            archive_symlink = root / relative_path
            if archive_symlink.exists():
                continue

            archive_symlink.parent.mkdir(parents=True, exist_ok=True)
            archive_symlink.symlink_to(dataset_file)
            print(f"Making link from {archive_symlink} -> {dataset_file}")

        return str(root)


class ExtractArchives(PrepareVisionDataset[VD, P]):
    """Extract some archives files in a subfolder of the `root` directory."""

    def __init__(self, archives: dict[str, str | Path]):
        """
        Parameters
        ----------

        - archives:
            A mapping from an archive name to path where the archive
            should be extracted (relative to the 'root' dir).
            The destination paths need to be relative.
        """
        self.archives = {glob: Path(path) for glob, path in archives.items()}

    @runs_on_local_main_process_first
    def __call__(self, root: str | Path, *dataset_args: P.args, **dataset_kwargs: P.kwargs) -> str:
        for archive, dest in self.archives.items():
            archive = Path(archive)
            assert not dest.is_absolute()

            dest = root / dest
            print(f"Extracting {archive} in {dest}")
            if archive.suffix == ".zip":
                with ZipFile(root / archive) as zf:
                    zf.extractall(str(dest))
            else:
                unpack_archive(archive, extract_dir=dest)

        return str(root)


class MoveFiles(PrepareVisionDataset[VD, P]):
    """Reorganize datasets' files in the `root` directory."""

    def __init__(self, files: dict[str, str | Path]):
        """
        Parameters
        ----------

        - files:
            A mapping from an archive and a destination's path where the result
            should be moved and replaced.

            If the destination path's leaf is "*", the destination's parent will be used to hold
            the file. If not, the destination will be used as the target for the move.
            The files are moved in sequence. The destination's path should be relative.
        """
        self.files = [(glob, Path(path)) for glob, path in files.items()]

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path,
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        root = Path(root)
        for glob, dest in self.files:
            assert not dest.is_absolute()
            dest = root / dest
            for entry in root.glob(glob):
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Avoid replacing dest by itself
                if dest.name == "*" and entry != dest.parent:
                    entry.replace(dest.parent / entry.name)
                elif dest.name != "*" and entry != dest:
                    entry.replace(dest)

        return str(root)


class CopyTree(CallDatasetConstructor[VD, P]):
    """Copies a tree of files from the cluster to the `root` directory."""

    def __init__(
        self,
        dataset_type: Callable[Concatenate[str, P], VD],
        relative_paths_to_dirs: dict[str, str | Path],
        ignore_filenames: Sequence[str] = (".git",),
    ):
        self.dataset_type = dataset_type
        self.relative_paths_to_dirs = {
            relative_path: Path(path) for relative_path, path in relative_paths_to_dirs.items()
        }
        self.ignore_dirs = ignore_filenames

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *constructor_args: P.args,
        **constructor_kwargs: P.kwargs,
    ):
        assert all(directory.exists() for directory in self.relative_paths_to_dirs.values())

        root = Path(root)
        for relative_path, tree in self.relative_paths_to_dirs.items():
            dest_dir = root / relative_path
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                tree,
                dest_dir,
                ignore=shutil.ignore_patterns(*self.ignore_dirs),
                dirs_exist_ok=True,
            )

        return super()(root, *constructor_args, **constructor_kwargs)


class Compose(PrepareVisionDataset[VD_co, P]):
    class Stop(Exception):
        pass

    def __init__(self, *callables: PrepareVisionDataset[VD_co, P]) -> None:
        self.callables = callables

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        try:
            for c in self.callables:
                # TODO: Check that nesting `runs_on_local_main_process_first` decorators isn't a
                # problem.
                root = c(root, *dataset_args, **dataset_kwargs)
        except self.Stop:
            pass
        return str(root)


class StopOnSucess(PrepareVisionDataset[VD, P]):
    """Raises a special Stop exception when calling function doesn't raise an exception.

    If an exception of a type matching one in `exceptions` is raised by the function, the exception
    is ignored. Other exceptions are raised.
    """

    def __init__(
        self, function: PrepareVisionDataset[VD, P], exceptions: Sequence[type[Exception]] = ()
    ):
        self.function = function
        self.exceptions = exceptions

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        with contextlib.suppress(*self.exceptions):
            self.function(root, *dataset_args, **dataset_kwargs)
            raise Compose.Stop()
        return str(root)


# NOTE: For some datasets, we have datasets stored in folders with the same structure. This here is
# only really used to prevent repeating a bit of code in the dictionary below.
# TODO: Find an exception to this rule and design this dict with that in mind.
standardized_torchvision_datasets_dir = {
    Cluster.Mila: Path("/network/datasets"),
    Cluster.Beluga: Path("~/project/rpp-bengioy/data/curated").expanduser().resolve(),
}

CocoType = TypeVar("CocoType", tvd.CocoCaptions, tvd.CocoDetection, default=tvd.CocoDetection)


def check_coco_is_setup(
    dataset_type: Callable[P, CocoType] = tvd.CocoDetection,
    variant: Literal["captions", "instances", "panoptic", "person_keypoints", "stuff"]
    | None = None,
):
    if variant is None:
        if dataset_type is tvd.CocoCaptions:
            variant = "captions"

    def _check_coco_setup(
        root: str | Path,
        annFile: str = "annotations/captions_train2017.json",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> str:
        dataset = dataset_type(
            f"{root}/train2017",
            f"{root}/annotations/{variant}_train2017.json",
            *args,
            **kwargs,
        )
        dataset[0]
        dataset = dataset_type(
            f"{root}/val2017",
            f"{root}/annotations/{variant}_val2017.json",
            *args,
            **kwargs,
        )
        dataset[0]

        return str(root)

    return _check_coco_setup


prepare_torchvision_datasets: dict[type, dict[Cluster, PrepareVisionDataset]] = {
    tvd.Caltech101: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.Caltech101, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/caltech101"),
            # Torchvision will look into a caltech101 directory to
            # preprocess the dataset
            MoveFiles({"*": "caltech101/*"}),
            CallDatasetConstructor(tvd.Caltech101),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Caltech256: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.Caltech256, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/caltech256"),
            # Torchvision will look into a caltech256 directory to
            # preprocess the dataset
            MoveFiles({"*": "caltech256/*"}),
            CallDatasetConstructor(tvd.Caltech256),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CelebA: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.CelebA, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/celeba"),
            # Torchvision will look into a celeba directory to preprocess
            # the dataset
            MoveFiles(
                {
                    "Anno/**/*": "celeba/*",
                    "Eval/**/*": "celeba/*",
                    "Img/**/*": "celeba/*",
                }
            ),
            CallDatasetConstructor(tvd.CelebA),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CIFAR10: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.CIFAR10, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(
                {"cifar-10-python.tar.gz": f"{datasets_dir}/cifar10/cifar-10-python.tar.gz"}
            ),
            CallDatasetConstructor(tvd.CIFAR10),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CIFAR100: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.CIFAR100, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(
                {"cifar-100-python.tar.gz": f"{datasets_dir}/cifar100/cifar-100-python.tar.gz"}
            ),
            CallDatasetConstructor(tvd.CIFAR100),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Cityscapes: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.Cityscapes, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/cityscapes"),
            CallDatasetConstructor(tvd.Cityscapes),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    # TODO: CocoCaptions is a bit weird.
    # - If we prepare everything right, we still have to call the constructor with
    # root=<root>/train2017.
    tvd.CocoCaptions: {
        cluster: Compose(
            # BUG: This actually only checks that the annotation file is present and can be loaded!
            # Therefore, we index the created dataset to see if it actually works.
            StopOnSucess(
                # CallDatasetConstructor(tvd.CocoCaptions, verify=True),
                check_coco_is_setup(tvd.CocoCaptions),
                exceptions=[FileNotFoundError],
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/coco/2017"),
            ExtractArchives(
                archives={
                    "test2017.zip": ".",
                    "train2017.zip": ".",
                    "val2017.zip": ".",
                    "annotations/annotations_trainval2017.zip": ".",
                    "annotations/image_info_test2017.zip": ".",
                    "annotations/panoptic_annotations_trainval2017.zip": ".",
                    "annotations/stuff_annotations_trainval2017.zip": ".",
                },
            ),
            check_coco_is_setup(tvd.CocoCaptions),
            # lambda root, annFile, *args, **kwargs: (tvd.CocoCaptions),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.CocoDetection: {
        cluster: Compose(
            StopOnSucess(
                check_coco_is_setup(tvd.CocoDetection, variant="stuff"),
                # CallDatasetConstructor(tvd.CocoDetection, verify=True),
                exceptions=[FileNotFoundError],
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/coco/2017"),
            ExtractArchives(
                archives={
                    "test2017.zip": ".",
                    "train2017.zip": ".",
                    "val2017.zip": ".",
                    "annotations/annotations_trainval2017.zip": ".",
                    "annotations/image_info_test2017.zip": ".",
                    "annotations/panoptic_annotations_trainval2017.zip": ".",
                    "annotations/stuff_annotations_trainval2017.zip": ".",
                }
            ),
            # CallDatasetConstructor(tvd.CocoDetection),
            check_coco_is_setup(tvd.CocoDetection, variant="stuff"),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.FashionMNIST: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.FashionMNIST, verify=True),
                exceptions=[RuntimeError],
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/fashionmnist"),
            # Torchvision will look into a FashionMNIST/raw directory to
            # preprocess the dataset
            MoveFiles({"*": "FashionMNIST/raw/*"}),
            CallDatasetConstructor(tvd.FashionMNIST),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.INaturalist: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.INaturalist, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/inat"),
            # Torchvision will look for those files to preprocess the
            # dataset
            MoveFiles(
                {
                    "train.tar.gz": "2021_train.tgz",
                    "train_mini.tar.gz": "2021_train_mini.tgz",
                    "val.tar.gz": "2021_valid.tgz",
                }
            ),
            CallDatasetConstructor(tvd.INaturalist),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.ImageNet: {
        # TODO: Write a customized `PrepareVisionDataset` for ImageNet that uses Olexa's magic tar
        # command.
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.ImageNet, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/imagenet"),
            CallDatasetConstructor(tvd.ImageNet),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.KMNIST: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.KMNIST, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/kmnist"),
            # Torchvision will look into a KMNIST/raw directory to
            # preprocess the dataset
            MoveFiles({"*": "KMNIST/raw/*"}),
            CallDatasetConstructor(tvd.KMNIST),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.MNIST: {
        # On the Mila and Beluga cluster we have archives which are extracted
        # into 4 "raw" binary files. We do need to match the expected directory
        # structure of the torchvision MNIST dataset though.  NOTE: On Beluga,
        # we also have the MNIST 'raw' files in
        # /project/rpp-bengioy/data/MNIST/raw, no archives.
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.MNIST, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/mnist"),
            # Torchvision will look into a raw directory to preprocess the
            # dataset
            MoveFiles({"*": "raw/*"}),
            CallDatasetConstructor(tvd.MNIST),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.Places365: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.Places365, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/places365"),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/places365.var/places365_challenge"),
            MoveFiles({"256/*.tar": "./*", "large/*.tar": "./*"}),
            CallDatasetConstructor(tvd.Places365),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.QMNIST: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.QMNIST, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/qmnist"),
            # Torchvision will look into a QMNIST/raw directory to
            # preprocess the dataset
            MoveFiles({"*": "QMNIST/raw/*"}),
            CallDatasetConstructor(tvd.QMNIST),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.STL10: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.STL10, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_folder}/stl10"),
            CallDatasetConstructor(tvd.STL10),
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dir.items()
    },
    tvd.SVHN: {
        cluster: Compose(
            StopOnSucess(CallDatasetConstructor(tvd.SVHN, verify=True), exceptions=[RuntimeError]),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/svhn"),
            CallDatasetConstructor(tvd.SVHN),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
    tvd.UCF101: {
        cluster: Compose(
            StopOnSucess(
                CallDatasetConstructor(tvd.UCF101, verify=True), exceptions=[RuntimeError]
            ),
            MakeSymlinksToDatasetFiles(f"{datasets_dir}/ucf101"),
            ExtractArchives(
                {
                    "UCF101.rar": ".",
                    "UCF101TrainTestSplits-RecognitionTask.zip": ".",
                }
            ),
            CallDatasetConstructor(tvd.UCF101),
        )
        for cluster, datasets_dir in standardized_torchvision_datasets_dir.items()
    },
}
"""Dataset preparation functions per dataset type, per cluster."""


command_line_args_for_dataset: MutableMapping[type[tvd.VisionDataset], type[DatasetArguments]] = {}


@dataclass
class DatasetArguments(Generic[VD]):
    """Keyword arguments for the dataset."""

    # root: Path = get_slurm_tmpdir() / "datasets"
    # """Root directory where images are downloaded to."""

    def __init_subclass__(cls, dataset_class: type[tvd.VisionDataset] | None = None) -> None:
        if not dataset_class:
            from typing import get_args

            dataset_class = get_args(cls.__orig_bases__[0])[0]  # type: ignore
            # TODO: Get the bound programmatically to avoid hardcoding `Dataset` here.
        if not (inspect.isclass(dataset_class) and issubclass(dataset_class, tvd.VisionDataset)):
            raise RuntimeError(
                "Your test class needs to pass the class under test to the generic base class.\n"
                "for example: `class TestMyDataset(DatasetTests[MyDataset]):`\n"
                f"(Got {dataset_class})"
            )
        command_line_args_for_dataset[dataset_class] = cls


@dataclass
class CocoCaptionsArgs(DatasetArguments[tvd.CocoCaptions]):
    """Command-line arguments used when preparing the CocoCaptions dataset."""

    root: Path = get_slurm_tmpdir() / "datasets"

    # TODO: Should we set a default value here to make things easier?
    annFile: str = "annotations/captions_train2017.json"
    """Path to json annotation file."""

    def __post_init__(self):
        self.annFile = f"{self.root}/{self.annFile}"


@dataclass
class CocoDetectionArgs(DatasetArguments[tvd.CocoDetection]):
    """Command-line arguments used when preparing the CocoCaptions dataset."""

    # TODO: Should we set a default value here to make things easier?
    annFile: str = "annotations/instances_train2017.json"
    """Path to json annotation file."""
