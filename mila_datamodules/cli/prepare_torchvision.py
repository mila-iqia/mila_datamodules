from __future__ import annotations

import inspect
import shutil
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable, Generic, Sequence, TypeVar
from zipfile import ZipFile

import torchvision.datasets as tvd
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.cli.utils import is_local_main, runs_on_local_main_process_first
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_slurm_tmpdir

logger = get_logger(__name__)
# from simple_parsing import ArgumentParser
SLURM_TMPDIR = get_slurm_tmpdir()
P = ParamSpec("P", default=Any)
VD = TypeVar("VD", bound=tvd.VisionDataset)
C = TypeVar("C", bound=Callable)

current_cluster = Cluster.current_or_error()


class AbstractPrepareVisionDataset(Generic[VD, P]):
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        del root, dataset_args, dataset_kwargs
        raise NotImplementedError


class PrepareVisionDataset(AbstractPrepareVisionDataset[VD, P]):
    def __init__(self, dataset_type: Callable[Concatenate[str, P], VD]):
        self.dataset_type = dataset_type

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
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
            dataset_kwargs["download"] = True

        logger.debug(
            f"Using dataset constructor: {self.dataset_type} with args {dataset_args}, and "
            f"kwargs {dataset_kwargs}"
        )
        dataset_instance = self.dataset_type(str(root), *dataset_args, **dataset_kwargs)
        if is_local_main():
            print(dataset_instance)
        return str(root)


class SymlinkDatasetFiles(AbstractPrepareVisionDataset[VD, P]):
    """Creates symlinks to the datasets' files in the `root` directory."""

    def __init__(self, source: str | Path = None, files: dict[str, str | Path] = None):
        """
        Parameters
        ----------

        - source:
            Source directory to clone the drectories and files hieararchy. If
            only source is defined, the entire tree will be cloned.
        - files:
            A mapping from a path to where the symlink to the archive should be created
            (relative to the 'root' directory) to the actual path to the archive on the cluster.
        """
        if source is not None:
            source = Path(source).expanduser().resolve()

        if source is not None and files is not None:
            self.relative_paths_to_files = {
                k: Path(v).relative_to(source) for k, v in files.items()
            }
        elif source is not None:
            self.relative_paths_to_files = {
                _f.relative_to(source): _f
                for _f in SymlinkDatasetFiles._recursive_list_files(Path(source))
            }
        else:
            self.relative_paths_to_files = {k: Path(v) for k, v in files.items()}

    @staticmethod
    def _recursive_list_files(root: Path):
        for entry in root.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_file():
                yield entry
            if entry.is_dir():
                yield from SymlinkDatasetFiles._recursive_list_files(entry)

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        for relative_path, archive in self.relative_paths_to_files.items():
            assert archive.exists()
            # Make a symlink in the local scratch directory to the archive on the network.
            archive_symlink = root / relative_path
            if archive_symlink.exists():
                continue

            archive_symlink.parent.mkdir(parents=True, exist_ok=True)
            archive_symlink.symlink_to(archive)
            print(f"Making link from {archive_symlink} -> {archive}")

        return str(root)


class ExtractArchives(AbstractPrepareVisionDataset[VD, P]):
    """Reorganize datasets' files in the `root` directory."""

    def __init__(
        self,
        archives: dict[str, str | Path] = dict(),
    ):
        """
        Parameters
        ----------

        - archives:
            A mapping from an archive name to path where the archive
            should be extracted (relative to the 'root' dir).
            The destination paths need to be relative.
        """
        self.archives = [(glob, Path(path)) for glob, path in archives]

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        for archive, dest in self.reorganize_files:
            assert not dest.is_absolute()
            if archive.suffix == ".zip":
                with ZipFile(archive) as zf:
                    zf.extractall(str(dest))
            else:
                raise ValueError("Unsupported archive type")

        return str(root)


class MoveFiles(AbstractPrepareVisionDataset[VD, P]):
    """Reorganize datasets' files in the `root` directory."""

    def __init__(
        self,
        files: list[str, str | Path] = dict(),
    ):
        """
        Parameters
        ----------

        - files:
            A list of pair of an archive and a destination's path where the result
            should be moved and replace. If the destination path's leaf is "*",
            the destination's parent will be used to hold the file. If not, the
            destination will be used as the target for the move. The move are
            executed in sequence. The destination's path should be relative.
        """
        self.files = [(glob, Path(path)) for glob, path in files]

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        for glob, dest in self.files:
            assert not dest.is_absolute()
            for entry in root.glob(glob):
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.name == "*":
                    entry.replace(root / dest.parent / entry.name)
                else:
                    entry.replace(root / dest)

        return str(root)


class CopyTree(PrepareVisionDataset[VD, P]):
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


class Compose(AbstractPrepareVisionDataset[VD, P]):
    def __init__(self, callables: list[AbstractPrepareVisionDataset]) -> None:
        self.callables = callables

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *dataset_args: P.args,
        **dataset_kwargs: P.kwargs,
    ) -> str:
        for c in self.callables:
            # TODO: Check that nesting `runs_on_local_main_process_first` decorators isn't a
            # problem.
            root = c(root, *dataset_args, **dataset_kwargs)
        return root


# NOTE: For some datasets, we have datasets stored in folders with the same structure. This here is
# only really used to prevent repeating a bit of code in the dictionary below.
# TODO: Find an exception to this rule and design this dict with that in mind.
standardized_torchvision_datasets_dirs = {
    Cluster.Mila: "/network/datasets",
    Cluster.Beluga: "~/project/rpp-bengioy/data/curated",
}

prepare_torchvision_datasets: dict[type, dict[Cluster, PrepareVisionDataset]] = {
    tvd.Caltech101: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/caltech101",
                ),
                MoveFiles(
                    # Torchvision will look into a caltech101 directory to
                    # preprocess the dataset
                    files=[("*", "caltech101/*")]
                ),
                PrepareVisionDataset(tvd.Caltech101),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.Caltech256: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/caltech256",
                ),
                MoveFiles(
                    # Torchvision will look into a caltech256 directory to
                    # preprocess the dataset
                    files=[("*", "caltech256/*")]
                ),
                PrepareVisionDataset(tvd.Caltech256),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.CelebA: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/celeba",
                ),
                MoveFiles(
                    # Torchvision will look into a celeba directory to preprocess
                    # the dataset
                    files=[
                        ("Anno/**/*", "celeba/*"),
                        ("Eval/**/*", "celeba/*"),
                        ("Img/**/*", "celeba/*"),
                    ]
                ),
                PrepareVisionDataset(tvd.CelebA),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.CIFAR10: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/cifar10",
                ),
                PrepareVisionDataset(tvd.CIFAR10),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.CIFAR100: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/cifar100",
                ),
                PrepareVisionDataset(tvd.CIFAR100),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.Cityscapes: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/cityscapes",
                ),
                PrepareVisionDataset(tvd.Cityscapes),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.CocoCaptions: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/coco/2017",
                ),
                ExtractArchives(
                    archives=[
                        ("test2017.zip", "."),
                        ("train2017.zip", "."),
                        ("val2017.zip", "."),
                        ("annotations/annotations_trainval2017.zip", "."),
                        ("annotations/image_info_test2017.zip", "."),
                        ("annotations/panoptic_annotations_trainval2017.zip", "."),
                        ("annotations/stuff_annotations_trainval2017.zip", "."),
                    ]
                ),
                PrepareVisionDataset(tvd.CocoCaptions),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.CocoDetection: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/coco/2017",
                ),
                ExtractArchives(
                    archives=[
                        ("test2017.zip", "."),
                        ("train2017.zip", "."),
                        ("val2017.zip", "."),
                        ("annotations/annotations_trainval2017.zip", "."),
                        ("annotations/image_info_test2017.zip", "."),
                        ("annotations/panoptic_annotations_trainval2017.zip", "."),
                        ("annotations/stuff_annotations_trainval2017.zip", "."),
                    ]
                ),
                PrepareVisionDataset(tvd.CocoDetection),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.FashionMNIST: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/fashionmnist",
                ),
                MoveFiles(
                    # Torchvision will look into a FashionMNIST/raw directory to
                    # preprocess the dataset
                    files=[("*", "FashionMNIST/raw/*")]
                ),
                PrepareVisionDataset(tvd.FashionMNIST),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.INaturalist: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/inat",
                ),
                MoveFiles(
                    # Torchvision will look for those files to preprocess the
                    # dataset
                    files=[
                        ("train.tar.gz", "2021_train.tgz"),
                        ("train_mini.tar.gz", "2021_train_mini.tgz"),
                        ("val.tar.gz", "2021_valid.tgz"),
                    ]
                ),
                PrepareVisionDataset(tvd.INaturalist),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.ImageNet: {
        # TODO: Write a customized `PrepareVisionDataset` for ImageNet that uses Olexa's magic tar
        # command.
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/imagenet",
                ),
                PrepareVisionDataset(tvd.ImageNet),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.KMNIST: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/kmnist",
                ),
                MoveFiles(
                    # Torchvision will look into a KMNIST/raw directory to
                    # preprocess the dataset
                    files=[("*", "KMNIST/raw/*")]
                ),
                PrepareVisionDataset(tvd.KMNIST),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.MNIST: {
        # On the Mila and Beluga cluster we have archives which are extracted
        # into 4 "raw" binary files. We do need to match the expected directory
        # structure of the torchvision MNIST dataset though.  NOTE: On Beluga,
        # we also have the MNIST 'raw' files in
        # /project/rpp-bengioy/data/MNIST/raw, no archives.
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/mnist",
                ),
                MoveFiles(
                    # Torchvision will look into a raw directory to preprocess the
                    # dataset
                    files=[("*", "raw/*")]
                ),
                PrepareVisionDataset(tvd.MNIST),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.Places365: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/places365",
                ),
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/places365.var/places365_challenge",
                ),
                MoveFiles(
                    files=[
                        ("256/*.tar", "./*"),
                        ("large/*.tar", "./*"),
                    ]
                ),
                PrepareVisionDataset(tvd.Places365),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.QMNIST: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/qmnist",
                ),
                MoveFiles(
                    # Torchvision will look into a QMNIST/raw directory to
                    # preprocess the dataset
                    files=[("*", "QMNIST/raw/*")]
                ),
                PrepareVisionDataset(tvd.QMNIST),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.STL10: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/stl10",
                ),
                PrepareVisionDataset(tvd.STL10),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.SVHN: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/svhn",
                ),
                PrepareVisionDataset(tvd.SVHN),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
    tvd.UCF101: {
        cluster: Compose(
            [
                SymlinkDatasetFiles(
                    source=f"{datasets_folder}/ucf101",
                ),
                ExtractArchives(
                    archives=[
                        # TODO: add support for rar archives
                        ("UCF101.rar", "."),
                        ("UCF101TrainTestSplits-RecognitionTask.zip", "."),
                    ]
                ),
                PrepareVisionDataset(tvd.UCF101),
            ]
        )
        for cluster, datasets_folder in standardized_torchvision_datasets_dirs.items()
    },
}
"""Dataset preparation functions per dataset type, per cluster."""
