from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path
from typing import Callable, Generic, Sequence, TypeVar

import torchvision.datasets as tvd
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.cli.utils import is_local_main, runs_on_local_main_process_first
from mila_datamodules.clusters.cluster import Cluster

# from simple_parsing import ArgumentParser
SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
P = ParamSpec("P")
VD = TypeVar("VD", bound=tvd.VisionDataset)
C = TypeVar("C", bound=Callable)

current_cluster = Cluster.current_or_error()


class PrepareVisionDataset(Generic[VD, P]):
    def __init__(self, dataset_type: Callable[Concatenate[str, P], VD]):
        self.dataset_type = dataset_type

    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *constructor_args: P.args,
        **constructor_kwargs: P.kwargs,
    ) -> str:
        """Use the dataset constructor to prepare the dataset in the `root` directory.

        If the dataset has a `download` argument in its constructor, it will be set to `True` so
        the archives are extracted.

        NOTE: This should only really be called after the actual dataset preparation has been done
        in a subclass's `__call__` method.

        Returns `root` (as a string).
        """
        Path(root).mkdir(parents=True, exist_ok=True)

        constructor_kwargs = constructor_kwargs.copy()  # type: ignore
        if "download" in inspect.signature(self.dataset_type).parameters:
            constructor_kwargs["download"] = True

        print(
            f"Using dataset constructor: {self.dataset_type} with args {constructor_args}, and kwargs {constructor_kwargs}"
        )
        dataset_instance = self.dataset_type(str(root), *constructor_args, **constructor_kwargs)
        if is_local_main():
            print(dataset_instance)
        return str(root)


class SymlinkArchives(PrepareVisionDataset[VD, P]):
    """Creates symlinks to the archives in the `root` directory."""

    def __init__(
        self,
        dataset_type: Callable[Concatenate[str, P], VD],
        relative_paths_to_archives: dict[str, str | Path],
    ):
        """

        Parameters
        ----------

        - dataset_type:
            Callable that returns an instance of a `torchvision.datasets.VisionDataset`.

        - relative_paths_to_archives:
            A mapping from a relative path where the symlink to the archive should be created
            (relative to the 'root' directory) to the actual path to the archive on the cluster.
        """
        self.dataset_type = dataset_type
        self.relative_paths_to_archives = {
            k: Path(v) for k, v in relative_paths_to_archives.items()
        }

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *constructor_args: P.args,
        **constructor_kwargs: P.kwargs,
    ) -> str:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        for relative_path, archive in self.relative_paths_to_archives.items():
            assert archive.exists()
            # Make a symlink in the local scratch directory to the archive on the network.
            archive_symlink = root / relative_path
            if archive_symlink.exists():
                continue

            archive_symlink.parent.mkdir(parents=True, exist_ok=True)
            archive_symlink.symlink_to(archive)
            print(f"Making link from {archive_symlink} -> {archive}")
        # TODO: Check that nesting `runs_on_local_main_process_first` decorators isn't a problem.
        return super().__call__(root, *constructor_args, **constructor_kwargs)


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


# NOTE: For some datasets, we have datasets stored in folders with the same structure. This here is
# only really used to prevent repeating a bit of code in the dictionary below.
# TODO: Find an exception to this rule and design this dict with that in mind.
standardized_torchvision_dataset_dirs = {
    Cluster.Mila: "/network/datasets",
    Cluster.Beluga: "/project/rpp-bengioy/data/curated",
}

prepare_torchvision_datasets: dict[type, dict[Cluster, PrepareVisionDataset]] = {
    tvd.MNIST: {
        # On the Mila and Beluga cluster we have archives which are extracted into 4 "raw" binary
        # files. We do need to match the expected directory structure of the torchvision MNIST
        # dataset though.
        # NOTE: On Beluga, we also have the MNIST 'raw' files in /project/rpp-bengioy/data/MNIST/raw, no archives.
        cluster: SymlinkArchives(
            tvd.MNIST,
            {
                f"MNIST/raw/{archive.name}": archive
                for archive in Path(f"{dataset_folder}/mnist").glob("*.gz")
            },
        )
        for cluster, dataset_folder in standardized_torchvision_dataset_dirs.items()
    },
    tvd.CIFAR10: {
        cluster: SymlinkArchives(
            tvd.CIFAR10,
            {"cifar-10-python.tar.gz": f"{dataset_folder}/cifar10/cifar-10-python.tar.gz"},
        )
        for cluster, dataset_folder in standardized_torchvision_dataset_dirs.items()
    },
    tvd.CIFAR100: {
        cluster: SymlinkArchives(
            tvd.CIFAR100,
            {"cifar-100-python.tar.gz": f"{dataset_folder}/cifar100/cifar-100-python.tar.gz"},
        )
        for cluster, dataset_folder in standardized_torchvision_dataset_dirs.items()
    },
    tvd.ImageNet: {
        # TODO: Write a customized `PrepareVisionDataset` for ImageNet that uses Olexa's magic tar
        # command.
        cluster: SymlinkArchives(
            tvd.ImageNet,
            {
                "ILSVRC2012_devkit_t12.tar.gz": f"{dataset_folder}/imagenet/ILSVRC2012_devkit_t12.tar.gz",
                "ILSVRC2012_img_train.tar": f"{dataset_folder}/imagenet/ILSVRC2012_img_train.tar",
                "ILSVRC2012_img_val.tar": f"{dataset_folder}/imagenet/ILSVRC2012_img_val.tar",
            },
        )
        for cluster, dataset_folder in standardized_torchvision_dataset_dirs.items()
    },
}
"""Dataset preparation functions per dataset type, per cluster."""
