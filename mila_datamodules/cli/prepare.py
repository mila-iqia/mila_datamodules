from __future__ import annotations
from argparse import ArgumentParser
import os
from pathlib import Path
import shutil
from typing import Callable, Generic, Sequence, TypeVar
from typing_extensions import ParamSpec, Concatenate
import inspect

from mila_datamodules.clusters.cluster import Cluster
import torchvision.datasets as tvd

from .utils import runs_on_local_main_process_first, is_local_main

# from simple_parsing import ArgumentParser
SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
P = ParamSpec("P")
VD = TypeVar("VD", bound=tvd.VisionDataset)
C = TypeVar("C", bound=Callable)

current_cluster = Cluster.current_or_error()


class PrepareVisionDataset(Generic[VD, P]):
    def __init__(self, dataset_type: Callable[Concatenate[str, P], VD]):
        self.dataset_type = dataset_type

    @runs_on_local_main_process_first
    def __call__(
        self,
        root: str | Path = SLURM_TMPDIR / "datasets",
        *constructor_args: P.args,
        **constructor_kwargs: P.kwargs,
    ) -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        dataset_instance = self.dataset_type(str(root), *constructor_args, **constructor_kwargs)
        if is_local_main():
            print(dataset_instance)
        return str(root)


class SymlinkArchives(PrepareVisionDataset[VD, P]):
    def __init__(
        self,
        dataset_type: Callable[Concatenate[str, P], VD],
        relative_paths_to_archives: dict[str, Path],
    ):
        self.dataset_type = dataset_type
        self.relative_paths_to_archives = relative_paths_to_archives

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

        # Use the dataset constructor to extract the archives.
        # Note: need to use `download=True` for torchvision to extract the archive.
        constructor_kwargs = constructor_kwargs.copy()  # type: ignore
        if "download" in inspect.signature(self.dataset_type).parameters:
            constructor_kwargs["download"] = True

        _dataset_instance = self.dataset_type(str(root), *constructor_args, **constructor_kwargs)
        if is_local_main():
            print(_dataset_instance)

        return str(root)


class CopyTree(PrepareVisionDataset[VD, P]):
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

        # Use the dataset constructor to extract the archives.
        # Note: need to use `download=True` for torchvision to extract the archive.
        constructor_kwargs = constructor_kwargs.copy()  # type: ignore
        if "download" in inspect.signature(self.dataset_type).parameters:
            constructor_kwargs["download"] = True

        _dataset_instance = self.dataset_type(str(root), *constructor_args, **constructor_kwargs)
        if is_local_main():
            print(_dataset_instance)

        return str(root)


prepare_torchvision_dataset: dict[type, dict[Cluster, PrepareVisionDataset]] = {
    tvd.MNIST: {
        # On the Mila cluster we have archives which are extracted into 4 "raw" binary files. Easy peasy.
        # NOTE: On Beluga, we have the MNIST 'raw' files in /project/rpp-bengioy/data/MNIST/raw, no archives.
        Cluster.Mila: SymlinkArchives(
            tvd.MNIST,
            {
                f"MNIST/raw/{archive.name}": archive
                for archive in Path("/network/datasets/mnist").glob("*.gz")
            },
        ),
        Cluster.Beluga: CopyTree(tvd.MNIST, {"MNIST": "/project/rpp-bengioy/data/MNIST"}),
    },
    tvd.CIFAR10: {
        Cluster.Mila: SymlinkArchives(
            tvd.CIFAR10,
            {"cifar-10-python.tar.gz": Path("/network/datasets/cifar10/cifar-10-python.tar.gz")},
        ),
        Cluster.Beluga: SymlinkArchives(
            tvd.CIFAR10,
            {
                "cifar-10-python.tar.gz": Path(
                    "/project/rpp-bengioy/data/curated/cifar10/cifar-10-python.tar.gz"
                ),
            },
        ),
    },
}
"""
A mapping from the relative path where the symlink to the archive should be created (relative to
a 'root' directory) to the actual path of the archive on the network.
"""

# TODO: For the datasets we don't have archives for, we could either list the locations where the
# extracted version can be found, or we could download the archive in $SCRATCH.


# cifar10_archives = {
#     Cluster.Mila: "/network/datasets/cifar10/cifar-10-python.tar.gz",
#     Cluster.Beluga: "/project/rpp-bengioy/data/curated/cifar10/cifar-10-python.tar.gz",
# }


# def prepare_cifar10(
#     dataset: type[tvd.CIFAR10] = tvd.CIFAR10,
#     root: Path = SLURM_TMPDIR / "datasets",
# ) -> Path:
#     cluster = Cluster.current_or_error()
#     archive = Path(cifar10_archives[cluster])
#     assert archive.exists()

#     with local_main_process_first():
#         if is_local_main():
#             root.mkdir(parents=True, exist_ok=True)

#             archive_name = archive.name
#             # Make a symlink in the local scratch directory to the archive on the network.
#             (Path(root) / archive_name).symlink_to(archive)
#             print(f"Making link from {root / archive_name} -> {archive}")

#         # Note: need to use `download=True` for torchvision to extract the archive.
#         dataset(str(root), download=True)

#     return root


# def prepare_mnist_beluga(
#     dataset: type[tvd.MNIST] = tvd.MNIST, root: Path = SLURM_TMPDIR / "datasets"
# ) -> Path:
#     shutil.copytree("/project/rpp-bengioy/data/MNIST", root / "MNIST")
#     return root


# def prepare_mnist_mila(
#     dataset: type[tvd.MNIST] = tvd.MNIST,
#     root: Path = SLURM_TMPDIR / "datasets",
# ) -> Path:
#     relative_path_to_archive = {
#         f"MNIST/raw/{archive.name}": archive
#         for archive in Path("/network/datasets/mnist").glob("*.gz")
#     }
#     with local_main_process_first():
#         root.mkdir(parents=True, exist_ok=True)

#         for relative_path, archive in relative_path_to_archive.items():
#             assert archive.exists()
#             # Make a symlink in the local scratch directory to the archive on the network.
#             archive_symlink = root / relative_path
#             if archive_symlink.exists():
#                 continue

#             archive_symlink.parent.mkdir(parents=True, exist_ok=True)
#             archive_symlink.symlink_to(archive)
#             print(f"Making link from {archive_symlink} -> {archive}")

#         # Note: need to use `download=True` for torchvision to extract the archive.
#         _ = dataset(str(root), download=True)

#     return root


# prepare_dataset_functions = {
#     tvd.MNIST: {
#         Cluster.Mila: prepare_mnist_mila,
#         Cluster.Beluga: prepare_mnist_beluga,
#     },
#     tvd.CIFAR10: {
#         Cluster.Mila: prepare_cifar10,
#         Cluster.Beluga: prepare_cifar10,
#     },
# }


def prepare(argv: list[str] | None = None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(
        title="dataset", description="Which dataset to prepare", dest="dataset"
    )

    cifar10_parser = subparsers.add_parser("cifar10", help="Prepare the CIFAR10 dataset")
    cifar10_parser.add_argument("--root", type=Path, default=SLURM_TMPDIR / "datasets")
    cifar10_parser.set_defaults(function=prepare_torchvision_dataset[tvd.CIFAR10][current_cluster])

    mnist_parser = subparsers.add_parser("mnist", help="Prepare the mnist dataset")
    mnist_parser.add_argument("--root", type=Path, default=SLURM_TMPDIR / "datasets")
    mnist_parser.set_defaults(function=prepare_torchvision_dataset[tvd.MNIST][current_cluster])

    args = parser.parse_args(argv)

    args_dict = vars(args)
    dataset = args_dict.pop("dataset")
    function = args_dict.pop("function")
    kwargs = args_dict

    new_root = function(**kwargs)
    print(f"The {dataset} dataset can now be read from the following directory: {new_root}")


if __name__ == "__main__":
    prepare()
