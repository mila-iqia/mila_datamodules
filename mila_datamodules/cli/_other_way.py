from __future__ import annotations
from argparse import ArgumentParser
import os
from pathlib import Path
import shutil
from typing import Callable, Generic, Sequence, TypeVar
from typing_extensions import ParamSpec, Concatenate
import inspect
from simple_parsing import field

from mila_datamodules.clusters.cluster import Cluster
import torchvision.datasets as tvd

from mila_datamodules.cli.utils import (
    runs_on_local_main_process_first,
    is_local_main,
    local_main_process_first,
)

SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])


cifar10_archives = {
    Cluster.Mila: "/network/datasets/cifar10/cifar-10-python.tar.gz",
    Cluster.Beluga: "/project/rpp-bengioy/data/curated/cifar10/cifar-10-python.tar.gz",
}


def prepare_cifar10(
    dataset: type[tvd.CIFAR10] = tvd.CIFAR10,
    root: Path = SLURM_TMPDIR / "datasets",
) -> Path:
    cluster = Cluster.current_or_error()
    archive = Path(cifar10_archives[cluster])
    assert archive.exists()

    with local_main_process_first():
        if is_local_main():
            root.mkdir(parents=True, exist_ok=True)

            archive_name = archive.name
            # Make a symlink in the local scratch directory to the archive on the network.
            (Path(root) / archive_name).symlink_to(archive)
            print(f"Making link from {root / archive_name} -> {archive}")

        # Note: need to use `download=True` for torchvision to extract the archive.
        dataset(str(root), download=True)

    return root


def prepare_mnist_beluga(
    dataset: type[tvd.MNIST] = tvd.MNIST, root: Path = SLURM_TMPDIR / "datasets"
) -> Path:
    shutil.copytree("/project/rpp-bengioy/data/MNIST", root / "MNIST")
    return root


def prepare_mnist_mila(
    dataset: type[tvd.MNIST] = tvd.MNIST,
    root: Path = SLURM_TMPDIR / "datasets",
) -> Path:
    relative_path_to_archive = {
        f"MNIST/raw/{archive.name}": archive
        for archive in Path("/network/datasets/mnist").glob("*.gz")
    }
    with local_main_process_first():
        root.mkdir(parents=True, exist_ok=True)

        for relative_path, archive in relative_path_to_archive.items():
            assert archive.exists()
            # Make a symlink in the local scratch directory to the archive on the network.
            archive_symlink = root / relative_path
            if archive_symlink.exists():
                continue

            archive_symlink.parent.mkdir(parents=True, exist_ok=True)
            archive_symlink.symlink_to(archive)
            print(f"Making link from {archive_symlink} -> {archive}")

        # Note: need to use `download=True` for torchvision to extract the archive.
        _ = dataset(str(root), download=True)

    return root


prepare_dataset_functions = {
    tvd.MNIST: {
        Cluster.Mila: prepare_mnist_mila,
        Cluster.Beluga: prepare_mnist_beluga,
    },
    tvd.CIFAR10: {
        Cluster.Mila: prepare_cifar10,
        Cluster.Beluga: prepare_cifar10,
    },
}
