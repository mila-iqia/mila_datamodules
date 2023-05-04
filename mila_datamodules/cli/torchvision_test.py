""" TODO: Tests for the CLI. """
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pytest
import simple_parsing
import torchvision.datasets as tvd
from torchvision.datasets import VisionDataset
from typing_extensions import Concatenate, ParamSpec

from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.torchvision import PrepareDatasetFn, prepare_torchvision_datasets
from mila_datamodules.cli.types import VD
from mila_datamodules.clusters import CURRENT_CLUSTER

P = ParamSpec("P", default=Any)
no_internet = pytest.mark.disable_socket

pytestmark = no_internet

datasets_to_preparation_function: dict[type[VisionDataset], Callable[..., str]] = {
    dataset_type: cluster_to_function[CURRENT_CLUSTER]
    for dataset_type, cluster_to_function in prepare_torchvision_datasets.items()
    if CURRENT_CLUSTER in cluster_to_function
}


def get_preparation_function(
    dataset_type: Callable[Concatenate[str, P], VD],
) -> PrepareDatasetFn[VD, P]:
    return datasets_to_preparation_function[dataset_type]


command_line_args_for_tests: dict[type[VisionDataset], list[str]] = {
    tvd.UCF101: ["--frames_per_clip=5"],
    tvd.INaturalist: ["--version=2021_valid"],
}


@pytest.mark.parametrize(
    ("dataset_type", "argv"),
    [
        pytest.param(
            dataset,
            argv,
            marks=pytest.mark.skipif(
                "-vvv" not in sys.argv, reason="This dataset takes a long time to prepare."
            ),
        )
        if dataset in (tvd.ImageNet, tvd.Places365)
        else (dataset, argv)
        for dataset in datasets_to_preparation_function
        for argv in command_line_args_for_tests.get(dataset, [""])
    ],
)
def test_prepare_dataset(
    dataset_type: Callable[Concatenate[str, P], VD],
    argv: str | None,
    fake_slurm_tmpdir: Path,
):
    dataset_preparation_function = get_preparation_function(dataset_type=dataset_type)
    assert len(list(fake_slurm_tmpdir.iterdir())) == 0  # start with an empty SLURM_TMPDIR.

    # TODO: Need to adapt this tests for the datasets that need additional arguments (e.g. COCO
    # needs `annFile`)
    from mila_datamodules.cli.torchvision import command_line_args_for_dataset

    args: DatasetArguments[VD] | None = None
    command_line_arguments = command_line_args_for_dataset.get(dataset_type)
    if command_line_arguments:
        if isinstance(command_line_arguments, type):
            if not argv:
                args = command_line_arguments(root=fake_slurm_tmpdir)
            else:
                args = simple_parsing.parse(
                    f"--root={fake_slurm_tmpdir} " + command_line_arguments, args=argv
                )
        else:
            args = command_line_arguments
        dataset_kwargs = args.to_dataset_kwargs()
        if "root" in dataset_kwargs:
            dataset_kwargs.pop("root")
    else:
        dataset_kwargs = {}

    new_root = dataset_preparation_function(fake_slurm_tmpdir, **dataset_kwargs)
    new_dataset_kwargs = dataset_kwargs.copy()
    dataset_instance = dataset_type(new_root, **new_dataset_kwargs)
    dataset_instance[0]
