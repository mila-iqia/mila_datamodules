""" TODO: Tests for the CLI. """

import pytest
from torchvision.datasets import VisionDataset
from typing import Any, Callable
from pathlib import Path
import sys
from typing_extensions import Concatenate, ParamSpec
import torchvision.datasets as tvd
from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.cli.prepare_torchvision import (
    VD,
    prepare_torchvision_datasets,
    PrepareVisionDataset,
)

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
) -> PrepareVisionDataset[VD, P]:
    return datasets_to_preparation_function[dataset_type]


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param(
            dataset,
            marks=pytest.mark.skipif(
                "-vvv" not in sys.argv, reason="This dataset takes a long time to prepare."
            ),
        )
        if dataset in (tvd.ImageNet,)
        else dataset
        for dataset in datasets_to_preparation_function
    ],
)
def test_prepare_dataset(
    dataset_type: Callable[Concatenate[str, P], VD],
    fake_slurm_tmpdir: Path,
):
    dataset_preparation_function = get_preparation_function(dataset_type=dataset_type)
    assert len(list(fake_slurm_tmpdir.iterdir())) == 0  # start with an empty SLURM_TMPDIR.
    new_root = dataset_preparation_function(root=fake_slurm_tmpdir)
    dataset_type(new_root)
