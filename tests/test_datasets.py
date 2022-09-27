from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path
from typing import Callable

import pytest
from torchvision.datasets import VisionDataset

import mila_datamodules.vision.datasets
from mila_datamodules.clusters import Cluster

datasets = {
    k: v
    for k, v in vars(mila_datamodules.vision.datasets).items()
    if inspect.isclass(v) and issubclass(v, VisionDataset)
}

datasets[mila_datamodules.vision.datasets.EMNIST] = partial(
    mila_datamodules.vision.datasets.EMNIST, split="mnist"
)


# TODO: Adapt this test for datasets like EMNIST that require more arguments
@pytest.mark.parametrize("dataset_cls", datasets.values())
def test_dataset_creation(dataset_cls: type[VisionDataset] | Callable[..., VisionDataset]):
    p = "fake_path"
    dataset = dataset_cls(root=p)
    assert not Path(p).exists()
    _ = dataset[0]
    assert len(dataset) > 0
    assert dataset.root == str(Cluster.current().slurm_tmpdir / "data")
