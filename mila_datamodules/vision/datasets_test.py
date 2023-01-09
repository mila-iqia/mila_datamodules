"""Tests for the `mila_datamodules.datasets` module.

Checks that the 'optimized' constructors work on the current cluster.
"""
from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path

import pytest
from torchvision.datasets import VisionDataset

import mila_datamodules.vision.datasets
from mila_datamodules.registry_test import check_dataset_creation_works

datasets = {
    k: v
    for k, v in vars(mila_datamodules.vision.datasets).items()
    if inspect.isclass(v) and issubclass(v, VisionDataset)
}

# TODO: Need to stop doing this kind of hard-coded fixing and listing of stuff.
datasets["EMNIST"] = partial(datasets.pop("EMNIST"), split="mnist")
datasets["BinaryEMNIST"] = partial(datasets.pop("BinaryEMNIST"), split="mnist")

# Dataset takes a bit longer to copy.
dataset_names = list(datasets.keys())
dataset_names = [
    pytest.param(
        dataset_name,
        # Put the tests for a given dataset in the same group, so that they (eventually) run on the
        # same node (same process for now, until we figure out how to distribute the tests).
        marks=[pytest.mark.xdist_group(name=dataset_name)]
        + ([pytest.mark.timeout(120)] if dataset_name == "CelebA" else []),
    )
    for dataset_name in dataset_names
]


# TODO: Make this quicker to test. Each test currently copies the entire dataset to SLURM_TMPDIR.
@pytest.mark.timeout(30)
@pytest.mark.parametrize("dataset_name", dataset_names)
def test_optimized_dataset_creation(dataset_name: str, tmp_path: Path):
    """Test that the dataset can be created, with the optimizations (copies/etc)."""
    dataset_cls = datasets[dataset_name]
    bad_root = str(tmp_path / "fake_path")
    dataset = check_dataset_creation_works(dataset_cls, root=bad_root)
    assert dataset.root != bad_root
    assert not dataset.root.startswith(bad_root)
    assert not Path(bad_root).exists()
    # NOTE: Doesn't always hold. For example, for datasets like fashion-mnist, the data doesn't
    # really need to be moved to SLURM_TMPDIR, since we're loading it into memory anyway...
    # assert dataset.root == str(Cluster.current().slurm_tmpdir / "data")
