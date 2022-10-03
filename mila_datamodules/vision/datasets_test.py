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

# Takes a bit longer to copy.
datasets["CelebA"] = pytest.param(datasets.pop("CelebA"), marks=pytest.mark.timeout(120))


# TODO: Adapt this test for datasets like EMNIST that require more arguments
@pytest.mark.parametrize("dataset_name", datasets.keys())
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
