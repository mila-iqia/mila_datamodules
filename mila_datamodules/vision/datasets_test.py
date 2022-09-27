"""Tests for the `mila_datamodules.datasets` module.

Checks that the 'optimized' constructors work on the current cluster.
"""
from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path
from typing import Callable

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
datasets[mila_datamodules.vision.datasets.EMNIST] = partial(  # type: ignore
    mila_datamodules.vision.datasets.EMNIST, split="mnist"
)


# TODO: Adapt this test for datasets like EMNIST that require more arguments
@pytest.mark.parametrize("dataset_cls", datasets.values())
def test_dataset_creation(
    dataset_cls: type[VisionDataset] | Callable[..., VisionDataset], tmp_path: Path
):
    """Test that the dataset can be created, with the optimizations (copies/etc)."""
    bad_root = str(tmp_path / "fake_path")
    dataset = check_dataset_creation_works(dataset_cls, root=bad_root)
    assert dataset.root != bad_root
    assert not dataset.root.startswith(bad_root)
    assert not Path(bad_root).exists()
    # NOTE: Doesn't always hold. For example, for datasets like fashion-mnist, the data doesn't
    # really need to be moved to SLURM_TMPDIR, since we're loading it into memory anyway...
    # assert dataset.root == str(Cluster.current().slurm_tmpdir / "data")
