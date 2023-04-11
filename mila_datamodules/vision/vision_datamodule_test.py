from __future__ import annotations

import itertools
from typing import ClassVar

import pytest
from torch import Tensor
from torch.utils.data import DataLoader

from .vision_datamodule import VisionDataModule

without_internet = pytest.mark.disable_socket
# TODO: Add marks to skip the ultra slow datamodules unless a --slow argument is passed to pytest?
# TODO: Create a slurm job that runs these unit tests.


@without_internet
class VisionDataModuleTests:
    """Set of unit tests for vision datamodules."""

    DataModule: ClassVar[type[VisionDataModule]]

    def test_datamodule_creation(self):
        # BUG: Need to pass the Generator to the dataloader, otherwise the batches change between
        # runs.
        datamodule = self.DataModule()
        # TODO: If we don't know where the dataset is stored, but it's simple to download, then
        # maybe we should just let it download itself into SCRATCH first, then copy to slurm
        # tmpdir, right?
        datamodule.prepare_data()
        datamodule.setup()

        train_dataloader = datamodule.train_dataloader()
        assert isinstance(train_dataloader, DataLoader)
        for x, y in itertools.islice(train_dataloader, 2):
            assert x.shape[0] == train_dataloader.batch_size
            if datamodule.dims:
                assert x.shape[1:] == datamodule.dims
            if isinstance(y, Tensor):
                assert y.shape[0] == train_dataloader.batch_size

        val_dataloader = datamodule.val_dataloader()
        assert isinstance(val_dataloader, DataLoader)
        for x, y in itertools.islice(val_dataloader, 2):
            assert x.shape[0] == val_dataloader.batch_size
            if datamodule.dims:
                assert x.shape[1:] == datamodule.dims
            if isinstance(y, Tensor):
                assert y.shape[0] == val_dataloader.batch_size

        test_dataloader = datamodule.test_dataloader()
        assert isinstance(test_dataloader, DataLoader)
        for x, y in itertools.islice(test_dataloader, 2):
            assert x.shape[0] == test_dataloader.batch_size
            if isinstance(y, Tensor):
                assert x.shape[1:] == datamodule.dims

    def test_train_val_splits_dont_overlap(self):
        datamodule = self.DataModule()
        datamodule.prepare_data()
        datamodule.setup()
        from torch.utils.data import Subset

        assert isinstance(datamodule.dataset_train, Subset)
        assert isinstance(datamodule.dataset_val, Subset)
        train_indices = set(datamodule.dataset_train.indices)
        val_indices = set(datamodule.dataset_val.indices)
        assert train_indices.isdisjoint(val_indices)
