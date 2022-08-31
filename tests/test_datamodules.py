import inspect
import itertools

import pytest
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import mila_datamodules.vision

datamodules = {
    k: v
    for k, v in vars(mila_datamodules.vision).items()
    if inspect.isclass(v) and issubclass(v, LightningDataModule)
}


@pytest.mark.parametrize("datamodule_cls", datamodules.values())
def test_datamodule_creation(datamodule_cls: type[LightningDataModule]):
    datamodule = datamodule_cls()
    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    assert isinstance(train_dataloader, DataLoader)
    for x, y in itertools.islice(train_dataloader, 2):
        assert x.shape[0] == train_dataloader.batch_size
        assert x.shape[1:] == datamodule.dims

    val_dataloader = datamodule.val_dataloader()
    assert isinstance(val_dataloader, DataLoader)
    for x, y in itertools.islice(val_dataloader, 2):
        assert x.shape[0] == val_dataloader.batch_size
        assert x.shape[1:] == datamodule.dims

    test_dataloader = datamodule.test_dataloader()
    assert isinstance(test_dataloader, DataLoader)
    for x, y in itertools.islice(test_dataloader, 2):
        assert x.shape[0] == test_dataloader.batch_size
        assert x.shape[1:] == datamodule.dims
