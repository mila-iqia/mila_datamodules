import inspect
from pathlib import Path

import pytest
from torchvision.datasets import VisionDataset

import mila_datamodules.vision.datasets

datasets = {
    k: v
    for k, v in vars(mila_datamodules.vision.datasets).items()
    if inspect.isclass(v) and issubclass(v, VisionDataset)
}


@pytest.mark.parametrize("dataset_cls", datasets.values())
def test_dataset_creation(dataset_cls: type[VisionDataset]):
    p = "fake_path"
    dataset = dataset_cls(root=p)
    assert not Path(p).exists()
    x, y = dataset[0]
