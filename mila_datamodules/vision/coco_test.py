import itertools
from importlib.util import find_spec
from typing import ClassVar

import pytest
from torch import Tensor

from .coco import CocoCaptionsDataModule
from .vision_datamodule import VisionDataModule
from .vision_datamodule_test import VisionDataModuleTests

PYCOCOTOOLS_INSTALLED = find_spec("pycocotools") is not None

coco_required = pytest.mark.xfail(not PYCOCOTOOLS_INSTALLED, reason="pycocotools isn't installed")


@coco_required
@pytest.mark.timeout(300)
class TestCoco(VisionDataModuleTests):
    DataModule: ClassVar[type[VisionDataModule]] = CocoCaptionsDataModule

    def test_dataset_items(self):
        datamodule = self.DataModule(batch_size=1)
        datamodule.prepare_data()
        datamodule.setup()

        train_dataloader = datamodule.train_dataloader()
        for x, y in itertools.islice(train_dataloader, 2):
            assert isinstance(x, Tensor)
            # TODO: Should we make this y 'cleaner', e.g. by making it into a list[str] ?
            y: list[tuple[str]]
            assert isinstance(y, list)
            assert all(isinstance(y_i, tuple) for y_i in y)
            assert all(isinstance(y_i_j, str) for y_i in y for y_i_j in y_i)
