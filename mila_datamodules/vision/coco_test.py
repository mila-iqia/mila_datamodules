import itertools
from typing import ClassVar

from torch import Tensor

from .coco import CocoCaptionsDataModule
from .vision_datamodule import VisionDataModule
from .vision_datamodule_test import VisionDataModuleTests


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
            assert isinstance(y, list)
            assert all(isinstance(y_i, tuple) for y_i in y)
            assert all(isinstance(y_i_j, str) for y_i in y for y_i_j in y_i)
