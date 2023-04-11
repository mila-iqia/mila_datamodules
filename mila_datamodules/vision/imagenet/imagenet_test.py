import pytest

from ..vision_datamodule_test import VisionDataModuleTests
from .imagenet import ImagenetDataModule

# note: This isn't quite right: ImagenetDataModule doesn't inherit from VisionDataModule!


class TestImagenetDataModule(VisionDataModuleTests):
    DataModule = ImagenetDataModule

    def test_train_val_splits_dont_overlap(self):
        pytest.skip("ImagenetDataModule doesn't have a dataset_train and dataset_val attribute.")
