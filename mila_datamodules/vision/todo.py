# TODO:
import http
import inspect

import cv2
from pl_bolts.datamodules.binary_emnist_datamodule import BinaryEMNISTDataModule
from pl_bolts.datamodules.binary_mnist_datamodule import BinaryMNISTDataModule

# from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule, TinyCIFAR10DataModule
# from pl_bolts.datamodules.cityscapes_datamodule import CityscapesDataModule
from pl_bolts.datamodules.emnist_datamodule import EMNISTDataModule

# from pl_bolts.datamodules.experience_source import DiscountedExperienceSource, ExperienceSource, ExperienceSourceDataset
from pl_bolts.datamodules.fashion_mnist_datamodule import FashionMNISTDataModule

# from pl_bolts.datamodules.imagenet_datamodule import ImagenetDataModule
from pl_bolts.datamodules.kitti_datamodule import KittiDataModule

# from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pl_bolts.datamodules.sklearn_datamodule import (
    SklearnDataModule,
    SklearnDataset,
    TensorDataset,
)

# from pl_bolts.datamodules.sr_datamodule import TVTDataModule
from pl_bolts.datamodules.ssl_imagenet_datamodule import SSLImagenetDataModule
from pl_bolts.datamodules.stl10_datamodule import STL10DataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datamodules.vocdetection_datamodule import VOCDetectionDataModule
from pl_bolts.datasets.kitti_dataset import KittiDataset
from pytorch_lightning import LightningDataModule

from mila_datamodules.vision.datasets import _adapt_dataset

successes = []
failures = []
for v in LightningDataModule.__subclasses__():
    if v is VisionDataModule:
        continue
    k = v.__qualname__
    print(k, v)
    try:
        datamodule = v("/network/datasets/torchvision")
        datamodule.prepare_data()
        print(datamodule)
    except (OSError, AttributeError) as err:
        print(err)
        failures.append(v)
    else:
        print(f"Success for {k}")
        successes.append(v)

print(successes)
print(failures)
# for failure in failures:
#     class Foo(failure):
#         dataset_cls = _adapt_dataset(failure.dataset_cls)
#     datamodule = Foo("/network/datasets/torchvision")
#     datamodule.prepare_data()
