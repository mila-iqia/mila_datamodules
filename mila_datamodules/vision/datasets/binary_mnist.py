"""Re-implementation of the BinaryMNIST from pl_bolts.datasets, which has some issues."""
from __future__ import annotations

import os

from PIL import Image
from pl_bolts.datasets import BinaryEMNIST as _BinaryEMNIST
from pl_bolts.datasets import BinaryMNIST as _BinaryMNIST

from .adapted_datasets import adapt_dataset

# TODO: Reformulate this as just MNIST + a transform?


class BinaryMNIST(_BinaryMNIST):
    """Fixes 2 bugs:

    1.  Looks for the data in a `BinaryMNIST` folder, which is totally uncessary. Could reuse the
        data from MNIST.
    2.  When no transform is used, an error is raised when converting the image to binary.
    """

    @property
    def raw_folder(self) -> str:
        # Note: This reuses the data from MNIST. Base class would look for a `BinaryMNIST` folder.
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        # Note: This reuses the data from MNIST. Base class would look for a `BinaryMNIST` folder.
        return os.path.join(self.root, "MNIST", "processed")

    def __getitem__(self, idx: int) -> tuple:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return _fixed_getitem(self, idx)


class BinaryEMNIST(_BinaryEMNIST):
    @property
    def raw_folder(self) -> str:
        # Note: This reuses the data from EMNIST. Base class would look for a `BinaryEMNIST` folder.
        return os.path.join(self.root, "EMNIST", "raw")

    @property
    def processed_folder(self) -> str:
        # Note: This reuses the data from EMNIST. Base class would look for a `BinaryEMNIST` folder.
        return os.path.join(self.root, "EMNIST", "processed")

    def __getitem__(self, idx: int) -> tuple:
        return _fixed_getitem(self, idx)


BinaryMNIST = adapt_dataset(BinaryMNIST)
BinaryEMNIST = adapt_dataset(BinaryEMNIST)


def _fixed_getitem(dataset: BinaryMNIST | BinaryEMNIST, idx: int):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    # if not _TORCHVISION_AVAILABLE:  # pragma: no cover
    #     raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

    img, target = dataset.data[idx], int(dataset.targets[idx])

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image

    image_array = img.numpy()
    # binary
    image_array[image_array < 0.5] = 0.0
    image_array[image_array >= 0.5] = 1.0

    img = Image.fromarray(image_array, mode="L")

    if dataset.transform is not None:
        img = dataset.transform(img)

    if dataset.target_transform is not None:
        target = dataset.target_transform(target)
    return img, target
