"""Re-implementation of the BinaryMNIST from pl_bolts.datasets, which has some issues."""
from __future__ import annotations

import os

from PIL import Image
from torchvision.datasets.mnist import MNIST

# TODO: Reformulate this as just MNIST + a transform.


class BinaryMNIST(MNIST):
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

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        #     raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        image_array = img.numpy()
        # binary
        image_array[image_array < 0.5] = 0.0
        image_array[image_array >= 0.5] = 1.0

        img = Image.fromarray(image_array, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
