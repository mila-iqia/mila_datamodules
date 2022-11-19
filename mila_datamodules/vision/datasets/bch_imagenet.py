import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import bcachefs as bch
import torch
from PIL import Image
from benzina.torch.dataset.dataset import ClassificationDatasetMixin
import pl_bolts.datasets.imagenet_dataset as bolts_imagenet
import torchvision.datasets.imagenet as imagenet


def pil_loader(bchfs: bch.bcachefs.Filesystem, path: str) -> Image.Image:
    with bchfs.open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class BchImageNet(ClassificationDatasetMixin, imagenet.ImageNet):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Bcachefs filesystem image.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, image: str, split: str = "train", **kwargs: Any) -> None:
        self.split = imagenet.verify_str_arg(split, "split", ("train", "val"))
        with bch.mount(image) as bchfs:
            self._cursor = bchfs.cd(self.split)

        if "root" in kwargs:
            del kwargs["root"]
        if "loader" not in kwargs:
            kwargs["loader"] = pil_loader

        with self.patch_imagenet_load_meta_file():
            # When using the _cursor of ClassificationDatasetMixin, the relative
            # image folder is "."
            imagenet.ImageNet.__init__(self, "/", split=split, **kwargs)

    def parse_archives(self) -> None:
        pass

    def load_meta_file(self, root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
        if file is None:
            file = imagenet.META_FILE
        file = os.path.join(root, file)

        if self._cursor.isfile(file):
            with self._cursor.open(file) as _f:
                return torch.load(_f)
        else:
            msg = (
                "The meta file {} is not present in the root directory or is corrupted. "
                "This file is automatically created by the ImageNet dataset."
            )
            raise RuntimeError(msg.format(file, root))

    @contextmanager
    def patch_imagenet_load_meta_file(self):
        load_meta_file = imagenet.load_meta_file
        imagenet.load_meta_file = self.load_meta_file
        yield
        imagenet.load_meta_file = load_meta_file


class BchUnlabeledImagenet(BchImageNet, bolts_imagenet.UnlabeledImagenet):
    def __init__(self, image: str, split: str = "train", **kwargs: Any) -> None:
        self.split = imagenet.verify_str_arg(split, "split", ("train", "val", "test"))
        with bch.mount(image) as bchfs:
            # [train], [val] --> [train, val], [test]
            if split == "train" or split == "val":
                self._cursor = bchfs.cd("train")
            else:
                self._cursor = bchfs.cd("val")

        if "root" in kwargs:
            del kwargs["root"]
        if "loader" not in kwargs:
            kwargs["loader"] = pil_loader

        with self.patch_imagenet_load_meta_file():
            # When using the _cursor of ClassificationDatasetMixin, the relative
            # image folder is "."
            bolts_imagenet.UnlabeledImagenet.__init__(self, "/", split=split, **kwargs)

    @contextmanager
    def patch_imagenet_load_meta_file(self):
        load_meta_file = imagenet.load_meta_file
        bolts_load_meta_file = bolts_imagenet.load_meta_file
        imagenet.load_meta_file = self.load_meta_file
        bolts_imagenet.load_meta_file = self.load_meta_file
        yield
        imagenet.load_meta_file = load_meta_file
        bolts_imagenet.load_meta_file = bolts_load_meta_file