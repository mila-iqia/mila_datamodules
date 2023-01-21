from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable, ClassVar, Literal

import torch
from pl_bolts.datamodules.imagenet_datamodule import imagenet_normalization
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions, VisionDataset
from torchvision.io.image import read_image
from torchvision.transforms import Compose, ToTensor

from mila_datamodules.clusters import Cluster, get_slurm_tmpdir
from mila_datamodules.vision import VisionDataModule

coco_archives_root = {Cluster.Mila: "/network/datasets/coco/2017"}
"""Path on each cluster where the 'train2017.zip' and 'val2017.zip' archives are stored."""


captions_train_annFile_location = {
    Cluster.Mila: "/network/datasets/torchvision/annotations/captions_train2017.json",
    # TODO: Find where the annotation files are stored on the other clusters.
    # ClusterType.BELUGA: "?",
}

# Note: There doesn't seem to be a 'test' annotation file. So here 'val' means "test", and 'test'
# means "predict".
captions_test_annFile_location = {
    Cluster.Mila: "/network/datasets/torchvision/annotations/captions_val2017.json",
    # TODO:
    # ClusterType.BELUGA: "?",
}


class CocoCaptionsDataModule(VisionDataModule):
    """Datamodule for the COCO image caption dataset.

    Raw dataset items are tuples of the form (image, captions), where `image` is a PIL image, and
    `captions`
    varying shape

    TODO: Images don't have a fixed dimensionality, and I don't know what the 'standard crop'
    would be for this dataset.
    """

    name: ClassVar[str] = "cococaptions"
    dataset_cls: ClassVar[type[VisionDataset]] = CocoCaptions
    #: A tuple describing the shape of the data
    # TODO: This is wrong! The height and width of the images isn't constant.
    dims: ClassVar[tuple[int, ...]] = ()

    def __init__(
        self,
        data_dir: str | None = None,
        val_split: int | float = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 1,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Callable | nn.Module | None = None,
        val_transforms: Callable | nn.Module | None = None,
        test_transforms: Callable | nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            data_dir,
            val_split,
            num_workers,
            normalize,
            batch_size,
            seed,
            shuffle,
            pin_memory,
            drop_last,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            **kwargs,
        )
        # TODO: Raise a warning if the user attempts to pass a data dir value not under SLURM_TMPDIR.
        # If the user passes a different directory in SLURM_TMPDIR, then use it.
        self.data_dir = Path(data_dir or get_slurm_tmpdir() / "coco")
        self.dataset_train: Dataset[tuple[Tensor, list[tuple[str]]]] | None = None
        self.dataset_val: Dataset[tuple[Tensor, list[tuple[str]]]] | None = None
        self.dataset_test: Dataset[tuple[Tensor, list[tuple[str]]]] | None = None
        self.dataset_predict: Dataset[Tensor] | None = None

    def prepare_data(self):
        """Extracts the COCO archives into `self.data_dir` (SLURM_TMPDIR/coco by default)."""
        cluster = Cluster.current()
        if cluster not in coco_archives_root:
            cluster_name = cluster.name if cluster else "local"
            github_issue_url = (
                f"https://github.com/lebrice/mila_datamodules/issues/new?"
                f"labels={cluster_name}&template=feature_request.md&"
                f"title=Feature%20request:%20COCO%20on%20{cluster_name}%20cluster"
            )
            raise NotImplementedError(
                f"Don't know in which directory to get the '[train,val,test]2017.zip' archives "
                f"for the COCO dataset on cluster {cluster}!\n"
                f"If you know where it can be found on this cluster, please create an issue at "
                f"{github_issue_url} so we can add support for it!"
            )
        coco_archives_dir = Path(coco_archives_root[cluster])
        _extract_missing("train2017.zip", self.data_dir, coco_archives_dir)
        _extract_missing("val2017.zip", self.data_dir, coco_archives_dir)
        _extract_missing("test2017.zip", self.data_dir, coco_archives_dir)

        # Extract the annotations. (This might be a bit extra. We could just copy the files also..)
        with zipfile.ZipFile(
            coco_archives_dir / "annotations/annotations_trainval2017.zip", "r"
        ) as zip_file:
            members = None
            if (self.data_dir / "annotations").exists():
                members = set(zip_file.namelist()) - {
                    str(p) for p in (self.data_dir / "annotations").iterdir()
                }
            zip_file.extractall(self.data_dir, members=members)

    def setup(self):
        # NOTE: It's a bit too complicated to try to reuse the stuff from the base class.
        train_root = str(self.data_dir / "train2017")
        test_root = str(self.data_dir / "val2017")
        predict_root = str(self.data_dir / "test2017")

        train_ann_file = str(self.data_dir / "annotations/captions_train2017.json")
        test_ann_file = str(self.data_dir / "annotations/captions_val2017.json")

        test_transforms = self.test_transforms or self.default_transforms()
        self.dataset_test = CocoCaptions(
            root=test_root, annFile=test_ann_file, transform=test_transforms
        )
        # NOTE: In order to get the transforms right, we'll have to instantiate this twice, with
        # the train transforms, and once again with the val transforms.
        # TODO: If we create a DataModule for COCODetection based on this one, we should pass
        # transforms= (with an s) to the CocoCaptions constructor, so the transforms are synced
        # between the image and the segmentation mask.
        train_transforms = self.train_transforms or self.default_transforms()
        dataset_trainval = CocoCaptions(
            root=train_root, annFile=train_ann_file, transform=train_transforms
        )
        val_transforms = self.val_transforms or self.default_transforms()
        dataset_trainval_val = CocoCaptions(
            root=train_root, annFile=train_ann_file, transform=val_transforms
        )
        # Fork the RNG just to be 100% sure we don't have any effect on the global RNG state.
        with torch.random.fork_rng():
            # NOTE: Inside `super()._split_dataset`, the RNG is seeded with `self.seed`. So this is
            # supposed to be fine.
            self.dataset_train = self._split_dataset(dataset_trainval, train=True)
            self.dataset_val = self._split_dataset(dataset_trainval_val, train=False)

        self.dataset_predict = UnsupervisedImageDataset(
            root=predict_root, transforms=self.test_transforms
        )

    def default_transforms(self) -> Callable:
        """Default transformations to use when none are provided."""

        transforms: list[Callable] = [ToTensor()]
        if self.normalize:
            # TODO: Seems like people use the normalization from ImageNet on COCO? Not 100% sure.
            transforms.append(imagenet_normalization())
        return Compose(transforms)

    def predict_dataloader(self) -> DataLoader:
        assert self.dataset_predict is not None
        return self._data_loader(self.dataset_predict)


class UnsupervisedImageDataset(Dataset):
    """Simple dataset for a folder containing images."""

    def __init__(self, root: str | Path, transforms: Callable | None = None, extension="jpg"):
        self.root = Path(root)
        self.files = list(str(p) for p in self.root.rglob(f"*.{extension}"))
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image = read_image(self.files[idx])
        if self.transforms:
            return self.transforms(image)
        return image

    def __len__(self):
        return len(self.files)


def _extract_missing(
    archive_name: Literal["train2017.zip", "val2017.zip", "test2017.zip"],
    dest_dir: Path,
    coco_archives_dir: Path,
) -> None:
    folder_name, _, _ = archive_name.partition(".")
    with zipfile.ZipFile(coco_archives_dir / archive_name, "r") as zip_file:
        files = set(zip_file.namelist()) - {f"{folder_name}/"}
        existing_files = set()
        if dest_dir.exists():
            existing_files = {
                str(p.relative_to(dest_dir)) for p in dest_dir.glob(f"{folder_name}/*.jpg")
            }
        missing_files = files - existing_files
        if missing_files:
            print(
                f"Extracting {len(missing_files)} files from {archive_name} to {dest_dir}/{folder_name} ..."
            )
            zip_file.extractall(dest_dir, members=missing_files)
