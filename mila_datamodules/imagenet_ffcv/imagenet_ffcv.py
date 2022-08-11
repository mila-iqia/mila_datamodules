""" ImageNet datamodule that uses FFCV. """

from __future__ import annotations

import typing
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable, TypeVar

import cv2  # noqa
import ffcv
import ffcv.transforms
import numpy as np
import torch
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from pl_bolts.datasets import UnlabeledImagenet
from torch import nn
from torch.utils.data import DataLoader
from typing_extensions import TypeGuard

from ..imagenet import ImagenetDataModule
from .ffcv_config import DatasetWriterConfig, FfcvLoaderConfig, ImageResolutionConfig

if typing.TYPE_CHECKING:
    from pytorch_lightning import Trainer


# FIXME: This is ugly. Replace with this, if possible.
# from pl_bolts.datamodules.imagenet_datamodule import imagenet_normalization
# IMAGENET_MEAN = imagenet_normalization().mean * 255
# IMAGENET_STD = imagenet_normalization().std * 255

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256


class ImagenetFfcvDataModule(ImagenetDataModule):
    """Wrapper around the ImageNetDataModule that uses ffcv for the Train dataloader.

    1. Copies the Imagenet dataset to SLURM_TMPDIR (same as parent class)
    2. Writes the dataset in ffcv format at SLURM_TMPRID/imagenet/train.ffcv
    3. Train dataloader reads from that file.

    The image resolution can be changed dynamically at each epoch (to match the ffcv-imagenet repo)
    based on the values in a configuration class. This can also be turned off.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        meta_dir: str | None = None,
        num_imgs_per_val_class: int = 50,
        image_size: int = 224,
        num_workers: int | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: nn.Module | Callable | None = None,
        val_transforms: nn.Module | Callable | None = None,
        test_transforms: nn.Module | Callable | None = None,
        ffcv_train_transforms: Sequence[Operation] | None = None,
        img_resolution_config: ImageResolutionConfig | None = None,
        writer_config: DatasetWriterConfig | None = None,
        loader_config: FfcvLoaderConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            meta_dir=meta_dir,
            num_imgs_per_val_class=num_imgs_per_val_class,
            image_size=image_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            train_transforms=None,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
        )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        default_ffcv_train_transforms = [
            ffcv.transforms.RandomHorizontalFlip(),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(self.device, non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),  # type: ignore
        ]

        if train_transforms is None:
            if ffcv_train_transforms is None:
                # No ffcv transform or torchvision transforms were passed, use the ffcv equivalent
                # to the usual torchvision transforms.
                ffcv_train_transforms = default_ffcv_train_transforms
            else:
                ffcv_train_transforms = ffcv_train_transforms
        elif self.device.type == "cuda" and not isinstance(train_transforms, nn.Module):
            raise RuntimeError(
                f"Can't use cuda and old-style torchvision transforms. Upgrade "
                f"torchvision to a more recent version and pass a nn.Sequential instead."
            )
        else:
            if ffcv_train_transforms is None:
                # Only using torchvision transforms.
                pass
            else:
                # Using both torchvision transforms and ffcv transforms.
                pass

        self.ffcv_train_transforms: Sequence[Operation] = ffcv_train_transforms or []
        if isinstance(train_transforms, nn.Module):
            self._train_transforms = train_transforms.to(self.device)

        self.img_resolution_config = img_resolution_config or ImageResolutionConfig()
        self.writer_config = writer_config or DatasetWriterConfig(
            subset=-1,
            write_mode="smart",
            max_resolution=224,
            compress_probability=None,
            jpeg_quality=90,
            num_workers=self.num_workers,
            chunk_size=100,
        )
        self.loader_config = loader_config or FfcvLoaderConfig(
            # NOTE: Can't use QUASI_RANDOM when using distributed=True atm.
            order=OrderOption.QUASI_RANDOM,  # type: ignore
            os_cache=False,
            drop_last=True,
            distributed=False,
            batches_ahead=3,
            seed=1234,
        )
        # TODO: Incorporate a hash of the writer config into the name of the ffcv file.
        self._train_file = Path(self.data_dir) / f"train.ffcv"
        self.save_hyperparameters()
        # Note: defined in the LightningDataModule class, gets set when using a Trainer.
        self.trainer: Trainer | None = None

    def prepare_data(self) -> None:
        super().prepare_data()
        # NOTE: Might not need to do this for val/test, since we can just always use the standard,
        # regular dataloaders. Just need to make sure taht the train/validation split is done the
        # same way for both.
        if not _done_file(self._train_file).exists():
            # Writes train.ffcv
            # train.ffcv_done.txt doesn't exist, so we need to rewrite the train.ffcv file
            _write_dataset(
                super().train_dataloader(),
                self._train_file,
                writer_config=self.writer_config,
            )
            _done_file(self._train_file).touch()

    def train_dataloader(self) -> Iterable:
        current_epoch = self.current_epoch
        res = self.img_resolution_config.get_resolution(current_epoch)
        print(
            (f"Epoch {current_epoch}: " if current_epoch is not None else "")
            + f"Loading images at {res}x{res} resolution"
        )
        image_pipeline: list[Operation] = [
            RandomResizedCropRGBImageDecoder((res, res)),
            *self.ffcv_train_transforms,
        ]
        label_pipeline: list[Operation] = [
            IntDecoder(),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.Squeeze(),
            ffcv.transforms.ToDevice(self.device, non_blocking=True),
        ]
        loader = Loader(
            str(self._train_file),
            pipelines={"image": image_pipeline, "label": label_pipeline},
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **self.loader_config,
        )

        if self._train_transforms:
            # Apply the Torchvision transforms after the FFCV transforms.
            return ApplyTransformLoader(loader, self._train_transforms)
        return loader

    def val_dataloader(self) -> DataLoader:
        return super().val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return super().test_dataloader()

    @property
    def current_epoch(self) -> int | None:
        """The current training epoch if using a Trainer of PyTorchLightning, else None."""
        if self.trainer is not None:
            return self.trainer.current_epoch
        return None


from typing import Tuple, TypeVar

from torch import nn
from torch.utils.data import DataLoader

T = TypeVar("T")
O = TypeVar("O")
L = TypeVar("L")


class ApplyTransformLoader(Iterable[tuple[O, L]]):
    def __init__(self, data: Iterable[tuple[T, L]], transform: Callable[[T], O]):
        self.data_source = data
        self.transform = transform

    def __iter__(self) -> Iterable[tuple[O, L]]:
        for x, y in self.data_source:
            yield self.transform(x), y

    def __len__(self) -> int:
        return len(self.data_source)  # type: ignore


def _write_dataset(
    dataloader: DataLoader, dataset_ffcv_path: Path, writer_config: DatasetWriterConfig
) -> None:
    dataset = dataloader.dataset
    assert isinstance(dataset, UnlabeledImagenet)
    # NOTE: We write the dataset without any transforms.
    dataset.transform = None
    dataset.label_transform = None
    dataset_ffcv_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_done_file = _done_file(dataset_ffcv_path)
    if not dataset_done_file.exists():
        print(f"Writing dataset in FFCV format at {dataset_ffcv_path}")
        writer_config.write(dataset, dataset_ffcv_path)
        dataset_done_file.touch()


def _done_file(path: Path) -> Path:
    return path.with_name(path.name + "_done.txt")


def _is_sequence_of_operators(ops: Any) -> TypeGuard[Sequence[Operation]]:
    return isinstance(ops, (list, tuple)) and all(
        isinstance(op, Operation) for op in ops
    )
