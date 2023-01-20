"""ImageNet datamodule that uses FFCV."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, TypedDict, TypeVar

import numpy as np
import torch
from pl_bolts.datasets import UnlabeledImagenet
from torch import Tensor, distributed, nn
from torch.utils.data import DataLoader

try:
    import ffcv.transforms
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.basics import IntDecoder
    from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.writer import DatasetWriter
except ImportError:
    pass


if typing.TYPE_CHECKING:
    from ffcv.traversal_order.base import TraversalOrder
    from pytorch_lightning import Trainer
    from ffcv.fields import Field
    from ffcv.pipeline.operation import Operation

from mila_datamodules.clusters import get_scratch_dir

from .imagenet import ImagenetDataModule

# FIXME: I don't like hard-coded values.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


@dataclass(frozen=True, unsafe_hash=True)
class DatasetWriterConfig:
    """Arguments to give the FFCV DatasetWriter."""

    max_resolution: int = 224
    """Max image side length."""

    num_workers: int = field(default=16, hash=False)
    """ Number of workers to use. """

    chunk_size: int = 100
    """ Chunk size for writing. """

    write_mode: Literal["raw", "smart", "jpg"] = "smart"

    jpeg_quality: int = 90
    """ Quality of JPEG images. """

    subset: int = -1
    """ How many images to use (-1 for all). """

    compress_probability: float | None = None

    def get_hash(self) -> str:
        """Returns a string hash of this config's values.

        NOTE: Could be used to retrieve a dataset file previously written with the same parameters.
        """
        config_dict = dataclasses.asdict(self)
        # Remove num_workers from hash since it doesn't change the resulting file contents.
        for f in dataclasses.fields(self):
            if not f.hash:
                config_dict.pop(f.name)
        writer_config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(writer_config_str.encode("utf-8")).hexdigest()


class FfcvLoaderConfig(TypedDict, total=False):
    os_cache: bool
    """ Leverages the operating system for caching purposes. This is beneficial when there is
    enough memory to cache the dataset and/or when multiple processes on the same machine training
    using the same dataset. See https://docs.ffcv.io/performance_guide.html for more information.
    """

    order: TraversalOrder
    """Traversal order, one of: SEQUENTIAL, RANDOM, QUASI_RANDOM
    QUASI_RANDOM is a random order that tries to be as uniform as possible while minimizing the
    amount of data read from the disk. Note that it is mostly useful when `os_cache=False`.
    Currently unavailable in distributed mode.
    """

    distributed: bool
    """For distributed training (multiple GPUs). Emulates the behavior of DistributedSampler from
    PyTorch.
    """

    seed: int
    """Random seed for batch ordering."""

    indices: Sequence[int]
    """Subset of dataset by filtering only some indices. """

    custom_fields: Mapping[str, type[Field]]
    """Dictionary informing the loader of the types associated to fields that are using a custom
    type.
    """

    drop_last: bool
    """Drop non-full batch in each iteration."""

    batches_ahead: int
    """Number of batches prepared in advance; balances latency and memory. """

    recompile: bool
    """Recompile every iteration. This is necessary if the implementation of some augmentations
    are expected to change during training.
    """


@dataclass(frozen=True, unsafe_hash=True)
class ImageResolutionConfig:
    """Configuration for the resolution of the images when loading from the written ffcv
    dataset."""

    min_res: int = 160
    """the minimum (starting) resolution"""

    max_res: int = 224
    """the maximum (final) resolution"""

    end_ramp: int = 0
    """ when to stop interpolating resolution. Set to 0 to disable this feature. """

    start_ramp: int = 0
    """ when to start interpolating resolution """

    def get_resolution(self, epoch: int | None) -> int:
        """Copied over from the FFCV example, where they ramp up the resolution during training.

        If `epoch` is None, or `end_ramp` is 0, the ramp-up is disabled and give back the max res.
        """
        assert self.min_res <= self.max_res
        if epoch is None:
            return self.max_res
        if epoch >= self.end_ramp:
            return self.max_res

        if epoch <= self.start_ramp:
            return self.min_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [self.start_ramp, self.end_ramp], [self.min_res, self.max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res


def default_ffcv_train_transforms(
    device: torch.device,
) -> list[Operation | nn.Module]:
    return [
        ffcv.transforms.RandomHorizontalFlip(),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(device, non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),  # type: ignore
    ]


class ImagenetFfcvDataModule(ImagenetDataModule):
    """Wrapper around the ImageNetDataModule that uses ffcv for the Train dataloader.

    Iterating over the dataloader can be a *lot* (~10x) faster than PyTorch (especially so if the
    image resolution is ramped up, see below).


    1. Copies the Imagenet dataset to SLURM_TMPDIR (same as parent class)
    2. Writes the dataset in ffcv format at SLURM_TMPRID/imagenet/train.ffcv
    3. Train dataloader reads from that file.

    The image resolution can be changed dynamically at each epoch (to match the ffcv-imagenet repo)
    based on the values in a configuration class.

    BUG: Using this DataModule with a PyTorch-Lightning Trainer is not currently performing well,
    even though the
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
        ffcv_train_transforms: Sequence[Operation | nn.Module] | None = None,
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
            # NOTE: RANDOM is a LOT slower than QUASI_RANDOM.
            order=OrderOption.QUASI_RANDOM,  # type: ignore
            os_cache=False,  # todo: investigate this option more.
            drop_last=self.drop_last,
            distributed=False,
            batches_ahead=3,
            seed=1234,
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if train_transforms is None and ffcv_train_transforms is None:
            # No ffcv transform or torchvision transforms were passed, use the ffcv equivalent
            # to the usual torchvision transforms.
            ffcv_train_transforms = default_ffcv_train_transforms(self.device)
        elif (
            train_transforms
            and not isinstance(train_transforms, nn.Module)
            and self.device.type == "cuda"
        ):
            raise RuntimeError(
                "Can't use cuda and old-style torchvision transforms. Upgrade "
                "torchvision to a more recent version and pass a nn.Sequential instead."
            )
        elif ffcv_train_transforms is None:
            # Only using torchvision transforms.
            pass
        else:
            # Using both torchvision transforms and ffcv transforms.
            pass

        self.ffcv_train_transforms = ffcv_train_transforms or []
        if isinstance(train_transforms, nn.Module):
            train_transforms = train_transforms.to(self.device)
        self.train_transforms = train_transforms

        # TODO: the validation split still requires the main dataset to be copied over,
        # even if we have a cached copy of the train ffcv dataset on SCRATCH.
        self._train_file = Path(self.data_dir) / "train.ffcv"
        self._val_file = Path(self.data_dir) / "val.ffcv"

        self.save_hyperparameters()
        # Note: defined in the LightningDataModule class, gets set when using a Trainer.
        self.trainer: Trainer | None = None
        self._current_epoch: int | None = None

    def prepare_data(self) -> None:
        super().prepare_data()

        # NOTE: We don't do this for val/test, since we can just always use the standard pytorch
        # train_done.txt and dataloaders.
        train_dataset = super().train_dataloader().dataset
        assert isinstance(train_dataset, UnlabeledImagenet)
        self._setup_ffcv_file_in_slurmtmpdir(
            dataset=train_dataset,
            ffcv_file=self._train_file,
            writer_config=self.writer_config,
            split="train",
        )
        # NOTE: Could also do this to use FFCV for the validation dataset. Not doing this atm.
        # valid_dataset = super().val_dataloader().dataset
        # assert isinstance(valid_dataset, UnlabeledImagenet)
        # # NOTE: Using the same writer config as the train dataset atm.
        # self._setup_ffcv_file_in_slurmtmpdir(
        #     dataset=valid_dataset,
        #     ffcv_file=self._val_file,
        #     writer_config=self.writer_config,
        #     split="valid",
        # )

    def _setup_ffcv_file_in_slurmtmpdir(
        self,
        dataset: UnlabeledImagenet,
        ffcv_file: Path,
        writer_config: DatasetWriterConfig,
        split: Literal["train", "valid"],
    ) -> None:
        if _done_file(ffcv_file).exists():
            return
        writer_hash = writer_config.get_hash()
        ffcv_file_on_scratch = SCRATCH / "imagenet" / f"{writer_hash}_{split}.ffcv"
        if ffcv_file_on_scratch.exists():
            print(
                f"Copying existing ffcv file from SCRATCH to SLURM_TMPDIR "
                f"({ffcv_file_on_scratch} -> {ffcv_file})"
            )
            shutil.copy(ffcv_file_on_scratch, ffcv_file)
            _done_file(ffcv_file).touch()
            return
        # If done txt file doesn't exist and we don't already have this file on scratch, we need
        # to rewrite the train.ffcv file
        _write_dataset(
            dataset,
            ffcv_file,
            writer_config=writer_config,
        )
        # Copy the ffcv file to $SCRATCH for the next time.
        # TODO: Should we remove other files, so we don't take too much space in SCRATCH?
        ffcv_file_on_scratch.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying ffcv file to SCRATCH for future runs ({ffcv_file_on_scratch})")
        shutil.copy(ffcv_file, ffcv_file_on_scratch)

        _done_file(ffcv_file).touch()

    def setup(self, stage: str) -> None:
        # Note: base doesn't really do anything with this here.
        super().setup(stage)
        if self.loader_config.get("distributed"):
            # TODO: Still a bit experimental. Don't rely on this being correct.
            this_gpu = self.trainer.local_rank if self.trainer else 0
            distributed.init_process_group(
                "nccl", rank=this_gpu, world_size=torch.cuda.device_count()
            )
            torch.cuda.set_device(this_gpu)

    def train_dataloader(self) -> Iterable[tuple[Tensor, Tensor]]:
        current_epoch = self.current_epoch
        res = self.img_resolution_config.get_resolution(current_epoch)
        print(
            (f"Epoch {current_epoch}: " if current_epoch is not None else "")
            + f"Loading images at {res}x{res} resolution"
        )
        image_pipeline: list[Operation | nn.Module] = [
            RandomResizedCropRGBImageDecoder((res, res)),
            *self.ffcv_train_transforms,
        ]
        label_pipeline: list[Operation | nn.Module] = [
            IntDecoder(),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.Squeeze(),
        ]
        if self.trainer is None:
            # When using PyTorch-Lightning, we don't want to add this ToDevice operation, because
            # PL already does it.
            label_pipeline.append(ffcv.transforms.ToDevice(self.device, non_blocking=True))

        use_extra_tv_transforms = False
        if isinstance(self.train_transforms, nn.Module):
            # Include the 'new-style' transform modules in the image pipeline of ffcv.
            image_pipeline.append(self.train_transforms)
        elif self.train_transforms:
            # We have some 'old-style' torchvision transforms.
            use_extra_tv_transforms = True

        loader = Loader(
            str(self._train_file),
            pipelines={"image": image_pipeline, "label": label_pipeline},
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **self.loader_config,
        )

        if use_extra_tv_transforms:
            assert self.train_transforms
            # Apply the Torchvision transforms after the FFCV transforms.
            return ApplyTransformLoader(loader, self.train_transforms)
        return loader

    def val_dataloader(self) -> DataLoader:
        return super().val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return super().test_dataloader()

    @property
    def current_epoch(self) -> int | None:
        """The current training epoch if using a Trainer of PyTorchLightning, else None."""
        if self._current_epoch is not None:
            return self._current_epoch
        if self.trainer is not None:
            return self.trainer.current_epoch
        return None

    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        """Manually set the current training epoch if not using a Trainer of PyTorchLightning."""
        self._current_epoch = value


X_in = TypeVar("X_in")
X = TypeVar("X")
Y = TypeVar("Y")


class ApplyTransformLoader(Iterable[tuple[X, Y]]):
    def __init__(self, data: Iterable[tuple[X_in, Y]], transform: Callable[[X_in], X]):
        self.data_source = data
        self.transform = transform

    def __iter__(self) -> Iterable[tuple[X, Y]]:
        for x, y in self.data_source:
            yield self.transform(x), y

    def __len__(self) -> int:
        return len(self.data_source)  # type: ignore


def _write_dataset(
    dataset: UnlabeledImagenet,
    dataset_ffcv_path: Path,
    writer_config: DatasetWriterConfig,
) -> None:
    assert isinstance(dataset, UnlabeledImagenet)
    # NOTE: We write the dataset without any transforms.
    # TODO: Make sure that the dataset.transforms is actually where the transforms are, and that
    # we're correctly removing those. Otherwise we're writing the dataset with some
    # transformations!
    assert hasattr(dataset, "transform")
    dataset.transform = None
    if hasattr(dataset, "label_transform"):
        dataset.label_transform = None  # type: ignore
    assert hasattr(dataset, "target_transform")
    dataset.target_transform = None
    dataset_done_file = _done_file(dataset_ffcv_path)
    if dataset_done_file.exists():
        return

    print(f"Writing dataset in FFCV format at {dataset_ffcv_path}")
    dataset_ffcv_path.parent.mkdir(parents=True, exist_ok=True)
    writer_config = writer_config
    write_path = dataset_ffcv_path
    write_path = Path(write_path)
    writer = DatasetWriter(
        str(write_path),
        {
            "image": RGBImageField(
                write_mode=writer_config.write_mode,
                max_resolution=writer_config.max_resolution,
                compress_probability=writer_config.compress_probability or 0.0,
                jpeg_quality=writer_config.jpeg_quality,
            ),
            "label": IntField(),
        },
        num_workers=writer_config.num_workers,
    )
    if writer_config.subset == -1:
        indices = list(range(len(dataset)))  # type: ignore
    else:
        indices = list(range(writer_config.subset))
    writer.from_indexed_dataset(dataset, chunksize=writer_config.chunk_size, indices=list(indices))
    dataset_done_file.touch()


def _done_file(path: Path) -> Path:
    return path.with_name(path.name + "_done.txt")
