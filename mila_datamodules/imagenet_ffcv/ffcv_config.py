from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import cv2  # noqa
from ffcv.fields import Field, IntField, RGBImageField
from ffcv.loader import OrderOption
from ffcv.writer import DatasetWriter
from torch.utils.data import Dataset


@dataclass(frozen=True, unsafe_hash=True)
class DatasetWriterConfig:
    """Arguments to give the FFCV DatasetWriter."""

    max_resolution: int
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

    def write(self, dataset: Dataset, write_path: str | Path):
        write_path = Path(write_path)
        writer = DatasetWriter(
            str(write_path),
            {
                "image": RGBImageField(
                    write_mode=self.write_mode,
                    max_resolution=self.max_resolution,
                    compress_probability=self.compress_probability or 0.0,
                    jpeg_quality=self.jpeg_quality,
                ),
                "label": IntField(),
            },
            num_workers=self.num_workers,
        )
        writer.from_indexed_dataset(dataset, chunksize=self.chunk_size)


@dataclass(frozen=True, unsafe_hash=True)
class ImageResolutionConfig:
    """Configuration for the resolution of the images when loading from the written ffcv dataset."""

    min_res: int = 160
    """the minimum (starting) resolution"""

    max_res: int = 224
    """the maximum (starting) resolution"""

    end_ramp: int = 0
    """ when to stop interpolating resolution """

    start_ramp: int = 0
    """ when to start interpolating resolution """

    def get_resolution(self, epoch: int | None) -> int:
        """Copied over from the FFCV example, where they ramp up the resolution during training."""
        assert self.min_res <= self.max_res
        if epoch is None:
            return self.max_res

        if epoch <= self.start_ramp:
            return self.min_res

        if epoch >= self.end_ramp:
            return self.max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp(
            [epoch], [self.start_ramp, self.end_ramp], [self.min_res, self.max_res]
        )
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res


class FfcvLoaderConfig(TypedDict, total=False):
    os_cache: bool
    """ Leverages the operating for caching purposes. This is beneficial when there is enough
    memory to cache the dataset and/or when multiple processes on the same machine training using
    the same dataset. See https://docs.ffcv.io/performance_guide.html for more information.
    """

    order: Literal[OrderOption.RANDOM, OrderOption.SEQUENTIAL]
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
    """Dictonary informing the loader of the types associated to fields that are using a custom
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
