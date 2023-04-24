from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import (
    Literal,
)

import torchvision.datasets as tvd
from typing_extensions import TypeVar

from mila_datamodules.cli.blocks import (
    Compose,
    ExtractArchives,
    MakeSymlinksToDatasetFiles,
    StopOnSucess,
)
from mila_datamodules.cli.torchvision.dataset_args import DatasetArguments
from mila_datamodules.clusters.utils import get_slurm_tmpdir

from .types import P

logger = get_logger(__name__)
# from simple_parsing import ArgumentParser
SLURM_TMPDIR = get_slurm_tmpdir()


CocoType = TypeVar("CocoType", tvd.CocoCaptions, tvd.CocoDetection, default=tvd.CocoDetection)
CocoVariant = Literal["captions", "instances", "panoptic", "person_keypoints", "stuff"]


def check_coco_is_setup(
    dataset_type: type[CocoType],
    variant: CocoVariant,
):
    def _check_coco_setup(
        root: str | Path,
        annFile: str = "annotations/captions_train2017.json",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> str:
        dataset = dataset_type(
            f"{root}/train2017",
            f"{root}/annotations/{variant}_train2017.json",
            *args,
            **kwargs,
        )
        dataset[0]
        dataset = dataset_type(
            f"{root}/val2017",
            f"{root}/annotations/{variant}_val2017.json",
            *args,
            **kwargs,
        )
        dataset[0]

        return str(root)

    return _check_coco_setup


# TODO: CocoCaptions is a bit weird.
# - If we prepare everything right, we still have to call the constructor with
# root=<root>/train2017.


def _prepare_coco(
    dataset_type: type[CocoType], datasets_dir: Path, variant: CocoVariant = "stuff"
):
    return Compose(
        StopOnSucess(
            check_coco_is_setup(dataset_type, variant=variant),
            continue_if_raised=[FileNotFoundError],
        ),
        MakeSymlinksToDatasetFiles(f"{datasets_dir}/coco/2017"),
        ExtractArchives(
            archives={
                "test2017.zip": ".",
                "train2017.zip": ".",
                "val2017.zip": ".",
                "annotations/annotations_trainval2017.zip": ".",
                "annotations/image_info_test2017.zip": ".",
                "annotations/panoptic_annotations_trainval2017.zip": ".",
                "annotations/stuff_annotations_trainval2017.zip": ".",
            }
        ),
        check_coco_is_setup(dataset_type, variant=variant),
    )


def PrepareCocoDetection(datasets_dir: Path, variant: CocoVariant = "stuff"):
    return _prepare_coco(tvd.CocoDetection, datasets_dir, variant)


def PrepareCocoCaptions(datasets_dir: Path, variant: CocoVariant = "captions"):
    return _prepare_coco(tvd.CocoCaptions, datasets_dir, variant)


@dataclass
class CocoDetectionArgs(DatasetArguments[tvd.CocoDetection]):
    """Command-line arguments used when preparing the CocoCaptions dataset."""

    root: Path = get_slurm_tmpdir() / "datasets"

    annFile: str = ""
    """Path to json annotation file.

    Auto-selected based on `variant`, or can be passed explicitly.
    """

    # NOTE: These args below are not actually passed to the dataset constructor. They are used to
    # create the dataset arguments in a more user-friendly way.

    variant: CocoVariant = "stuff"
    """Which variant of the Coco dataset to use."""

    split: Literal["train", "val"] = "train"
    """Which split to prepare."""

    def __post_init__(self):
        if not self.annFile:
            annFile = f"annotations/{self.variant}_{self.split}2017.json"
            self.annFile = f"{self.root}/{annFile}"

    def to_dataset_kwargs(self) -> dict:
        dataset_kwargs = dataclasses.asdict(self)
        dataset_kwargs.pop("variant")
        dataset_kwargs.pop("split")
        return dataset_kwargs
