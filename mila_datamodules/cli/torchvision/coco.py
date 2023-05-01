from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Literal

import torchvision.datasets as tvd
from typing_extensions import TypeVar

from mila_datamodules.cli.blocks import (
    CallDatasetConstructor,
    Compose,
    ExtractArchives,
    MakeSymlinksToDatasetFiles,
    StopOnSuccess,
)
from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.clusters.utils import get_slurm_tmpdir

from ..types import P

logger = get_logger(__name__)
# from simple_parsing import ArgumentParser
SLURM_TMPDIR = get_slurm_tmpdir()


CocoType = TypeVar("CocoType", tvd.CocoCaptions, tvd.CocoDetection, default=tvd.CocoDetection)
CocoVariant = Literal["captions", "instances", "panoptic", "person_keypoints", "stuff"]
CocoSplit = Literal["train", "val", "test", "unlabeled"]


def _check_coco_is_setup(dataset_type: type[CocoType], variant: CocoVariant, split: CocoSplit):
    def _check_coco_setup(
        root: str | Path,
        annFile: str = "annotations/captions_train2017.json",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> str:
        fn = CallDatasetConstructor(dataset_type, get_index=0)
        fn(
            f"{root}/{split}2017",
            annFile=f"{root}/annotations/{variant}_{split}2017.json",
            *args,
            **kwargs,
        )
        return f"{root}/{split}2017"

    return _check_coco_setup


# TODO: CocoCaptions is a bit weird.
# - If we prepare everything right, we still have to call the constructor with
# root=<root>/train2017.


def prepare_coco(
    dataset_type: type[CocoType],
    root: str | Path,
    variant: CocoVariant,
    split: CocoSplit,
):
    return Compose(
        StopOnSuccess(
            _check_coco_is_setup(dataset_type, variant=variant, split=split),
            continue_if_raised=FileNotFoundError,
        ),
        MakeSymlinksToDatasetFiles(f"{root}/coco/2017"),
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
        _check_coco_is_setup(dataset_type, variant=variant, split=split),
    )


def PrepareCocoDetection(datasets_dir: Path, variant: CocoVariant, split: CocoSplit):
    return prepare_coco(tvd.CocoDetection, datasets_dir, variant=variant, split=split)


def PrepareCocoCaptions(datasets_dir: Path, variant: CocoVariant, split: CocoSplit):
    return prepare_coco(tvd.CocoCaptions, datasets_dir, variant=variant, split=split)


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

    # TODO: Check other splits than just train and val.
    split: CocoSplit = "train"
    """Which split to prepare."""

    def to_dataset_kwargs(self) -> dict:
        """Returns the dataset constructor arguments that are to be passed to the dataset
        preparation function."""
        dataset_kwargs = dataclasses.asdict(self)
        dataset_kwargs.pop("variant")
        dataset_kwargs.pop("split")

        if not self.annFile:
            annFile = f"annotations/{self.variant}_{self.split}2017.json"
            dataset_kwargs["annFile"] = f"{self.root}/{annFile}"

        return dataset_kwargs

    def code_to_use(self) -> str:
        slurm_tmpdir = get_slurm_tmpdir()
        coco_type = "CocoCaptions" if self.variant == "captions" else "CocoDetection"
        # FIXME: Remove this assertion
        assert self.root == slurm_tmpdir / "datasets", "fixme"
        return (
            f"{coco_type}(\n"
            + ("""    root=f"{os.environ['SLURM_TMPDIR']}/datasets/""" + f"{self.split}2017,\n")
            + (
                """    annFile=f"{os.environ['SLURM_TMPDIR']}/"""
                + str(Path(self.annFile).relative_to(slurm_tmpdir))
                + '"\n'
            )
            + ")"
        )


@dataclass
class CocoCaptionArgs(CocoDetectionArgs):
    variant: CocoVariant = "captions"
