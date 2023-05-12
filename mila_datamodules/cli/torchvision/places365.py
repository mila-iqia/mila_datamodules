from __future__ import annotations

import dataclasses
from dataclasses import field
from pathlib import Path

import torchvision.datasets as tvd
from typing_extensions import Literal

from mila_datamodules.blocks import (
    CallDatasetConstructor,
    Compose,
    CopyFiles,
    MakeSymlinksToDatasetFiles,
    SkipRestIfThisWorks,
)
from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.clusters.utils import get_slurm_tmpdir

Split = Literal["train-standard", "train-challenge", "val"]


# TODO: Add arguments to select which splits and version of Places365 to prepare:
# (train-standard, train-challenge, val) x (small=True, small=False)
def prepare_places365(datasets_dir: Path):
    return Compose(
        SkipRestIfThisWorks(
            CallDatasetConstructor(tvd.Places365),
            continue_if_raised=(FileNotFoundError, RuntimeError),
        ),
        CopyFiles(
            {
                "categories_places365.txt": f"{datasets_dir}/places365/categories_places365.txt",
            }
        ),
        MakeSymlinksToDatasetFiles(
            {
                "filelist_places365-standard.tar": (
                    f"{datasets_dir}/places365/filelist_places365-standard.tar"
                ),
                "filelist_places365-challenge.tar": (
                    f"{datasets_dir}/places365.var/places365_challenge/filelist_places365-challenge.tar"
                ),
                "test_256.tar": f"{datasets_dir}/places365/256/test_256.tar",
                "test_large.tar": f"{datasets_dir}/places365/large/test_large.tar",
                "val_256.tar": f"{datasets_dir}/places365/256/val_256.tar",
                "val_large.tar": f"{datasets_dir}/places365/large/val_large.tar",
                "train_256_places365standard.tar": (
                    f"{datasets_dir}/places365/256/train_256_places365standard.tar"
                ),
                "train_large_places365standard.tar": (
                    f"{datasets_dir}/places365/large/train_large_places365standard.tar"
                ),
                "lmdb_places365standard.tar": (
                    f"{datasets_dir}/places365/lmdb/lmdb_places365standard.tar"
                ),
                "places365standard_easyformat.tar": (
                    f"{datasets_dir}/places365/easyformat/places365standard_easyformat.tar"
                ),
                "train_256_places365challenge.tar": (
                    f"{datasets_dir}/places365.var/places365_challenge/256/train_256_places365challenge.tar"
                ),
                "train_large_places365challenge.tar": (
                    f"{datasets_dir}/places365.var/places365_challenge/large/train_large_places365challenge.tar"
                ),
                # NOTE: Duplicates of the files above:
                # "categories_places365.txt": (
                #     f"{datasets_dir}/places365.var/places365_challenge/categories_places365.txt"
                # ),
                # "test_256.tar": (
                #     f"{datasets_dir}/places365.var/places365_challenge/256/test_256.tar"
                # ),
                # "val_256.tar": (
                #     f"{datasets_dir}/places365.var/places365_challenge/256/val_256.tar"
                # ),
                # "test_large.tar": (
                #     f"{datasets_dir}/places365.var/places365_challenge/large/test_large.tar"
                # ),
                # "val_large.tar": (
                #     f"{datasets_dir}/places365.var/places365_challenge/large/val_large.tar"
                # ),
            }
        ),
        # NOTE: Calling tvd.Places365(..., download=True) raises an error if the archives have
        # already been extracted.
        CallDatasetConstructor(
            tvd.Places365,
            extract_and_verify_archives=True,
        ),
    )


@dataclasses.dataclass
class Places365Args(DatasetArguments[tvd.Places365]):
    """Command-line arguments used when preparing the CocoCaptions dataset."""

    root: Path = field(default_factory=lambda: get_slurm_tmpdir() / "datasets")

    split: Literal["train-standard", "train-challenge", "val"] = "train-standard"

    small: bool = False

    def to_dataset_kwargs(self) -> dict:
        """Returns the dataset constructor arguments that are to be passed to the dataset
        preparation function."""
        return dataclasses.asdict(self)
