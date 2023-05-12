from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.clusters.utils import get_slurm_tmpdir
from mila_datamodules.types import VD


@dataclass
class VisionDatasetArgs(DatasetArguments[VD]):
    root: Path = field(default_factory=lambda: get_slurm_tmpdir() / "datasets")


def dataset_name(dataset_type: type[VD]) -> str:
    return getattr(dataset_type, "__name__", str(dataset_type)).lower()
