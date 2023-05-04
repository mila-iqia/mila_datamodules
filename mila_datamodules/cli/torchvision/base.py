from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mila_datamodules.cli.dataset_args import DatasetArguments
from mila_datamodules.cli.types import VD
from mila_datamodules.clusters.utils import get_slurm_tmpdir


@dataclass
class VisionDatasetArgs(DatasetArguments[VD]):
    root: Path = field(default_factory=lambda: get_slurm_tmpdir() / "datasets")
