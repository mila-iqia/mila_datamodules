from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import (
    Generic,
)

from ._types import VD


@dataclass
class DatasetArguments(Generic[VD]):
    """Arguments for the dataset preparation."""

    # root: Path = get_slurm_tmpdir() / "datasets"
    # """Root directory where images are downloaded to."""

    def to_dataset_kwargs(self) -> dict:
        return dataclasses.asdict(self)
