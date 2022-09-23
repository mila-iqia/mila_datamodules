from __future__ import annotations

import enum
import os
from logging import getLogger as get_logger
from pathlib import Path

from mila_datamodules.clusters.utils import setup_slurm_env_variables

logger = get_logger(__name__)


class ClusterType(enum.Enum):
    """Enum of the different clusters available."""

    MILA = enum.auto()
    CEDAR = enum.auto()
    BELUGA = enum.auto()
    GRAHAM = enum.auto()
    NARVAL = enum.auto()

    @classmethod
    def current(cls) -> ClusterType:
        setup_slurm_env_variables()
        cluster_name = os.environ["SLURM_CLUSTER_NAME"]
        if cluster_name == "mila":
            return cls.MILA
        # TODO: Double-check the value in other clusters:
        if cluster_name == "beluga":
            return cls.BELUGA
        if cluster_name == "graham":
            return cls.GRAHAM
        if cluster_name == "cedar":
            return cls.CEDAR
        if cluster_name == "narval":
            return cls.NARVAL

        raise NotImplementedError(f"Unknown cluster: {cluster_name}")

    @property
    def slurm_tmpdir(self) -> Path:
        """Returns the 'fast' directory where files should be stored for quick read/writtes."""
        return Path(os.environ["SLURM_TMPDIR"])

    @property
    def scratch(self) -> Path:
        """Returns the writeable directory where checkpoints / code / general data should be
        stored."""
        return Path(os.environ["SCRATCH"])
