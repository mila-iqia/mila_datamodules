from __future__ import annotations

import enum
import os
from logging import getLogger as get_logger
from pathlib import Path

from mila_datamodules.clusters.utils import setup_slurm_env_variables

logger = get_logger(__name__)


class Cluster(enum.Enum):
    """Enum of the different clusters available."""

    Mila = enum.auto()
    Cedar = enum.auto()
    Beluga = enum.auto()
    Graham = enum.auto()
    Narval = enum.auto()
    _local = enum.auto()

    @classmethod
    def current(cls) -> Cluster:
        setup_slurm_env_variables()
        cluster_name = os.environ["SLURM_CLUSTER_NAME"]
        if cluster_name == "mila":
            return cls.Mila
        # TODO: Double-check the value of this environment variable in other clusters:
        if cluster_name == "beluga":
            return cls.Beluga
        if cluster_name == "graham":
            return cls.Graham
        if cluster_name == "cedar":
            return cls.Cedar
        if cluster_name == "narval":
            return cls.Narval
        if cluster_name == "local":
            return cls._local
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
