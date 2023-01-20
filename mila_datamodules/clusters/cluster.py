from __future__ import annotations

import enum
import os
from logging import getLogger as get_logger
from pathlib import Path

from mila_datamodules.clusters.utils import setup_slurm_env_variables

logger = get_logger(__name__)


def current_cluster_name() -> str | None:
    if "cc_cluster" in os.environ:
        return os.environ["cc_cluster"]
    if "/home/mila" in str(Path.home()):
        return "mila"
    try:
        slurm_env_vars = setup_slurm_env_variables()
        return slurm_env_vars.SLURM_CLUSTER_NAME
    except NotImplementedError:
        return None


class Cluster(enum.Enum):
    """Enum of the different clusters available."""

    Mila = enum.auto()
    Cedar = enum.auto()
    Beluga = enum.auto()
    Graham = enum.auto()
    Narval = enum.auto()

    # _local_debug = enum.auto()
    """ Fake SLURM cluster for local debugging.
    
    Uses the values of the `FAKE_SCRATCH` and `FAKE_SLURM_TMPDIR` environment variables.    
    """

    @classmethod
    def current(cls) -> Cluster | None:
        """Returns the current cluster when called on a SLURM cluster and `None` otherwise."""
        # TODO: Find a more reliable way of determining if we are on a SLURM cluster.

        cluster_name = current_cluster_name()

        if not cluster_name:
            # Not on a SLURM cluster.
            return None
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
