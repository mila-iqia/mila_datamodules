from __future__ import annotations

import enum
import functools
import os
from logging import getLogger as get_logger
from pathlib import Path
from shutil import which


logger = get_logger(__name__)


@functools.cache
def on_slurm_cluster() -> bool:
    """Return `True` if the current process is running on a SLURM cluster."""
    return which("srun") is not None


def current_cluster_name() -> str | None:
    if "CC_CLUSTER" in os.environ:
        return os.environ["CC_CLUSTER"]
    if "/home/mila" in str(Path.home()):
        return "mila"
    return None


def on_compute_node() -> bool:
    return on_slurm_cluster() and ("SLURM_JOB_ID" in os.environ or "SLURM_JOBID" in os.environ)


def on_login_node() -> bool:
    return on_slurm_cluster() and not on_compute_node()


class Cluster(enum.Enum):
    """Enum of the different clusters available."""

    Mila = enum.auto()
    Cedar = enum.auto()
    Beluga = enum.auto()
    Graham = enum.auto()
    Narval = enum.auto()

    # _local_debug = enum.auto()
    # """ IDEA: Fake SLURM cluster for local debugging.
    # Uses the values of the `FAKE_SCRATCH` and `FAKE_SLURM_TMPDIR` environment variables.
    # """

    @classmethod
    def current(cls) -> Cluster | None:
        """Returns the current cluster when called on a SLURM cluster and `None` otherwise."""
        # TODO: Find a more reliable way of determining if we are on a SLURM cluster.
        if not on_slurm_cluster():
            return None
        cluster_name = current_cluster_name()
        if cluster_name is None:
            raise RuntimeError(
                "On a SLURM cluster, but could not determine the cluster name! "
                "Please make an issue on the `mila-datamodules` repository."
            )
        return cls[cluster_name.capitalize()]

    @property
    def slurm_tmpdir(self) -> Path:
        """Returns the 'fast' directory where files should be stored for quick read/writtes."""
        return Path(os.environ["SLURM_TMPDIR"])

    @property
    def scratch(self) -> Path:
        """Returns the writeable directory where checkpoints / code / general data should be
        stored."""
        return Path(os.environ["SCRATCH"])
