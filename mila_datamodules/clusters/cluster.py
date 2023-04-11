from __future__ import annotations

import enum
from logging import getLogger as get_logger

logger = get_logger(__name__)


class Cluster(enum.Enum):
    """Enum of the different clusters available."""

    Mila = "mila"
    Cedar = "cedar"
    Beluga = "beluga"
    Graham = "graham"
    Narval = "narval"
    _mock = "_mock"
    """ TODO: IDEA: Fake SLURM cluster for local debugging.
    Uses the values of the `FAKE_SCRATCH` and `FAKE_SLURM_TMPDIR` environment variables.
    """

    @classmethod
    def current_or_error(cls) -> Cluster:
        """Returns the current cluster.

        Raises an error when called outside of a SLURM cluster.
        """
        current = cls.current()
        if current is None:
            from mila_datamodules.errors import NotOnSlurmClusterError

            raise NotOnSlurmClusterError(
                "The `current_cluster` method (and everything that uses it) can only be used "
                "on a SLURM cluster."
            )
        return current

    # TODO: Decide whether this should return None or _local when called on a non-SLURM cluster.
    @classmethod
    def current(cls) -> Cluster | None:
        """Returns the current cluster when called on a SLURM cluster and `None` otherwise."""
        from mila_datamodules.clusters.utils import (
            current_cluster_name,
            on_fake_slurm_cluster,
            on_slurm_cluster,
        )

        if on_fake_slurm_cluster():
            return cls._mock
        elif not on_slurm_cluster():
            return None
        cluster_name = current_cluster_name()
        if cluster_name is None:
            raise RuntimeError(
                "On a SLURM cluster, but could not determine the cluster name! "
                "Please make an issue on the `mila-datamodules` repository."
            )
        return cls[cluster_name.capitalize()]
