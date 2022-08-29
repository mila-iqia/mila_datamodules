from __future__ import annotations

import enum
import itertools
import os
import socket
import subprocess
import sys
from logging import getLogger as get_logger
from pathlib import Path

from .utils import setup_slurm_env_variables

logger = get_logger(__name__)


class ClusterType(enum.Enum):
    """Enum of the different clusters available."""

    MILA = enum.auto()
    CEDAR = enum.auto()
    BELUGA = enum.auto()
    GRAHAM = enum.auto()

    @classmethod
    def current(cls) -> ClusterType:
        setup_slurm_env_variables()
        cluster_name = os.environ["SLURM_CLUSTER_NAME"]

        full_hostname = socket.getfqdn()
        if cluster_name == "mila" or full_hostname.endswith(".server.mila.quebec"):
            return cls.MILA
        if cluster_name == "beluga" or full_hostname.endswith(".beluga.computecanada.ca"):
            return cls.BELUGA

        raise NotImplementedError(f"Unknown cluster: {cluster_name}, hostname: {full_hostname}")

    @property
    def torchvision_dir(self) -> Path:
        if self is ClusterType.MILA:
            return Path("/network/datasets/torchvision")
        raise NotImplementedError(self)  # todo

    @property
    def fast_data_dir(self) -> Path:
        """Returns the 'fast' directory where datasets are stored for quick read/writtes."""
        return Path(os.environ["SLURM_TMPDIR"])


current = ClusterType.current()
