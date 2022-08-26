from __future__ import annotations

import enum
import os
import socket
from pathlib import Path
from typing import TypeVar

from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset


class ClusterType(enum.Enum):
    """Enum of the different clusters available."""

    MILA = enum.auto()
    CEDAR = enum.auto()
    BELUGA = enum.auto()
    GRAHAM = enum.auto()

    @classmethod
    def current(cls) -> ClusterType:
        cluster_name = os.environ.get("SLURM_CLUSTER_NAME")
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
