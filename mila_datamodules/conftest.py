from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from mila_datamodules.clusters.utils import (
    on_slurm_cluster,
)

TEST_SEED = 123


@pytest.fixture(autouse=True)
def seed():
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    yield TEST_SEED


def pytest_xdist_auto_num_workers(config):
    """Return the number of workers to spawn when ``--numprocesses=auto`` is given in the command-
    line."""
    if on_slurm_cluster():
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return os.cpu_count()
