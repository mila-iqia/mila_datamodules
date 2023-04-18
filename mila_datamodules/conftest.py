from __future__ import annotations

import os
import random
import shutil

import numpy as np
import pytest
import torch
from filelock import FileLock

from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import (
    get_scratch_dir,
    get_slurm_tmpdir,
    on_slurm_cluster,
)

TEST_SEED = 123


@pytest.fixture()
def seed():
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    yield TEST_SEED


# @pytest.fixture(autouse=True)
# def datadir():
#     return Path(__file__).parent.parent / "data"


seeded = pytest.mark.usefixtures("seed")


def pytest_xdist_auto_num_workers(config):
    """Return the number of workers to spawn when ``--numprocesses=auto`` is given in the command-
    line."""
    if on_slurm_cluster():
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return os.cpu_count()


# Note: Turning this off for now, so I can test ImageNet quicker.
@pytest.fixture(autouse=False, scope="session")
def clear_slurm_tmpdir():
    """Clears the SLURM_TMPDIR/data before running any tests.

    This is unfortunately necessary so that each test run is unaffected by other runs. I was having
    weird errors because the datasets were half-moved to SLURM_TMPDIR, so reading them on the next
    run would fail.
    """
    slurm_tmpdir = get_slurm_tmpdir(default="")
    if slurm_tmpdir.exists() and slurm_tmpdir.is_dir() and len(list(slurm_tmpdir.iterdir())):
        with FileLock(slurm_tmpdir / "data.lock"):
            if (slurm_tmpdir / "data").exists():
                # NOTE: The `chmod_recursive` doesn't seem to work atm. It's safe to assume we're
                # on Linux, so this is fine for now.
                os.system(f"chmod --recursive +rwx {slurm_tmpdir}/data")
                shutil.rmtree(slurm_tmpdir / "data")
    yield


@pytest.fixture(autouse=True, scope="session")
def scratch_data_dir(tmp_path_factory: pytest.TempPathFactory):
    """Creates a fake $SCRATCH directory in SLURM_TMPDIR.

    This is used so the tests don't actually use the real $SCRATCH directory.
    """
    # TODO: Fix this Not sure what it should be doing anymore!
    cluster = Cluster.current()
    if cluster is Cluster._mock:
        yield
        return  # We don't need to do anything if we're on the mock cluster.
    if cluster is None:
        yield
        return  # We can't do anything when we're not on a cluster.

    fake_scratch_data_dir = get_slurm_tmpdir() / "fake_scratch"

    if fake_scratch_data_dir.exists():
        # Make sure it's completely empty.
        # BUG: This may fail when running multiple tests in parallel.
        try:
            shutil.rmtree(fake_scratch_data_dir)
        except FileNotFoundError:
            pass
    # NOTE: exist_ok here because we might be parallelizing the tests with multiple workers.
    fake_scratch_data_dir.mkdir(parents=False, exist_ok=True)

    # save a copy for safekeeping.
    original_scratch = os.environ.get("SCRATCH")

    os.environ["SCRATCH"] = str(fake_scratch_data_dir)
    assert cluster
    assert get_scratch_dir() == fake_scratch_data_dir

    yield

    if original_scratch is not None:
        os.environ["SCRACTH"] = original_scratch
