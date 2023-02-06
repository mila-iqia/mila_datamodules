from __future__ import annotations

import os
import random
import shutil

import numpy as np
import pytest
import torch
from filelock import FileLock

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import (
    get_scratch_dir,
    get_slurm_tmpdir,
    on_slurm_cluster,
)
from mila_datamodules.registry import dataset_roots_per_cluster, is_stored_on_cluster
from mila_datamodules.vision.imagenet.imagenet import num_cpus_to_use

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
        return num_cpus_to_use()
    return os.cpu_count()


@pytest.fixture(scope="session", autouse=True)
def patch_dataset_for_local_cluster():
    if CURRENT_CLUSTER is not Cluster._mock:
        yield
        return

    all_datasets: set[type] = set(
        sum([list(d.keys()) for d in dataset_roots_per_cluster.values()], [])
    )
    # Add the dataset in the `_local` cluster if it is stored in the FAKE_SCRATCH directory.
    dataset_roots_per_cluster[Cluster._mock] = {
        dataset: os.environ["FAKE_SCRATCH"]
        for dataset in all_datasets
        if is_stored_on_cluster(dataset_cls=dataset, cluster=Cluster._mock)
    }
    yield
    dataset_roots_per_cluster[Cluster._mock] = {}


@pytest.fixture(autouse=True, scope="session")
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
                # NOTE: The `chmod_recursive` doesn't seem to work atm. It's safe to assume we're on Linux,
                # so this is fine for now.
                # chmod_recursive(SLURM_TMPDIR / "data", 0o644)
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
        shutil.rmtree(fake_scratch_data_dir)
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
