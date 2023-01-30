from __future__ import annotations

import os
import shutil

import pytest
from filelock import FileLock

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_slurm_tmpdir
from mila_datamodules.registry import dataset_roots_per_cluster, is_stored_on_cluster


@pytest.fixture(scope="session", autouse=True)
def patch_dataset_for_local_cluster():
    if CURRENT_CLUSTER is not Cluster._mock:
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
    data_dir_when_not_on_cluster = tmp_path_factory.mktemp("scratch_data_dir")
    fake_scratch_data_dir = get_slurm_tmpdir(default=data_dir_when_not_on_cluster) / "fake_scratch"

    if fake_scratch_data_dir.exists():
        # Make sure it's completely empty.
        shutil.rmtree(fake_scratch_data_dir)
    original_scratch = os.environ.get("SCRATCH")
    # NOTE: exist_ok here because we might be parallelizing the tests with multiple workers.
    fake_scratch_data_dir.mkdir(parents=False, exist_ok=True)

    os.environ["SCRATCH"] = str(fake_scratch_data_dir)
    if CURRENT_CLUSTER:
        assert CURRENT_CLUSTER.scratch == fake_scratch_data_dir
    yield

    if original_scratch is not None:
        os.environ["SCRACTH"] = original_scratch
