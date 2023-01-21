from __future__ import annotations

import os
import shutil
from typing import Sequence

import pytest
from filelock import FileLock
from torch.utils.data import Dataset

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_slurm_tmpdir, on_slurm_cluster
from mila_datamodules.registry import dataset_roots_per_cluster, is_stored_on_cluster


@pytest.fixture(scope="session", autouse=True)
def patch_dataset_for_local_cluster():
    # TODO: @lebrice: This is super hard-coded just for my machine.
    if CURRENT_CLUSTER is not Cluster._local:
        return

    all_datasets: set[type] = set(
        sum([list(d.keys()) for d in dataset_roots_per_cluster.values()], [])
    )
    # Add the dataset in the `_local` cluster if it is stored in the FAKE_SCRATCH directory.
    dataset_roots_per_cluster[Cluster._local] = {
        dataset: os.environ["FAKE_SCRATCH"]
        for dataset in all_datasets
        if is_stored_on_cluster(dataset_cls=dataset, cluster=Cluster._local)
    }
    dataset_roots_per_cluster[Cluster._local] = {}


def skip_if_not_stored_on_current_cluster(dataset: type[Dataset]):
    return pytest.mark.skipif(
        not is_stored_on_cluster(dataset, CURRENT_CLUSTER),
        reason=f"Dataset isn't stored on {CURRENT_CLUSTER} cluster",
    )


# TODO: Remove this, or make it clear that it's actually partially running the test (uses `xfail`).


def xfail_if_not_stored_on_current_cluster(dataset: type[Dataset]):
    return pytest.mark.xfail(
        condition=not is_stored_on_cluster(dataset, CURRENT_CLUSTER),
        reason=f"Dataset isn't stored on {CURRENT_CLUSTER} cluster",
    )


def only_runs_on_cluster(cluster: Cluster | Sequence[Cluster]):
    """When `cluster` is None, only runs when we're on any SLURM cluster.

    When `cluster` is set, then only runs when we're on that specific cluster.
    """
    clusters = [cluster] if isinstance(cluster, Cluster) else list(cluster)
    reason = f"Test only runs on {'|'.join(c.name for c in clusters)}"
    return pytest.mark.skipif(
        CURRENT_CLUSTER not in clusters,
        reason=reason,
    )


def needs_slurm_cluster():
    """When `cluster` is None, only runs when we're on any SLURM cluster.

    When `cluster` is set, then only runs when we're on that specific cluster.
    """
    return pytest.mark.skipif(not on_slurm_cluster(), reason="Test only runs on SLURM clusters")


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
