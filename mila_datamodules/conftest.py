import os
import shutil

import pytest
from filelock import FileLock

from mila_datamodules.clusters import CURRENT_CLUSTER, SLURM_TMPDIR


@pytest.fixture(autouse=True, scope="session")
def clear_slurm_tmpdir():
    """Clears the SLURM_TMPDIR/data before running any tests.

    This is unfortunately necessary so that each test run is unaffected by other runs. I was having
    weird errors because the datasets were half-moved to SLURM_TMPDIR, so reading them on the next
    run would fail.
    """
    with FileLock(SLURM_TMPDIR / "data.lock"):
        if (SLURM_TMPDIR / "data").exists():
            # NOTE: The `chmod_recursive` doesn't seem to work atm. It's safe to assume we're on Linux,
            # so this is fine for now.
            # chmod_recursive(SLURM_TMPDIR / "data", 0o644)
            os.system(f"chmod --recursive +rwx {SLURM_TMPDIR}/data")
            shutil.rmtree(SLURM_TMPDIR / "data")
    yield


@pytest.fixture(autouse=True, scope="session", params=["mila", "beluga"])
def cluster(request):
    """A fixture that makes all the tests run on all the clusters!"""
    host: str = request.param
    if host != CURRENT_CLUSTER.name.lower():
        pytest.skip(f"Runs on the {host} cluster (we're on {CURRENT_CLUSTER.name})")
    yield host
