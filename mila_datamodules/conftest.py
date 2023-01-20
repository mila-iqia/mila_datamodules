import os
import shutil

import pytest
from filelock import FileLock

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.utils import get_slurm_tmpdir


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


@pytest.fixture(autouse=True, scope="session", params=["mila", "beluga"])
def cluster(request):
    """A fixture that makes all the tests run on all the clusters!"""
    host: str = request.param
    if host != CURRENT_CLUSTER.name.lower():
        pytest.skip(f"Runs on the {host} cluster (we're on {CURRENT_CLUSTER.name})")
    yield host
