import os
import shutil

import pytest

from mila_datamodules.clusters import SLURM_TMPDIR


@pytest.fixture(autouse=True, scope="session")
def clear_slurm_tmpdir():
    """Clears the SLURM_TMPDIR before running any tests.

    This is unfortunately necessary so that each test run is unaffected by other runs. I was having
    weird errors because the datasets were half-moved to SLURM_TMPDIR, so reading them on the next
    run would fail.
    """
    if (SLURM_TMPDIR / "data").exists():
        # NOTE: The `chmod_recursive` doesn't seem to work atm. It's safe to assume we're on Linux,
        # so this is fine for now.
        # chmod_recursive(SLURM_TMPDIR / "data", 0o644)
        os.system(f"chmod --recursive +rwx {SLURM_TMPDIR}/data")
        shutil.rmtree(SLURM_TMPDIR / "data")
    yield
