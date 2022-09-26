"""Set of functions for creating torchvision datasets when on the Mila cluster.

IDEA: later on, we could also add some functions for loading torchvision models from a cached
directory.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from torch.utils.data import Dataset
from typing_extensions import ParamSpec

D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
C = Callable[P, D]

logger = get_logger(__name__)


def setup_slurm_env_variables(vars_to_ignore: Sequence[str] = ()) -> None:
    """Sets the slurm-related environment variables inside the current shell if they are not set.

    Executes `env | grep SLURM` inside a `srun --pty /bin/bash` sub-command (assuming that no other
    such command is being run). Then, extracts the variables from the outputs and sets them in
    `os.environ`, if not already present.

    if `vars_to_ignore` is provided, those variables are not set.
    """
    if "SLURM_CLUSTER_NAME" in os.environ:
        # SLURM-related environment variables have already been set. Ignoring.
        return
    # TODO: Having issues when running this with multiple processes, e.g. when using `pytest -n 4`.
    # Perhaps we could store a simple job_{SLURM_JOBID}.txt file with the environment variables,
    # and reuse it between workers?

    temp_dir = tempfile.gettempdir()
    if "SLURM_JOBID" in os.environ:
        SLURM_JOBID = os.environ["SLURM_JOBID"]
        temp_file = Path(temp_dir) / f"env_vars_{SLURM_JOBID}.txt"
    else:
        SLURM_JOBID = None
        temp_file = Path(temp_dir) / "env_vars_temp.txt"

    if temp_file.exists():
        lines = temp_file.read_text().splitlines()
        if SLURM_JOBID is None:
            # Set the SLURM_JOBID, and rename this file to the proper name.
            for line in lines:
                if line.startswith("SLURM_JOBID="):
                    _, _, SLURM_JOBID_str = line.strip().partition("=")
                    SLURM_JOBID = int(SLURM_JOBID_str)
                    temp_file.rename(Path(temp_dir) / f"env_vars_{SLURM_JOBID}.txt")
                    break
    else:
        command = "srun env | grep SLURM"
        logger.info("Extracting SLURM environment variables... ")
        try:
            with temp_file.open("w") as f:
                logger.debug(f"> {command}")
                subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    timeout=5,  # max 5 seconds (this is plenty as far as I can tell).
                    stdout=f,
                )
                lines = temp_file.read_text().split()
                logger.info("done!")

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Unable to extract SLURM environment variables. Check that there isn't already a "
                "`srun --pty /bin/bash` command running (there can only be one at any given time)."
            )

    assert lines
    # Read and copy the environment variables.
    for line in lines:
        key, _, value = line.partition("=")
        if key in vars_to_ignore:
            continue
        logger.debug(f"Setting {line}")
        os.environ.setdefault(key, value)
