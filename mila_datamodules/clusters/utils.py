"""Set of functions for creating torchvision datasets when on the Mila cluster.

IDEA: later on, we could also add some functions for loading torchvision models from a cached
directory.
"""
from __future__ import annotations
from dataclasses import dataclass, fields

import os
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Sequence, TypeVar
import subprocess
from torch.utils.data import Dataset
from typing_extensions import ParamSpec
import tempfile
from filelock import FileLock

D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
C = Callable[P, D]

logger = get_logger(__name__)


@dataclass(frozen=True)
class SlurmEnvVariables:
    """Slurm environment variables that this package cares about.

    TODO: To use the package outside a SLURM cluster, the users should only have to set these
    variables locally.
    """

    SCRATCH: Path = Path(os.environ.get("SCRATCH", Path.cwd() / "scratch"))

    SLURM_JOBID: int = int(os.environ.get("SLURM_JOBID", 0))

    SLURM_TMPDIR: Path = Path(
        os.environ["SLURM_TMPDIR"]
        if "SLURM_TMPDIR" in os.environ
        else tempfile.gettempdir()
    )

    SLURM_CLUSTER_NAME: str = os.environ.get("SLURM_CLUSTER_NAME", "local")


def setup_slurm_env_variables(vars_to_ignore: Sequence[str] = ()) -> SlurmEnvVariables:
    """Sets the slurm-related environment variables inside `os.environ` if they are not set.

    Executes `env | grep SLURM` inside a `srun --pty /bin/bash` sub-command (assuming that no other
    such command is being run). Then, extracts the variables from the outputs and sets them in
    `os.environ`, if not already present.

    if `vars_to_ignore` is provided, those variables are not set.

    Returns a `SlurmEnvVariables` object with the variables that were set.
    """
    if "SLURM_CLUSTER_NAME" in os.environ:
        # SLURM-related environment variables have already been set. Ignoring.
        return SlurmEnvVariables()

    temp_dir = tempfile.gettempdir()
    if "SLURM_JOBID" in os.environ:
        # Only the SLURM_JOBID is set when running this with `mila code`, for example.
        SLURM_JOBID = os.environ["SLURM_JOBID"]
        temp_file = Path(temp_dir) / f"env_vars_{SLURM_JOBID}.txt"
    else:
        SLURM_JOBID = None
        temp_file = Path(temp_dir) / "env_vars_temp.txt"

    # Using a lockfile so that even when running this with multiple workers, they share a single
    # call to the `srun` command.
    with FileLock(temp_file.with_suffix(".lock")):
        if temp_file.exists():
            # We are not the first process to run this function. We can just read the file that was
            # created by another process.
            lines = temp_file.read_text().splitlines()
        else:
            # Extract only the slurm-related environment variables.
            # TODO: Extracting all env variables, not just the slurm-related ones could be useful,
            # for example when running multi-node or multi-gpu jobs.
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
                    "Unable to extract SLURM environment variables. Check that there isn't "
                    "already a `srun --pty /bin/bash` command running (there can only be one at "
                    "any given time)."
                )
            except subprocess.CalledProcessError:
                temp_file.unlink(missing_ok=True)
                raise NotImplementedError(
                    "Unable to extract SLURM environment variables. This package only currently "
                    "works on SLURM clusters. In the near future, we will add support for running "
                    "this outside a cluster by setting the environment variables in the "
                    f"{SlurmEnvVariables.__name__} class locally."
                )

    assert lines

    if SLURM_JOBID is None:
        # Set the SLURM_JOBID, and rename this file to the proper name.
        for line in lines:
            if line.startswith("SLURM_JOBID="):
                _, _, SLURM_JOBID_str = line.strip().partition("=")
                SLURM_JOBID = int(SLURM_JOBID_str)
                temp_file.rename(Path(temp_dir) / f"env_vars_{SLURM_JOBID}.txt")
                break

    # Read and copy the environment variables.
    for line in lines:
        key, _, value = line.partition("=")
        if key in vars_to_ignore:
            continue
        logger.debug(f"Setting {line}")
        os.environ.setdefault(key, value)

    return SlurmEnvVariables()
