"""Environment variables that are set on SLURM clusters."""
import functools
import ipaddress
import os
import subprocess
import tempfile
from logging import getLogger as get_logger
from pathlib import Path
from typing import Optional, Sequence

import pydantic
from filelock import FileLock
from pydantic import BaseSettings
from pydantic.dataclasses import dataclass
from pydantic.fields import Field

logger = get_logger(__name__)


@dataclass(frozen=True, init=False)
class SlurmEnvVariables(BaseSettings):
    """Slurm environment variables that this package cares about.

    TODO: To use the package outside a SLURM cluster, the users should only have to set these
    variables locally.
    """

    # TODO: Figure out a good default value to use for these variables when not on a SLURM cluster.
    SCRATCH: Path
    SLURM_TMPDIR: Path
    SLURM_JOBID: int
    SLURM_CLUSTER_NAME: str = Field(
        default_factory=lambda: os.environ.get(
            "SLURM_CLUSTER_NAME", os.environ.get("SLURM_WORKING_CLUSTER", ":").split(":")[0]
        )
    )

    # ------------
    # Other environment variables that we don't currently use, but are set on SLURM clusters.
    # NOTE: Having all these variables isn't really useful at the moment, and it will make it more
    # difficult to "mock" this when running on a local machine. There again, it might be fine to
    # only use this class when in a SLURM cluster.
    # ------------

    SLURM_CONF: Path
    SLURM_PRIO_PROCESS: int
    SLURM_UMASK: str
    SLURM_JOB_NAME: str
    SLURM_JOB_CPUS_PER_NODE: int = 1  # NOTE: Appears to be missing when in a second `srun` step.
    SLURM_NTASKS: int
    SLURM_NPROCS: int
    SLURM_JOB_ID: int
    SLURM_STEP_ID: int
    SLURM_STEPID: int
    SLURM_NNODES: int
    SLURM_NODELIST: str
    # NOTE: The Literal would work, but the partitions keep changing over time, so it's probably
    # best to just keep it as a string.
    SLURM_JOB_PARTITION: str  # Literal["unkillable-cpu", "main", "unkillable", "long"]
    SLURM_TASKS_PER_NODE: int  # = 1
    SLURM_SRUN_COMM_PORT: int
    SLURM_JOB_UID: int
    SLURM_JOB_USER: str
    SLURM_WORKING_CLUSTER: str
    SLURM_JOB_NODELIST: str
    SLURM_STEP_NODELIST: str
    SLURM_STEP_NUM_NODES: int
    SLURM_STEP_NUM_TASKS: int
    SLURM_STEP_TASKS_PER_NODE: int
    SLURM_STEP_LAUNCHER_PORT: int
    SLURM_SRUN_COMM_HOST: ipaddress.IPv4Address
    SLURM_TOPOLOGY_ADDR: str
    SLURM_TOPOLOGY_ADDR_PATTERN: str
    SLURM_CPUS_ON_NODE: int
    SLURM_CPU_BIND: str
    SLURM_CPU_BIND_LIST: str
    SLURM_CPU_BIND_TYPE: str
    SLURM_CPU_BIND_VERBOSE: str
    SLURM_TASK_PID: int
    SLURM_NODEID: int
    SLURM_PROCID: int
    SLURM_LOCALID: int
    SLURM_LAUNCH_NODE_IPADDR: ipaddress.IPv4Address
    SLURM_GTIDS: int
    SLURM_JOB_GID: int
    SLURMD_NODENAME: str

    # NOTE: These variables appear to be missing when in a second `srun` step (e.g. in the console)
    # of a VSCode window create with the `mila code` command.
    SLURM_SUBMIT_DIR: Optional[Path] = None
    SLURM_SUBMIT_HOST: Optional[str] = None
    SRUN_DEBUG: int = 3
    SLURM_JOB_NUM_NODES: int = 1
    SLURM_JOB_ACCOUNT: Optional[str] = None
    SLURM_JOB_QOS: str = "normal"


@dataclass(frozen=True, init=False)
class DdpEnvVariables(BaseSettings):
    RANK: int
    LOCAL_RANK: int
    WORLD_SIZE: int
    MASTER_ADDR: ipaddress.IPv4Address
    MASTER_PORT: int


def in_ddp_context() -> bool:
    try:
        DdpEnvVariables()
        return True
    except pydantic.ValidationError:
        return False


@functools.cache
def run_job_step_to_get_slurm_env_variables(
    vars_to_ignore: Sequence[str] = (),
) -> SlurmEnvVariables:
    """Sets the slurm-related environment variables inside `os.environ` if they are not set.

    Executes `env | grep SLURM` inside a `srun --pty /bin/bash` sub-command (assuming that no other
    such command is being run). Then, extracts the variables from the outputs and sets them in
    `os.environ`, if not already present.

    if `vars_to_ignore` is provided, those variables are not set.

    Returns a `SlurmEnvVariables` object with the variables that were set.
    """
    if "SLURM_CLUSTER_NAME" in os.environ or "SLURM_WORKING_CLUSTER" in os.environ:
        # SLURM-related environment variables have already been set.
        # Extract them from `os.environ` using the BaseSettings class of pydantic.
        return SlurmEnvVariables()

    # TODO: Make sure the location for this file is correct. Also make sure this works with
    # multiple workers, as it did before when it was using the node-wide `tempfile.gettempdir()`
    # instead of the user-local `tempfile.mkdtemp()`.
    temp_dir = tempfile.mkdtemp(prefix="slurm_env_vars_")
    if "SLURM_JOBID" in os.environ:
        # Only the SLURM_JOBID is set when running this in the VsCode console window created with
        # `mila code`, for example.
        SLURM_JOBID = os.environ["SLURM_JOBID"]
        temp_file = Path(temp_dir) / f"env_vars_{SLURM_JOBID}.txt"
    else:
        SLURM_JOBID = None
        temp_file = Path(temp_dir) / "env_vars_temp.txt"

    # Using a lockfile so that even when running this with multiple workers, they share a single
    # call to the `srun` command.
    print(f"Using temp file: {temp_file}")
    with FileLock(temp_file.with_suffix(".lock")):
        if temp_file.exists():
            # We are not the first process to run this function. We can just read the file that was
            # created by another process.
            lines = temp_file.read_text().splitlines()
        else:
            # Extract only the slurm-related environment variables.
            # TODO: Extracting all env variables, not just the slurm-related ones could be useful,
            # for example when running multi-node or multi-gpu jobs.
            logger.info("Extracting SLURM environment variables... ")
            # NOTE: Need to pass --account for DRAC cluster!
            command = "srun --overlap env | grep SLURM"
            try:
                with temp_file.open("w") as f:
                    logger.debug(f"Using temp file: {temp_file}")
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

    if not lines:
        raise RuntimeError(
            f"The temporary file with the output of the `srun` command ({temp_file!r}) is empty!"
        )

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

    # Use the values in `os.environ` to populate this object.
    return SlurmEnvVariables()
