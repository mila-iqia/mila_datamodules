"""Set of functions for creating torchvision datasets when on the Mila cluster.

IDEA: later on, we could also add some functions for loading torchvision models from a cached
directory.
"""
from __future__ import annotations

import functools
import os
from logging import getLogger as get_logger
from pathlib import Path
from shutil import which
from typing import TypeVar

T = TypeVar("T")

logger = get_logger(__name__)


def on_slurm_cluster() -> bool:
    return on_real_slurm_cluster() or on_fake_slurm_cluster()


@functools.cache
def on_real_slurm_cluster() -> bool:
    """Return `True` if the current process is running on a SLURM cluster."""
    return which("srun") is not None


def on_fake_slurm_cluster() -> bool:
    return (
        not on_real_slurm_cluster()
        and "FAKE_SCRATCH" in os.environ
        and "FAKE_SLURM_TMPDIR" in os.environ
    )


def current_cluster_name() -> str | None:
    if "CC_CLUSTER" in os.environ:
        return os.environ["CC_CLUSTER"]
    if Path("/home/mila").exists():
        return "mila"
    return None


def on_compute_node() -> bool:
    return on_slurm_cluster() and ("SLURM_JOB_ID" in os.environ or "SLURM_JOBID" in os.environ)


def on_login_node() -> bool:
    return on_slurm_cluster() and not on_compute_node()


def in_job_but_not_in_job_step_so_no_slurm_env_vars() -> bool:
    """Returns `True` if this process is being executed inside another shell of the job (e.g. when
    using `mila code`, the vscode shell doesn't have the SLURM environment variables set)."""
    if not on_slurm_cluster():
        return False
    return "SLURM_JOB_ID" in os.environ and "SLURM_TMPDIR" not in os.environ


def get_scratch_dir(default: str | Path | None = None) -> Path:
    """Returns the path to the scratch directory on the current cluster, or `default` otherwise."""
    return Path(_get_env_var("SCRATCH", default=default))


def get_slurm_tmpdir(default: str | Path | None = None) -> Path:
    """Returns the path to the SLURM_TMPDIR directory on the current cluster, or `default` when not
    on a cluster."""
    # NOTE: This variable is a little bit different.
    return Path(_get_env_var("SLURM_TMPDIR", default=default))


def _get_env_var(
    var_name: str, default: T | None = None, mock_var_prefix: str = "FAKE_"
) -> str | T:
    if in_job_but_not_in_job_step_so_no_slurm_env_vars():
        from mila_datamodules.clusters.env_variables import run_job_step_to_get_slurm_env_variables

        run_job_step_to_get_slurm_env_variables()
    if var_name in os.environ:
        return os.environ[var_name]
    if default is not None:
        return default
    fake_var_name = f"{mock_var_prefix}{var_name}"
    if fake_var_name in os.environ:
        return os.environ[fake_var_name]
    raise RuntimeError(
        f"Could not retrieve the {var_name} environment variable. If running outside a SLURM "
        f"cluster, either pass a value for the `default` argument, or set the `{fake_var_name}` "
        f"environment variable."
    )
