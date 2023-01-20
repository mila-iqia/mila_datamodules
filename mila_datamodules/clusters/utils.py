"""Set of functions for creating torchvision datasets when on the Mila cluster.

IDEA: later on, we could also add some functions for loading torchvision models from a cached
directory.
"""
from __future__ import annotations
import os
from logging import getLogger as get_logger
from pathlib import Path
from typing import TypeVar

from mila_datamodules.clusters.cluster import on_compute_node, on_slurm_cluster
from mila_datamodules.clusters.env_variables import SlurmEnvVariables, setup_slurm_env_variables

T = TypeVar("T")

logger = get_logger(__name__)


def in_job_process_without_slurm_env_vars() -> bool:
    """Returns `True` if this process is being executed inside another shell of the job (e.g. when
    using `mila code`, the vscode shell doesn't have the SLURM environment variables set).
    """
    if not on_slurm_cluster():
        return False
    return "SLURM_JOB_ID" in os.environ and "SLURM_TMPDIR" not in os.environ


def get_scratch_dir(default: str | Path | None = None) -> Path:
    """Returns the path to the scratch directory on the current cluster, or `default` otherwise.
    If the current machine is not on the Mila cluster, returns `default`.
    """
    if in_job_process_without_slurm_env_vars():
        setup_slurm_env_variables()
    return Path(_get_env_var("SCRATCH", default=default))


def get_slurm_tmpdir(default: str | Path | None = None) -> Path:
    """Returns the path to the SLURM_TMPDIR directory on the current cluster, or `default` when not
    on a cluster.
    """
    # NOTE: This variable is a little bit different.
    if in_job_process_without_slurm_env_vars():
        setup_slurm_env_variables()
    return Path(_get_env_var("SLURM_TMPDIR", default=default))


def _get_env_var(
    var_name: str, default: T | None = None, fake_var_prefix: str = "FAKE_"
) -> str | T:
    if var_name in os.environ:
        return os.environ[var_name]
    if default is not None:
        return default
    fake_var_name = f"{fake_var_prefix}{var_name}"
    if fake_var_name in os.environ:
        return os.environ[fake_var_name]
    raise RuntimeError(
        f"Could not retrieve the {var_name} environment variable. If running outside a SLURM "
        f"cluster, either pass a value for the `default` argument, or set the `{fake_var_name}` "
        f"environment variable."
    )
